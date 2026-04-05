"""
BFGS and L-BFGS optimizers on Metal via MLX metal_kernel (JIT).

Port of nvMolKit's BFGS minimizer to Apple Metal.

BFGS: Dense inverse Hessian, O(n²) memory.
L-BFGS: Limited-memory variant, stores last m (s,y) pairs, O(mn) memory.

Both use the same cubic-interpolation backtracking line search (RDKit/nvMolKit).

Reference: Numerical Recipes, §10.7 (BFGS) and §10.6 (line search).
           Nocedal & Wright, Numerical Optimization, §7.2 (L-BFGS).
nvMolKit: https://github.com/NVIDIA-Digital-Bio/nvMolKit/tree/main/src/minimizer
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import mlx.core as mx

FUNCTOL = 1e-4
MOVETOL = 1e-7
TOLX = 4.0 * 3e-8
EPS_HESSIAN = 3e-8
MAX_LINE_SEARCH_ITERS = 100
MAX_STEP_FACTOR = 100.0

# ---------------------------------------------------------------------------
# Metal kernel sources (JIT)
# ---------------------------------------------------------------------------

_HESSIAN_IDENTITY_SOURCE = """
uint idx = thread_position_in_grid.x;
uint dim = dim_buf[0];
uint total = dim * dim;
for (uint k = idx; k < total; k += dim) {
    H[k] = H_in[k];
}
if (idx < dim) {
    H[idx * dim + idx] = 1.0f;
}
"""

_MATVEC_SOURCE = """
uint row = thread_position_in_grid.x;
uint dim = dim_buf[0];
if (row >= dim) { return; }
float sum = 0.0f;
for (uint j = 0; j < dim; j++) {
    sum += H[row * dim + j] * g[j];
}
d[row] = -sum;
"""

_BFGS_UPDATE_SOURCE = """
uint tid = thread_position_in_grid.x;
uint dim = dim_buf[0];
uint total = dim * dim;
if (tid >= total) { return; }
uint i = tid / dim;
uint j = tid % dim;

float fac_val = fac[0];
float fad_val = fad[0];
float fae_val = fae[0];

float update = fac_val * xi[i] * xi[j]
             - fad_val * hdg[i] * hdg[j]
             + fae_val * dg_upd[i] * dg_upd[j];
H_out[tid] = H[tid] + update;
"""

_COMPUTE_HDG_SOURCE = """
uint row = thread_position_in_grid.x;
uint dim = dim_buf[0];
if (row >= dim) { return; }
float sum = 0.0f;
for (uint j = 0; j < dim; j++) {
    sum += H[row * dim + j] * dg[j];
}
hdg[row] = sum;
"""

_DOT_PRODUCT_SOURCE = """
uint tid = thread_position_in_grid.x;
uint n = n_buf[0];
if (tid >= n) { return; }
out[tid] = a[tid] * b[tid];
"""

_AXPY_SOURCE = """
uint tid = thread_position_in_grid.x;
uint n = n_buf[0];
if (tid >= n) { return; }
float alpha_val = alpha[0];
y[tid] = x[tid] + alpha_val * d[tid];
"""


# ---------------------------------------------------------------------------
# Kernel cache
# ---------------------------------------------------------------------------

_kernel_cache = {}


def _get_kernel(name, source, input_names, output_names):
    key = name
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Kernel wrappers
# ---------------------------------------------------------------------------

def _set_hessian_identity(dim: int) -> mx.array:
    """Initialize dim x dim identity Hessian."""
    H = mx.zeros((dim * dim,), dtype=mx.float32)
    mx.eval(H)
    k = _get_kernel("bfgs_hessian_id", _HESSIAN_IDENTITY_SOURCE,
                     ["H_in", "dim_buf"], ["H"])
    H = k(
        inputs=[H, mx.array([dim], dtype=mx.uint32)],
        grid=(dim, 1, 1),
        threadgroup=(min(256, dim), 1, 1),
        output_shapes=[(dim * dim,)],
        output_dtypes=[mx.float32],
    )[0]
    return H


def _matvec_neg(H: mx.array, g: mx.array, dim: int) -> mx.array:
    """d = -H @ g (search direction)."""
    k = _get_kernel("bfgs_matvec", _MATVEC_SOURCE,
                     ["H", "g", "dim_buf"], ["d"])
    d = k(
        inputs=[H, g, mx.array([dim], dtype=mx.uint32)],
        grid=(dim, 1, 1),
        threadgroup=(min(256, dim), 1, 1),
        output_shapes=[(dim,)],
        output_dtypes=[mx.float32],
    )[0]
    return d


def _compute_hdg(H: mx.array, dg: mx.array, dim: int) -> mx.array:
    """hdg = H @ dg (Hessian times gradient difference)."""
    k = _get_kernel("bfgs_hdg", _COMPUTE_HDG_SOURCE,
                     ["H", "dg", "dim_buf"], ["hdg"])
    hdg = k(
        inputs=[H, dg, mx.array([dim], dtype=mx.uint32)],
        grid=(dim, 1, 1),
        threadgroup=(min(256, dim), 1, 1),
        output_shapes=[(dim,)],
        output_dtypes=[mx.float32],
    )[0]
    return hdg


def _elementwise_dot(a: mx.array, b: mx.array, n: int) -> mx.array:
    """Element-wise product (for reduction to dot product)."""
    k = _get_kernel("bfgs_dot", _DOT_PRODUCT_SOURCE,
                     ["a", "b", "n_buf"], ["out"])
    out = k(
        inputs=[a, b, mx.array([n], dtype=mx.uint32)],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def _bfgs_hessian_update(
    H: mx.array, xi: mx.array, hdg: mx.array, dg_upd: mx.array,
    fac: mx.array, fad: mx.array, fae: mx.array, dim: int,
) -> mx.array:
    """Rank-2 BFGS update: H += fac*xi*xi' - fad*hdg*hdg' + fae*dg_upd*dg_upd'."""
    k = _get_kernel("bfgs_update", _BFGS_UPDATE_SOURCE,
                     ["H", "xi", "hdg", "dg_upd", "fac", "fad", "fae", "dim_buf"],
                     ["H_out"])
    total = dim * dim
    H_out = k(
        inputs=[H, xi, hdg, dg_upd, fac, fad, fae,
                mx.array([dim], dtype=mx.uint32)],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
    )[0]
    return H_out


def _axpy(x: mx.array, d: mx.array, alpha: mx.array, n: int) -> mx.array:
    """y = x + alpha * d."""
    k = _get_kernel("bfgs_axpy", _AXPY_SOURCE,
                     ["x", "d", "alpha", "n_buf"], ["y"])
    y = k(
        inputs=[x, d, alpha, mx.array([n], dtype=mx.uint32)],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]
    return y


# ---------------------------------------------------------------------------
# BFGS result
# ---------------------------------------------------------------------------

@dataclass
class BfgsResult:
    x: np.ndarray
    energy: float
    grad_norm: float
    n_iters: int
    converged: bool


# ---------------------------------------------------------------------------
# Energy/gradient function type
# ---------------------------------------------------------------------------

EnergyGradFn = Callable[[mx.array], tuple[mx.array, mx.array]]


# ---------------------------------------------------------------------------
# BFGS minimizer (single system)
# ---------------------------------------------------------------------------

def _scale_grad_nvmolkit(grad_np: np.ndarray, scale_grads: bool) -> tuple[np.ndarray, float]:
    """
    Gradient scaling matching nvMolKit's scaleGradKernel exactly.

    1. Multiply gradient by 0.1 (gradScale = 0.1)
    2. Find max of the (already-scaled) gradient
    3. If max > 10.0, halve gradScale repeatedly until max*gradScale <= 10.0,
       then apply this extra scaling to the gradient

    Returns (scaled_grad, cumulative_grad_scale).
    """
    if not scale_grads:
        return grad_np.copy(), 1.0

    g = grad_np.copy()
    grad_scale = 0.1
    g *= grad_scale

    max_g = float(np.max(g))
    if max_g > 10.0:
        extra = 1.0
        while max_g * extra > 10.0:
            extra *= 0.5
        g *= extra
        grad_scale *= extra

    return g, grad_scale


def bfgs_minimize(
    x0: mx.array,
    energy_grad_fn: EnergyGradFn,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    scale_grads: bool = False,
) -> BfgsResult:
    """
    BFGS minimization on Metal.

    Matches nvMolKit's BfgsBatchMinimizer::minimize() loop:
      1. H⁻¹ = I, compute E₀, g₀
      2. scaleGrad(g₀)                     [nvMolKit: scaleGrad(preLoop=true)]
      3. d = -g_scaled                     [nvMolKit: copyAndInvert]
      4. setMaxStep()
      Loop:
        5. lineSearch(d, g_scaled, E)       [nvMolKit: doLineSearch*]
        6. setDirection(): xi=x_new-x, dGrad=g_scaled, check TOLX
        7. Compute new gradient, scaleGrad  [nvMolKit: gFunc()+scaleGrad(false)]
        8. updateDGrad(): dGrad=g_new_scaled-g_old_scaled, check gradTol
           convergence uses max(E*gradScale, 1.0) as denominator
        9. updateHessian(): BFGS rank-2 on scaled gradients
        10. d = -H⁻¹ @ g_scaled

    Args:
        x0: initial position vector, shape (dim,), float32.
        energy_grad_fn: callable(x) → (energy, grad), both mx.array.
        max_iters: maximum BFGS iterations.
        grad_tol: gradient convergence tolerance.
        scale_grads: apply nvMolKit-style gradient scaling (for force fields).

    Returns:
        BfgsResult with optimized coordinates, energy, grad norm, iterations.
    """
    dim = int(x0.shape[0])
    x = mx.array(x0, dtype=mx.float32)

    H = _set_hessian_identity(dim)
    mx.eval(H)

    energy, grad_raw = energy_grad_fn(x)
    mx.eval(energy, grad_raw)

    # --- scaleGrad(preLoop=true) ---
    grad_scaled_np, grad_scale = _scale_grad_nvmolkit(np.array(grad_raw), scale_grads)
    grad = mx.array(grad_scaled_np, dtype=mx.float32)
    mx.eval(grad)

    # --- d = -grad_scaled (copyAndInvert) ---
    d = _matvec_neg(H, grad, dim)
    mx.eval(d)

    # --- setMaxStep ---
    x_np = np.array(x)
    sum_sq = float(np.sum(x_np ** 2))
    max_step = MAX_STEP_FACTOR * max(np.sqrt(sum_sq), float(dim))

    converged = False
    n_iter = 0

    for iteration in range(max_iters):
        n_iter = iteration + 1

        # --- Scale direction if too large (initializeLineSearch) ---
        d_np = np.array(d)
        d_norm = float(np.sqrt(np.sum(d_np ** 2)))
        if d_norm > max_step:
            d = d * mx.array(max_step / d_norm, dtype=mx.float32)
            mx.eval(d)
            d_np = np.array(d)

        # --- Line search slope uses SCALED gradient ---
        slope = float(np.dot(d_np, grad_scaled_np))

        if slope >= 0:
            d = -grad
            mx.eval(d)
            d_np = np.array(d)
            slope = float(np.dot(d_np, grad_scaled_np))
            if slope >= 0:
                break

        x_np = np.array(x)
        test_vals = np.abs(d_np) / np.maximum(np.abs(x_np), 1.0)
        max_tv = float(np.max(test_vals))
        lambda_min = MOVETOL / max_tv if max_tv > 0 else 1e-12

        # --- Backtracking line search (energy is UNSCALED) ---
        lam = 1.0
        lam2 = 0.0
        f_old = float(np.array(energy).flat[0])
        f2 = 0.0

        x_new = _axpy(x, d, mx.array([lam], dtype=mx.float32), dim)
        mx.eval(x_new)

        ls_converged = False
        for ls_iter in range(MAX_LINE_SEARCH_ITERS):
            e_new, _ = energy_grad_fn(x_new)
            mx.eval(e_new)
            f_new = float(np.array(e_new).flat[0])

            if f_new - f_old <= FUNCTOL * lam * slope:
                ls_converged = True
                break

            if lam < lambda_min:
                break

            if ls_iter == 0:
                tmp_lam = -slope / (2.0 * (f_new - f_old - slope))
            else:
                rhs1 = f_new - f_old - lam * slope
                rhs2 = f2 - f_old - lam2 * slope
                dl = lam - lam2
                if abs(dl) < 1e-30:
                    tmp_lam = 0.5 * lam
                else:
                    a = (rhs1 / (lam * lam) - rhs2 / (lam2 * lam2)) / dl
                    b = (-lam2 * rhs1 / (lam * lam) + lam * rhs2 / (lam2 * lam2)) / dl
                    if a == 0.0:
                        tmp_lam = -slope / (2.0 * b)
                    else:
                        disc = b * b - 3.0 * a * slope
                        if disc < 0.0:
                            tmp_lam = 0.5 * lam
                        elif b <= 0.0:
                            tmp_lam = (-b + np.sqrt(disc)) / (3.0 * a)
                        else:
                            tmp_lam = -slope / (b + np.sqrt(disc))
                    if tmp_lam > 0.5 * lam:
                        tmp_lam = 0.5 * lam

            lam2 = lam
            f2 = f_new
            lam = max(tmp_lam, 0.1 * lam)

            x_new = _axpy(x, d, mx.array([lam], dtype=mx.float32), dim)
            mx.eval(x_new)

        # --- doLineSearchPostLoop: if not converged, revert position ---
        if not ls_converged:
            x_new = _axpy(x, d, mx.array([lam], dtype=mx.float32), dim)
            mx.eval(x_new)

        # --- setDirection: xi = x_new - x, update positions ---
        xi = x_new - x
        mx.eval(xi)
        xi_np = np.array(xi)
        x_new_np = np.array(x_new)
        test_step = float(np.max(np.abs(xi_np) / np.maximum(np.abs(x_new_np), 1.0)))

        # dGrad stores the OLD scaled gradient for later differencing
        dgrad_np = grad_scaled_np.copy()

        x = x_new
        energy = e_new

        if test_step < TOLX:
            converged = True
            break

        # --- Compute new gradient + scaleGrad ---
        _, grad_raw_new = energy_grad_fn(x)
        mx.eval(grad_raw_new)
        grad_scaled_np, grad_scale = _scale_grad_nvmolkit(np.array(grad_raw_new), scale_grads)
        grad = mx.array(grad_scaled_np, dtype=mx.float32)
        mx.eval(grad)

        # --- updateDGrad: dGrad = g_new_scaled - g_old_scaled ---
        dgrad_np = grad_scaled_np - dgrad_np

        # Convergence check: max(|g_scaled| * max(|x|, 1)) / max(E * gradScale, 1)
        e_val = float(np.array(energy).flat[0])
        den = max(e_val * grad_scale, 1.0)
        test_grad = float(np.max(
            np.abs(grad_scaled_np) * np.maximum(np.abs(x_new_np), 1.0)
        ) / den)
        if test_grad < grad_tol:
            converged = True
            break

        # --- updateHessian: BFGS rank-2 update on SCALED gradients ---
        dg = mx.array(dgrad_np, dtype=mx.float32)
        mx.eval(dg)

        hdg = _compute_hdg(H, dg, dim)
        mx.eval(hdg)

        fac_val = float(mx.sum(_elementwise_dot(dg, xi, dim)).item())
        fae_val = float(mx.sum(_elementwise_dot(dg, hdg, dim)).item())
        sum_dg = float(mx.sum(_elementwise_dot(dg, dg, dim)).item())
        sum_xi = float(mx.sum(_elementwise_dot(xi, xi, dim)).item())

        if fac_val > np.sqrt(EPS_HESSIAN * sum_dg * sum_xi):
            fac_inv = 1.0 / fac_val
            fad_inv = 1.0 / fae_val

            dg_upd = mx.array(fac_inv * np.array(xi) - fad_inv * np.array(hdg),
                              dtype=mx.float32)
            mx.eval(dg_upd)

            H = _bfgs_hessian_update(
                H, xi, hdg, dg_upd,
                mx.array([fac_inv], dtype=mx.float32),
                mx.array([fad_inv], dtype=mx.float32),
                mx.array([fae_val], dtype=mx.float32),
                dim,
            )
            mx.eval(H)

        # --- New search direction from SCALED gradient ---
        d = _matvec_neg(H, grad, dim)
        mx.eval(d)

    return BfgsResult(
        x=np.array(x),
        energy=float(np.array(energy).flat[0]),
        grad_norm=float(np.linalg.norm(grad_scaled_np)),
        n_iters=n_iter,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Shared line search (used by both BFGS and L-BFGS)
# ---------------------------------------------------------------------------

def _line_search(
    x: mx.array,
    d: mx.array,
    grad: mx.array,
    energy: mx.array,
    energy_grad_fn: EnergyGradFn,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array, bool]:
    """
    Backtracking line search with cubic interpolation.

    Returns (x_new, e_new, g_new, step_too_small).
    """
    d_np = np.array(d)
    x_np = np.array(x)
    slope = float(np.dot(d_np, np.array(grad)))

    if slope >= 0:
        return x, energy, grad, True

    test_vals = np.abs(d_np) / np.maximum(np.abs(x_np), 1.0)
    max_test = float(np.max(test_vals))
    lambda_min = MOVETOL / max_test if max_test > 0 else 1e-12

    lam = 1.0
    lam2 = 0.0
    f_old = float(energy.item())
    f2 = 0.0

    x_new = _axpy(x, d, mx.array([lam], dtype=mx.float32), dim)
    mx.eval(x_new)

    for ls_iter in range(MAX_LINE_SEARCH_ITERS):
        e_new, g_new = energy_grad_fn(x_new)
        mx.eval(e_new, g_new)
        f_new = float(e_new.item())

        if f_new - f_old <= FUNCTOL * lam * slope:
            return x_new, e_new, g_new, False

        if lam < lambda_min:
            return x, energy, grad, True

        if ls_iter == 0:
            tmp_lam = -slope / (2.0 * (f_new - f_old - slope))
        else:
            rhs1 = f_new - f_old - lam * slope
            rhs2 = f2 - f_old - lam2 * slope
            a = (rhs1 / (lam * lam) - rhs2 / (lam2 * lam2)) / (lam - lam2)
            b = (-lam2 * rhs1 / (lam * lam) + lam * rhs2 / (lam2 * lam2)) / (lam - lam2)
            if a == 0.0:
                tmp_lam = -slope / (2.0 * b)
            else:
                disc = b * b - 3.0 * a * slope
                if disc < 0.0:
                    tmp_lam = 0.5 * lam
                elif b <= 0.0:
                    tmp_lam = (-b + np.sqrt(disc)) / (3.0 * a)
                else:
                    tmp_lam = -slope / (b + np.sqrt(disc))
            if tmp_lam > 0.5 * lam:
                tmp_lam = 0.5 * lam

        lam2 = lam
        f2 = f_new
        lam = max(tmp_lam, 0.1 * lam)

        x_new = _axpy(x, d, mx.array([lam], dtype=mx.float32), dim)
        mx.eval(x_new)

    return x_new, e_new, g_new, False


# ---------------------------------------------------------------------------
# L-BFGS two-loop recursion (Nocedal & Wright Algorithm 7.4)
# ---------------------------------------------------------------------------

def _lbfgs_direction(
    grad: np.ndarray,
    s_hist: list[np.ndarray],
    y_hist: list[np.ndarray],
    rho_hist: list[float],
) -> np.ndarray:
    """
    L-BFGS two-loop recursion to compute d = -H_k * grad.

    Uses the last m (s, y) pairs to implicitly represent the inverse Hessian.
    H_0 is set as γI where γ = s'y / y'y (scaled identity).
    """
    m = len(s_hist)
    q = grad.copy()
    alpha = np.zeros(m)

    for i in range(m - 1, -1, -1):
        alpha[i] = rho_hist[i] * np.dot(s_hist[i], q)
        q -= alpha[i] * y_hist[i]

    if m > 0:
        sy = np.dot(s_hist[-1], y_hist[-1])
        yy = np.dot(y_hist[-1], y_hist[-1])
        gamma = sy / (yy + 1e-30)
    else:
        gamma = 1.0

    r = gamma * q

    for i in range(m):
        beta = rho_hist[i] * np.dot(y_hist[i], r)
        r += s_hist[i] * (alpha[i] - beta)

    return -r


def lbfgs_minimize(
    x0: mx.array,
    energy_grad_fn: EnergyGradFn,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    m: int = 10,
    scale_grads: bool = False,
) -> BfgsResult:
    """
    L-BFGS minimization on Metal, matching nvMolKit flow.

    Uses limited-memory BFGS (two-loop recursion) for the direction
    computation, with Metal kernels for energy/gradient and axpy.

    When scale_grads=True, follows nvMolKit's gradient scaling:
      - After each gradient computation, apply scaleGrad (0.1x, cap at 10)
      - All L-BFGS history (s, y) operates on SCALED gradients
      - Convergence test uses max(E * gradScale, 1.0) as denominator

    Memory: O(m * dim) instead of O(dim²) for full BFGS.

    Args:
        x0: initial position vector, shape (dim,), float32.
        energy_grad_fn: callable(x) → (energy, grad), both mx.array.
        max_iters: maximum L-BFGS iterations.
        grad_tol: gradient convergence tolerance.
        m: number of history pairs to store (default 10).
        scale_grads: apply nvMolKit-style gradient scaling (for force fields).

    Returns:
        BfgsResult with optimized coordinates, energy, grad norm, iterations.
    """
    dim = int(x0.shape[0])
    x = mx.array(x0, dtype=mx.float32)

    # --- Initial energy and gradient ---
    energy, grad_raw = energy_grad_fn(x)
    mx.eval(energy, grad_raw)

    # --- scaleGrad(preLoop=true) ---
    grad_scaled_np, grad_scale = _scale_grad_nvmolkit(np.array(grad_raw), scale_grads)

    x_np = np.array(x)
    sum_sq = float(np.sum(x_np ** 2))
    max_step = MAX_STEP_FACTOR * max(np.sqrt(sum_sq), float(dim))

    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []

    converged = False
    n_iter = 0

    for iteration in range(max_iters):
        n_iter = iteration + 1

        # --- L-BFGS direction via two-loop recursion on SCALED gradient ---
        d_np = _lbfgs_direction(grad_scaled_np, s_hist, y_hist, rho_hist)

        d_norm = float(np.linalg.norm(d_np))
        if d_norm > max_step:
            d_np *= max_step / d_norm

        d = mx.array(d_np.astype(np.float32))
        mx.eval(d)

        # --- Line search slope uses SCALED gradient ---
        slope = float(np.dot(d_np, grad_scaled_np))

        if slope >= 0:
            d_np = -grad_scaled_np
            d_norm = float(np.linalg.norm(d_np))
            if d_norm > max_step:
                d_np *= max_step / d_norm
            d = mx.array(d_np.astype(np.float32))
            mx.eval(d)
            slope = float(np.dot(d_np, grad_scaled_np))
            if slope >= 0:
                break

        x_np = np.array(x)
        test_vals = np.abs(d_np) / np.maximum(np.abs(x_np), 1.0)
        max_tv = float(np.max(test_vals))
        lambda_min = MOVETOL / max_tv if max_tv > 0 else 1e-12

        # --- Backtracking line search (energy only) ---
        lam = 1.0
        lam2 = 0.0
        f_old = float(np.array(energy).flat[0])
        f2 = 0.0

        x_new = _axpy(x, d, mx.array([lam], dtype=mx.float32), dim)
        mx.eval(x_new)

        ls_converged = False
        for ls_iter in range(MAX_LINE_SEARCH_ITERS):
            e_new, _ = energy_grad_fn(x_new)
            mx.eval(e_new)
            f_new = float(np.array(e_new).flat[0])

            if f_new - f_old <= FUNCTOL * lam * slope:
                ls_converged = True
                break

            if lam < lambda_min:
                break

            if ls_iter == 0:
                tmp_lam = -slope / (2.0 * (f_new - f_old - slope))
            else:
                rhs1 = f_new - f_old - lam * slope
                rhs2 = f2 - f_old - lam2 * slope
                dl = lam - lam2
                if abs(dl) < 1e-30:
                    tmp_lam = 0.5 * lam
                else:
                    a = (rhs1 / (lam * lam) - rhs2 / (lam2 * lam2)) / dl
                    b = (-lam2 * rhs1 / (lam * lam) + lam * rhs2 / (lam2 * lam2)) / dl
                    if a == 0.0:
                        tmp_lam = -slope / (2.0 * b)
                    else:
                        disc = b * b - 3.0 * a * slope
                        if disc < 0.0:
                            tmp_lam = 0.5 * lam
                        elif b <= 0.0:
                            tmp_lam = (-b + np.sqrt(disc)) / (3.0 * a)
                        else:
                            tmp_lam = -slope / (b + np.sqrt(disc))
                    if tmp_lam > 0.5 * lam:
                        tmp_lam = 0.5 * lam

            lam2 = lam
            f2 = f_new
            lam = max(tmp_lam, 0.1 * lam)

            x_new = _axpy(x, d, mx.array([lam], dtype=mx.float32), dim)
            mx.eval(x_new)

        if not ls_converged:
            x_new = _axpy(x, d, mx.array([lam], dtype=mx.float32), dim)
            mx.eval(x_new)

        # --- setDirection: xi = x_new - x, check step convergence ---
        xi_np = np.array(x_new) - np.array(x)
        x_new_np = np.array(x_new)
        test_step = float(np.max(np.abs(xi_np) / np.maximum(np.abs(x_new_np), 1.0)))

        dgrad_old_np = grad_scaled_np.copy()

        x = x_new
        energy = e_new

        if test_step < TOLX:
            converged = True
            break

        # --- Compute new gradient + scaleGrad ---
        _, grad_raw_new = energy_grad_fn(x)
        mx.eval(grad_raw_new)
        grad_scaled_np, grad_scale = _scale_grad_nvmolkit(
            np.array(grad_raw_new), scale_grads
        )

        # --- updateDGrad: dGrad = g_new_scaled - g_old_scaled ---
        # Convergence: max(|g_scaled| * max(|x|, 1)) / max(E * gradScale, 1)
        e_val = float(np.array(energy).flat[0])
        den = max(e_val * grad_scale, 1.0)
        test_grad = float(np.max(
            np.abs(grad_scaled_np) * np.maximum(np.abs(x_new_np), 1.0)
        ) / den)
        if test_grad < grad_tol:
            converged = True
            break

        # --- Update L-BFGS history with SCALED gradients ---
        s_k = xi_np
        y_k = grad_scaled_np - dgrad_old_np
        sy = float(np.dot(s_k, y_k))

        if sy > EPS_HESSIAN * float(np.dot(y_k, y_k)):
            if len(s_hist) >= m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s_k)
            y_hist.append(y_k)
            rho_hist.append(1.0 / sy)

    return BfgsResult(
        x=np.array(x),
        energy=float(np.array(energy).flat[0]),
        grad_norm=float(np.linalg.norm(grad_scaled_np)),
        n_iters=n_iter,
        converged=converged,
    )
