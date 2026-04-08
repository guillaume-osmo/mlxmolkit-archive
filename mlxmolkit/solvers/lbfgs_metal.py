"""
GPU-parallel L-BFGS on Metal via MLX.

Two modes:
  1. Single: L-BFGS direction for one molecule (replaces numpy _lbfgs_direction)
  2. Batched: L-BFGS directions for N molecules simultaneously (fully vectorized)

The two-loop recursion is expressed as pure MLX array ops, running entirely
on Metal GPU. For N molecules with history m and d DOF:
  - Forward loop: m iterations of dot products + axpy, all N in parallel
  - Backward loop: same pattern
  - Total: O(m × N × d) — fully vectorized, no Python per-molecule loops

Based on:
  - Nocedal & Wright, "Numerical Optimization" §7.2 (L-BFGS)
  - Columbia GPU L-BFGS-B paper: panel-panel multiplication via parallel reduction
  - Cuby4 PP-LBFGS: sparse Hessian preconditioning
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx


def lbfgs_direction(
    grad: mx.array,
    s_hist: list[mx.array],
    y_hist: list[mx.array],
    rho_hist: list[mx.array] | list[float],
) -> mx.array:
    """L-BFGS two-loop recursion on MLX Metal.

    Computes d = -H_k · g where H_k is the implicit inverse Hessian
    from the last m (s, y) pairs. H_0 = γI (scaled identity).

    Args:
        grad: (d,) gradient vector
        s_hist: list of (d,) step vectors s_i = x_{i+1} - x_i (newest last)
        y_hist: list of (d,) gradient change vectors y_i = g_{i+1} - g_i
        rho_hist: list of scalars ρ_i = 1 / (y_i · s_i)

    Returns:
        d: (d,) search direction = -H_k · g
    """
    m = len(s_hist)
    if m == 0:
        return -grad

    q = grad  # don't copy — MLX is lazy
    alpha = [None] * m

    # Forward loop: newest to oldest
    for i in range(m - 1, -1, -1):
        rho_i = rho_hist[i] if isinstance(rho_hist[i], mx.array) else mx.array(rho_hist[i])
        a = rho_i * mx.sum(s_hist[i] * q)
        alpha[i] = a
        q = q - a * y_hist[i]

    # Initial Hessian: H_0 = γI
    sy = mx.sum(s_hist[-1] * y_hist[-1])
    yy = mx.sum(y_hist[-1] * y_hist[-1])
    gamma = sy / (yy + 1e-30)
    r = gamma * q

    # Backward loop: oldest to newest
    for i in range(m):
        rho_i = rho_hist[i] if isinstance(rho_hist[i], mx.array) else mx.array(rho_hist[i])
        beta = rho_i * mx.sum(y_hist[i] * r)
        r = r + s_hist[i] * (alpha[i] - beta)

    return -r


def lbfgs_direction_batch(
    grads: mx.array,
    s_hist: mx.array,
    y_hist: mx.array,
    rho_hist: mx.array,
) -> mx.array:
    """Batched L-BFGS two-loop recursion — N molecules in parallel on Metal.

    All N molecules share the same history depth m but can have different
    (s, y, ρ) values. The two-loop recursion is fully vectorized.

    Args:
        grads: (N, d) gradient vectors for N molecules
        s_hist: (N, m, d) step history (newest at index m-1)
        y_hist: (N, m, d) gradient change history
        rho_hist: (N, m) inverse curvature scalars

    Returns:
        directions: (N, d) search directions for all N molecules
    """
    N, m, d = s_hist.shape

    q = grads  # (N, d)
    alpha_list = []

    # Forward loop: i = m-1 ... 0 (newest to oldest)
    for i in range(m - 1, -1, -1):
        # α_i = ρ_i × (s_i · q) for each molecule
        a = rho_hist[:, i] * mx.sum(s_hist[:, i, :] * q, axis=1)  # (N,)
        alpha_list.append(a)  # store in reverse order
        q = q - a[:, None] * y_hist[:, i, :]  # (N, d)

    alpha_list.reverse()  # now alpha_list[i] corresponds to history index i

    # Initial Hessian: γ = (s · y) / (y · y) per molecule (newest pair)
    sy = mx.sum(s_hist[:, -1, :] * y_hist[:, -1, :], axis=1)  # (N,)
    yy = mx.sum(y_hist[:, -1, :] * y_hist[:, -1, :], axis=1)  # (N,)
    gamma = sy / (yy + 1e-30)  # (N,)
    r = gamma[:, None] * q  # (N, d)

    # Backward loop: i = 0 ... m-1 (oldest to newest)
    for i in range(m):
        beta = rho_hist[:, i] * mx.sum(y_hist[:, i, :] * r, axis=1)  # (N,)
        r = r + s_hist[:, i, :] * (alpha_list[i] - beta)[:, None]  # (N, d)

    return -r


class BatchLBFGS:
    """Batched L-BFGS optimizer for N molecules on Metal.

    Maintains per-molecule history and computes all directions in parallel.

    Usage:
        opt = BatchLBFGS(N=100, d=30, m=10)
        for step in range(max_steps):
            # energy_grad_fn computes all N in one Metal dispatch
            energies, grads = energy_grad_fn(positions)
            directions = opt.step(positions, grads)
            # line search (batched)
            positions = positions + alpha * directions
    """

    def __init__(self, N: int, d: int, m: int = 10):
        """
        Args:
            N: number of molecules
            d: degrees of freedom per molecule (3 * n_atoms)
            m: L-BFGS history depth
        """
        self.N = N
        self.d = d
        self.m = m
        self.iteration = 0

        # History buffers: (N, m, d) — circular buffer
        self._s = mx.zeros((N, m, d))
        self._y = mx.zeros((N, m, d))
        self._rho = mx.zeros((N, m))
        self._filled = 0  # how many history slots are actually filled

        # Previous state
        self._prev_x = None
        self._prev_g = None

    def step(
        self,
        x: mx.array,
        grad: mx.array,
        converged_mask: mx.array | None = None,
    ) -> mx.array:
        """Compute L-BFGS search directions for all N molecules.

        Args:
            x: (N, d) current positions
            grad: (N, d) current gradients
            converged_mask: (N,) bool — skip converged molecules

        Returns:
            directions: (N, d) search directions
        """
        if self._prev_x is not None:
            # Update history
            s_k = x - self._prev_x  # (N, d)
            y_k = grad - self._prev_g  # (N, d)
            sy = mx.sum(s_k * y_k, axis=1)  # (N,)

            # Only update if sy > 0 (curvature condition)
            valid = sy > 1e-20
            rho_k = mx.where(valid, 1.0 / (sy + 1e-30), mx.zeros_like(sy))

            # Shift history: drop oldest, append newest
            if self._filled < self.m:
                idx = self._filled
                self._filled += 1
            else:
                # Shift left (drop oldest at index 0)
                self._s = mx.concatenate([self._s[:, 1:, :], s_k[:, None, :]], axis=1)
                self._y = mx.concatenate([self._y[:, 1:, :], y_k[:, None, :]], axis=1)
                self._rho = mx.concatenate([self._rho[:, 1:], rho_k[:, None]], axis=1)
                idx = -1  # already appended

            if idx >= 0:
                # Place new vectors at the right index
                # Build mask for this slot and update
                mask_3d = mx.zeros_like(self._s)
                mask_3d = mask_3d.at[:, idx, :].add(mx.ones((self.N, self.d)))
                self._s = mx.where(mask_3d > 0, mx.broadcast_to(s_k[:, None, :], mask_3d.shape), self._s)
                self._y = mx.where(mask_3d > 0, mx.broadcast_to(y_k[:, None, :], mask_3d.shape), self._y)

                mask_1d = mx.zeros_like(self._rho)
                mask_1d = mask_1d.at[:, idx].add(mx.ones(self.N))
                self._rho = mx.where(mask_1d > 0, mx.broadcast_to(rho_k[:, None], mask_1d.shape), self._rho)

        self._prev_x = x
        self._prev_g = grad

        # Compute directions
        if self._filled == 0:
            # First iteration: steepest descent
            return -grad

        # Use only filled portion of history
        k = self._filled
        directions = lbfgs_direction_batch(
            grad,
            self._s[:, :k, :],
            self._y[:, :k, :],
            self._rho[:, :k],
        )

        # Zero out converged molecules
        if converged_mask is not None:
            directions = mx.where(converged_mask[:, None], mx.zeros_like(directions), directions)

        self.iteration += 1
        return directions

    def reset(self):
        """Clear history (e.g., when restarting)."""
        self._s = mx.zeros_like(self._s)
        self._y = mx.zeros_like(self._y)
        self._rho = mx.zeros_like(self._rho)
        self._filled = 0
        self._prev_x = None
        self._prev_g = None
        self.iteration = 0


def batched_backtracking_line_search(
    x: mx.array,
    d: mx.array,
    grad: mx.array,
    energy: mx.array,
    energy_fn,
    c1: float = 1e-4,
    max_ls: int = 20,
) -> tuple[mx.array, mx.array, mx.array]:
    """Batched backtracking line search for N molecules on Metal.

    Armijo condition: f(x + α·d) ≤ f(x) + c1·α·(g·d)

    Args:
        x: (N, d) positions
        d: (N, d) search directions
        grad: (N, d) gradients
        energy: (N,) current energies
        energy_fn: callable (N,d) → (N,) energies
        c1: Armijo constant
        max_ls: max line search iterations

    Returns:
        x_new: (N, d) updated positions
        e_new: (N,) new energies
        alpha: (N,) step sizes
    """
    slope = mx.sum(grad * d, axis=1)  # (N,)  directional derivative
    alpha = mx.ones(x.shape[0])       # (N,)  start with α=1

    x_new = x + alpha[:, None] * d
    e_new = energy_fn(x_new)
    mx.eval(e_new)

    sufficient = e_new <= energy + c1 * alpha * slope

    for _ in range(max_ls):
        if mx.all(sufficient):
            break
        # Halve alpha for molecules that don't satisfy Armijo
        alpha = mx.where(sufficient, alpha, 0.5 * alpha)
        x_new = x + alpha[:, None] * d
        e_new = energy_fn(x_new)
        mx.eval(e_new)
        sufficient = e_new <= energy + c1 * alpha * slope

    return x_new, e_new, alpha
