"""
Batched BFGS and L-BFGS on Metal — optimize N molecules in parallel.

Port of nvMolKit's batched BFGS backend to Apple Metal via MLX.
All molecules share a single GPU dispatch for energy/gradient computation.

Architecture:
  - Positions, gradients, energies for all molecules packed into flat arrays
  - atom_starts[i] gives the starting atom index for molecule i
  - One Metal kernel dispatch computes energy/gradient for ALL molecules
  - Python loop orchestrates BFGS iterations, per-molecule convergence tracked

nvMolKit: https://github.com/NVIDIA-Digital-Bio/nvMolKit/tree/main/src/minimizer
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import mlx.core as mx

from mlxmolkit.bfgs_metal import (
    FUNCTOL, MOVETOL, TOLX, EPS_HESSIAN,
    MAX_LINE_SEARCH_ITERS, MAX_STEP_FACTOR,
    BfgsResult, _lbfgs_direction,
)


# ---------------------------------------------------------------------------
# Batch result
# ---------------------------------------------------------------------------

@dataclass
class BatchBfgsResult:
    """Result of batched BFGS optimization."""
    positions: np.ndarray
    energies: np.ndarray
    grad_norms: np.ndarray
    n_iters: int
    converged: np.ndarray
    n_molecules: int


# ---------------------------------------------------------------------------
# Batched DG energy/gradient Metal kernel
# ---------------------------------------------------------------------------

_BATCH_DG_ENERGY_GRAD_SOURCE = """
uint tid = thread_position_in_grid.x;
uint total_atoms = total_atoms_buf[0];
uint total_pairs = total_pairs_buf[0];
uint n_mols = n_mols_buf[0];
uint dim_val = 3u;

if (tid >= total_atoms * dim_val) {{ return; }}

uint global_atom = tid / dim_val;
uint coord = tid % dim_val;

// Find which molecule this atom belongs to
uint mol_id = 0;
for (uint m = 0; m < n_mols; m++) {{
    if (global_atom < (uint)atom_starts[m + 1]) {{
        mol_id = m;
        break;
    }}
}}

uint atom_offset = (uint)atom_starts[mol_id];
uint pair_offset = (uint)pair_starts[mol_id];
uint n_pairs_mol = (uint)pair_starts[mol_id + 1] - pair_offset;
uint local_atom = global_atom - atom_offset;

float grad_val = 0.0f;
float e_contrib = 0.0f;

for (uint p = 0; p < n_pairs_mol; p++) {{
    uint pair_idx = pair_offset + p;
    uint a = (uint)pairs[pair_idx * 2] + atom_offset;
    uint b = (uint)pairs[pair_idx * 2 + 1] + atom_offset;

    bool is_a = (a == global_atom);
    bool is_b = (b == global_atom);
    if (!is_a && !is_b) continue;

    float dx = pos[a * dim_val + 0] - pos[b * dim_val + 0];
    float dy = pos[a * dim_val + 1] - pos[b * dim_val + 1];
    float dz = pos[a * dim_val + 2] - pos[b * dim_val + 2];
    float dist = sqrt(dx * dx + dy * dy + dz * dz + 1e-12f);
    float target = targets[pair_idx];
    float diff = dist - target;

    float my_coord_diff;
    if (coord == 0u) my_coord_diff = dx;
    else if (coord == 1u) my_coord_diff = dy;
    else my_coord_diff = dz;

    if (!is_a) my_coord_diff = -my_coord_diff;

    grad_val += 2.0f * diff * my_coord_diff / dist;

    if (coord == 0u && is_a) {{
        e_contrib += diff * diff;
    }}
}}

grad_out[tid] = grad_val;
if (coord == 0u) {{
    energy_parts[global_atom] = e_contrib;
}}
"""

_batch_dg_kernel = None


def _get_batch_dg_kernel():
    global _batch_dg_kernel
    if _batch_dg_kernel is None:
        _batch_dg_kernel = mx.fast.metal_kernel(
            name="batch_dg_energy_grad",
            input_names=[
                "pos", "pairs", "targets",
                "atom_starts", "pair_starts",
                "total_atoms_buf", "total_pairs_buf", "n_mols_buf",
            ],
            output_names=["grad_out", "energy_parts"],
            source=_BATCH_DG_ENERGY_GRAD_SOURCE,
            ensure_row_contiguous=True,
        )
    return _batch_dg_kernel


def make_batch_distgeom_energy_grad(
    all_pairs: list[np.ndarray],
    all_targets: list[np.ndarray],
    atom_counts: list[int],
):
    """
    Create a batched energy/gradient function for N molecules.

    Args:
        all_pairs: list of (n_pairs_i, 2) int32 arrays (local atom indices).
        all_targets: list of (n_pairs_i,) float32 target distance arrays.
        atom_counts: list of atom counts per molecule.

    Returns:
        Callable(pos_flat) → (energies, grad), where:
          pos_flat: (total_atoms * 3,) float32
          energies: (n_mols,) float32
          grad: (total_atoms * 3,) float32
    """
    n_mols = len(atom_counts)

    atom_starts = np.zeros(n_mols + 1, dtype=np.int32)
    for i, c in enumerate(atom_counts):
        atom_starts[i + 1] = atom_starts[i] + c
    total_atoms = int(atom_starts[-1])

    pair_starts = np.zeros(n_mols + 1, dtype=np.int32)
    flat_pairs = []
    flat_targets = []
    for i in range(n_mols):
        n_p = len(all_targets[i])
        pair_starts[i + 1] = pair_starts[i] + n_p
        flat_pairs.append(all_pairs[i].astype(np.int32).reshape(-1, 2))
        flat_targets.append(all_targets[i].astype(np.float32))
    total_pairs = int(pair_starts[-1])

    pairs_flat = np.concatenate(flat_pairs, axis=0).flatten()
    targets_flat = np.concatenate(flat_targets)

    pairs_mx = mx.array(pairs_flat)
    targets_mx = mx.array(targets_flat)
    atom_starts_mx = mx.array(atom_starts)
    pair_starts_mx = mx.array(pair_starts)
    total_atoms_buf = mx.array([total_atoms], dtype=mx.uint32)
    total_pairs_buf = mx.array([total_pairs], dtype=mx.uint32)
    n_mols_buf = mx.array([n_mols], dtype=mx.uint32)

    k = _get_batch_dg_kernel()
    dim3 = total_atoms * 3

    def batch_energy_grad_fn(pos_flat: mx.array) -> tuple[mx.array, mx.array]:
        grad_out, energy_parts = k(
            inputs=[
                pos_flat, pairs_mx, targets_mx,
                atom_starts_mx, pair_starts_mx,
                total_atoms_buf, total_pairs_buf, n_mols_buf,
            ],
            grid=(dim3, 1, 1),
            threadgroup=(min(256, dim3), 1, 1),
            output_shapes=[(dim3,), (total_atoms,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        return energy_parts, grad_out

    return batch_energy_grad_fn, atom_starts, total_atoms, n_mols


# ---------------------------------------------------------------------------
# Batched L-BFGS optimizer
# ---------------------------------------------------------------------------

def lbfgs_minimize_batch(
    all_pairs: list[np.ndarray],
    all_targets: list[np.ndarray],
    atom_counts: list[int],
    x0: Optional[np.ndarray] = None,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    m: int = 10,
    seed: int = 42,
) -> BatchBfgsResult:
    """
    Batched L-BFGS: optimize N molecules in parallel.

    All molecules' energy/gradient computed in a single Metal dispatch.
    Each molecule has its own L-BFGS history and convergence state.

    Args:
        all_pairs: list of (n_pairs_i, 2) int32 arrays per molecule.
        all_targets: list of (n_pairs_i,) float32 arrays per molecule.
        atom_counts: list of atom counts per molecule.
        x0: optional initial positions (total_atoms*3,) float32.
        max_iters: maximum iterations.
        grad_tol: gradient convergence tolerance.
        m: L-BFGS history size.
        seed: random seed for initial coordinates.

    Returns:
        BatchBfgsResult with per-molecule results.
    """
    batch_fn, atom_starts, total_atoms, n_mols = make_batch_distgeom_energy_grad(
        all_pairs, all_targets, atom_counts,
    )
    dim3 = total_atoms * 3

    if x0 is None:
        np.random.seed(seed)
        x0 = np.random.randn(dim3).astype(np.float32) * 0.5

    x = mx.array(x0, dtype=mx.float32)

    # Per-molecule state
    mol_slices = []
    for i in range(n_mols):
        s = atom_starts[i] * 3
        e = atom_starts[i + 1] * 3
        mol_slices.append((s, e))

    mol_dim = [e - s for s, e in mol_slices]
    mol_converged = np.zeros(n_mols, dtype=bool)

    # Per-molecule L-BFGS history
    s_hists = [[] for _ in range(n_mols)]
    y_hists = [[] for _ in range(n_mols)]
    rho_hists = [[] for _ in range(n_mols)]

    # Per-molecule max_step
    max_steps = np.zeros(n_mols)
    x_np = np.array(x)
    for i in range(n_mols):
        s, e = mol_slices[i]
        ss = float(np.sum(x_np[s:e] ** 2))
        max_steps[i] = MAX_STEP_FACTOR * max(np.sqrt(ss), float(mol_dim[i]))

    # Initial energy and gradient (single GPU dispatch for all molecules)
    energy_parts, grad = batch_fn(x)
    mx.eval(energy_parts, grad)

    # Per-molecule energies
    ep_np = np.array(energy_parts)
    mol_energies = np.zeros(n_mols)
    for i in range(n_mols):
        mol_energies[i] = float(np.sum(ep_np[atom_starts[i]:atom_starts[i + 1]]))

    n_iter = 0

    for iteration in range(max_iters):
        n_iter = iteration + 1

        if np.all(mol_converged):
            break

        grad_np = np.array(grad)
        x_np = np.array(x)

        # Compute per-molecule directions via L-BFGS two-loop recursion
        d_np = np.zeros(dim3, dtype=np.float32)
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            g_i = grad_np[s:e]
            d_i = _lbfgs_direction(g_i, s_hists[i], y_hists[i], rho_hists[i])
            d_norm = float(np.linalg.norm(d_i))
            if d_norm > max_steps[i]:
                d_i *= max_steps[i] / d_norm
            d_np[s:e] = d_i

        d = mx.array(d_np, dtype=mx.float32)
        mx.eval(d)

        # Check slopes per molecule, reset to steepest descent if needed
        slopes = np.zeros(n_mols)
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            slopes[i] = float(np.dot(d_np[s:e], grad_np[s:e]))
            if slopes[i] >= 0:
                d_np[s:e] = -grad_np[s:e]
                slopes[i] = float(np.dot(d_np[s:e], grad_np[s:e]))

        d = mx.array(d_np, dtype=mx.float32)
        mx.eval(d)

        # Compute per-molecule lambda_min
        lambda_mins = np.zeros(n_mols)
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            tv = np.abs(d_np[s:e]) / np.maximum(np.abs(x_np[s:e]), 1.0)
            mt = float(np.max(tv)) if len(tv) > 0 else 0.0
            lambda_mins[i] = MOVETOL / mt if mt > 0 else 1e-12

        # --- Batched line search ---
        lams = np.ones(n_mols)
        lam2s = np.zeros(n_mols)
        f_olds = mol_energies.copy()
        f2s = np.zeros(n_mols)
        ls_done = mol_converged.copy()

        for ls_iter in range(MAX_LINE_SEARCH_ITERS):
            if np.all(ls_done):
                break

            # Build x_new for all molecules
            x_new_np = x_np.copy()
            for i in range(n_mols):
                if ls_done[i]:
                    continue
                s, e = mol_slices[i]
                x_new_np[s:e] = x_np[s:e] + lams[i] * d_np[s:e]

            x_new = mx.array(x_new_np, dtype=mx.float32)
            mx.eval(x_new)
            ep_new, g_new = batch_fn(x_new)
            mx.eval(ep_new, g_new)
            ep_new_np = np.array(ep_new)

            new_energies = np.zeros(n_mols)
            for i in range(n_mols):
                new_energies[i] = float(np.sum(ep_new_np[atom_starts[i]:atom_starts[i + 1]]))

            for i in range(n_mols):
                if ls_done[i]:
                    continue

                f_new = new_energies[i]

                if f_new - f_olds[i] <= FUNCTOL * lams[i] * slopes[i]:
                    ls_done[i] = True
                    continue

                if lams[i] < lambda_mins[i]:
                    # Revert
                    s, e = mol_slices[i]
                    x_new_np[s:e] = x_np[s:e]
                    ls_done[i] = True
                    new_energies[i] = f_olds[i]
                    continue

                if ls_iter == 0:
                    tmp = -slopes[i] / (2.0 * (f_new - f_olds[i] - slopes[i]))
                else:
                    rhs1 = f_new - f_olds[i] - lams[i] * slopes[i]
                    rhs2 = f2s[i] - f_olds[i] - lam2s[i] * slopes[i]
                    dl = lams[i] - lam2s[i]
                    if abs(dl) < 1e-30:
                        tmp = 0.5 * lams[i]
                    else:
                        a = (rhs1 / (lams[i] ** 2) - rhs2 / (lam2s[i] ** 2)) / dl
                        b = (-lam2s[i] * rhs1 / (lams[i] ** 2) + lams[i] * rhs2 / (lam2s[i] ** 2)) / dl
                        if a == 0.0:
                            tmp = -slopes[i] / (2.0 * b)
                        else:
                            disc = b * b - 3.0 * a * slopes[i]
                            if disc < 0.0:
                                tmp = 0.5 * lams[i]
                            elif b <= 0.0:
                                tmp = (-b + np.sqrt(disc)) / (3.0 * a)
                            else:
                                tmp = -slopes[i] / (b + np.sqrt(disc))
                        if tmp > 0.5 * lams[i]:
                            tmp = 0.5 * lams[i]

                lam2s[i] = lams[i]
                f2s[i] = f_new
                lams[i] = max(tmp, 0.1 * lams[i])

        # Rebuild final x_new, e_new, g_new after line search
        x_new_np_final = x_np.copy()
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            x_new_np_final[s:e] = x_np[s:e] + lams[i] * d_np[s:e]

        x_new = mx.array(x_new_np_final, dtype=mx.float32)
        mx.eval(x_new)
        ep_new, g_new = batch_fn(x_new)
        mx.eval(ep_new, g_new)
        ep_new_np = np.array(ep_new)
        g_new_np = np.array(g_new)

        # Per-molecule post-step updates
        x_new_np = np.array(x_new)
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]

            xi_i = x_new_np[s:e] - x_np[s:e]
            test_step = float(np.max(np.abs(xi_i) / np.maximum(np.abs(x_new_np[s:e]), 1.0)))
            if test_step < TOLX:
                mol_converged[i] = True
                continue

            new_e = float(np.sum(ep_new_np[atom_starts[i]:atom_starts[i + 1]]))
            mol_energies[i] = new_e
            g_i_new = g_new_np[s:e]
            den = max(new_e, 1.0)
            test_grad = float(
                np.max(np.abs(g_i_new) * np.maximum(np.abs(x_new_np[s:e]), 1.0) / den)
            )
            if test_grad < grad_tol:
                mol_converged[i] = True
                continue

            # Update L-BFGS history
            s_k = xi_i
            y_k = g_i_new - grad_np[s:e]
            sy = float(np.dot(s_k, y_k))

            if sy > EPS_HESSIAN * float(np.dot(y_k, y_k)):
                if len(s_hists[i]) >= m:
                    s_hists[i].pop(0)
                    y_hists[i].pop(0)
                    rho_hists[i].pop(0)
                s_hists[i].append(s_k.copy())
                y_hists[i].append(y_k.copy())
                rho_hists[i].append(1.0 / sy)

        x = x_new
        grad = g_new

    # Final per-molecule results
    x_final = np.array(x)
    g_final = np.array(grad)
    ep_final_np = np.array(mx.eval(batch_fn(x)[0]) or batch_fn(x)[0])

    # Recompute final energies
    ep_out, _ = batch_fn(x)
    mx.eval(ep_out)
    ep_np_final = np.array(ep_out)

    energies_out = np.zeros(n_mols)
    grad_norms_out = np.zeros(n_mols)
    for i in range(n_mols):
        energies_out[i] = float(np.sum(ep_np_final[atom_starts[i]:atom_starts[i + 1]]))
        s, e = mol_slices[i]
        grad_norms_out[i] = float(np.linalg.norm(g_final[s:e]))

    return BatchBfgsResult(
        positions=x_final,
        energies=energies_out,
        grad_norms=grad_norms_out,
        n_iters=n_iter,
        converged=mol_converged,
        n_molecules=n_mols,
    )


# ---------------------------------------------------------------------------
# Batched full BFGS optimizer
# ---------------------------------------------------------------------------

def bfgs_minimize_batch(
    all_pairs: list[np.ndarray],
    all_targets: list[np.ndarray],
    atom_counts: list[int],
    x0: Optional[np.ndarray] = None,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    seed: int = 42,
) -> BatchBfgsResult:
    """
    Batched full BFGS: optimize N molecules in parallel with dense inverse Hessians.

    Each molecule has its own dim×dim inverse Hessian. Energy/gradient for all
    molecules computed in a single Metal dispatch.

    Args:
        all_pairs: list of (n_pairs_i, 2) int32 arrays per molecule.
        all_targets: list of (n_pairs_i,) float32 arrays per molecule.
        atom_counts: list of atom counts per molecule.
        x0: optional initial positions (total_atoms*3,) float32.
        max_iters: maximum iterations.
        grad_tol: gradient convergence tolerance.
        seed: random seed for initial coordinates.

    Returns:
        BatchBfgsResult with per-molecule results.
    """
    batch_fn, atom_starts, total_atoms, n_mols = make_batch_distgeom_energy_grad(
        all_pairs, all_targets, atom_counts,
    )
    dim3 = total_atoms * 3

    if x0 is None:
        np.random.seed(seed)
        x0 = np.random.randn(dim3).astype(np.float32) * 0.5

    x = mx.array(x0, dtype=mx.float32)

    mol_slices = []
    for i in range(n_mols):
        s = atom_starts[i] * 3
        e = atom_starts[i + 1] * 3
        mol_slices.append((s, e))

    mol_dim = [e - s for s, e in mol_slices]
    mol_converged = np.zeros(n_mols, dtype=bool)

    # Per-molecule dense inverse Hessian (NumPy for simplicity, GPU for energy)
    H_list = [np.eye(d, dtype=np.float32) for d in mol_dim]

    max_steps = np.zeros(n_mols)
    x_np = np.array(x)
    for i in range(n_mols):
        s, e = mol_slices[i]
        ss = float(np.sum(x_np[s:e] ** 2))
        max_steps[i] = MAX_STEP_FACTOR * max(np.sqrt(ss), float(mol_dim[i]))

    energy_parts, grad = batch_fn(x)
    mx.eval(energy_parts, grad)
    ep_np = np.array(energy_parts)

    mol_energies = np.zeros(n_mols)
    for i in range(n_mols):
        mol_energies[i] = float(np.sum(ep_np[atom_starts[i]:atom_starts[i + 1]]))

    n_iter = 0

    for iteration in range(max_iters):
        n_iter = iteration + 1
        if np.all(mol_converged):
            break

        grad_np = np.array(grad)
        x_np = np.array(x)

        # Per-molecule directions: d = -H @ g
        d_np = np.zeros(dim3, dtype=np.float32)
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            d_i = -H_list[i] @ grad_np[s:e]
            d_norm = float(np.linalg.norm(d_i))
            if d_norm > max_steps[i]:
                d_i *= max_steps[i] / d_norm
            d_np[s:e] = d_i

        slopes = np.zeros(n_mols)
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            slopes[i] = float(np.dot(d_np[s:e], grad_np[s:e]))
            if slopes[i] >= 0:
                d_np[s:e] = -grad_np[s:e]
                slopes[i] = float(np.dot(d_np[s:e], grad_np[s:e]))

        lambda_mins = np.zeros(n_mols)
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            tv = np.abs(d_np[s:e]) / np.maximum(np.abs(x_np[s:e]), 1.0)
            mt = float(np.max(tv)) if len(tv) > 0 else 0.0
            lambda_mins[i] = MOVETOL / mt if mt > 0 else 1e-12

        # Batched line search (same as L-BFGS batch)
        lams = np.ones(n_mols)
        lam2s = np.zeros(n_mols)
        f_olds = mol_energies.copy()
        f2s = np.zeros(n_mols)
        ls_done = mol_converged.copy()

        for ls_iter in range(MAX_LINE_SEARCH_ITERS):
            if np.all(ls_done):
                break

            x_new_np = x_np.copy()
            for i in range(n_mols):
                if ls_done[i]:
                    continue
                s, e = mol_slices[i]
                x_new_np[s:e] = x_np[s:e] + lams[i] * d_np[s:e]

            x_new = mx.array(x_new_np, dtype=mx.float32)
            mx.eval(x_new)
            ep_new, g_new_ls = batch_fn(x_new)
            mx.eval(ep_new, g_new_ls)
            ep_new_np = np.array(ep_new)

            new_energies = np.zeros(n_mols)
            for i in range(n_mols):
                new_energies[i] = float(np.sum(ep_new_np[atom_starts[i]:atom_starts[i + 1]]))

            for i in range(n_mols):
                if ls_done[i]:
                    continue
                f_new = new_energies[i]
                if f_new - f_olds[i] <= FUNCTOL * lams[i] * slopes[i]:
                    ls_done[i] = True
                    continue
                if lams[i] < lambda_mins[i]:
                    ls_done[i] = True
                    continue
                if ls_iter == 0:
                    tmp = -slopes[i] / (2.0 * (f_new - f_olds[i] - slopes[i]))
                else:
                    rhs1 = f_new - f_olds[i] - lams[i] * slopes[i]
                    rhs2 = f2s[i] - f_olds[i] - lam2s[i] * slopes[i]
                    dl = lams[i] - lam2s[i]
                    if abs(dl) < 1e-30:
                        tmp = 0.5 * lams[i]
                    else:
                        a = (rhs1 / (lams[i] ** 2) - rhs2 / (lam2s[i] ** 2)) / dl
                        b = (-lam2s[i] * rhs1 / (lams[i] ** 2) + lams[i] * rhs2 / (lam2s[i] ** 2)) / dl
                        if a == 0.0:
                            tmp = -slopes[i] / (2.0 * b)
                        else:
                            disc = b * b - 3.0 * a * slopes[i]
                            if disc < 0.0:
                                tmp = 0.5 * lams[i]
                            elif b <= 0.0:
                                tmp = (-b + np.sqrt(disc)) / (3.0 * a)
                            else:
                                tmp = -slopes[i] / (b + np.sqrt(disc))
                            if tmp > 0.5 * lams[i]:
                                tmp = 0.5 * lams[i]
                lam2s[i] = lams[i]
                f2s[i] = f_new
                lams[i] = max(tmp, 0.1 * lams[i])

        # Final positions after line search
        x_new_np_final = x_np.copy()
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            x_new_np_final[s:e] = x_np[s:e] + lams[i] * d_np[s:e]

        x_new = mx.array(x_new_np_final, dtype=mx.float32)
        mx.eval(x_new)
        ep_new, g_new = batch_fn(x_new)
        mx.eval(ep_new, g_new)
        g_new_np = np.array(g_new)
        ep_new_np = np.array(ep_new)
        x_new_np = np.array(x_new)

        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]

            xi_i = x_new_np[s:e] - x_np[s:e]
            test_step = float(np.max(np.abs(xi_i) / np.maximum(np.abs(x_new_np[s:e]), 1.0)))
            if test_step < TOLX:
                mol_converged[i] = True
                continue

            new_e = float(np.sum(ep_new_np[atom_starts[i]:atom_starts[i + 1]]))
            mol_energies[i] = new_e
            g_i_new = g_new_np[s:e]
            den = max(new_e, 1.0)
            test_grad = float(
                np.max(np.abs(g_i_new) * np.maximum(np.abs(x_new_np[s:e]), 1.0) / den)
            )
            if test_grad < grad_tol:
                mol_converged[i] = True
                continue

            # BFGS Hessian update
            s_k = xi_i
            y_k = g_i_new - grad_np[s:e]
            sy = float(np.dot(s_k, y_k))

            if sy > np.sqrt(EPS_HESSIAN * float(np.dot(y_k, y_k)) * float(np.dot(s_k, s_k))):
                H = H_list[i]
                Hy = H @ y_k
                fac = 1.0 / sy
                fae = float(np.dot(y_k, Hy))
                fad = 1.0 / fae
                dg_upd = fac * s_k - fad * Hy
                H_list[i] = (
                    H
                    + fac * np.outer(s_k, s_k)
                    - fad * np.outer(Hy, Hy)
                    + fae * np.outer(dg_upd, dg_upd)
                )

        x = x_new
        grad = g_new

    # Final results
    x_final = np.array(x)
    g_final = np.array(grad)
    ep_out, _ = batch_fn(x)
    mx.eval(ep_out)
    ep_np_final = np.array(ep_out)

    energies_out = np.zeros(n_mols)
    grad_norms_out = np.zeros(n_mols)
    for i in range(n_mols):
        energies_out[i] = float(np.sum(ep_np_final[atom_starts[i]:atom_starts[i + 1]]))
        s, e = mol_slices[i]
        grad_norms_out[i] = float(np.linalg.norm(g_final[s:e]))

    return BatchBfgsResult(
        positions=x_final,
        energies=energies_out,
        grad_norms=grad_norms_out,
        n_iters=n_iter,
        converged=mol_converged,
        n_molecules=n_mols,
    )
