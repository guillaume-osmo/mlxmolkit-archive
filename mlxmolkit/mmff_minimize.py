"""
MMFF94 optimization for N×k conformers using the fused mega Metal kernel.

The mega-kernel computes energy + gradient for all conformers in one
dispatch.  The L-BFGS loop runs on CPU, calling the mega-kernel each
iteration.  Parameters are shared per-molecule via the dispatch table.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import mlx.core as mx

from .mmff_params import MMFFParams, extract_mmff_params
from .mmff_metal_kernel import (
    pack_multi_mol_for_metal,
    mmff_energy_grad_metal_mega,
)


def mmff_minimize_nk(
    mmff_params_list: List[MMFFParams],
    conf_counts: List[int],
    positions_3d: np.ndarray,
    *,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    step_tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MMFF94 L-BFGS on all N×k conformers with shared params.

    Parameters
    ----------
    mmff_params_list : list of MMFFParams
        One per molecule (shared across k conformers).
    conf_counts : list of int
        k_i conformers per molecule.
    positions_3d : np.ndarray
        Flat (total_coords,) float32 — all conformers concatenated.

    Returns
    -------
    out_pos : np.ndarray
    energies : np.ndarray, shape (C,)
    converged : np.ndarray, shape (C,) bool
    """
    C = sum(conf_counts)

    # Pack params for mega-kernel (shared per molecule)
    idx_buf, param_buf, all_meta, dispatch, pos_offsets, dims_per_mol = (
        pack_multi_mol_for_metal(mmff_params_list, conf_counts)
    )
    total_coords = int(positions_3d.shape[0])

    # Build per-conformer slices
    conf_slices = []
    pos = 0
    for mol_idx, (p, k) in enumerate(zip(mmff_params_list, conf_counts)):
        dim = p.n_atoms * 3
        for _ in range(k):
            conf_slices.append((pos, pos + dim))
            pos += dim

    # L-BFGS state per conformer
    x = positions_3d.copy()
    m = 10  # history depth
    s_hists = [[] for _ in range(C)]
    y_hists = [[] for _ in range(C)]
    rho_hists = [[] for _ in range(C)]
    converged = np.zeros(C, dtype=bool)
    final_energies = np.zeros(C, dtype=np.float32)

    # Initial energy + gradient
    x_mx = mx.array(x)
    e_mx, g_mx = mmff_energy_grad_metal_mega(
        idx_buf, param_buf, all_meta, dispatch, pos_offsets,
        x_mx, C, total_coords,
    )
    mx.eval(e_mx, g_mx)
    energies = np.array(e_mx)
    grad = np.array(g_mx)
    final_energies[:] = energies

    for iteration in range(max_iters):
        if np.all(converged):
            break

        # L-BFGS direction per conformer
        d = np.zeros_like(x)
        for c in range(C):
            if converged[c]:
                continue
            s0, s1 = conf_slices[c]
            g_c = grad[s0:s1].copy()
            q = g_c.copy()

            # Two-loop recursion
            alphas = []
            for j in range(len(s_hists[c]) - 1, -1, -1):
                a = rho_hists[c][j] * np.dot(s_hists[c][j], q)
                q -= a * y_hists[c][j]
                alphas.append(a)
            alphas.reverse()

            if s_hists[c]:
                sy = np.dot(s_hists[c][-1], y_hists[c][-1])
                yy = np.dot(y_hists[c][-1], y_hists[c][-1])
                gamma = sy / max(yy, 1e-30)
                q *= gamma

            for j in range(len(s_hists[c])):
                beta = rho_hists[c][j] * np.dot(y_hists[c][j], q)
                q += (alphas[j] - beta) * s_hists[c][j]

            d[s0:s1] = -q

        # Line search (simple backtracking for all conformers)
        lam = np.ones(C, dtype=np.float32)
        x_old = x.copy()
        grad_old = grad.copy()
        e_old = energies.copy()

        for ls in range(20):
            x_new = x_old.copy()
            for c in range(C):
                if converged[c]:
                    continue
                s0, s1 = conf_slices[c]
                x_new[s0:s1] = x_old[s0:s1] + lam[c] * d[s0:s1]

            x_mx = mx.array(x_new)
            e_mx, g_mx = mmff_energy_grad_metal_mega(
                idx_buf, param_buf, all_meta, dispatch, pos_offsets,
                x_mx, C, total_coords,
            )
            mx.eval(e_mx, g_mx)
            e_new = np.array(e_mx)

            all_ok = True
            for c in range(C):
                if converged[c]:
                    continue
                s0, s1 = conf_slices[c]
                slope = np.dot(d[s0:s1], grad_old[s0:s1])
                if e_new[c] <= e_old[c] + 1e-4 * lam[c] * slope:
                    pass  # accepted
                else:
                    lam[c] *= 0.5
                    all_ok = False
            if all_ok:
                break

        x = x_new
        grad = np.array(g_mx)
        energies = e_new
        final_energies[:] = energies

        # Update L-BFGS history + convergence check
        for c in range(C):
            if converged[c]:
                continue
            s0, s1 = conf_slices[c]
            s_k = x[s0:s1] - x_old[s0:s1]
            y_k = grad[s0:s1] - grad_old[s0:s1]
            ys = np.dot(y_k, s_k)

            # Step convergence
            max_step = np.max(np.abs(s_k) / np.maximum(np.abs(x[s0:s1]), 1.0))
            max_grad = np.max(np.abs(grad[s0:s1]) * np.maximum(np.abs(x[s0:s1]), 1.0))
            if max_step < step_tol or max_grad / max(abs(energies[c]), 1.0) < grad_tol:
                converged[c] = True
                continue

            if ys > 1e-10:
                s_hists[c].append(s_k)
                y_hists[c].append(y_k)
                rho_hists[c].append(1.0 / ys)
                if len(s_hists[c]) > m:
                    s_hists[c].pop(0)
                    y_hists[c].pop(0)
                    rho_hists[c].pop(0)

    return x, final_energies, converged
