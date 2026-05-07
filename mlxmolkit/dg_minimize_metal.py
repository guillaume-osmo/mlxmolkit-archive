"""
Batched L-BFGS minimizer for 4D Distance Geometry on Metal.

Takes a BatchedDGSystem, generates random 4D initial coordinates,
and minimizes the DG energy using batched L-BFGS with Metal kernel
energy/gradient.  Handles stages 1-2 (DG minimize) and stage 4
(4th dimension collapse) of the ETKDG pipeline.

Reuses the proven L-BFGS two-loop recursion from bfgs_metal.py
with a single Metal dispatch for all molecules' energy+gradient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mlx.core as mx

from mlxmolkit.bfgs_metal import (
    FUNCTOL, MOVETOL, TOLX, EPS_HESSIAN,
    MAX_LINE_SEARCH_ITERS, MAX_STEP_FACTOR,
    _lbfgs_direction,
)
from mlxmolkit.dg_extract import BatchedDGSystem
from mlxmolkit.dg_energy_metal import make_dg4d_energy_grad


@dataclass
class DGMinimizeResult:
    """Result of batched DG minimization."""
    positions: np.ndarray     # (n_atoms_total * dim,) float32
    energies: np.ndarray      # (n_mols,) float32
    grad_norms: np.ndarray    # (n_mols,) float32
    n_iters: int
    converged: np.ndarray     # (n_mols,) bool
    n_mols: int
    dim: int
    atom_starts: np.ndarray   # (n_mols+1,) int32


def dg_minimize_batch(
    system: BatchedDGSystem,
    x0: np.ndarray | None = None,
    fourth_dim_weight: float = 10.0,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    m: int = 10,
    seed: int = 42,
) -> DGMinimizeResult:
    """
    Batched L-BFGS DG minimization on Metal.

    Minimizes the 4D distance geometry energy for all molecules
    simultaneously.  Energy+gradient for every molecule computed
    in a single Metal kernel dispatch.

    Args:
        system: Batched DG parameters from batch_dg_params().
        x0: Optional initial 4D coordinates (n_atoms_total * dim,).
            If None, random Gaussian coordinates are generated.
        fourth_dim_weight: Weight for 4th dimension penalty.
        max_iters: Maximum L-BFGS iterations.
        grad_tol: Gradient convergence tolerance.
        m: L-BFGS history depth.
        seed: Random seed for initial coordinates.
    """
    energy_grad_fn, atom_starts, n_atoms_total, n_mols, dim = (
        make_dg4d_energy_grad(system, fourth_dim_weight)
    )
    total_coords = n_atoms_total * dim

    if x0 is None:
        rng = np.random.RandomState(seed)
        x0 = rng.randn(total_coords).astype(np.float32) * 1.0

    x = mx.array(x0, dtype=mx.float32)

    # Per-molecule slices into the flat coordinate vector
    mol_slices = []
    for i in range(n_mols):
        s = int(atom_starts[i]) * dim
        e = int(atom_starts[i + 1]) * dim
        mol_slices.append((s, e))

    mol_dim = [e - s for s, e in mol_slices]
    mol_converged = np.zeros(n_mols, dtype=bool)

    # Per-molecule L-BFGS history
    s_hists: list[list[np.ndarray]] = [[] for _ in range(n_mols)]
    y_hists: list[list[np.ndarray]] = [[] for _ in range(n_mols)]
    rho_hists: list[list[float]] = [[] for _ in range(n_mols)]

    # Per-molecule max step
    x_np = np.array(x)
    max_steps = np.zeros(n_mols)
    for i in range(n_mols):
        s, e = mol_slices[i]
        ss = float(np.sum(x_np[s:e] ** 2))
        max_steps[i] = MAX_STEP_FACTOR * max(np.sqrt(ss), float(mol_dim[i]))

    # Initial energy/gradient
    energy_parts, grad = energy_grad_fn(x)
    mx.eval(energy_parts, grad)
    ep_np = np.array(energy_parts)
    mol_energies = np.zeros(n_mols)
    for i in range(n_mols):
        a_s = int(atom_starts[i])
        a_e = int(atom_starts[i + 1])
        mol_energies[i] = float(np.sum(ep_np[a_s:a_e]))

    n_iter = 0

    for iteration in range(max_iters):
        n_iter = iteration + 1
        if np.all(mol_converged):
            break

        grad_np = np.array(grad)
        x_np = np.array(x)

        # Per-molecule L-BFGS direction
        d_np = np.zeros(total_coords, dtype=np.float32)
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

        # Check slopes, reset to steepest descent if positive
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

        # Per-molecule lambda_min
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

            x_new_np = x_np.copy()
            for i in range(n_mols):
                if ls_done[i]:
                    continue
                s, e = mol_slices[i]
                x_new_np[s:e] = x_np[s:e] + lams[i] * d_np[s:e]

            x_new = mx.array(x_new_np, dtype=mx.float32)
            mx.eval(x_new)
            ep_new, _ = energy_grad_fn(x_new)
            mx.eval(ep_new)
            ep_new_np = np.array(ep_new)

            new_energies = np.zeros(n_mols)
            for i in range(n_mols):
                a_s = int(atom_starts[i])
                a_e = int(atom_starts[i + 1])
                new_energies[i] = float(np.sum(ep_new_np[a_s:a_e]))

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
                        a_c = (rhs1 / (lams[i] ** 2) - rhs2 / (lam2s[i] ** 2)) / dl
                        b_c = (-lam2s[i] * rhs1 / (lams[i] ** 2)
                               + lams[i] * rhs2 / (lam2s[i] ** 2)) / dl
                        if a_c == 0.0:
                            tmp = -slopes[i] / (2.0 * b_c)
                        else:
                            disc = b_c * b_c - 3.0 * a_c * slopes[i]
                            if disc < 0.0:
                                tmp = 0.5 * lams[i]
                            elif b_c <= 0.0:
                                tmp = (-b_c + np.sqrt(disc)) / (3.0 * a_c)
                            else:
                                tmp = -slopes[i] / (b_c + np.sqrt(disc))
                        if tmp > 0.5 * lams[i]:
                            tmp = 0.5 * lams[i]

                lam2s[i] = lams[i]
                f2s[i] = f_new
                lams[i] = max(tmp, 0.1 * lams[i])

        # Final positions after line search
        x_new_np = x_np.copy()
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            x_new_np[s:e] = x_np[s:e] + lams[i] * d_np[s:e]

        x_new = mx.array(x_new_np, dtype=mx.float32)
        mx.eval(x_new)
        ep_new, g_new = energy_grad_fn(x_new)
        mx.eval(ep_new, g_new)
        g_new_np = np.array(g_new)
        ep_new_np = np.array(ep_new)
        x_new_np_final = np.array(x_new)

        # Per-molecule convergence + L-BFGS history update
        for i in range(n_mols):
            if mol_converged[i]:
                continue
            s, e = mol_slices[i]
            a_s = int(atom_starts[i])
            a_e = int(atom_starts[i + 1])

            xi_i = x_new_np_final[s:e] - x_np[s:e]
            test_step = float(np.max(
                np.abs(xi_i) / np.maximum(np.abs(x_new_np_final[s:e]), 1.0)
            ))
            if test_step < TOLX:
                mol_converged[i] = True
                continue

            new_e = float(np.sum(ep_new_np[a_s:a_e]))
            mol_energies[i] = new_e
            g_i_new = g_new_np[s:e]
            den = max(new_e, 1.0)
            test_grad = float(np.max(
                np.abs(g_i_new) * np.maximum(np.abs(x_new_np_final[s:e]), 1.0) / den
            ))
            if test_grad < grad_tol:
                mol_converged[i] = True
                continue

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

    # Final energy evaluation
    ep_out, g_out = energy_grad_fn(x)
    mx.eval(ep_out, g_out)
    ep_np_final = np.array(ep_out)
    g_final = np.array(g_out)

    energies_out = np.zeros(n_mols)
    grad_norms_out = np.zeros(n_mols)
    for i in range(n_mols):
        a_s = int(atom_starts[i])
        a_e = int(atom_starts[i + 1])
        energies_out[i] = float(np.sum(ep_np_final[a_s:a_e]))
        s, e = mol_slices[i]
        grad_norms_out[i] = float(np.linalg.norm(g_final[s:e]))

    return DGMinimizeResult(
        positions=np.array(x),
        energies=energies_out,
        grad_norms=grad_norms_out,
        n_iters=n_iter,
        converged=mol_converged,
        n_mols=n_mols,
        dim=dim,
        atom_starts=atom_starts,
    )


def dg_collapse_4th_dim(
    system: BatchedDGSystem,
    positions_4d: np.ndarray,
    fourth_dim_weight: float = 100.0,
    max_iters: int = 100,
    grad_tol: float = 1e-4,
    m: int = 10,
) -> DGMinimizeResult:
    """
    Stage 4: Collapse 4th dimension by re-minimizing with heavy 4th-dim penalty.

    Takes 4D coordinates from stage 2 and minimizes again with a much
    larger fourth_dim_weight to drive the 4th coordinate to zero.
    """
    return dg_minimize_batch(
        system=system,
        x0=positions_4d,
        fourth_dim_weight=fourth_dim_weight,
        max_iters=max_iters,
        grad_tol=grad_tol,
        m=m,
    )


def extract_3d_coords(
    positions_4d: np.ndarray,
    atom_starts: np.ndarray,
    n_mols: int,
    dim: int = 4,
) -> np.ndarray:
    """Extract first 3 coordinates from 4D positions.

    Returns (n_atoms_total * 3,) float32 with 4th dimension stripped.
    """
    n_atoms_total = int(atom_starts[-1])
    pos_3d = np.zeros(n_atoms_total * 3, dtype=np.float32)
    for i in range(n_atoms_total):
        pos_3d[i * 3: i * 3 + 3] = positions_4d[i * dim: i * dim + 3]
    return pos_3d


def generate_random_4d_coords(
    system: BatchedDGSystem,
    seed: int = 42,
    scale: float = 1.0,
) -> np.ndarray:
    """Stage 1: Generate random 4D initial coordinates for all conformers."""
    rng = np.random.RandomState(seed)
    return rng.randn(system.n_atoms_total * system.dim).astype(np.float32) * scale
