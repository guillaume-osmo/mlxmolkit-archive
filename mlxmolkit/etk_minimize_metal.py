"""
Batched L-BFGS minimizer for 3D ETK (Experimental Torsion Knowledge) on Metal.

Stage 5 of ETKDG: refine 3D coordinates to match CSD torsion preferences,
planarity at sp2 centers, and 1-4 distance constraints.

Takes 3D coordinates (from stages 1-4) and an ETK energy function,
runs batched L-BFGS to minimize the ETK energy.
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
from mlxmolkit.etk_extract import BatchedETKSystem
from mlxmolkit.etk_energy_metal import make_etk_energy_grad


@dataclass
class ETKMinimizeResult:
    """Result of batched ETK minimization."""
    positions: np.ndarray     # (n_atoms_total * 3,) float32
    energies: np.ndarray      # (n_mols,) float32
    grad_norms: np.ndarray    # (n_mols,) float32
    n_iters: int
    converged: np.ndarray     # (n_mols,) bool
    n_mols: int
    atom_starts: np.ndarray


def etk_minimize_batch(
    system: BatchedETKSystem,
    x0_3d: np.ndarray,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    m: int = 10,
) -> ETKMinimizeResult:
    """
    Batched L-BFGS ETK minimization on Metal (stage 5).

    Minimizes the 3D ETK energy (CSD torsions + improper torsions +
    1-4 distance constraints) for all conformers simultaneously.

    Args:
        system: Batched ETK parameters from batch_etk_params().
        x0_3d: Initial 3D coordinates (n_atoms_total * 3,) from stage 4.
        max_iters: Maximum L-BFGS iterations.
        grad_tol: Gradient convergence tolerance.
        m: L-BFGS history depth.
    """
    energy_grad_fn, atom_starts, n_atoms_total, n_mols = (
        make_etk_energy_grad(system)
    )
    total_coords = n_atoms_total * 3

    x = mx.array(x0_3d.astype(np.float32), dtype=mx.float32)

    mol_slices = []
    for i in range(n_mols):
        s = int(atom_starts[i]) * 3
        e = int(atom_starts[i + 1]) * 3
        mol_slices.append((s, e))

    mol_dim = [e - s for s, e in mol_slices]
    mol_converged = np.zeros(n_mols, dtype=bool)

    s_hists: list[list[np.ndarray]] = [[] for _ in range(n_mols)]
    y_hists: list[list[np.ndarray]] = [[] for _ in range(n_mols)]
    rho_hists: list[list[float]] = [[] for _ in range(n_mols)]

    x_np = np.array(x)
    max_steps = np.zeros(n_mols)
    for i in range(n_mols):
        s, e = mol_slices[i]
        ss = float(np.sum(x_np[s:e] ** 2))
        max_steps[i] = MAX_STEP_FACTOR * max(np.sqrt(ss), float(mol_dim[i]))

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

    # Final evaluation
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

    return ETKMinimizeResult(
        positions=np.array(x),
        energies=energies_out,
        grad_norms=grad_norms_out,
        n_iters=n_iter,
        converged=mol_converged,
        n_mols=n_mols,
        atom_starts=atom_starts,
    )
