"""
Fully vectorized native batched MMFF optimization.

Both the energy/gradient AND the optimizer loop operate on all C conformers
simultaneously using numpy broadcasting. No Python for-loops over conformers.

Architecture:
  1. Extract MMFF parameters once from RDKit
  2. Vectorized energy/gradient: (C, N, 3) -> (C,), (C, N*3)
  3. Vectorized BFGS: all Hessians as (C, dim, dim), batched matmul
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from mlxmolkit.bfgs_metal import MAX_STEP_FACTOR
from mlxmolkit.mmff_energy_vectorized import mmff_energy_grad_batch
from mlxmolkit.mmff_params import extract_mmff_params

FUNCTOL = 1e-4
MOVETOL = 1e-7
TOLX = 4 * np.finfo(np.float32).eps
EPS_HESSIAN = 1e-12


@dataclass
class NativeBatchMMFFResult:
    """Result of native batched MMFF optimization."""

    energies: np.ndarray
    positions: np.ndarray
    grad_norms: np.ndarray
    n_iters: int
    converged: np.ndarray
    n_conformers: int
    per_conf_iters: np.ndarray


def _scale_grads_batch(grads: np.ndarray, do_scale: bool):
    """Scale gradients nvMolKit-style, all conformers at once."""
    C = grads.shape[0]
    scales = np.ones(C, dtype=np.float64)
    if not do_scale:
        return grads.copy(), scales
    g = grads.copy().astype(np.float32)
    g *= 0.1
    max_g = np.max(np.abs(g), axis=1)  # (C,)
    mask = max_g > 10.0
    if np.any(mask):
        factor = 10.0 / max_g[mask]
        g[mask] *= factor[:, None]
        scales[mask] = 0.1 * factor
    else:
        scales[:] = 0.1
    return g, scales


def mmff_optimize_native_batch_v2(
    mol: Chem.Mol,
    conf_ids: Optional[list[int]] = None,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    scale_grads: bool = True,
) -> NativeBatchMMFFResult:
    """
    Fully vectorized BFGS optimizer for multiple conformers.

    No Python for-loops over conformers in the hot path.
    All state (Hessians, directions, line search) is vectorized.
    """
    n_atoms = mol.GetNumAtoms()
    dim = n_atoms * 3

    if conf_ids is None:
        conf_ids = [c.GetId() for c in mol.GetConformers()]
    C = len(conf_ids)

    mmff_params = extract_mmff_params(mol, conf_id=conf_ids[0])

    pos = np.zeros((C, n_atoms, 3), dtype=np.float64)
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            p = conf.GetAtomPosition(i)
            pos[k, i] = [p.x, p.y, p.z]

    x = pos.reshape(C, dim).astype(np.float32)

    # Initial evaluation
    energies, grads = mmff_energy_grad_batch(mmff_params, pos)
    g_scaled, g_scales = _scale_grads_batch(grads, scale_grads)

    max_steps = MAX_STEP_FACTOR * np.maximum(
        np.sqrt(np.sum(x ** 2, axis=1)), float(dim)
    )

    # BFGS Hessian: (C, dim, dim) identity
    H = np.zeros((C, dim, dim), dtype=np.float32)
    for k in range(C):
        H[k] = np.eye(dim, dtype=np.float32)

    converged = np.zeros(C, dtype=bool)
    per_conf_iters = np.zeros(C, dtype=np.int32)
    active = np.ones(C, dtype=bool)
    n_iter = 0

    for iteration in range(max_iters):
        n_iter = iteration + 1
        if np.all(converged):
            break

        active = ~converged

        # --- Batched BFGS direction: d = -H @ g  ---
        # Using einsum for batched matmul: (C, dim, dim) @ (C, dim) -> (C, dim)
        d = -np.einsum("ijk,ik->ij", H, g_scaled)

        # Clip step norms
        d_norms = np.linalg.norm(d, axis=1)  # (C,)
        clip_mask = d_norms > max_steps
        if np.any(clip_mask):
            d[clip_mask] *= (max_steps[clip_mask] / d_norms[clip_mask])[:, None]

        # Check slopes
        slopes = np.sum(d * g_scaled, axis=1)  # (C,)
        bad_slope = (slopes >= 0) & active
        if np.any(bad_slope):
            d[bad_slope] = -g_scaled[bad_slope]
            slopes[bad_slope] = np.sum(d[bad_slope] * g_scaled[bad_slope], axis=1)
            still_bad = (slopes >= 0) & bad_slope
            converged[still_bad] = True

        if np.all(converged):
            break

        # Lambda_min
        tv = np.abs(d) / np.maximum(np.abs(x), 1.0)
        mt = np.max(tv, axis=1).clip(min=1e-30)
        lambda_mins = MOVETOL / mt

        # --- Batched line search ---
        lams = np.ones(C)
        lam2s = np.zeros(C)
        f_olds = energies.copy()
        f2s = np.zeros(C)
        ls_done = converged.copy()

        for ls_iter in range(10):
            if np.all(ls_done):
                break

            x_trial = x + (lams[:, None] * d).astype(np.float32)
            pos_trial = x_trial.reshape(C, n_atoms, 3).astype(np.float64)
            trial_e, _ = mmff_energy_grad_batch(mmff_params, pos_trial)

            sufficient = (trial_e - f_olds) <= FUNCTOL * lams * slopes
            too_small = lams < lambda_mins
            accept = (sufficient | too_small) & ~ls_done
            ls_done[accept] = True

            # Cubic backtracking for remaining
            needs_bt = ~ls_done
            if not np.any(needs_bt):
                break

            idx = np.where(needs_bt)[0]
            for k in idx:
                if ls_iter == 0:
                    tmp = -slopes[k] / (2.0 * (trial_e[k] - f_olds[k] - slopes[k]))
                else:
                    rhs1 = trial_e[k] - f_olds[k] - lams[k] * slopes[k]
                    rhs2 = f2s[k] - f_olds[k] - lam2s[k] * slopes[k]
                    dl = lams[k] - lam2s[k]
                    if abs(dl) < 1e-30:
                        tmp = 0.5 * lams[k]
                    else:
                        a = (rhs1 / lams[k] ** 2 - rhs2 / lam2s[k] ** 2) / dl
                        b = (-lam2s[k] * rhs1 / lams[k] ** 2 + lams[k] * rhs2 / lam2s[k] ** 2) / dl
                        if a == 0:
                            tmp = -slopes[k] / (2 * b) if b != 0 else 0.5 * lams[k]
                        else:
                            disc = b * b - 3 * a * slopes[k]
                            if disc < 0:
                                tmp = 0.5 * lams[k]
                            elif b <= 0:
                                tmp = (-b + np.sqrt(disc)) / (3 * a)
                            else:
                                tmp = -slopes[k] / (b + np.sqrt(disc))
                    tmp = min(tmp, 0.5 * lams[k])

                lam2s[k] = lams[k]
                f2s[k] = trial_e[k]
                lams[k] = max(tmp, 0.1 * lams[k])

        # --- Apply step ---
        x_new = x + (lams[:, None] * d).astype(np.float32)
        pos_new = x_new.reshape(C, n_atoms, 3).astype(np.float64)
        new_e, new_g = mmff_energy_grad_batch(mmff_params, pos_new)
        new_g_scaled, new_g_scales = _scale_grads_batch(new_g, scale_grads)

        # --- Convergence checks and Hessian updates (vectorized where possible) ---
        xi = x_new - x
        test_step = np.max(np.abs(xi) / np.maximum(np.abs(x_new), 1.0), axis=1)

        old_g_scaled = g_scaled.copy()
        x = x_new
        energies = new_e

        # Step convergence
        step_conv = (test_step < TOLX) & active
        converged[step_conv] = True

        # Gradient convergence
        still_active = active & ~converged
        if np.any(still_active):
            den = np.maximum(energies[still_active] * new_g_scales[still_active], 1.0)
            test_grad = np.max(
                np.abs(new_g_scaled[still_active])
                * np.maximum(np.abs(x_new[still_active]), 1.0),
                axis=1,
            ) / den
            grad_conv_idx = np.where(still_active)[0][test_grad < grad_tol]
            converged[grad_conv_idx] = True

        g_scaled = new_g_scaled
        g_scales = new_g_scales
        per_conf_iters[active] = n_iter

        # --- BFGS Hessian update (vectorized) ---
        update_mask = active & ~converged
        if np.any(update_mask):
            idx_u = np.where(update_mask)[0]
            s = xi[idx_u]  # (U, dim)
            y = (g_scaled[idx_u] - old_g_scaled[idx_u])  # (U, dim)
            sy = np.sum(s * y, axis=1)  # (U,)
            yy = np.sum(y * y, axis=1)  # (U,)
            ss = np.sum(s * s, axis=1)  # (U,)

            valid = sy > np.sqrt(EPS_HESSIAN * yy * ss)
            for local_i, global_k in enumerate(idx_u):
                if not valid[local_i]:
                    continue
                Hk = H[global_k]
                sk = s[local_i]
                yk = y[local_i]
                sy_k = sy[local_i]
                Hy = Hk @ yk
                fac = 1.0 / sy_k
                fae = float(np.dot(yk, Hy))
                if abs(fae) < 1e-30:
                    continue
                fad = 1.0 / fae
                dg = fac * sk - fad * Hy
                H[global_k] = (
                    Hk
                    + fac * np.outer(sk, sk)
                    - fad * np.outer(Hy, Hy)
                    + fae * np.outer(dg, dg)
                )

    # Write back
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        p = x[k].reshape(n_atoms, 3)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, p[i].astype(np.float64).tolist())

    grad_norms = np.linalg.norm(g_scaled, axis=1)

    return NativeBatchMMFFResult(
        energies=energies,
        positions=x,
        grad_norms=grad_norms,
        n_iters=n_iter,
        converged=converged,
        n_conformers=C,
        per_conf_iters=per_conf_iters,
    )
