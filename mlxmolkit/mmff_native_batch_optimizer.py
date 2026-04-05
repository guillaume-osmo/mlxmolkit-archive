"""
Native batched MMFF optimization: N conformers optimized simultaneously
without any RDKit callback during the optimization loop.

Architecture (matching nvMolKit's approach):
  1. Extract MMFF parameters ONCE from RDKit (mmff_params.py)
  2. Compute energy/gradient on CPU via vectorized numpy (mmff_energy_vectorized.py)
  3. Batched L-BFGS/BFGS loop with per-conformer convergence

Compared to the thread-parallel RDKit callback approach, this is faster because:
  - No per-call ForceField recreation overhead
  - Vectorized numpy operations across all conformers simultaneously
  - All positions processed as a single (C, N, 3) array
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from mlxmolkit.bfgs_metal import (
    EPS_HESSIAN,
    FUNCTOL,
    MAX_LINE_SEARCH_ITERS,
    MAX_STEP_FACTOR,
    MOVETOL,
    TOLX,
    _lbfgs_direction,
    _scale_grad_nvmolkit,
)
from mlxmolkit.mmff_energy_vectorized import mmff_energy_grad_batch
from mlxmolkit.mmff_params import extract_mmff_params


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


def _cubic_backtrack(
    f_old: float,
    f_new: float,
    slope: float,
    lam: float,
    lam2: float,
    f2: float,
    ls_iter: int,
) -> float:
    if ls_iter == 0:
        return -slope / (2.0 * (f_new - f_old - slope))
    rhs1 = f_new - f_old - lam * slope
    rhs2 = f2 - f_old - lam2 * slope
    dl = lam - lam2
    if abs(dl) < 1e-30:
        return 0.5 * lam
    a = (rhs1 / (lam**2) - rhs2 / (lam2**2)) / dl
    b = (-lam2 * rhs1 / (lam**2) + lam * rhs2 / (lam2**2)) / dl
    if a == 0.0:
        tmp = -slope / (2.0 * b)
    else:
        disc = b * b - 3.0 * a * slope
        if disc < 0.0:
            tmp = 0.5 * lam
        elif b <= 0.0:
            tmp = (-b + np.sqrt(disc)) / (3.0 * a)
        else:
            tmp = -slope / (b + np.sqrt(disc))
    return min(tmp, 0.5 * lam)


def mmff_optimize_native_batch(
    mol: Chem.Mol,
    conf_ids: Optional[list[int]] = None,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    method: str = "lbfgs",
    m: int = 10,
    scale_grads: bool = True,
) -> NativeBatchMMFFResult:
    """
    Optimize multiple conformers with no RDKit callback.

    Parameters are extracted once from RDKit, then the entire optimization
    loop runs on vectorized numpy. All C conformers are processed as a
    single (C, N, 3) array per energy/gradient evaluation.

    Args:
        mol: RDKit molecule with conformers embedded.
        conf_ids: which conformers to optimize (default: all).
        max_iters: maximum optimizer iterations.
        grad_tol: gradient convergence tolerance.
        method: "lbfgs" (default) or "bfgs".
        m: L-BFGS history size.
        scale_grads: nvMolKit-style gradient scaling.

    Returns:
        NativeBatchMMFFResult with per-conformer energies, positions, convergence.
    """
    n_atoms = mol.GetNumAtoms()
    dim = n_atoms * 3

    if conf_ids is None:
        conf_ids = [c.GetId() for c in mol.GetConformers()]
    n_conf = len(conf_ids)

    mmff_params = extract_mmff_params(mol, conf_id=conf_ids[0])

    positions = np.zeros((n_conf, n_atoms, 3), dtype=np.float64)
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            p = conf.GetAtomPosition(i)
            positions[k, i] = [p.x, p.y, p.z]

    def batch_eval(pos_3d: np.ndarray, active: np.ndarray):
        """Evaluate energy/gradient for active conformers."""
        energies, grads = mmff_energy_grad_batch(mmff_params, pos_3d)
        return energies, grads

    # Initial evaluation
    active = np.ones(n_conf, dtype=bool)
    mol_energies, grad_all = batch_eval(positions, active)

    x_all = positions.reshape(n_conf, dim).astype(np.float32)
    grad_scaled_all = np.zeros_like(grad_all)
    grad_scale_all = np.ones(n_conf, dtype=np.float64)
    for k in range(n_conf):
        grad_scaled_all[k], grad_scale_all[k] = _scale_grad_nvmolkit(
            grad_all[k], scale_grads,
        )

    max_steps = np.zeros(n_conf)
    for k in range(n_conf):
        ss = float(np.sum(x_all[k] ** 2))
        max_steps[k] = MAX_STEP_FACTOR * max(np.sqrt(ss), float(dim))

    converged = np.zeros(n_conf, dtype=bool)
    per_conf_iters = np.zeros(n_conf, dtype=np.int32)

    if method == "lbfgs":
        s_hists = [[] for _ in range(n_conf)]
        y_hists = [[] for _ in range(n_conf)]
        rho_hists = [[] for _ in range(n_conf)]
    else:
        H_list = [np.eye(dim, dtype=np.float32) for _ in range(n_conf)]

    n_iter = 0

    for iteration in range(max_iters):
        n_iter = iteration + 1
        if np.all(converged):
            break

        d_all = np.zeros((n_conf, dim), dtype=np.float32)
        for k in range(n_conf):
            if converged[k]:
                continue
            if method == "lbfgs":
                d_k = _lbfgs_direction(
                    grad_scaled_all[k],
                    s_hists[k],
                    y_hists[k],
                    rho_hists[k],
                )
            else:
                d_k = -H_list[k] @ grad_scaled_all[k]

            d_norm = float(np.linalg.norm(d_k))
            if d_norm > max_steps[k]:
                d_k *= max_steps[k] / d_norm
            d_all[k] = d_k

        slopes = np.zeros(n_conf)
        for k in range(n_conf):
            if converged[k]:
                continue
            slopes[k] = float(np.dot(d_all[k], grad_scaled_all[k]))
            if slopes[k] >= 0:
                d_all[k] = -grad_scaled_all[k]
                slopes[k] = float(np.dot(d_all[k], grad_scaled_all[k]))
                if slopes[k] >= 0:
                    converged[k] = True

        if np.all(converged):
            break

        lambda_mins = np.zeros(n_conf)
        for k in range(n_conf):
            if converged[k]:
                continue
            tv = np.abs(d_all[k]) / np.maximum(np.abs(x_all[k]), 1.0)
            mt = float(np.max(tv)) if len(tv) > 0 else 0.0
            lambda_mins[k] = MOVETOL / mt if mt > 0 else 1e-12

        lams = np.ones(n_conf)
        lam2s = np.zeros(n_conf)
        f_olds = mol_energies.copy()
        f2s = np.zeros(n_conf)
        ls_done = converged.copy()

        for ls_iter in range(MAX_LINE_SEARCH_ITERS):
            if np.all(ls_done):
                break

            x_trial = x_all.copy()
            for k in range(n_conf):
                if ls_done[k]:
                    continue
                x_trial[k] = x_all[k] + lams[k] * d_all[k]

            pos_trial = x_trial.reshape(n_conf, n_atoms, 3).astype(np.float64)
            trial_energies, _ = batch_eval(pos_trial, ~ls_done)

            for k in range(n_conf):
                if ls_done[k]:
                    continue
                f_new = trial_energies[k]
                if f_new - f_olds[k] <= FUNCTOL * lams[k] * slopes[k]:
                    ls_done[k] = True
                    continue
                if lams[k] < lambda_mins[k]:
                    ls_done[k] = True
                    continue
                tmp = _cubic_backtrack(
                    f_olds[k],
                    f_new,
                    slopes[k],
                    lams[k],
                    lam2s[k],
                    f2s[k],
                    ls_iter,
                )
                lam2s[k] = lams[k]
                f2s[k] = f_new
                lams[k] = max(tmp, 0.1 * lams[k])

        x_new = x_all.copy()
        for k in range(n_conf):
            if converged[k]:
                continue
            x_new[k] = x_all[k] + lams[k] * d_all[k]

        pos_new = x_new.reshape(n_conf, n_atoms, 3).astype(np.float64)
        new_energies, new_grads = batch_eval(pos_new, ~converged)

        old_grad_scaled = grad_scaled_all.copy()

        for k in range(n_conf):
            if converged[k]:
                continue
            per_conf_iters[k] = n_iter

            xi_k = x_new[k] - x_all[k]
            test_step = float(
                np.max(np.abs(xi_k) / np.maximum(np.abs(x_new[k]), 1.0))
            )

            x_all[k] = x_new[k]
            mol_energies[k] = new_energies[k]

            if test_step < TOLX:
                converged[k] = True
                continue

            grad_scaled_all[k], grad_scale_all[k] = _scale_grad_nvmolkit(
                new_grads[k],
                scale_grads,
            )

            e_val = mol_energies[k]
            den = max(e_val * grad_scale_all[k], 1.0)
            test_grad = float(
                np.max(
                    np.abs(grad_scaled_all[k])
                    * np.maximum(np.abs(x_new[k]), 1.0)
                )
                / den
            )
            if test_grad < grad_tol:
                converged[k] = True
                continue

            dgrad_k = grad_scaled_all[k] - old_grad_scaled[k]

            if method == "lbfgs":
                s_k = xi_k
                y_k = dgrad_k
                sy = float(np.dot(s_k, y_k))
                if sy > EPS_HESSIAN * float(np.dot(y_k, y_k)):
                    if len(s_hists[k]) >= m:
                        s_hists[k].pop(0)
                        y_hists[k].pop(0)
                        rho_hists[k].pop(0)
                    s_hists[k].append(s_k.copy())
                    y_hists[k].append(y_k.copy())
                    rho_hists[k].append(1.0 / sy)
            else:
                s_k = xi_k
                y_k = dgrad_k
                sy = float(np.dot(s_k, y_k))
                if sy > np.sqrt(
                    EPS_HESSIAN
                    * float(np.dot(y_k, y_k))
                    * float(np.dot(s_k, s_k))
                ):
                    H = H_list[k]
                    Hy = H @ y_k
                    fac = 1.0 / sy
                    fae = float(np.dot(y_k, Hy))
                    fad = 1.0 / fae
                    dg_upd = fac * s_k - fad * Hy
                    H_list[k] = (
                        H
                        + fac * np.outer(s_k, s_k)
                        - fad * np.outer(Hy, Hy)
                        + fae * np.outer(dg_upd, dg_upd)
                    )

    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        pos = x_all[k].reshape(n_atoms, 3)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, pos[i].astype(np.float64).tolist())

    grad_norms = np.array(
        [float(np.linalg.norm(grad_scaled_all[k])) for k in range(n_conf)]
    )

    return NativeBatchMMFFResult(
        energies=mol_energies.astype(np.float64),
        positions=x_all,
        grad_norms=grad_norms,
        n_iters=n_iter,
        converged=converged,
        n_conformers=n_conf,
        per_conf_iters=per_conf_iters,
    )
