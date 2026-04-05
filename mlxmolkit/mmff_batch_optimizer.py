"""
Batched MMFF optimization: N conformers in parallel.

Thread-parallel RDKit energy/gradient + batched L-BFGS/BFGS loop.

Architecture (matching nvMolKit's batched minimizer pattern):
  - All conformer positions packed in a single flat array
  - Per-conformer convergence tracked independently
  - RDKit MMFF energy/gradient computed in parallel via ThreadPoolExecutor
    (RDKit's C++ core releases the GIL, so threads give real parallelism)
  - L-BFGS/BFGS direction, line search, and Hessian updates run on CPU
    with the same batched structure as bfgs_batch_metal.py

Compared to sequential mmff_optimize_confs, this gives ~Nx speedup where
N = min(n_conformers, n_cpu_threads) for the energy/gradient bottleneck.

Reference:
  - nvMolKit batched: https://github.com/NVIDIA-Digital-Bio/nvMolKit/tree/main/src/minimizer
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import mlx.core as mx

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from mlxmolkit.bfgs_metal import (
    FUNCTOL, MOVETOL, TOLX, EPS_HESSIAN,
    MAX_LINE_SEARCH_ITERS, MAX_STEP_FACTOR,
    BfgsResult, _lbfgs_direction, _scale_grad_nvmolkit,
)


@dataclass
class BatchMMFFResult:
    """Result of batched MMFF optimization."""
    energies: np.ndarray
    positions: np.ndarray
    grad_norms: np.ndarray
    n_iters: int
    converged: np.ndarray
    n_conformers: int
    per_conf_iters: np.ndarray


def _make_mmff_closure(mol: Chem.Mol, conf_id: int, mmff_props):
    """
    Create a thread-safe MMFF energy/gradient closure for one conformer.

    Each closure operates on its own RWMol copy so calls are independent
    and safe for concurrent use from different threads.
    """
    mol_copy = Chem.RWMol(mol)
    n_atoms = mol_copy.GetNumAtoms()

    def energy_grad(pos_flat: np.ndarray) -> tuple[float, np.ndarray]:
        conf = mol_copy.GetConformer(conf_id)
        pos = pos_flat.reshape(n_atoms, 3).astype(np.float64)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, pos[i].tolist())
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol_copy, mmff_props, confId=conf_id,
        )
        e = ff.CalcEnergy()
        g = np.array(ff.CalcGrad(), dtype=np.float32)
        return e, g

    return energy_grad


def _cubic_backtrack(
    f_old: float, f_new: float, slope: float,
    lam: float, lam2: float, f2: float, ls_iter: int,
) -> float:
    """Cubic-interpolation backtracking step (same as bfgs_metal.py)."""
    if ls_iter == 0:
        return -slope / (2.0 * (f_new - f_old - slope))
    rhs1 = f_new - f_old - lam * slope
    rhs2 = f2 - f_old - lam2 * slope
    dl = lam - lam2
    if abs(dl) < 1e-30:
        return 0.5 * lam
    a = (rhs1 / (lam ** 2) - rhs2 / (lam2 ** 2)) / dl
    b = (-lam2 * rhs1 / (lam ** 2) + lam * rhs2 / (lam2 ** 2)) / dl
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
    if tmp > 0.5 * lam:
        tmp = 0.5 * lam
    return tmp


def mmff_optimize_batch(
    mol: Chem.Mol,
    conf_ids: Optional[list[int]] = None,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    method: str = "lbfgs",
    m: int = 10,
    scale_grads: bool = True,
    n_threads: int = 0,
) -> BatchMMFFResult:
    """
    Optimize multiple conformers of a molecule in parallel.

    All conformers share MMFF properties. Energy/gradient is computed via
    thread-parallel RDKit calls (C++ releases the GIL). The BFGS/L-BFGS
    loop orchestrates all conformers in lockstep with per-conformer
    convergence tracking.

    Args:
        mol: RDKit molecule with conformers embedded.
        conf_ids: which conformers to optimize (default: all).
        max_iters: maximum optimizer iterations.
        grad_tol: gradient convergence tolerance.
        method: "lbfgs" (default, faster) or "bfgs".
        m: L-BFGS history size.
        scale_grads: nvMolKit-style gradient scaling.
        n_threads: threads for parallel energy/gradient (0 = auto).

    Returns:
        BatchMMFFResult with per-conformer energies, positions, convergence.
    """
    mmff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if mmff_props is None:
        raise ValueError("Could not compute MMFF properties for molecule")

    n_atoms = mol.GetNumAtoms()
    dim = n_atoms * 3

    if conf_ids is None:
        conf_ids = [c.GetId() for c in mol.GetConformers()]
    n_conf = len(conf_ids)

    if n_threads <= 0:
        n_threads = min(n_conf, os.cpu_count() or 4)

    closures = [_make_mmff_closure(mol, cid, mmff_props) for cid in conf_ids]

    # Pack all conformer positions into flat arrays
    x_all = np.zeros((n_conf, dim), dtype=np.float32)
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            p = conf.GetAtomPosition(i)
            x_all[k, i * 3] = p.x
            x_all[k, i * 3 + 1] = p.y
            x_all[k, i * 3 + 2] = p.z

    # ------------------------------------------------------------------
    # Thread-parallel energy/gradient helper
    # ------------------------------------------------------------------
    pool = ThreadPoolExecutor(max_workers=n_threads)

    def _eval_single(args):
        k, pos_flat = args
        return closures[k](pos_flat)

    def parallel_energy_grad(
        x: np.ndarray, active: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute energy/gradient for all active conformers in parallel."""
        energies = np.zeros(n_conf, dtype=np.float64)
        grads = np.zeros((n_conf, dim), dtype=np.float32)

        tasks = [(k, x[k]) for k in range(n_conf) if active[k]]
        if not tasks:
            return energies, grads

        results = list(pool.map(_eval_single, tasks))
        for (k, _), (e, g) in zip(tasks, results):
            energies[k] = e
            grads[k] = g
        return energies, grads

    # ------------------------------------------------------------------
    # Initial energy/gradient
    # ------------------------------------------------------------------
    active = np.ones(n_conf, dtype=bool)
    mol_energies, grad_all = parallel_energy_grad(x_all, active)

    # Scale gradients
    grad_scaled_all = np.zeros_like(grad_all)
    grad_scale_all = np.ones(n_conf, dtype=np.float64)
    for k in range(n_conf):
        grad_scaled_all[k], grad_scale_all[k] = _scale_grad_nvmolkit(
            grad_all[k], scale_grads,
        )

    # Max step per conformer
    max_steps = np.zeros(n_conf)
    for k in range(n_conf):
        ss = float(np.sum(x_all[k] ** 2))
        max_steps[k] = MAX_STEP_FACTOR * max(np.sqrt(ss), float(dim))

    # Per-conformer state
    converged = np.zeros(n_conf, dtype=bool)
    per_conf_iters = np.zeros(n_conf, dtype=np.int32)

    if method == "lbfgs":
        s_hists = [[] for _ in range(n_conf)]
        y_hists = [[] for _ in range(n_conf)]
        rho_hists = [[] for _ in range(n_conf)]
    else:
        H_list = [np.eye(dim, dtype=np.float32) for _ in range(n_conf)]

    n_iter = 0

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------
    for iteration in range(max_iters):
        n_iter = iteration + 1
        if np.all(converged):
            break

        # --- Compute search directions ---
        d_all = np.zeros((n_conf, dim), dtype=np.float32)
        for k in range(n_conf):
            if converged[k]:
                continue
            if method == "lbfgs":
                d_k = _lbfgs_direction(
                    grad_scaled_all[k], s_hists[k], y_hists[k], rho_hists[k],
                )
            else:
                d_k = -H_list[k] @ grad_scaled_all[k]

            d_norm = float(np.linalg.norm(d_k))
            if d_norm > max_steps[k]:
                d_k *= max_steps[k] / d_norm
            d_all[k] = d_k

        # --- Check slopes, reset to steepest descent if needed ---
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

        # --- Compute lambda_min per conformer ---
        lambda_mins = np.zeros(n_conf)
        for k in range(n_conf):
            if converged[k]:
                continue
            tv = np.abs(d_all[k]) / np.maximum(np.abs(x_all[k]), 1.0)
            mt = float(np.max(tv)) if len(tv) > 0 else 0.0
            lambda_mins[k] = MOVETOL / mt if mt > 0 else 1e-12

        # --- Batched line search ---
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

            ls_active = ~ls_done
            trial_energies, _ = parallel_energy_grad(x_trial, ls_active)

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
                    f_olds[k], f_new, slopes[k],
                    lams[k], lam2s[k], f2s[k], ls_iter,
                )
                lam2s[k] = lams[k]
                f2s[k] = f_new
                lams[k] = max(tmp, 0.1 * lams[k])

        # --- Apply final lambda to get new positions ---
        x_new = x_all.copy()
        for k in range(n_conf):
            if converged[k]:
                continue
            x_new[k] = x_all[k] + lams[k] * d_all[k]

        # --- Compute new energy/gradient at final positions ---
        new_active = ~converged
        new_energies, new_grads = parallel_energy_grad(x_new, new_active)

        # --- Per-conformer post-step updates ---
        old_grad_scaled = grad_scaled_all.copy()

        for k in range(n_conf):
            if converged[k]:
                continue
            per_conf_iters[k] = n_iter

            xi_k = x_new[k] - x_all[k]
            test_step = float(np.max(
                np.abs(xi_k) / np.maximum(np.abs(x_new[k]), 1.0)
            ))

            x_all[k] = x_new[k]
            mol_energies[k] = new_energies[k]

            if test_step < TOLX:
                converged[k] = True
                continue

            # Scale new gradient
            grad_scaled_all[k], grad_scale_all[k] = _scale_grad_nvmolkit(
                new_grads[k], scale_grads,
            )

            # Gradient convergence check
            e_val = mol_energies[k]
            den = max(e_val * grad_scale_all[k], 1.0)
            test_grad = float(np.max(
                np.abs(grad_scaled_all[k])
                * np.maximum(np.abs(x_new[k]), 1.0)
            ) / den)
            if test_grad < grad_tol:
                converged[k] = True
                continue

            # Update optimizer state
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
                    EPS_HESSIAN * float(np.dot(y_k, y_k)) * float(np.dot(s_k, s_k))
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

    pool.shutdown(wait=False)

    # --- Write optimized positions back to conformers ---
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        pos = x_all[k].reshape(n_atoms, 3)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, pos[i].astype(np.float64).tolist())

    grad_norms = np.array([
        float(np.linalg.norm(grad_scaled_all[k])) for k in range(n_conf)
    ])

    return BatchMMFFResult(
        energies=mol_energies.astype(np.float64),
        positions=x_all,
        grad_norms=grad_norms,
        n_iters=n_iter,
        converged=converged,
        n_conformers=n_conf,
        per_conf_iters=per_conf_iters,
    )


def mmff_optimize_molecules_batch(
    mols: list[Chem.Mol],
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    method: str = "lbfgs",
    m: int = 10,
    scale_grads: bool = True,
    n_threads: int = 0,
) -> list[BatchMMFFResult]:
    """
    Optimize conformers for a list of molecules.

    Each molecule's conformers are optimized in parallel via
    mmff_optimize_batch. Molecules are processed sequentially
    (each molecule fully utilizes all threads for its conformers).

    Args:
        mols: list of RDKit molecules with conformers.
        max_iters: max optimizer iterations per conformer.
        grad_tol: gradient tolerance.
        method: "lbfgs" (default) or "bfgs".
        m: L-BFGS history size.
        scale_grads: nvMolKit gradient scaling.
        n_threads: threads for parallel energy/gradient (0 = auto).

    Returns:
        List of BatchMMFFResult, one per molecule.
    """
    results = []
    for mol in mols:
        r = mmff_optimize_batch(
            mol, max_iters=max_iters, grad_tol=grad_tol,
            method=method, m=m, scale_grads=scale_grads,
            n_threads=n_threads,
        )
        results.append(r)
    return results
