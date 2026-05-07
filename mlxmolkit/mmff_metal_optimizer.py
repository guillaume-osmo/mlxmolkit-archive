"""
Batched MMFF L-BFGS optimizer using the fused Metal kernel.

Three modes:
  1. Single molecule, many conformers: mmff_optimize_metal_batch()
  2. Many molecules, Python L-BFGS:    mmff_optimize_metal_multi_mol()
  3. Many molecules, fused GPU opt:    mmff_optimize_metal_fused_multi_mol()

Mode 3 runs the entire optimization in a single Metal kernel launch —
each GPU thread runs all iterations of steepest descent internally.
Zero Python loop overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np
from rdkit import Chem

from mlxmolkit.mmff_metal_kernel import (
    mmff_energy_grad_metal,
    mmff_energy_grad_metal_mega,
    mmff_optimize_metal_fused,
    pack_multi_mol_for_metal,
    pack_params_for_metal,
)
from mlxmolkit.mmff_params import extract_mmff_params

EPS_CURV = 1e-10


@dataclass
class MetalBatchResult:
    """Result of fused Metal MMFF optimization."""

    energies: np.ndarray
    positions: np.ndarray
    grad_norms: np.ndarray
    n_iters: int
    n_conformers: int


def _lbfgs_direction_batch(
    grad: mx.array,
    s_hist: list[mx.array],
    y_hist: list[mx.array],
    rho_hist: list[mx.array],
) -> mx.array:
    """Vectorized L-BFGS two-loop recursion for C conformers."""
    m = len(s_hist)
    q = grad
    alphas = []
    for i in range(m - 1, -1, -1):
        a_i = rho_hist[i] * mx.sum(s_hist[i] * q, axis=1, keepdims=True)
        q = q - a_i * y_hist[i]
        alphas.append(a_i)

    if m > 0:
        sy = mx.sum(s_hist[-1] * y_hist[-1], axis=1, keepdims=True)
        yy = mx.sum(y_hist[-1] * y_hist[-1], axis=1, keepdims=True) + 1e-30
        gamma = sy / yy
    else:
        gamma = mx.array(1.0)

    r = gamma * q
    for i in range(m):
        beta = rho_hist[i] * mx.sum(y_hist[i] * r, axis=1, keepdims=True)
        r = r + s_hist[i] * (alphas[m - 1 - i] - beta)

    return -r


def mmff_optimize_metal_batch(
    mol: Chem.Mol,
    conf_ids: Optional[list[int]] = None,
    max_iters: int = 200,
    eval_freq: int = 10,
    max_step_per_atom: float = 0.3,
    lbfgs_m: int = 10,
    grad_scale: float = 0.1,
) -> MetalBatchResult:
    """
    Batched L-BFGS MMFF optimization using the fused Metal kernel.

    Args:
        mol: RDKit molecule with embedded conformers.
        conf_ids: conformer IDs to optimize (default: all).
        max_iters: maximum L-BFGS iterations.
        eval_freq: GPU sync frequency (10 = sync every 10 iters).
        max_step_per_atom: max displacement per atom per step (Angstrom).
        lbfgs_m: L-BFGS history pairs.
        grad_scale: gradient scaling factor.

    Returns:
        MetalBatchResult with final energies, positions, grad norms.
    """
    n_atoms = mol.GetNumAtoms()
    dim = n_atoms * 3

    if conf_ids is None:
        conf_ids = [c.GetId() for c in mol.GetConformers()]
    C = len(conf_ids)

    # --- Extract and pack params for Metal ---
    np_params = extract_mmff_params(mol, conf_id=conf_ids[0])
    idx_buf, param_buf, meta = pack_params_for_metal(np_params)

    pos_np = np.zeros((C, n_atoms, 3), dtype=np.float32)
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            p = conf.GetAtomPosition(i)
            pos_np[k, i] = [p.x, p.y, p.z]

    # --- Flat positions for optimizer ---
    x = mx.array(pos_np.reshape(C, dim))
    mx.eval(x)

    # Initial energy + gradient via Metal kernel
    pos3d = mx.reshape(x, (C, n_atoms, 3))
    e_init, g = mmff_energy_grad_metal(idx_buf, param_buf, meta, pos3d)
    mx.eval(e_init, g)
    g_sc = g * grad_scale

    # Track best-known energy/positions (monotone guard)
    best_e = e_init  # (C,)
    best_x = x  # (C, dim)

    # L-BFGS state
    s_hist: list[mx.array] = []
    y_hist: list[mx.array] = []
    rho_hist: list[mx.array] = []
    max_step = mx.array(max_step_per_atom, dtype=mx.float32)
    gs = mx.array(grad_scale, dtype=mx.float32)
    one = mx.array(1.0, dtype=mx.float32)

    # --- Optimization loop with energy guard ---
    for it in range(max_iters):
        d = _lbfgs_direction_batch(g_sc, s_hist, y_hist, rho_hist)

        d3 = mx.reshape(d, (C, n_atoms, 3))
        atom_disp = mx.sqrt(mx.sum(d3 * d3, axis=-1, keepdims=True) + 1e-30)
        d = mx.reshape(d3 * mx.minimum(max_step / atom_disp, one), (C, dim))

        x_new = x + d

        pos3d_new = mx.reshape(x_new, (C, n_atoms, 3))
        e_new, g_new = mmff_energy_grad_metal(idx_buf, param_buf, meta, pos3d_new)
        g_new_sc = g_new * gs

        s_k = x_new - x
        y_k = g_new_sc - g_sc
        sy = mx.sum(s_k * y_k, axis=1, keepdims=True)
        rho_k = mx.where(sy > EPS_CURV, 1.0 / (sy + 1e-30), mx.zeros_like(sy))

        if len(s_hist) >= lbfgs_m:
            s_hist.pop(0)
            y_hist.pop(0)
            rho_hist.pop(0)
        s_hist.append(s_k)
        y_hist.append(y_k)
        rho_hist.append(rho_k)

        x = x_new
        g_sc = g_new_sc

        if (it + 1) % eval_freq == 0:
            mx.eval(x, g_sc, e_new, *s_hist, *y_hist, *rho_hist)
            # Monotone guard: remember best positions per conformer
            improved = mx.expand_dims(e_new < best_e, axis=1)
            best_x = mx.where(improved, x, best_x)
            best_e = mx.minimum(e_new, best_e)
            mx.eval(best_x, best_e)

    # --- Final: use best-seen positions ---
    mx.eval(best_x)
    pos3d_final = mx.reshape(best_x, (C, n_atoms, 3))
    final_e, _ = mmff_energy_grad_metal(idx_buf, param_buf, meta, pos3d_final)
    g_norms = mx.sqrt(mx.sum(g_sc * g_sc, axis=1))
    mx.eval(final_e, g_norms)

    final_pos = np.array(best_x).reshape(C, n_atoms, 3)
    final_energies = np.array(final_e)
    final_gnorms = np.array(g_norms)

    # Write back to RDKit
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, final_pos[k, i].astype(float).tolist())

    return MetalBatchResult(
        energies=final_energies,
        positions=final_pos,
        grad_norms=final_gnorms,
        n_iters=max_iters,
        n_conformers=C,
    )


# ---------------------------------------------------------------------------
# Multi-molecule parallel optimisation via mega-kernel
# ---------------------------------------------------------------------------


@dataclass
class MultiMolResult:
    """Result of multi-molecule parallel MMFF optimisation."""

    mol_results: list[MetalBatchResult]


def mmff_optimize_metal_multi_mol(
    mol_specs: list[tuple[Chem.Mol, Optional[list[int]]]],
    max_iters: int = 200,
    eval_freq: int = 10,
    max_step_per_atom: float = 0.3,
    lbfgs_m: int = 10,
    grad_scale: float = 0.1,
) -> MultiMolResult:
    """
    Optimise many molecules simultaneously in a **single Metal kernel launch**
    with padded positions for vectorised L-BFGS.

    Architecture:
      - Positions padded to max_dim so all conformers form a clean
        (total_confs, max_dim) matrix.
      - One mega-kernel call per iteration.
      - Batched L-BFGS direction via _lbfgs_direction_batch() on the full
        (total_confs, max_dim) matrix — zero Python loops per iteration.
      - Padded elements carry zero gradient so they don't influence L-BFGS.

    Args:
        mol_specs: list of (mol, conf_ids) tuples.  If conf_ids is None
                   every embedded conformer is optimised.
        max_iters, eval_freq, max_step_per_atom, lbfgs_m, grad_scale:
            same semantics as mmff_optimize_metal_batch().

    Returns:
        MultiMolResult containing per-molecule MetalBatchResult objects.
    """
    gs_val = grad_scale

    # ---- 1. Extract params and initial positions -----------------------
    mols: list[Chem.Mol] = []
    all_conf_ids: list[list[int]] = []
    params_list: list = []
    conf_counts: list[int] = []

    for mol, conf_ids in mol_specs:
        if conf_ids is None:
            conf_ids = [c.GetId() for c in mol.GetConformers()]
        mols.append(mol)
        all_conf_ids.append(conf_ids)
        params_list.append(extract_mmff_params(mol, conf_id=conf_ids[0]))
        conf_counts.append(len(conf_ids))

    total_confs = sum(conf_counts)
    dims_per_mol = [p.n_atoms * 3 for p in params_list]
    max_dim = max(dims_per_mol)
    max_atoms = max_dim // 3
    total_padded = total_confs * max_dim

    # ---- 2. Pack parameters for mega-kernel ----------------------------
    # Override pos_offsets to use padded stride: cid * max_dim
    (
        idx_buf, param_buf, all_meta, dispatch, _pos_offsets_orig, _,
    ) = pack_multi_mol_for_metal(params_list, conf_counts)

    pos_offsets_np = np.arange(total_confs, dtype=np.uint32) * max_dim
    pos_offsets = mx.array(pos_offsets_np)

    # Build padded position matrix (total_confs, max_dim)
    pos_np = np.zeros((total_confs, max_dim), dtype=np.float32)
    conf_idx = 0
    for m_idx, (mol, cids) in enumerate(zip(mols, all_conf_ids)):
        n_atoms = mol.GetNumAtoms()
        for cid in cids:
            conf = mol.GetConformer(cid)
            for a in range(n_atoms):
                p = conf.GetAtomPosition(a)
                pos_np[conf_idx, a * 3] = p.x
                pos_np[conf_idx, a * 3 + 1] = p.y
                pos_np[conf_idx, a * 3 + 2] = p.z
            conf_idx += 1

    # ---- 3. Initial energy + gradient ----------------------------------
    x = mx.array(pos_np.ravel())
    e_init, g = mmff_energy_grad_metal_mega(
        idx_buf, param_buf, all_meta, dispatch, pos_offsets,
        x, total_confs, total_padded,
    )
    x = mx.reshape(x, (total_confs, max_dim))
    g_sc = mx.reshape(g, (total_confs, max_dim)) * gs_val
    mx.eval(x, g_sc, e_init)

    # Track best-known energy and positions for monotone guarantee
    best_e = e_init  # (total_confs,)
    best_x = x  # (total_confs, max_dim)

    max_step_mx = mx.array(max_step_per_atom, dtype=mx.float32)
    one_f = mx.array(1.0, dtype=mx.float32)

    s_hist: list[mx.array] = []
    y_hist: list[mx.array] = []
    rho_hist: list[mx.array] = []

    # ---- 4. Optimisation loop with energy guard ------------------------
    for it in range(max_iters):
        d = _lbfgs_direction_batch(g_sc, s_hist, y_hist, rho_hist)

        # Per-atom step capping
        d3 = mx.reshape(d, (total_confs, max_atoms, 3))
        atom_disp = mx.sqrt(
            mx.sum(d3 * d3, axis=-1, keepdims=True) + 1e-30
        )
        d = mx.reshape(
            d3 * mx.minimum(max_step_mx / atom_disp, one_f),
            (total_confs, max_dim),
        )

        x_new = x + d

        # Single mega-kernel call: energy + gradient
        x_flat = mx.reshape(x_new, (total_padded,))
        e_new, g_new = mmff_energy_grad_metal_mega(
            idx_buf, param_buf, all_meta, dispatch, pos_offsets,
            x_flat, total_confs, total_padded,
        )
        g_new_sc = mx.reshape(g_new, (total_confs, max_dim)) * gs_val

        # L-BFGS history update
        s_k = x_new - x
        y_k = g_new_sc - g_sc
        sy = mx.sum(s_k * y_k, axis=1, keepdims=True)
        rho_k = mx.where(
            sy > EPS_CURV, 1.0 / (sy + 1e-30), mx.zeros_like(sy)
        )

        if len(s_hist) >= lbfgs_m:
            s_hist.pop(0)
            y_hist.pop(0)
            rho_hist.pop(0)
        s_hist.append(s_k)
        y_hist.append(y_k)
        rho_hist.append(rho_k)

        x = x_new
        g_sc = g_new_sc

        # Periodic sync + energy guard
        if (it + 1) % eval_freq == 0:
            mx.eval(x, g_sc, e_new, *s_hist, *y_hist, *rho_hist)

            # Per-conformer monotone guard: keep best positions
            improved = mx.expand_dims(e_new < best_e, axis=1)
            best_x = mx.where(improved, x, best_x)
            best_e = mx.minimum(e_new, best_e)
            mx.eval(best_x, best_e)

    # At the end, revert ALL conformers to their best-seen positions
    # and recompute energy + gradient at those positions
    mx.eval(best_x)
    x = best_x
    x_flat = mx.reshape(x, (total_padded,))
    e_at_best, g_at_best = mmff_energy_grad_metal_mega(
        idx_buf, param_buf, all_meta, dispatch, pos_offsets,
        x_flat, total_confs, total_padded,
    )
    g_sc = mx.reshape(g_at_best, (total_confs, max_dim)) * gs_val
    mx.eval(x, g_sc, e_at_best)

    # Run a short "polishing" pass from best positions (fresh L-BFGS)
    s_hist = []
    y_hist = []
    rho_hist = []
    polish_iters = max_iters // 4
    for it in range(polish_iters):
        d = _lbfgs_direction_batch(g_sc, s_hist, y_hist, rho_hist)
        d3 = mx.reshape(d, (total_confs, max_atoms, 3))
        atom_disp = mx.sqrt(
            mx.sum(d3 * d3, axis=-1, keepdims=True) + 1e-30
        )
        d = mx.reshape(
            d3 * mx.minimum(max_step_mx / atom_disp, one_f),
            (total_confs, max_dim),
        )
        x_new = x + d
        x_flat = mx.reshape(x_new, (total_padded,))
        e_new, g_new = mmff_energy_grad_metal_mega(
            idx_buf, param_buf, all_meta, dispatch, pos_offsets,
            x_flat, total_confs, total_padded,
        )
        g_new_sc = mx.reshape(g_new, (total_confs, max_dim)) * gs_val

        s_k = x_new - x
        y_k = g_new_sc - g_sc
        sy = mx.sum(s_k * y_k, axis=1, keepdims=True)
        rho_k = mx.where(
            sy > EPS_CURV, 1.0 / (sy + 1e-30), mx.zeros_like(sy)
        )
        if len(s_hist) >= lbfgs_m:
            s_hist.pop(0)
            y_hist.pop(0)
            rho_hist.pop(0)
        s_hist.append(s_k)
        y_hist.append(y_k)
        rho_hist.append(rho_k)

        x = x_new
        g_sc = g_new_sc

        if (it + 1) % eval_freq == 0:
            mx.eval(x, g_sc, e_new, *s_hist, *y_hist, *rho_hist)
            improved = mx.expand_dims(e_new < best_e, axis=1)
            best_x = mx.where(improved, x, best_x)
            best_e = mx.minimum(e_new, best_e)
            mx.eval(best_x, best_e)

    # Final: use absolute best positions
    x = best_x

    # ---- 5. Final energies + write back --------------------------------
    mx.eval(x)
    x_flat = mx.reshape(x, (total_padded,))
    final_e, _ = mmff_energy_grad_metal_mega(
        idx_buf, param_buf, all_meta, dispatch, pos_offsets,
        x_flat, total_confs, total_padded,
    )
    g_norms_3d = mx.reshape(g_sc, (total_confs, max_atoms, 3))
    g_norms_per_atom = mx.sqrt(mx.sum(g_norms_3d ** 2, axis=-1))
    mx.eval(final_e, g_norms_per_atom)

    x_np = np.array(x)  # (total_confs, max_dim)
    e_np = np.array(final_e)
    gn_np = np.array(g_norms_per_atom)  # (total_confs, max_atoms)

    results: list[MetalBatchResult] = []
    conf_cursor = 0
    for m_idx, mol in enumerate(mols):
        C = conf_counts[m_idx]
        n_atoms = mol.GetNumAtoms()

        mol_pos = x_np[conf_cursor : conf_cursor + C, : n_atoms * 3].reshape(
            C, n_atoms, 3
        )
        mol_e = e_np[conf_cursor : conf_cursor + C]
        mol_gn = gn_np[conf_cursor : conf_cursor + C, :n_atoms]
        mol_gn_max = np.max(mol_gn, axis=1)

        for k, cid in enumerate(all_conf_ids[m_idx]):
            conf = mol.GetConformer(cid)
            for a in range(n_atoms):
                conf.SetAtomPosition(
                    a, mol_pos[k, a].astype(float).tolist()
                )

        results.append(
            MetalBatchResult(
                energies=mol_e,
                positions=mol_pos,
                grad_norms=mol_gn_max,
                n_iters=max_iters,
                n_conformers=C,
            )
        )
        conf_cursor += C

    return MultiMolResult(mol_results=results)


# ---------------------------------------------------------------------------
# Mode 3: Fully fused GPU optimiser (zero Python loop overhead)
# ---------------------------------------------------------------------------


def mmff_optimize_metal_fused_multi_mol(
    mol_specs: list[tuple[Chem.Mol, Optional[list[int]]]],
    max_iters: int = 500,
    max_step: float = 0.3,
    grad_lr: float = 0.1,
) -> MultiMolResult:
    """
    Optimise many molecules in a **single Metal kernel launch**.

    The entire optimisation loop (energy → gradient → SD step × max_iters)
    runs inside the GPU kernel.  Python only does parameter extraction
    and packing (once), then a single kernel call, then reads back results.

    Steepest descent converges slower than L-BFGS so default max_iters
    is higher (500 vs 200).

    Args:
        mol_specs: list of (mol, conf_ids) tuples.
        max_iters: gradient descent iterations (default 500).
        max_step: max per-atom displacement per step (Angstrom).
        grad_lr: gradient learning rate (step = min(max_step/gnorm, grad_lr)).

    Returns:
        MultiMolResult containing per-molecule MetalBatchResult objects.
    """
    mols: list[Chem.Mol] = []
    all_conf_ids: list[list[int]] = []
    params_list: list = []
    conf_counts: list[int] = []

    for mol, conf_ids in mol_specs:
        if conf_ids is None:
            conf_ids = [c.GetId() for c in mol.GetConformers()]
        mols.append(mol)
        all_conf_ids.append(conf_ids)
        params_list.append(extract_mmff_params(mol, conf_id=conf_ids[0]))
        conf_counts.append(len(conf_ids))

    total_confs = sum(conf_counts)
    dims_per_mol = [p.n_atoms * 3 for p in params_list]
    max_dim = max(dims_per_mol)
    total_padded = total_confs * max_dim

    # Pack parameters (mega-kernel dispatch table)
    (
        idx_buf, param_buf, all_meta, dispatch, _, _,
    ) = pack_multi_mol_for_metal(params_list, conf_counts)

    # Padded position offsets
    pos_offsets = mx.array(
        np.arange(total_confs, dtype=np.uint32) * max_dim
    )

    # Fill padded position buffer
    pos_np = np.zeros((total_confs, max_dim), dtype=np.float32)
    ci = 0
    for m_idx, (mol, cids) in enumerate(zip(mols, all_conf_ids)):
        n_atoms = mol.GetNumAtoms()
        for cid in cids:
            conf = mol.GetConformer(cid)
            for a in range(n_atoms):
                p = conf.GetAtomPosition(a)
                pos_np[ci, a * 3] = p.x
                pos_np[ci, a * 3 + 1] = p.y
                pos_np[ci, a * 3 + 2] = p.z
            ci += 1

    pos_flat = mx.array(pos_np.ravel())

    # ONE kernel call — entire optimization
    pos_out, energy_out, _ = mmff_optimize_metal_fused(
        idx_buf, param_buf, all_meta, dispatch, pos_offsets,
        pos_flat, total_confs, total_padded,
        max_iters=max_iters, max_step=max_step, grad_lr=grad_lr,
    )
    mx.eval(pos_out, energy_out)

    pos_out_np = np.array(pos_out).reshape(total_confs, max_dim)
    e_np = np.array(energy_out)

    # Unpack results per molecule and write back to RDKit
    results: list[MetalBatchResult] = []
    conf_cursor = 0
    for m_idx, mol in enumerate(mols):
        C = conf_counts[m_idx]
        n_atoms = mol.GetNumAtoms()

        mol_pos = pos_out_np[
            conf_cursor : conf_cursor + C, : n_atoms * 3
        ].reshape(C, n_atoms, 3)
        mol_e = e_np[conf_cursor : conf_cursor + C]

        for k, cid in enumerate(all_conf_ids[m_idx]):
            conf = mol.GetConformer(cid)
            for a in range(n_atoms):
                conf.SetAtomPosition(
                    a, mol_pos[k, a].astype(float).tolist()
                )

        results.append(
            MetalBatchResult(
                energies=mol_e,
                positions=mol_pos,
                grad_norms=mol_e * 0,  # not tracked in fused kernel
                n_iters=max_iters,
                n_conformers=C,
            )
        )
        conf_cursor += C

    return MultiMolResult(mol_results=results)
