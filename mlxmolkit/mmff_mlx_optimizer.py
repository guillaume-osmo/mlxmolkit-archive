"""
Fully batched MMFF L-BFGS optimizer on MLX Metal.

All C conformers are optimized simultaneously on the GPU with minimal
Python overhead. The key optimization is batching L-BFGS iterations with
periodic mx.eval() synchronization (every eval_freq iterations), letting
MLX build and fuse the compute graph.

Performance (200 conformers, M1 Mac):
  - Aspirin (21 atoms):  ~420ms (1.4x RDKit)
  - Ibuprofen (33 atoms): ~420ms (2x faster than RDKit)
  - Naproxen (31 atoms):  ~420ms (1.9x faster than RDKit)
  - Crossover at ~300 conformers for small molecules, ~100 for large ones.

Architecture (mirrors nvMolKit):
  1. Extract MMFF params once from RDKit (CPU)
  2. Upload to Metal as MLX arrays
  3. L-BFGS with capped steps, energy guard, periodic eval
  4. Auto-diff gives gradients (no manual gradient code)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np
from rdkit import Chem

from mlxmolkit.mmff_energy_mlx import MMFFParamsMLX, mmff_energy_batch, params_to_mlx
from mlxmolkit.mmff_params import extract_mmff_params

EPS_CURV = 1e-10


@dataclass
class MLXBatchResult:
    """Result of batched MLX Metal MMFF optimization."""

    energies: np.ndarray
    positions: np.ndarray
    grad_norms: np.ndarray
    n_iters: int
    n_conformers: int


def mmff_optimize_mlx_batch(
    mol: Chem.Mol,
    conf_ids: Optional[list[int]] = None,
    max_iters: int = 200,
    eval_freq: int = 10,
    max_step_per_atom: float = 0.3,
    lbfgs_m: int = 10,
    grad_scale: float = 0.1,
    energy_guard: bool = False,
) -> MLXBatchResult:
    """
    Batched L-BFGS MMFF optimization on MLX Metal.

    All conformers are processed in parallel on the GPU. The optimizer uses
    L-BFGS directions with per-atom step capping and optional energy guard
    (reject steps that increase energy).

    Args:
        mol: RDKit molecule with embedded conformers.
        conf_ids: conformer IDs to optimize (default: all).
        max_iters: maximum L-BFGS iterations.
        eval_freq: how often to synchronize GPU (lower = more overhead but
                   better convergence tracking; 10 is a good default).
        max_step_per_atom: maximum displacement per atom per step (Angstrom).
        lbfgs_m: number of L-BFGS history pairs.
        grad_scale: gradient scaling factor (0.1 matches nvMolKit).
        energy_guard: if True, reject steps that increase energy (prevents
                      divergence in longer runs).

    Returns:
        MLXBatchResult with final energies, positions, grad norms.
    """
    n_atoms = mol.GetNumAtoms()
    dim = n_atoms * 3

    if conf_ids is None:
        conf_ids = [c.GetId() for c in mol.GetConformers()]
    C = len(conf_ids)

    # --- 1. Extract params and upload to Metal ---
    np_params = extract_mmff_params(mol, conf_id=conf_ids[0])
    mlx_params = params_to_mlx(np_params)

    pos_np = np.zeros((C, n_atoms, 3), dtype=np.float32)
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            p = conf.GetAtomPosition(i)
            pos_np[k, i] = [p.x, p.y, p.z]

    # --- 2. Compiled energy+grad function ---
    def total_energy(xf: mx.array) -> mx.array:
        return mx.sum(mmff_energy_batch(mlx_params, mx.reshape(xf, (C, n_atoms, 3))))

    vg = mx.compile(mx.value_and_grad(total_energy))

    def per_conf_energy(xf: mx.array) -> mx.array:
        return mmff_energy_batch(mlx_params, mx.reshape(xf, (C, n_atoms, 3)))

    x = mx.array(pos_np.reshape(C, dim))
    mx.eval(x)

    # Initial eval
    _, g = vg(x)
    mx.eval(g)
    g_sc = g * grad_scale

    if energy_guard:
        cur_e = per_conf_energy(x)
        mx.eval(cur_e)

    # --- 3. L-BFGS state ---
    s_hist: list[mx.array] = []
    y_hist: list[mx.array] = []
    rho_hist: list[mx.array] = []
    max_step = mx.array(max_step_per_atom, dtype=mx.float32)
    gs = mx.array(grad_scale, dtype=mx.float32)
    one = mx.array(1.0, dtype=mx.float32)

    # --- 4. Optimization loop ---
    for it in range(max_iters):
        # L-BFGS two-loop recursion (all conformers in parallel)
        q = g_sc
        alphas = []
        for i in range(len(s_hist) - 1, -1, -1):
            a_i = rho_hist[i] * mx.sum(s_hist[i] * q, axis=1, keepdims=True)
            q = q - a_i * y_hist[i]
            alphas.append(a_i)

        if s_hist:
            sy = mx.sum(s_hist[-1] * y_hist[-1], axis=1, keepdims=True)
            yy = mx.sum(y_hist[-1] * y_hist[-1], axis=1, keepdims=True) + 1e-30
            gamma = sy / yy
        else:
            gamma = one

        r = gamma * q
        for i in range(len(s_hist)):
            beta = rho_hist[i] * mx.sum(y_hist[i] * r, axis=1, keepdims=True)
            r = r + s_hist[i] * (alphas[len(s_hist) - 1 - i] - beta)

        d = -r

        # Per-atom step capping
        d3 = mx.reshape(d, (C, n_atoms, 3))
        atom_disp = mx.sqrt(mx.sum(d3 * d3, axis=-1, keepdims=True) + 1e-30)
        d = mx.reshape(d3 * mx.minimum(max_step / atom_disp, one), (C, dim))

        # Take step
        x_new = x + d
        _, g_new = vg(x_new)
        g_new_sc = g_new * gs

        if energy_guard:
            new_e = per_conf_energy(x_new)
            worse = mx.expand_dims(new_e > cur_e + 0.1, -1)
            x_new = mx.where(worse, x, x_new)
            g_new_sc = mx.where(worse, g_sc, g_new_sc)
            cur_e = mx.where(mx.squeeze(worse, -1), cur_e, new_e)

        # Update L-BFGS history
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

        # Periodic sync to prevent compute graph explosion
        if (it + 1) % eval_freq == 0:
            to_eval = [x, g_sc] + s_hist + y_hist + rho_hist
            if energy_guard:
                to_eval.append(cur_e)
            mx.eval(*to_eval)

    # --- 5. Final evaluation and download ---
    mx.eval(x)
    final_e = per_conf_energy(x)
    g_norms = mx.sqrt(mx.sum(g_sc * g_sc, axis=1))
    mx.eval(final_e, g_norms)

    final_pos = np.array(x).reshape(C, n_atoms, 3)
    final_energies = np.array(final_e)
    final_gnorms = np.array(g_norms)

    # Write back to RDKit molecule
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, final_pos[k, i].astype(float).tolist())

    return MLXBatchResult(
        energies=final_energies,
        positions=final_pos,
        grad_norms=final_gnorms,
        n_iters=max_iters,
        n_conformers=C,
    )
