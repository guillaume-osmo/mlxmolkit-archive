"""
Full 7-stage ETKDG conformer generation pipeline on Metal/MLX.

Pipeline stages:
  1. Random 4D coordinate generation
  2. 4D Distance Geometry minimization (Metal L-BFGS)
  3. Tetrahedral + chirality checks
  4. 4th dimension collapse (Metal L-BFGS)
  5. ETK torsion minimization (Metal L-BFGS, 3D)
  6-7. Double bond geometry/stereo + distance matrix checks

Key design:
  - Preprocessing (bounds matrix, DG params, ETK params) computed ONCE
    per molecule and reused across all conformers and retries
  - N conformers batched into a single GPU dispatch
  - Failed conformers retried with new random seeds without recomputing
    molecular parameters
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.dg_extract import (
    get_bounds_matrix, extract_dg_params, batch_dg_params,
    DGParams, BatchedDGSystem,
)
from mlxmolkit.dg_minimize_metal import (
    dg_minimize_batch, dg_collapse_4th_dim, extract_3d_coords,
    generate_random_4d_coords,
)
from mlxmolkit.etk_extract import (
    extract_etk_params, batch_etk_params,
    ETKParams, BatchedETKSystem,
)
from mlxmolkit.etk_minimize_metal import etk_minimize_batch
from mlxmolkit.stereo_checks import run_stage3_checks, run_stage67_checks


@dataclass
class MoleculeCache:
    """Cached preprocessing for a single molecule (reused across conformers/retries)."""
    mol: Chem.Mol
    bounds_mat: np.ndarray
    dg_params: DGParams
    etk_params: ETKParams
    n_atoms: int


@dataclass
class ConformerResult:
    """Result for a single molecule's conformer generation."""
    smiles: str
    n_requested: int
    n_generated: int
    coords_3d: list[np.ndarray]     # list of (n_atoms, 3) arrays
    energies_dg: list[float]
    energies_etk: list[float]
    success: bool


@dataclass
class PipelineResult:
    """Result of the full pipeline for multiple molecules."""
    results: list[ConformerResult]
    total_conformers: int
    total_attempted: int


def _preprocess_molecule(smiles: str) -> MoleculeCache | None:
    """Preprocess a single molecule: extract bounds matrix, DG params, ETK params."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    try:
        bounds_mat = get_bounds_matrix(mol)
    except Exception:
        return None

    dg_params = extract_dg_params(mol, bounds_mat)
    etk_params = extract_etk_params(mol, bounds_mat)

    return MoleculeCache(
        mol=mol,
        bounds_mat=bounds_mat,
        dg_params=dg_params,
        etk_params=etk_params,
        n_atoms=mol.GetNumAtoms(),
    )


def _run_pipeline_batch(
    caches: list[MoleculeCache],
    n_confs_per_mol: list[int],
    seed: int = 42,
    dg_max_iters: int = 200,
    dg_grad_tol: float = 1e-3,
    collapse_weight: float = 1000.0,
    collapse_max_iters: int = 200,
    etk_max_iters: int = 200,
    etk_grad_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BatchedDGSystem, list[int]]:
    """
    Run stages 1-5 on a batch of (molecule, conformer) pairs.

    Returns (pos_3d, energies_dg, energies_etk, dg_system, mol_map)
    where mol_map[i] is the cache index for conformer i.
    """
    # Build flat lists of DG params and ETK params
    dg_params_list = []
    etk_params_list = []
    mol_map = []  # mol_map[i] = index into caches

    for cache_idx, (cache, n_c) in enumerate(zip(caches, n_confs_per_mol)):
        for _ in range(n_c):
            dg_params_list.append(cache.dg_params)
            etk_params_list.append(cache.etk_params)
            mol_map.append(cache_idx)

    total_confs = len(dg_params_list)
    if total_confs == 0:
        empty = np.zeros(0, dtype=np.float32)
        dummy_system = batch_dg_params([])  if False else None
        return empty, empty, empty, None, mol_map

    # Batch DG params
    dg_system = batch_dg_params(dg_params_list)

    # Stage 1: Random 4D coordinates
    x0 = generate_random_4d_coords(dg_system, seed=seed)

    # Stage 2: DG minimize
    res_dg = dg_minimize_batch(
        dg_system, x0=x0, max_iters=dg_max_iters, grad_tol=dg_grad_tol,
    )

    # Stage 4: 4D → 3D collapse
    res_col = dg_collapse_4th_dim(
        dg_system, res_dg.positions,
        fourth_dim_weight=collapse_weight,
        max_iters=collapse_max_iters,
    )

    pos_3d = extract_3d_coords(
        res_col.positions, res_col.atom_starts, res_col.n_mols, dim=4,
    )

    energies_dg = res_dg.energies

    # Stage 5: ETK minimize (3D)
    etk_system = batch_etk_params(etk_params_list, dg_system.atom_starts)
    res_etk = etk_minimize_batch(
        etk_system, pos_3d, max_iters=etk_max_iters, grad_tol=etk_grad_tol,
    )

    return res_etk.positions, energies_dg, res_etk.energies, dg_system, mol_map


def generate_conformers_pipeline(
    smiles_list: list[str],
    n_confs: int = 10,
    max_attempts: int = 5,
    seed: int = 42,
    dg_max_iters: int = 500,
    dg_grad_tol: float = 1e-4,
    collapse_weight: float = 1000.0,
    collapse_max_iters: int = 300,
    etk_max_iters: int = 200,
    etk_grad_tol: float = 1e-3,
    check_stereo: bool = True,
) -> PipelineResult:
    """
    Full 7-stage ETKDG conformer generation pipeline.

    Generates n_confs conformers per molecule with retry logic.
    Preprocessing is cached per molecule and reused across retries.

    Args:
        smiles_list: List of SMILES strings.
        n_confs: Number of conformers to generate per molecule.
        max_attempts: Maximum retry rounds for failed conformers.
        seed: Base random seed.
        dg_max_iters: Max L-BFGS iterations for DG minimization.
        dg_grad_tol: Gradient tolerance for DG minimization.
        collapse_weight: 4th dimension penalty weight for collapse.
        collapse_max_iters: Max iterations for 4D collapse.
        etk_max_iters: Max iterations for ETK minimization.
        etk_grad_tol: Gradient tolerance for ETK minimization.
        check_stereo: Whether to run stereochemistry checks.
    """
    # --- Preprocessing (once per molecule) ---
    caches: list[MoleculeCache | None] = []
    for smi in smiles_list:
        caches.append(_preprocess_molecule(smi))

    # Per-molecule tracking
    n_mols = len(smiles_list)
    accepted_coords: list[list[np.ndarray]] = [[] for _ in range(n_mols)]
    accepted_e_dg: list[list[float]] = [[] for _ in range(n_mols)]
    accepted_e_etk: list[list[float]] = [[] for _ in range(n_mols)]
    needed: list[int] = [n_confs if c is not None else 0 for c in caches]
    total_attempted = 0

    for attempt in range(max_attempts):
        # Build batch of molecules that still need conformers
        active_caches = []
        active_confs = []
        active_mol_idx = []

        for i in range(n_mols):
            if needed[i] <= 0 or caches[i] is None:
                continue
            active_caches.append(caches[i])
            active_confs.append(needed[i])
            active_mol_idx.append(i)

        if not active_caches:
            break

        current_seed = seed + attempt * 10000

        pos_3d, e_dg, e_etk, dg_system, mol_map = _run_pipeline_batch(
            active_caches, active_confs,
            seed=current_seed,
            dg_max_iters=dg_max_iters,
            dg_grad_tol=dg_grad_tol,
            collapse_weight=collapse_weight,
            collapse_max_iters=collapse_max_iters,
            etk_max_iters=etk_max_iters,
            etk_grad_tol=etk_grad_tol,
        )

        if dg_system is None:
            break

        total_confs = sum(active_confs)
        total_attempted += total_confs

        # Evaluate conformers: run checks, accept/reject
        atom_starts = dg_system.atom_starts
        for conf_idx in range(total_confs):
            cache_local_idx = mol_map[conf_idx]
            mol_idx = active_mol_idx[cache_local_idx]
            cache = caches[mol_idx]

            if needed[mol_idx] <= 0:
                continue

            a_s = int(atom_starts[conf_idx])
            a_e = int(atom_starts[conf_idx + 1])
            n_at = a_e - a_s

            coords = pos_3d[a_s * 3: a_e * 3].reshape(n_at, 3)

            passed = True
            if check_stereo:
                # Stage 3: tetrahedral + chirality
                if not run_stage3_checks(
                    pos_3d, cache.mol, cache.bounds_mat, atom_offset=a_s,
                ):
                    passed = False

                # Stages 6-7: double bond + distance matrix
                if passed and not run_stage67_checks(
                    pos_3d, cache.mol, cache.bounds_mat, atom_offset=a_s,
                ):
                    passed = False

            if passed:
                accepted_coords[mol_idx].append(coords.copy())
                accepted_e_dg[mol_idx].append(float(e_dg[conf_idx]))
                accepted_e_etk[mol_idx].append(float(e_etk[conf_idx]))
                needed[mol_idx] -= 1

    # Build results
    results = []
    total_generated = 0
    for i in range(n_mols):
        n_gen = len(accepted_coords[i])
        total_generated += n_gen
        results.append(ConformerResult(
            smiles=smiles_list[i],
            n_requested=n_confs,
            n_generated=n_gen,
            coords_3d=accepted_coords[i],
            energies_dg=accepted_e_dg[i],
            energies_etk=accepted_e_etk[i],
            success=n_gen >= n_confs,
        ))

    return PipelineResult(
        results=results,
        total_conformers=total_generated,
        total_attempted=total_attempted,
    )
