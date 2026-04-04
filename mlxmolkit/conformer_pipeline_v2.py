"""
N×k conformer generation pipeline with divide-and-conquer memory management.

Full pipeline: SMILES → DG (4D) → 4D→3D → ETK (3D) → MMFF94 (optional)

The divide-and-conquer queue splits conformers into GPU-sized batches.
Each batch: DG → extract 3D → ETK → (optional MMFF) → accumulate on CPU.
GPU memory is released between batches.

Constraints are stored ONCE per molecule (SharedConstraintBatch).
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import mlx.core as mx

from .dg_extract import DGParams, extract_dg_params, get_bounds_matrix
from .etk_extract import ETKParams, extract_etk_params
from .shared_batch import (
    SharedConstraintBatch,
    pack_shared_dg_batch,
    add_etk_to_batch,
    init_random_positions,
)
from .conformer_metal import dg_minimize_shared
from .etk_metal import etk_minimize_shared
from .mmff_params import MMFFParams, extract_mmff_params
from .mmff_minimize import mmff_minimize_nk


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConformerResult:
    """Result for one molecule."""
    n_atoms: int
    positions_3d: List[np.ndarray]   # list of (n_atoms, 3) arrays
    energies: List[float]
    converged: List[bool]


@dataclass
class PipelineResult:
    """Result for the full N×k pipeline."""
    molecules: List[ConformerResult]
    total_conformers: int
    total_time: float
    n_batches: int


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _get_free_memory_bytes() -> int:
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        page_size = 16384
        free_pages = 0
        for line in result.stdout.splitlines():
            if "Pages free" in line or "Pages speculative" in line:
                free_pages += int(line.split(":")[1].strip().rstrip("."))
        if free_pages > 0:
            return free_pages * page_size
    except Exception:
        pass
    return 4 * 1024 ** 3


def _compute_max_confs_per_batch(
    mol_n_atoms: List[int], dim: int = 4,
    max_memory_bytes: Optional[int] = None, lbfgs_m: int = 8,
) -> int:
    if max_memory_bytes is None:
        max_memory_bytes = _get_free_memory_bytes() // 2
    avg_atoms = int(np.mean(mol_n_atoms)) if mol_n_atoms else 30
    n_vars = avg_atoms * dim
    # pos + grad + dir + 3×scratch + 2×m×lbfgs + rho + alpha + outputs
    mem_per_conf = n_vars * 4 * 7 + 2 * lbfgs_m * n_vars * 4 + lbfgs_m * 8 + n_vars * 4 + 8
    return max(1, max_memory_bytes // max(mem_per_conf, 1))


# ---------------------------------------------------------------------------
# Chunk scheduler
# ---------------------------------------------------------------------------

def _build_chunk_schedule(
    n_confs_per_mol: List[int], max_confs_per_batch: int,
) -> List[List[tuple]]:
    chunks: List[List[tuple]] = []
    current_chunk: List[tuple] = []
    current_count = 0
    for mol_idx, k in enumerate(n_confs_per_mol):
        remaining, offset = k, 0
        while remaining > 0:
            space = max_confs_per_batch - current_count
            take = min(remaining, space)
            current_chunk.append((mol_idx, offset, offset + take))
            current_count += take
            offset += take
            remaining -= take
            if current_count >= max_confs_per_batch:
                chunks.append(current_chunk)
                current_chunk = []
                current_count = 0
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------

def _auto_iters(max_atoms: int, base: int, scale: float) -> int:
    """Scale iterations by largest molecule size. Small molecules converge
    early via in-kernel TOLX/grad checks — no wasted compute."""
    return max(base, int(base + scale * max_atoms))


def _process_chunk(
    chunk: List[tuple],
    dg_params_list: List[DGParams],
    etk_params_list: Optional[List[ETKParams]],
    mols_list: Optional[list],
    run_mmff: bool,
    seed_offset: int,
    dg_max_iters: int,
    etk_max_iters: int,
    mmff_max_iters: int,
    mmff_use_lbfgs: bool,
    fourth_dim_weight: float,
    chiral_weight: float,
) -> List[tuple]:
    """Run DG → 3D → ETK → MMFF on one chunk. Returns per-conformer results."""

    # Identify molecules in this chunk
    mol_k: dict[int, int] = {}
    mol_order: List[int] = []
    for mol_idx, c_start, c_end in chunk:
        if mol_idx not in mol_k:
            mol_k[mol_idx] = 0
            mol_order.append(mol_idx)
        mol_k[mol_idx] += c_end - c_start

    chunk_dg = [dg_params_list[m] for m in mol_order]
    chunk_k = [mol_k[m] for m in mol_order]
    C = sum(chunk_k)

    # Auto-scale iterations by largest molecule in this chunk.
    # Larger molecules have more constraints → harder energy landscape.
    # Small molecules converge early via in-kernel TOLX/grad checks.
    max_atoms = max(dg_params_list[m].n_atoms for m in mol_order)
    max_constraints = max(len(dg_params_list[m].dist_idx1) for m in mol_order)
    complexity = max(max_atoms, int(max_constraints ** 0.5))
    if dg_max_iters <= 0:
        dg_max_iters = _auto_iters(complexity, base=300, scale=20.0)
    if etk_max_iters <= 0:
        etk_max_iters = _auto_iters(complexity, base=150, scale=10.0)
    if mmff_max_iters <= 0:
        mmff_max_iters = _auto_iters(complexity, base=200, scale=15.0)

    # ---- Stage 1: DG minimize (4D) ----
    batch4 = pack_shared_dg_batch(chunk_dg, chunk_k, dim=4)
    pos4 = init_random_positions(batch4, seed=42 + seed_offset)
    dg_out, dg_e, dg_s = dg_minimize_shared(
        batch4, pos4, max_iters=dg_max_iters,
        fourth_dim_weight=fourth_dim_weight, chiral_weight=chiral_weight,
    )

    # ---- Stage 2: Extract 3D from 4D ----
    batch3 = pack_shared_dg_batch(chunk_dg, chunk_k, dim=3)
    pos3 = np.zeros(int(batch3.conf_atom_starts[-1]) * 3, dtype=np.float32)
    for c in range(C):
        n_a = batch3.mol_n_atoms[batch3.conf_to_mol[c]]
        s4 = int(batch4.conf_atom_starts[c]) * 4
        p4 = dg_out[s4:s4 + n_a * 4].reshape(n_a, 4)
        s3 = int(batch3.conf_atom_starts[c]) * 3
        pos3[s3:s3 + n_a * 3] = p4[:, :3].flatten()

    # ---- Stage 3: ETK minimize (3D) ----
    etk_e = np.zeros(C, dtype=np.float32)
    etk_s = np.ones(C, dtype=np.int32)
    if etk_params_list is not None:
        chunk_etk = [etk_params_list[m] for m in mol_order]
        add_etk_to_batch(batch3, chunk_etk)
        # Check if any ETK terms exist
        has_etk = (
            (batch3.etk_torsion_term_starts is not None and batch3.etk_torsion_term_starts[-1] > 0)
            or (batch3.etk_improper_term_starts is not None and batch3.etk_improper_term_starts[-1] > 0)
            or (batch3.etk_dist14_term_starts is not None and batch3.etk_dist14_term_starts[-1] > 0)
        )
        if has_etk:
            etk_out, etk_e, etk_s = etk_minimize_shared(
                batch3, pos3, max_iters=etk_max_iters,
            )
            pos3 = etk_out  # use ETK-refined positions

    # ---- Stage 4: MMFF94 optimization (3D) ----
    mmff_e = np.zeros(C, dtype=np.float32)
    if run_mmff and mols_list is not None:
        from rdkit.Chem import AllChem
        chunk_mmff = []
        # Use first conformer of each molecule for MMFF param extraction
        conf_cursor = 0
        for chunk_mol_ord, mol_idx in enumerate(mol_order):
            mol = mols_list[mol_idx]
            n_a = dg_params_list[mol_idx].n_atoms
            s3 = int(batch3.conf_atom_starts[conf_cursor]) * 3
            conf_pos = pos3[s3:s3 + n_a * 3].reshape(n_a, 3)
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            conf = mol.GetConformer(0)
            for a_idx in range(n_a):
                conf.SetAtomPosition(a_idx, conf_pos[a_idx].astype(float).tolist())
            try:
                chunk_mmff.append(extract_mmff_params(mol))
            except Exception:
                chunk_mmff = None
                break
            conf_cursor += mol_k[mol_idx]

        if chunk_mmff is not None:
            chunk_mmff_k = [mol_k[m] for m in mol_order]
            pos3, mmff_e, _ = mmff_minimize_nk(
                chunk_mmff, chunk_mmff_k, pos3,
                max_iters=mmff_max_iters,
                use_lbfgs=mmff_use_lbfgs,
            )

    # ---- Collect results per conformer ----
    results = []
    c = 0
    for mol_idx in mol_order:
        k = mol_k[mol_idx]
        for lk in range(k):
            n_a = dg_params_list[mol_idx].n_atoms
            s3 = int(batch3.conf_atom_starts[c]) * 3
            p3 = pos3[s3:s3 + n_a * 3].reshape(n_a, 3).copy()
            results.append((
                mol_idx, p3,
                float(mmff_e[c]) if run_mmff else float(dg_e[c]) + float(etk_e[c]),
                bool(dg_s[c] == 0),
            ))
            c += 1

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_conformers_nk(
    smiles_list: Sequence[str],
    n_confs_per_mol: int | List[int] = 10,
    *,
    max_confs_per_batch: Optional[int] = None,
    max_memory_bytes: Optional[int] = None,
    dg_max_iters: int = 0,
    etk_max_iters: int = 0,
    fourth_dim_weight: float = 0.1,
    chiral_weight: float = 1.0,
    variant: str = "ETKDGv2",
    run_mmff: bool = False,
    mmff_max_iters: int = 0,
    mmff_use_lbfgs: bool = False,
) -> PipelineResult:
    """Generate 3D conformers for N molecules x k conformers each.

    Full pipeline: SMILES → DG (4D) → 3D → ETK (3D) → MMFF94 (optional)

    Supports all ETKDG variants: DG, KDG, ETDG, ETKDG, ETKDGv2, ETKDGv3,
    srETKDGv3.  The variant controls which ETK terms are active.

    Parameters
    ----------
    smiles_list : list of str
        N SMILES strings.
    n_confs_per_mol : int or list of int
        k conformers per molecule.
    max_confs_per_batch : int, optional
        Max conformers per GPU batch (auto-computed from free memory).
    dg_max_iters : int
        L-BFGS iterations for DG stage. 0 (default) = auto-scale by
        molecule complexity: ``300 + 20 * max(n_atoms, sqrt(n_constraints))``.
        Small molecules converge early via in-kernel checks — no wasted compute.
    etk_max_iters : int
        L-BFGS iterations for ETK stage. 0 = auto-scale.
    variant : str
        ETKDG variant: DG, KDG, ETDG, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3.
    run_mmff : bool
        Whether to run MMFF94 force field optimization (default False).
    mmff_max_iters : int
        L-BFGS iterations for MMFF stage.
    """
    from rdkit import Chem

    t_start = time.time()
    N = len(smiles_list)
    k_list = [n_confs_per_mol] * N if isinstance(n_confs_per_mol, int) else list(n_confs_per_mol)

    # Determine which ETK stages to run based on variant
    run_etk = variant != "DG"

    # ---- Extract per-molecule params (CPU, once) ----
    mols, dg_params_list, etk_params_list, mmff_params_list_all, mol_n_atoms = [], [], [], [], []
    bmats = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi}")
        mol = Chem.AddHs(mol)
        mols.append(mol)
        bmat = get_bounds_matrix(mol)
        bmats.append(bmat)
        dg_params_list.append(extract_dg_params(mol, bmat, dim=4))
        if run_etk:
            etk_params_list.append(extract_etk_params(mol, bmat, variant=variant))
        mol_n_atoms.append(dg_params_list[-1].n_atoms)

    # MMFF extraction deferred — needs a conformer, we'll embed after DG+ETK
    # to avoid wasting time on a separate RDKit EmbedMolecule call

    # ---- Compute batch size ----
    if max_confs_per_batch is None:
        max_confs_per_batch = _compute_max_confs_per_batch(
            mol_n_atoms, dim=4, max_memory_bytes=max_memory_bytes,
        )

    # ---- Build chunk schedule ----
    chunks = _build_chunk_schedule(k_list, max_confs_per_batch)

    # ---- Process chunks (divide-and-conquer) ----
    mol_results = [
        ConformerResult(n_atoms=dg_params_list[i].n_atoms, positions_3d=[], energies=[], converged=[])
        for i in range(N)
    ]
    total_confs = 0
    for chunk_idx, chunk in enumerate(chunks):
        chunk_results = _process_chunk(
            chunk, dg_params_list,
            etk_params_list if run_etk else None,
            mols if run_mmff else None,
            run_mmff,
            seed_offset=chunk_idx * 10000,
            dg_max_iters=dg_max_iters,
            etk_max_iters=etk_max_iters,
            mmff_max_iters=mmff_max_iters,
            mmff_use_lbfgs=mmff_use_lbfgs,
            fourth_dim_weight=fourth_dim_weight,
            chiral_weight=chiral_weight,
        )
        for mol_idx, pos_3d, energy, converged in chunk_results:
            mol_results[mol_idx].positions_3d.append(pos_3d)
            mol_results[mol_idx].energies.append(energy)
            mol_results[mol_idx].converged.append(converged)
        total_confs += sum(c_end - c_start for _, c_start, c_end in chunk)

    return PipelineResult(
        molecules=mol_results,
        total_conformers=total_confs,
        total_time=time.time() - t_start,
        n_batches=len(chunks),
    )
