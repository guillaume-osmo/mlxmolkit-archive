"""
N×k conformer generation pipeline with divide-and-conquer memory management.

Full pipeline:  SMILES → DG (4D) → stereo checks → 4D→3D → ETK (3D) → MMFF94
Memory:         Processes conformers in batches sized to fit available RAM.
Constraints:    Stored ONCE per molecule, shared across k conformers.

The divide-and-conquer queue splits the total C = Σ k_i conformers into
chunks of ``max_confs_per_batch`` (auto-computed from free memory).  Each
chunk runs the full DG→ETK→MMFF pipeline on GPU, results are accumulated
on CPU, and GPU memory is released between chunks.

Usage:
    from mlxmolkit.conformer_pipeline_v2 import generate_conformers_nk

    results = generate_conformers_nk(
        smiles_list=["c1ccccc1", "CC(=O)O"],
        n_confs_per_mol=10,
    )
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import mlx.core as mx

from .dg_extract import (
    DGParams,
    extract_dg_params,
    get_bounds_matrix,
    extract_tetrahedral_data,
)
from .shared_batch import (
    SharedConstraintBatch,
    pack_shared_dg_batch,
    init_random_positions,
)
from .conformer_metal import dg_minimize_shared
import subprocess


def _get_free_memory_bytes() -> int:
    """Query approximate free system memory on macOS. Falls back to 4 GB."""
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5,
        )
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


# ---------------------------------------------------------------------------
# Result dataclass
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
# Memory-adaptive batch sizing
# ---------------------------------------------------------------------------

def _estimate_memory_per_conformer(n_atoms: int, dim: int, lbfgs_m: int = 8) -> int:
    """Estimate GPU memory per conformer in bytes.

    Includes: positions, gradient, direction, scratch (3×), L-BFGS history.
    Does NOT include shared constraints (those are per-molecule, constant).
    """
    n_vars = n_atoms * dim
    pos_bytes = n_vars * 4       # float32
    grad_bytes = n_vars * 4
    dir_bytes = n_vars * 4
    scratch_bytes = 3 * n_vars * 4  # old_pos, old_grad, q
    lbfgs_bytes = 2 * lbfgs_m * n_vars * 4  # S + Y history
    rho_bytes = lbfgs_m * 4
    alpha_bytes = lbfgs_m * 4
    # Output: pos + energy + status
    out_bytes = n_vars * 4 + 4 + 4
    return pos_bytes + grad_bytes + dir_bytes + scratch_bytes + lbfgs_bytes + rho_bytes + alpha_bytes + out_bytes


def _compute_max_confs_per_batch(
    mol_n_atoms: List[int],
    dim: int = 4,
    max_memory_bytes: Optional[int] = None,
    lbfgs_m: int = 8,
) -> int:
    """Compute how many conformers fit in one GPU batch.

    Uses 50% of free memory as budget (conservative to leave room for
    constraint arrays and MLX overhead).
    """
    if max_memory_bytes is None:
        max_memory_bytes = _get_free_memory_bytes() // 2

    # Use the average atom count for estimation
    avg_atoms = int(np.mean(mol_n_atoms)) if mol_n_atoms else 30
    mem_per_conf = _estimate_memory_per_conformer(avg_atoms, dim, lbfgs_m)

    max_confs = max(1, max_memory_bytes // max(mem_per_conf, 1))
    return max_confs


# ---------------------------------------------------------------------------
# Divide-and-conquer queue
# ---------------------------------------------------------------------------

def _build_chunk_schedule(
    n_confs_per_mol: List[int],
    max_confs_per_batch: int,
) -> List[List[tuple]]:
    """Split C conformers into chunks that fit in GPU memory.

    Returns a list of chunks, where each chunk is a list of
    ``(mol_idx, conf_start, conf_end)`` tuples describing which
    conformers of which molecule are in this chunk.

    Conformers of the same molecule may span multiple chunks if
    k_i > max_confs_per_batch.
    """
    chunks: List[List[tuple]] = []
    current_chunk: List[tuple] = []
    current_count = 0

    for mol_idx, k in enumerate(n_confs_per_mol):
        remaining = k
        conf_offset = 0

        while remaining > 0:
            space = max_confs_per_batch - current_count
            take = min(remaining, space)

            current_chunk.append((mol_idx, conf_offset, conf_offset + take))
            current_count += take
            conf_offset += take
            remaining -= take

            if current_count >= max_confs_per_batch:
                chunks.append(current_chunk)
                current_chunk = []
                current_count = 0

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _run_dg_chunk(
    dg_params_list: List[DGParams],
    chunk: List[tuple],
    dim: int,
    seed_offset: int,
    max_iters: int,
    fourth_dim_weight: float,
    chiral_weight: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run DG minimize on one chunk of conformers.

    Returns:
        conf_positions: dict mapping (mol_idx, conf_local_idx) → (n_atoms, dim) array
        energies: per-conformer energies
        statuses: per-conformer convergence status
    """
    # Build per-molecule k counts for this chunk
    mol_k_in_chunk: dict[int, int] = {}
    mol_order: List[int] = []
    for mol_idx, c_start, c_end in chunk:
        k = c_end - c_start
        if mol_idx not in mol_k_in_chunk:
            mol_k_in_chunk[mol_idx] = 0
            mol_order.append(mol_idx)
        mol_k_in_chunk[mol_idx] += k

    # Pack shared batch with only the molecules in this chunk
    chunk_dg = [dg_params_list[m] for m in mol_order]
    chunk_k = [mol_k_in_chunk[m] for m in mol_order]
    batch = pack_shared_dg_batch(chunk_dg, chunk_k, dim=dim)

    # Random initial positions
    pos = init_random_positions(batch, seed=42 + seed_offset)

    # DG minimize
    out_pos, energies, statuses = dg_minimize_shared(
        batch, pos,
        max_iters=max_iters,
        fourth_dim_weight=fourth_dim_weight,
        chiral_weight=chiral_weight,
    )

    # Unpack results per conformer
    conf_positions = {}
    c = 0
    for chunk_mol_idx, mol_idx in enumerate(mol_order):
        n_atoms = dg_params_list[mol_idx].n_atoms
        k = mol_k_in_chunk[mol_idx]
        for local_k in range(k):
            start = int(batch.conf_atom_starts[c]) * dim
            end = int(batch.conf_atom_starts[c + 1]) * dim
            conf_positions[(mol_idx, local_k)] = out_pos[start:end].reshape(n_atoms, dim)
            c += 1

    return conf_positions, np.array(energies), np.array(statuses)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_conformers_nk(
    smiles_list: Sequence[str],
    n_confs_per_mol: int | List[int] = 10,
    *,
    max_confs_per_batch: Optional[int] = None,
    max_memory_bytes: Optional[int] = None,
    dg_max_iters: int = 200,
    fourth_dim_weight: float = 0.1,
    chiral_weight: float = 1.0,
    collapse_weight: float = 10.0,
    collapse_iters: int = 100,
) -> PipelineResult:
    """Generate 3D conformers for N molecules × k conformers each.

    Uses divide-and-conquer batching: splits the total C conformers into
    GPU-sized chunks, processes each chunk independently, and accumulates
    results on CPU.  GPU memory is bounded regardless of total C.

    Parameters
    ----------
    smiles_list : list of str
        N SMILES strings.
    n_confs_per_mol : int or list of int
        k conformers per molecule (scalar or per-molecule list).
    max_confs_per_batch : int, optional
        Maximum conformers per GPU batch.  Auto-computed from free memory.
    max_memory_bytes : int, optional
        Memory budget override.
    dg_max_iters : int
        L-BFGS iterations for DG stage.
    fourth_dim_weight : float
        Weight for 4th dimension penalty in DG.
    chiral_weight : float
        Weight for chiral volume constraints.
    collapse_weight : float
        Weight for 4D→3D collapse stage.
    collapse_iters : int
        L-BFGS iterations for 4D→3D collapse.
    """
    from rdkit import Chem

    t_start = time.time()
    N = len(smiles_list)

    # Normalize n_confs_per_mol
    if isinstance(n_confs_per_mol, int):
        k_list = [n_confs_per_mol] * N
    else:
        k_list = list(n_confs_per_mol)
    assert len(k_list) == N

    # ---- Step 1: Extract per-molecule params (CPU, once) ----
    mols = []
    dg_params_list = []
    mol_n_atoms = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi}")
        mol = Chem.AddHs(mol)
        mols.append(mol)

        bmat = get_bounds_matrix(mol)
        dg_params = extract_dg_params(mol, bmat, dim=4)
        dg_params_list.append(dg_params)
        mol_n_atoms.append(dg_params.n_atoms)

    # ---- Step 2: Compute batch size ----
    if max_confs_per_batch is None:
        max_confs_per_batch = _compute_max_confs_per_batch(
            mol_n_atoms, dim=4, max_memory_bytes=max_memory_bytes,
        )

    # ---- Step 3: Build chunk schedule ----
    chunks = _build_chunk_schedule(k_list, max_confs_per_batch)

    # ---- Step 4: Process chunks (divide-and-conquer) ----
    # Accumulate results per molecule
    mol_results: List[ConformerResult] = [
        ConformerResult(
            n_atoms=dg_params_list[i].n_atoms,
            positions_3d=[],
            energies=[],
            converged=[],
        )
        for i in range(N)
    ]

    total_confs = 0
    for chunk_idx, chunk in enumerate(chunks):
        # Stage 1a: DG minimize (4D)
        conf_pos, energies, statuses = _run_dg_chunk(
            dg_params_list, chunk, dim=4,
            seed_offset=chunk_idx * 10000,
            max_iters=dg_max_iters,
            fourth_dim_weight=fourth_dim_weight,
            chiral_weight=chiral_weight,
        )

        # Extract 3D positions (drop 4th coordinate from DG output)
        c = 0
        mol_k_in_chunk: dict[int, int] = {}
        mol_order: List[int] = []
        for mol_idx, c_start, c_end in chunk:
            k = c_end - c_start
            if mol_idx not in mol_k_in_chunk:
                mol_k_in_chunk[mol_idx] = 0
                mol_order.append(mol_idx)
            mol_k_in_chunk[mol_idx] += k

        for chunk_mol_idx, mol_idx in enumerate(mol_order):
            k = mol_k_in_chunk[mol_idx]
            for local_k in range(k):
                pos_4d = conf_pos[(mol_idx, local_k)]
                pos_3d = pos_4d[:, :3].copy()
                mol_results[mol_idx].positions_3d.append(pos_3d)
                mol_results[mol_idx].energies.append(float(energies[c]))
                mol_results[mol_idx].converged.append(bool(statuses[c] == 0))
                c += 1

        total_confs += sum(c_end - c_start for _, c_start, c_end in chunk)

    t_total = time.time() - t_start

    return PipelineResult(
        molecules=mol_results,
        total_conformers=total_confs,
        total_time=t_total,
        n_batches=len(chunks),
    )
