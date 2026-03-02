"""
Butina clustering (greedy) using a CSR neighbor list.

Pipeline (nvMolKit-style): Morgan (CPU) → Fused Tanimoto→CSR (Metal) → Butina greedy (CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import mlx.core as mx


@dataclass
class ButinaResult:
    clusters: List[Tuple[int, ...]]
    cutoff: float


def butina_from_neighbor_list_csr(
    offsets: np.ndarray,
    indices: np.ndarray,
    n: int,
    cutoff: float,
) -> ButinaResult:
    """
    Fast Butina greedy from CSR neighbor list.

    Key optimization: update counts by iterating over REMOVED members' neighbors
    (O(cluster_size * avg_degree)) instead of scanning ALL alive molecules (O(N)).
    """
    counts = (offsets[1:] - offsets[:-1]).astype(np.int64)
    alive = np.ones(n, dtype=np.bool_)
    clusters: List[Tuple[int, ...]] = []

    masked_counts = counts.copy()

    while True:
        best = int(np.argmax(masked_counts))
        if masked_counts[best] < 0:
            break

        nbrs = indices[offsets[best]:offsets[best + 1]]
        alive_nbrs = nbrs[alive[nbrs]]
        members = np.empty(1 + len(alive_nbrs), dtype=np.int64)
        members[0] = best
        members[1:] = alive_nbrs

        alive[members] = False
        masked_counts[members] = -1
        clusters.append(tuple(members.tolist()))

        for m in members:
            m_nbrs = indices[offsets[m]:offsets[m + 1]]
            alive_m_nbrs = m_nbrs[alive[m_nbrs]]
            if len(alive_m_nbrs) > 0:
                counts[alive_m_nbrs] -= 1
                masked_counts[alive_m_nbrs] = counts[alive_m_nbrs]

    singletons = np.where(alive)[0]
    for s in singletons:
        clusters.append((int(s),))

    return ButinaResult(clusters=clusters, cutoff=cutoff)


def butina_from_similarity_matrix(sim: np.ndarray, cutoff: float) -> ButinaResult:
    """Butina from dense similarity matrix (for testing)."""
    N = sim.shape[0]
    nbrs = []
    for i in range(N):
        js = np.where(sim[i] >= cutoff)[0]
        js = js[js != i]
        nbrs.append(js)

    offsets = np.zeros(N + 1, dtype=np.int32)
    for i, js in enumerate(nbrs):
        offsets[i + 1] = offsets[i] + len(js)
    indices = np.concatenate(nbrs) if any(len(j) > 0 for j in nbrs) else np.array([], dtype=np.int64)
    return butina_from_neighbor_list_csr(offsets, indices, N, cutoff)


def butina_tanimoto_mlx(
    fp_bytes: mx.array,
    cutoff: float,
) -> ButinaResult:
    """
    Full fused pipeline: fp uint8 → uint32 → Fused Tanimoto→CSR (Metal) → Butina greedy (CPU).
    No N×N matrix materialized.
    """
    from .fp_uint32 import fp_uint8_to_uint32
    from .fused_tanimoto_nlist import fused_neighbor_list_metal

    fp_u32 = fp_uint8_to_uint32(fp_bytes)
    N = int(fp_u32.shape[0])
    offsets, indices = fused_neighbor_list_metal(fp_u32, cutoff)
    return butina_from_neighbor_list_csr(offsets, indices, N, cutoff)
