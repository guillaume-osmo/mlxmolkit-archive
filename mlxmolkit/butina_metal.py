"""
Butina clustering on Metal: build neighbor structure from sim matrix, then greedy on CPU.

Pipeline: sim matrix (GPU) → hit matrix / neighbor count (GPU) → Butina greedy (CPU on CSR).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from .butina import ButinaResult

# ---- Kernel: count neighbors per row (sim[i,j] >= cutoff and j != i) ----
_COUNT_NEIGHBORS_TEMPLATE = """
uint i = thread_position_in_grid.x;
uint N = sim_shape[0];
if (i >= N) {{ return; }}
float c = {cutoff}f;
int cnt = 0;
for (uint j = 0; j < N; j++) {{
  if (i != j && sim[i * N + j] >= c) {{ cnt++; }}
}}
num_neighbors[i] = cnt;
"""

# ---- Kernel: fill neighbor indices (CSR) ----
_FILL_NEIGHBORS_TEMPLATE = """
uint i = thread_position_in_grid.x;
uint N = sim_shape[0];
if (i >= N) {{ return; }}
float c = {cutoff}f;
int base = neighbor_offsets[i];
int pos = 0;
for (uint j = 0; j < N; j++) {{
  if (i != j && sim[i * N + j] >= c) {{
    neighbor_indices[base + pos] = (int)j;
    pos++;
  }}
}}
"""


def build_neighbor_list_metal(sim: mx.array, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build CSR neighbor list on GPU from similarity matrix.
    Cutoff is embedded in kernel source (avoids buffer issues).
    Returns (offsets, indices) as numpy arrays.
    """
    N = int(sim.shape[0])

    count_src = _COUNT_NEIGHBORS_TEMPLATE.format(cutoff=cutoff)
    count_kernel = mx.fast.metal_kernel(
        name="butina_count_neighbors",
        input_names=["sim"],
        output_names=["num_neighbors"],
        source=count_src,
        ensure_row_contiguous=True,
    )
    num_neighbors = count_kernel(
        inputs=[sim],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.int32],
    )[0]
    mx.eval(num_neighbors)

    n_np = np.array(num_neighbors)
    total = int(n_np.sum())
    if total == 0:
        offsets = np.zeros((N + 1,), dtype=np.int32)
        return offsets, np.array([], dtype=np.int32)

    offsets = np.zeros((N + 1,), dtype=np.int32)
    np.cumsum(n_np, out=offsets[1:])

    fill_src = _FILL_NEIGHBORS_TEMPLATE.format(cutoff=cutoff)
    fill_kernel = mx.fast.metal_kernel(
        name="butina_fill_neighbors",
        input_names=["sim", "neighbor_offsets"],
        output_names=["neighbor_indices"],
        source=fill_src,
        ensure_row_contiguous=True,
    )
    neighbor_offsets_mx = mx.array(offsets)
    neighbor_indices = fill_kernel(
        inputs=[sim, neighbor_offsets_mx],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.int32],
    )[0]
    mx.eval(neighbor_indices)

    return offsets, np.array(neighbor_indices)


def butina_from_similarity_metal(sim: mx.array, cutoff: float) -> ButinaResult:
    """
    Butina clustering: build neighbor list on GPU, then greedy on CPU.
    """
    from .butina import butina_from_neighbor_list_csr

    N = int(sim.shape[0])
    offsets, indices = build_neighbor_list_metal(sim, cutoff)
    return butina_from_neighbor_list_csr(offsets, indices, N, cutoff)
