"""
Fused Tanimoto + threshold + CSR neighbor list in a single Metal kernel.

No N×N similarity matrix is materialized — O(N) memory instead of O(N²).
Each GPU thread handles one row i: computes Tanimoto(i,j) for all j,
checks >= cutoff, and counts/fills neighbors.

Two-pass approach (like nvMolKit):
  Pass 1: count neighbors per row → allocate CSR arrays
  Pass 2: fill neighbor indices into pre-allocated CSR

Uses pre-computed per-row popcounts: union = cnt[i] + cnt[j] - popcount(a[i] & a[j]).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

_FUSED_COUNT_TEMPLATE = """
uint i = thread_position_in_grid.x;
uint N = a_shape[0];
uint nwords = a_shape[1];
if (i >= N) {{ return; }}

float c = {cutoff}f;
uint cnt_i = cnt[i];
int count = 0;

for (uint j = 0; j < N; j++) {{
    if (j == i) {{ continue; }}
    uint inter = 0;
    for (uint k = 0; k < nwords; k++) {{
        uint andv = a[i * nwords + k] & a[j * nwords + k];
        uint x = andv - ((andv >> 1) & 0x55555555u);
        x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
        x = (x + (x >> 4)) & 0x0F0F0F0Fu;
        inter += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    }}
    uint union_ = cnt_i + cnt[j] - inter;
    float sim = (float)inter / ((float)union_ + 1e-12f);
    if (sim >= c) {{ count++; }}
}}
num_neighbors[i] = count;
"""

_FUSED_FILL_TEMPLATE = """
uint i = thread_position_in_grid.x;
uint N = a_shape[0];
uint nwords = a_shape[1];
if (i >= N) {{ return; }}

float c = {cutoff}f;
uint cnt_i = cnt[i];
int base = offsets[i];
int pos = 0;

for (uint j = 0; j < N; j++) {{
    if (j == i) {{ continue; }}
    uint inter = 0;
    for (uint k = 0; k < nwords; k++) {{
        uint andv = a[i * nwords + k] & a[j * nwords + k];
        uint x = andv - ((andv >> 1) & 0x55555555u);
        x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
        x = (x + (x >> 4)) & 0x0F0F0F0Fu;
        inter += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    }}
    uint union_ = cnt_i + cnt[j] - inter;
    float sim = (float)inter / ((float)union_ + 1e-12f);
    if (sim >= c) {{
        indices[base + pos] = (int)j;
        pos++;
    }}
}}
"""


def fused_neighbor_list_metal(
    fp_u32: mx.array,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fused Tanimoto + threshold → CSR neighbor list in Metal.
    No N×N matrix materialized. O(N + edges) memory.

    Args:
        fp_u32: (N, nwords) uint32 packed fingerprints.
        cutoff: similarity threshold (e.g. 0.4 for distance_threshold=0.6).

    Returns:
        (offsets, indices): CSR arrays as numpy int32.
    """
    from .tanimoto_metal_u32 import popcount_rows_metal

    N = int(fp_u32.shape[0])

    cnt = popcount_rows_metal(fp_u32)
    mx.eval(cnt)

    # Pass 1: count neighbors per row
    count_src = _FUSED_COUNT_TEMPLATE.format(cutoff=cutoff)
    count_kernel = mx.fast.metal_kernel(
        name="fused_tanimoto_count",
        input_names=["a", "cnt"],
        output_names=["num_neighbors"],
        source=count_src,
        ensure_row_contiguous=True,
    )
    num_neighbors = count_kernel(
        inputs=[fp_u32, cnt],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.int32],
    )[0]
    mx.eval(num_neighbors)

    n_np = np.array(num_neighbors)
    total = int(n_np.sum())
    offsets = np.zeros(N + 1, dtype=np.int32)
    if total == 0:
        return offsets, np.array([], dtype=np.int32)

    np.cumsum(n_np, out=offsets[1:])

    # Pass 2: fill neighbor indices
    fill_src = _FUSED_FILL_TEMPLATE.format(cutoff=cutoff)
    fill_kernel = mx.fast.metal_kernel(
        name="fused_tanimoto_fill",
        input_names=["a", "cnt", "offsets"],
        output_names=["indices"],
        source=fill_src,
        ensure_row_contiguous=True,
    )
    offsets_mx = mx.array(offsets)
    neighbor_indices = fill_kernel(
        inputs=[fp_u32, cnt, offsets_mx],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(max(1, total),)],
        output_dtypes=[mx.int32],
    )[0]
    mx.eval(neighbor_indices)

    return offsets, np.array(neighbor_indices)
