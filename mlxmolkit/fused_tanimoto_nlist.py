"""
Fused Tanimoto + threshold + CSR neighbor list in a single Metal kernel.

No N×N similarity matrix is materialized — O(N) memory instead of O(N²).
Each GPU thread handles one row i: computes Tanimoto(i,j) for all j,
checks >= cutoff, and counts/fills neighbors.

Two-pass approach (like nvMolKit):
  Pass 1: count neighbors per row → allocate CSR arrays
  Pass 2: fill neighbor indices into pre-allocated CSR

Two variants:
  - Pre-compiled: kernel compiled once at import, cutoff as runtime buffer.
  - JIT: cutoff embedded in source string, recompiled per cutoff value.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

# ---------------------------------------------------------------------------
# Pre-compiled kernels (compiled ONCE, cutoff passed as buffer at runtime)
# ---------------------------------------------------------------------------

_COMPILED_COUNT_SOURCE = """
uint i = thread_position_in_grid.x;
uint N = a_shape[0];
uint nwords = a_shape[1];
if (i >= N) { return; }

float c = cutoff_buf[0];
uint cnt_i = cnt[i];
int count = 0;

for (uint j = 0; j < N; j++) {
    if (j == i) { continue; }
    uint inter = 0;
    for (uint k = 0; k < nwords; k++) {
        uint andv = a[i * nwords + k] & a[j * nwords + k];
        uint x = andv - ((andv >> 1) & 0x55555555u);
        x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
        x = (x + (x >> 4)) & 0x0F0F0F0Fu;
        inter += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    }
    uint union_ = cnt_i + cnt[j] - inter;
    float sim = (float)inter / ((float)union_ + 1e-12f);
    if (sim >= c) { count++; }
}
num_neighbors[i] = count;
"""

_COMPILED_FILL_SOURCE = """
uint i = thread_position_in_grid.x;
uint N = a_shape[0];
uint nwords = a_shape[1];
if (i >= N) { return; }

float c = cutoff_buf[0];
uint cnt_i = cnt[i];
int base = offsets[i];
int pos = 0;

for (uint j = 0; j < N; j++) {
    if (j == i) { continue; }
    uint inter = 0;
    for (uint k = 0; k < nwords; k++) {
        uint andv = a[i * nwords + k] & a[j * nwords + k];
        uint x = andv - ((andv >> 1) & 0x55555555u);
        x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
        x = (x + (x >> 4)) & 0x0F0F0F0Fu;
        inter += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    }
    uint union_ = cnt_i + cnt[j] - inter;
    float sim = (float)inter / ((float)union_ + 1e-12f);
    if (sim >= c) {
        indices[base + pos] = (int)j;
        pos++;
    }
}
"""

_compiled_count_kernel = None
_compiled_fill_kernel = None


def _get_compiled_count_kernel():
    global _compiled_count_kernel
    if _compiled_count_kernel is None:
        _compiled_count_kernel = mx.fast.metal_kernel(
            name="fused_tanimoto_count_compiled",
            input_names=["a", "cnt", "cutoff_buf"],
            output_names=["num_neighbors"],
            source=_COMPILED_COUNT_SOURCE,
            ensure_row_contiguous=True,
        )
    return _compiled_count_kernel


def _get_compiled_fill_kernel():
    global _compiled_fill_kernel
    if _compiled_fill_kernel is None:
        _compiled_fill_kernel = mx.fast.metal_kernel(
            name="fused_tanimoto_fill_compiled",
            input_names=["a", "cnt", "cutoff_buf", "offsets"],
            output_names=["indices"],
            source=_COMPILED_FILL_SOURCE,
            ensure_row_contiguous=True,
        )
    return _compiled_fill_kernel


# ---------------------------------------------------------------------------
# JIT kernels (cutoff embedded in source, recompiled per cutoff value)
# ---------------------------------------------------------------------------

_JIT_COUNT_TEMPLATE = """
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

_JIT_FILL_TEMPLATE = """
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_neighbor_list_metal(
    fp_u32: mx.array,
    cutoff: float,
    *,
    compiled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fused Tanimoto + threshold → CSR neighbor list in Metal.
    No N×N matrix materialized. O(N + edges) memory.

    Args:
        fp_u32: (N, nwords) uint32 packed fingerprints.
        cutoff: similarity threshold (e.g. 0.4 for distance_threshold=0.6).
        compiled: if True (default), use pre-compiled kernel (cutoff as buffer).
                  if False, JIT-compile with cutoff embedded in source.
    """
    if compiled:
        return _fused_compiled(fp_u32, cutoff)
    return _fused_jit(fp_u32, cutoff)


def _fused_compiled(fp_u32: mx.array, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compiled path: kernel compiled once, cutoff passed as runtime buffer."""
    from .tanimoto_metal_u32 import popcount_rows_metal

    N = int(fp_u32.shape[0])
    cnt = popcount_rows_metal(fp_u32)
    mx.eval(cnt)
    cutoff_buf = mx.array([cutoff], dtype=mx.float32)

    k_count = _get_compiled_count_kernel()
    num_neighbors = k_count(
        inputs=[fp_u32, cnt, cutoff_buf],
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

    k_fill = _get_compiled_fill_kernel()
    offsets_mx = mx.array(offsets)
    neighbor_indices = k_fill(
        inputs=[fp_u32, cnt, cutoff_buf, offsets_mx],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(max(1, total),)],
        output_dtypes=[mx.int32],
    )[0]
    mx.eval(neighbor_indices)

    return offsets, np.array(neighbor_indices)


def _fused_jit(fp_u32: mx.array, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """JIT path: cutoff embedded in source, recompiled each call."""
    from .tanimoto_metal_u32 import popcount_rows_metal

    N = int(fp_u32.shape[0])
    cnt = popcount_rows_metal(fp_u32)
    mx.eval(cnt)

    count_src = _JIT_COUNT_TEMPLATE.format(cutoff=cutoff)
    count_kernel = mx.fast.metal_kernel(
        name="fused_tanimoto_count_jit",
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

    fill_src = _JIT_FILL_TEMPLATE.format(cutoff=cutoff)
    fill_kernel = mx.fast.metal_kernel(
        name="fused_tanimoto_fill_jit",
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
