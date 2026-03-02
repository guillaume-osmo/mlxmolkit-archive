"""
Fused Metal Tanimoto kernel for uint32-packed fingerprints.

- 32-bit reads; optional tiled kernel with threadgroup memory for better throughput.
- union = cnt_i + cnt_j - inter (no OR+popcount); cnt_i/cnt_j computed in-kernel from tiles/rows.
- Options: upper triangle only, float16 output.
"""
from __future__ import annotations
from typing import Optional

import mlx.core as mx

TILE = 16
MAX_WORDS = 64  # 2048 bits

# Kernel: per-row popcount of fp (N, nwords) -> cnt (N,) uint32
_POPCOUNT_ROWS_SOURCE = """
uint tid = thread_position_in_grid.x;
uint N = a_shape[0];
uint nwords = a_shape[1];
if (tid >= N) { return; }
uint sum_ = 0;
for (uint k = 0; k < nwords; k++) {
  uint v = a[tid * nwords + k];
  uint x = v - ((v >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  x = x + (x >> 8);
  x = x + (x >> 16);
  sum_ += (x & 0x3Fu);
}
out[tid] = sum_;
"""

# Tiled kernel: AND-only; union = cnt_a[i] + cnt_b[j] - inter
_TANIMOTO_U32_TILED_SOURCE = """
uint tx = thread_position_in_threadgroup.x;
uint ty = thread_position_in_threadgroup.y;
uint bx = threadgroup_position_in_grid.x;
uint by = threadgroup_position_in_grid.y;
uint Na = a_shape[0];
uint Nb = b_shape[0];
uint nwords = a_shape[1];

threadgroup uint A_tile[16][64];
threadgroup uint B_tile[16][64];

uint ai = by * 16 + ty;
uint bj = bx * 16 + tx;

if (tx == 0) {
  if (ai < Na) {
    for (uint k = 0; k < nwords; k++) {
      A_tile[ty][k] = a[ai * nwords + k];
    }
  } else {
    for (uint k = 0; k < nwords; k++) {
      A_tile[ty][k] = 0u;
    }
  }
}
if (ty == 0) {
  if (bj < Nb) {
    for (uint k = 0; k < nwords; k++) {
      B_tile[tx][k] = b[bj * nwords + k];
    }
  } else {
    for (uint k = 0; k < nwords; k++) {
      B_tile[tx][k] = 0u;
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint i = ai;
uint j = bj;
if (i >= Na || j >= Nb) { return; }
uint cnt_i = 0, cnt_j = 0, inter = 0;
for (uint k = 0; k < nwords; k++) {
  uint va = A_tile[ty][k];
  uint vb = B_tile[tx][k];
  uint x;
  x = va - ((va >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  cnt_i += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  x = vb - ((vb >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  cnt_j += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  uint andv = va & vb;
  x = andv - ((andv >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  inter += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
}
uint union_ = cnt_i + cnt_j - inter;
out[i * Nb + j] = (float)inter / ((float)union_ + 1e-12f);
"""

# Tiled kernel, upper-triangle only: skip compute for j <= i, write 0
_TANIMOTO_U32_TILED_UPPER_SOURCE = """
uint tx = thread_position_in_threadgroup.x;
uint ty = thread_position_in_threadgroup.y;
uint bx = threadgroup_position_in_grid.x;
uint by = threadgroup_position_in_grid.y;
uint Na = a_shape[0];
uint Nb = b_shape[0];
uint nwords = a_shape[1];

threadgroup uint A_tile[16][64];
threadgroup uint B_tile[16][64];

uint ai = by * 16 + ty;
uint bj = bx * 16 + tx;

if (tx == 0) {
  if (ai < Na) {
    for (uint k = 0; k < nwords; k++) {
      A_tile[ty][k] = a[ai * nwords + k];
    }
  } else {
    for (uint k = 0; k < nwords; k++) {
      A_tile[ty][k] = 0u;
    }
  }
}
if (ty == 0) {
  if (bj < Nb) {
    for (uint k = 0; k < nwords; k++) {
      B_tile[tx][k] = b[bj * nwords + k];
    }
  } else {
    for (uint k = 0; k < nwords; k++) {
      B_tile[tx][k] = 0u;
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint i = ai;
uint j = bj;
if (i >= Na || j >= Nb) { return; }
if (j <= i) {
  out[i * Nb + j] = 0.0f;
  return;
}
uint cnt_i = 0, cnt_j = 0, inter = 0;
for (uint k = 0; k < nwords; k++) {
  uint va = A_tile[ty][k];
  uint vb = B_tile[tx][k];
  uint x;
  x = va - ((va >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  cnt_i += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  x = vb - ((vb >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  cnt_j += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  uint andv = va & vb;
  x = andv - ((andv >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  inter += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
}
uint union_ = cnt_i + cnt_j - inter;
out[i * Nb + j] = (float)inter / ((float)union_ + 1e-12f);
"""

# Naive 1D kernel: cnt_i, cnt_j from row popcount in-kernel; union = cnt_i + cnt_j - inter (no OR)
_TANIMOTO_U32_NAIVE_SOURCE = """
uint tid = thread_position_in_grid.x;
uint Na = a_shape[0];
uint Nb = b_shape[0];
uint nwords = a_shape[1];
if (tid >= Na * Nb) { return; }
uint i = tid / Nb;
uint j = tid % Nb;
uint cnt_i = 0, cnt_j = 0, inter = 0;
for (uint k = 0; k < nwords; k++) {
  uint va = a[i * nwords + k];
  uint vb = b[j * nwords + k];
  uint x;
  x = va - ((va >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  cnt_i += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  x = vb - ((vb >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  cnt_j += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  uint andv = va & vb;
  x = andv - ((andv >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  inter += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
}
uint union_ = cnt_i + cnt_j - inter;
out[tid] = (float)inter / ((float)union_ + 1e-12f);
"""

# Naive, upper-triangle only
_TANIMOTO_U32_NAIVE_UPPER_SOURCE = """
uint tid = thread_position_in_grid.x;
uint Na = a_shape[0];
uint Nb = b_shape[0];
uint nwords = a_shape[1];
if (tid >= Na * Nb) { return; }
uint i = tid / Nb;
uint j = tid % Nb;
if (j <= i) {
  out[tid] = 0.0f;
  return;
}
uint cnt_i = 0, cnt_j = 0, inter = 0;
for (uint k = 0; k < nwords; k++) {
  uint va = a[i * nwords + k];
  uint vb = b[j * nwords + k];
  uint x;
  x = va - ((va >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  cnt_i += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  x = vb - ((vb >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  cnt_j += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  uint andv = va & vb;
  x = andv - ((andv >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  inter += (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
}
uint union_ = cnt_i + cnt_j - inter;
out[tid] = (float)inter / ((float)union_ + 1e-12f);
"""

_popcount_rows_kernel = None
_tanimoto_u32_tiled_kernel = None
_tanimoto_u32_tiled_upper_kernel = None
_tanimoto_u32_naive_kernel = None
_tanimoto_u32_naive_upper_kernel = None


def _get_popcount_rows_kernel():
    global _popcount_rows_kernel
    if _popcount_rows_kernel is None:
        _popcount_rows_kernel = mx.fast.metal_kernel(
            name="popcount_rows_u32",
            input_names=["a"],
            output_names=["out"],
            source=_POPCOUNT_ROWS_SOURCE,
            ensure_row_contiguous=True,
        )
    return _popcount_rows_kernel


def popcount_rows_metal(fp_u32: mx.array) -> mx.array:
    """Per-row popcount of uint32-packed fp (N, nwords) -> (N,) uint32."""
    N = int(fp_u32.shape[0])
    k = _get_popcount_rows_kernel()
    return k(
        inputs=[fp_u32],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.uint32],
    )[0]


def _get_tiled_kernel(upper_triangle_only: bool):
    global _tanimoto_u32_tiled_kernel, _tanimoto_u32_tiled_upper_kernel
    if upper_triangle_only:
        if _tanimoto_u32_tiled_upper_kernel is None:
            _tanimoto_u32_tiled_upper_kernel = mx.fast.metal_kernel(
                name="tanimoto_metal_u32_tiled_upper",
                input_names=["a", "b"],
                output_names=["out"],
                source=_TANIMOTO_U32_TILED_UPPER_SOURCE,
                ensure_row_contiguous=True,
            )
        return _tanimoto_u32_tiled_upper_kernel
    if _tanimoto_u32_tiled_kernel is None:
        _tanimoto_u32_tiled_kernel = mx.fast.metal_kernel(
            name="tanimoto_metal_u32_tiled",
            input_names=["a", "b"],
            output_names=["out"],
            source=_TANIMOTO_U32_TILED_SOURCE,
            ensure_row_contiguous=True,
        )
    return _tanimoto_u32_tiled_kernel


def _get_naive_kernel(upper_triangle_only: bool):
    global _tanimoto_u32_naive_kernel, _tanimoto_u32_naive_upper_kernel
    if upper_triangle_only:
        if _tanimoto_u32_naive_upper_kernel is None:
            _tanimoto_u32_naive_upper_kernel = mx.fast.metal_kernel(
                name="tanimoto_metal_u32_naive_upper",
                input_names=["a", "b"],
                output_names=["out"],
                source=_TANIMOTO_U32_NAIVE_UPPER_SOURCE,
                ensure_row_contiguous=True,
            )
        return _tanimoto_u32_naive_upper_kernel
    if _tanimoto_u32_naive_kernel is None:
        _tanimoto_u32_naive_kernel = mx.fast.metal_kernel(
            name="tanimoto_metal_u32_naive",
            input_names=["a", "b"],
            output_names=["out"],
            source=_TANIMOTO_U32_NAIVE_SOURCE,
            ensure_row_contiguous=True,
        )
    return _tanimoto_u32_naive_kernel


def tanimoto_matrix_metal_u32(
    a: mx.array,
    b: Optional[mx.array] = None,
    *,
    use_tiled: bool = True,
    upper_triangle_only: bool = False,
    output_float16: bool = False,
) -> mx.array:
    """
    Pairwise Tanimoto on uint32-packed fingerprints.

    union = cnt_i + cnt_j - inter (no OR+popcount); cnt computed in-kernel from rows.

    - use_tiled: 16x16 tiled kernel (disabled by default; has correctness issues on some tile boundaries).
    - upper_triangle_only: only compute j > i (≈ half time + half effective memory).
    - output_float16: return float16 to halve bandwidth (default float32).
    """
    if b is None:
        b = a
    Na, nwords = int(a.shape[0]), int(a.shape[1])
    Nb = int(b.shape[0])

    if nwords > MAX_WORDS:
        use_tiled = False
    if use_tiled and Na != Nb:
        use_tiled = False

    if use_tiled:
        kernel = _get_tiled_kernel(upper_triangle_only)
        grid_x = ((Nb + TILE - 1) // TILE) * TILE
        grid_y = ((Na + TILE - 1) // TILE) * TILE
        out = kernel(
            inputs=[a, b],
            grid=(grid_x, grid_y, 1),
            threadgroup=(TILE, TILE, 1),
            output_shapes=[(Na, Nb)],
            output_dtypes=[mx.float32],
        )[0]
    else:
        kernel = _get_naive_kernel(upper_triangle_only)
        out = kernel(
            inputs=[a, b],
            grid=(Na * Nb, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(Na, Nb)],
            output_dtypes=[mx.float32],
        )[0]

    if output_float16:
        out = out.astype(mx.float16)
    return out
