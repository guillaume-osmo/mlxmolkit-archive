"""
Memory-adaptive divide-and-conquer Tanimoto neighbor extraction.

For large N (150k+), the fused single-dispatch kernel may hit GPU timeouts
or memory limits.  This module tiles both row and column dimensions into
sub-blocks sized to fit in available system memory, using the fast tiled
Metal kernel from tanimoto_metal_u32 per tile.

Pipeline per tile:
  1. Compute dense similarity sub-block on GPU (tiled Metal kernel).
  2. Threshold on GPU → uint8 mask (1 byte per pair vs 4 for float32).
  3. Transfer compact mask to CPU, extract sparse neighbor indices.
  4. mx.eval() between tiles to release GPU intermediates.

After all tiles: merge per-row neighbor lists into a single CSR structure
for Butina clustering.

Memory per tile: O(block_size^2 * nwords * 4)  (FP slices + sim output).
Total memory: O(N * nwords * 4)  (full FP on GPU, read-only) + O(N + edges) (CSR).
"""
from __future__ import annotations

import math
import subprocess
from typing import Optional, Tuple

import numpy as np
import mlx.core as mx


# ---------------------------------------------------------------------------
# Fused tiled Metal kernel: similarity + threshold → uint8 mask (1/0).
# Uses threadgroup shared memory for row/col tiles.
# union = cnt_i + cnt_j - inter (no OR+popcount needed).
# ---------------------------------------------------------------------------

_TILE = 16
_MAX_TILED_WORDS = 64  # threadgroup memory: 16 × 64 uint32 = 4 KB per tile side

_TILED_MASK_TEMPLATE = """
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

if (tx == 0) {{
  if (ai < Na) {{
    for (uint k = 0; k < nwords; k++) A_tile[ty][k] = a[ai * nwords + k];
  }} else {{
    for (uint k = 0; k < nwords; k++) A_tile[ty][k] = 0u;
  }}
}}
if (ty == 0) {{
  if (bj < Nb) {{
    for (uint k = 0; k < nwords; k++) B_tile[tx][k] = b[bj * nwords + k];
  }} else {{
    for (uint k = 0; k < nwords; k++) B_tile[tx][k] = 0u;
  }}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (ai >= Na || bj >= Nb) {{ return; }}
uint cnt_i = 0, cnt_j = 0, inter = 0;
for (uint k = 0; k < nwords; k++) {{
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
}}
uint union_ = cnt_i + cnt_j - inter;
float sim = (float)inter / ((float)union_ + 1e-12f);
out[ai * Nb + bj] = (sim >= {cutoff}f) ? (uint8_t)1 : (uint8_t)0;
"""

# Cache: cutoff → compiled kernel
_mask_kernels: dict[float, object] = {}


def _get_mask_kernel(cutoff: float):
    if cutoff not in _mask_kernels:
        src = _TILED_MASK_TEMPLATE.format(cutoff=cutoff)
        _mask_kernels[cutoff] = mx.fast.metal_kernel(
            name=f"tanimoto_tiled_mask_{hash(cutoff) & 0xFFFFFFFF:08x}",
            input_names=["a", "b"],
            output_names=["out"],
            source=src,
            ensure_row_contiguous=True,
        )
    return _mask_kernels[cutoff]


def _metal_mask_block(a_u32: mx.array, b_u32: mx.array, cutoff: float) -> mx.array:
    """Fused tiled Tanimoto + threshold → (Na, Nb) uint8 mask."""
    Na = int(a_u32.shape[0])
    Nb = int(b_u32.shape[0])
    k = _get_mask_kernel(cutoff)
    grid_x = ((Nb + _TILE - 1) // _TILE) * _TILE
    grid_y = ((Na + _TILE - 1) // _TILE) * _TILE
    return k(
        inputs=[a_u32, b_u32],
        grid=(grid_x, grid_y, 1),
        threadgroup=(_TILE, _TILE, 1),
        output_shapes=[(Na, Nb)],
        output_dtypes=[mx.uint8],
    )[0]


# ---------------------------------------------------------------------------
# Memory-adaptive helpers
# ---------------------------------------------------------------------------

def _get_free_memory_bytes() -> int:
    """Query approximate free system memory on macOS. Falls back to 4 GB."""
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5,
        )
        page_size = 16384  # Apple Silicon default
        free_pages = 0
        for line in result.stdout.splitlines():
            if "Pages free" in line or "Pages speculative" in line:
                free_pages += int(line.split(":")[1].strip().rstrip("."))
        if free_pages > 0:
            return free_pages * page_size
    except Exception:
        pass
    return 4 * 1024 ** 3


def _auto_block_size(nwords: int, max_memory_bytes: Optional[int] = None) -> int:
    """Compute tile side-length B so peak memory per tile fits in budget.

    Peak per tile ≈ B² × nwords × 4 (uint32 FP slices for both sides)
                  + B² × 4          (float32 similarity output)
                  + B²              (uint8 mask output)
                  ≈ B² × (4 * nwords + 5)
    """
    if max_memory_bytes is None:
        max_memory_bytes = _get_free_memory_bytes() // 4  # 25% of free RAM
    bytes_per_pair = 4 * nwords + 5
    max_pairs = max_memory_bytes // bytes_per_pair
    block = int(math.isqrt(max(1, max_pairs)))
    return max(16, block)


# ---------------------------------------------------------------------------
# Divide-and-conquer sparse neighbor extraction
# ---------------------------------------------------------------------------

def tanimoto_neighbors_blockwise(
    fp_u32: mx.array,
    cutoff: float,
    *,
    block_size: Optional[int] = None,
    max_memory_bytes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide-and-conquer neighbor extraction using tiled Metal kernels.

    Tiles both row and column dimensions into sub-blocks of at most
    ``block_size × block_size`` pairs.  Each tile runs the fused
    Metal tiled kernel (Tanimoto + threshold) and extracts only the
    sparse neighbor indices.  GPU intermediates are freed between tiles.

    Parameters
    ----------
    fp_u32 : mx.array, shape (N, nwords), dtype uint32
        uint32-packed fingerprints (use fp_uint8_to_uint32 to convert).
    cutoff : float
        Similarity threshold (keep pairs with sim >= cutoff).
    block_size : int, optional
        Tile side-length.  Auto-computed from free memory when None.
    max_memory_bytes : int, optional
        Memory budget for one tile.  Defaults to 25% of free system RAM.

    Returns
    -------
    offsets : np.ndarray, shape (N+1,), dtype int64
        CSR row-pointer array.
    indices : np.ndarray, shape (total_edges,), dtype int64
        Flat neighbor index array.
    """
    N = int(fp_u32.shape[0])
    nwords = int(fp_u32.shape[1])

    if block_size is None:
        block_size = _auto_block_size(nwords, max_memory_bytes)
    block_size = min(block_size, N)

    # Verify Metal kernel compiles
    _get_mask_kernel(cutoff)

    # Per-row accumulator
    row_nbrs: list[list[np.ndarray]] = [[] for _ in range(N)]

    for i0 in range(0, N, block_size):
        i1 = min(i0 + block_size, N)
        ai = fp_u32[i0:i1]

        for j0 in range(0, N, block_size):
            j1 = min(j0 + block_size, N)
            bj = fp_u32[j0:j1]

            mask_mx = _metal_mask_block(ai, bj, cutoff)
            mx.eval(mask_mx)
            mask_np = np.array(mask_mx).astype(np.bool_)
            del mask_mx

            bi = i1 - i0
            for li in range(bi):
                gi = i0 + li
                cols = np.flatnonzero(mask_np[li]).astype(np.int64) + j0
                cols = cols[cols != gi]  # exclude self
                if len(cols) > 0:
                    row_nbrs[gi].append(cols)

            del mask_np

    # Build CSR
    offsets = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        n_edges = sum(c.shape[0] for c in row_nbrs[i])
        offsets[i + 1] = offsets[i] + n_edges

    total_edges = int(offsets[N])
    indices = np.empty(total_edges, dtype=np.int64)
    pos = 0
    for i in range(N):
        for chunk in row_nbrs[i]:
            end = pos + chunk.shape[0]
            indices[pos:end] = chunk
            pos = end

    return offsets, indices
