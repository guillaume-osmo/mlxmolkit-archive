#!/usr/bin/env python3
"""
Benchmark: Metal fused kernel vs pure numpy for Tanimoto + threshold + CSR.

Shows the raw GPU acceleration factor on the same computation.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import mlx.core as mx


def numpy_tanimoto_csr(fp_u8: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure numpy: compute pairwise Tanimoto, threshold, build CSR.
    Materializes the full N×N similarity matrix in float32.
    """
    bits = np.unpackbits(fp_u8, axis=1, bitorder="little")
    cnt = bits.sum(axis=1)
    inter = bits @ bits.T
    union = cnt[:, None] + cnt[None, :] - inter
    sim = (inter / (union + 1e-12)).astype(np.float32)
    np.fill_diagonal(sim, 0.0)

    counts = np.sum(sim >= cutoff, axis=1).astype(np.int32)
    offsets = np.zeros(len(fp_u8) + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    indices = np.zeros(max(1, total), dtype=np.int32)
    for i in range(len(fp_u8)):
        js = np.where(sim[i] >= cutoff)[0]
        indices[offsets[i]:offsets[i + 1]] = js
    return offsets, indices


def metal_fused_csr(fp_u8: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Metal fused kernel: fp → uint32 → fused Tanimoto+threshold → CSR."""
    from mlxmolkit.fp_uint32 import fp_uint8_to_uint32
    from mlxmolkit.fused_tanimoto_nlist import fused_neighbor_list_metal

    fp_u32 = fp_uint8_to_uint32(mx.array(fp_u8))
    return fused_neighbor_list_metal(fp_u32, cutoff)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-max", type=int, default=2000)
    p.add_argument("--cutoff", type=float, default=0.4)
    p.add_argument("--nbits", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs for Metal JIT")
    args = p.parse_args()

    N = args.n_max
    nbytes = args.nbits // 8
    cutoff = args.cutoff

    np.random.seed(42)
    fp_u8 = np.random.randint(0, 256, (N, nbytes), dtype=np.uint8)

    print(f"N={N}, nbits={args.nbits}, cutoff={cutoff}")
    print(f"Sim matrix: {N * N * 4 / 1e6:.1f} MB (numpy materializes this, Metal doesn't)")
    print()

    # Warmup Metal JIT
    for _ in range(args.warmup):
        metal_fused_csr(fp_u8[:100], cutoff)

    # --- Metal fused ---
    t0 = time.time()
    off_metal, idx_metal = metal_fused_csr(fp_u8, cutoff)
    t_metal = time.time() - t0
    edges_metal = int(np.diff(off_metal).sum())

    # --- Pure numpy ---
    if N <= 10000:
        t0 = time.time()
        off_np, idx_np = numpy_tanimoto_csr(fp_u8, cutoff)
        t_numpy = time.time() - t0
        edges_np = int(np.diff(off_np).sum())

        # Verify results match
        counts_match = np.array_equal(np.diff(off_metal), np.diff(off_np))
        sets_match = True
        for i in range(N):
            s1 = set(idx_metal[off_metal[i]:off_metal[i + 1]].tolist())
            s2 = set(idx_np[off_np[i]:off_np[i + 1]].tolist())
            if s1 != s2:
                sets_match = False
                break
    else:
        t_numpy = None
        edges_np = None

    print("--- Metal (fused kernel, no N×N matrix) ---")
    print(f"  Time:  {t_metal:.4f}s")
    print(f"  Edges: {edges_metal:,}")
    print(f"  Mem:   {(N * 4 + edges_metal * 4) / 1e6:.2f} MB")
    print()

    if t_numpy is not None:
        print("--- Numpy (full N×N sim matrix in RAM) ---")
        print(f"  Time:  {t_numpy:.4f}s")
        print(f"  Edges: {edges_np:,}")
        print(f"  Mem:   {N * N * 4 / 1e6:.1f} MB")
        print()
        speedup = t_numpy / t_metal if t_metal > 0 else float("inf")
        print(f"Speedup: {speedup:.1f}x (Metal vs numpy)")
        print(f"Results match: counts={counts_match}, neighbor_sets={sets_match}")
    else:
        print(f"--- Numpy skipped (N={N} > 10000, would OOM or be too slow) ---")
        print(f"  Metal handles {N} in {t_metal:.2f}s with {(N * 4 + edges_metal * 4) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
