#!/usr/bin/env python3
"""
Benchmark: pre-compiled Metal kernel vs JIT (on-the-fly) Metal kernel.

- Compiled: kernel compiled ONCE at first call, cutoff passed as runtime buffer.
- JIT: cutoff embedded in source string, kernel recompiled every call.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import mlx.core as mx

from mlxmolkit.fp_uint32 import fp_uint8_to_uint32
from mlxmolkit.fused_tanimoto_nlist import fused_neighbor_list_metal
from mlxmolkit.butina import butina_from_neighbor_list_csr


def run_pipeline(fp_u32, cutoff, N, compiled: bool):
    t0 = time.time()
    offsets, indices = fused_neighbor_list_metal(fp_u32, cutoff, compiled=compiled)
    t_fused = time.time() - t0

    t0 = time.time()
    result = butina_from_neighbor_list_csr(offsets, indices, N, cutoff)
    t_butina = time.time() - t0

    return t_fused, t_butina, len(result.clusters), int(np.diff(offsets).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-max", type=int, default=5000)
    p.add_argument("--cutoff", type=float, default=0.4)
    p.add_argument("--nbits", type=int, default=1024)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()

    N = args.n_max
    nbytes = args.nbits // 8

    np.random.seed(42)
    fp_u8 = np.random.randint(0, 256, (N, nbytes), dtype=np.uint8)
    fp_u32 = fp_uint8_to_uint32(mx.array(fp_u8))

    print(f"N={N}, nbits={args.nbits}, cutoff={args.cutoff}, rounds={args.rounds}")
    print()

    # Warmup both paths (first call triggers compilation)
    print("Warming up (first call = Metal shader compilation)...")
    t0 = time.time()
    fused_neighbor_list_metal(fp_u32[:100], args.cutoff, compiled=True)
    t_warmup_compiled = time.time() - t0

    t0 = time.time()
    fused_neighbor_list_metal(fp_u32[:100], args.cutoff, compiled=False)
    t_warmup_jit = time.time() - t0

    print(f"  Compiled warmup (shader compile): {t_warmup_compiled:.3f}s")
    print(f"  JIT warmup (shader compile):      {t_warmup_jit:.3f}s")
    print()

    # Benchmark: compiled (kernel already cached)
    print("--- Pre-compiled (kernel cached, cutoff as buffer) ---")
    times_compiled = []
    for r in range(args.rounds):
        t_fused, t_butina, n_clust, n_edges = run_pipeline(fp_u32, args.cutoff, N, compiled=True)
        times_compiled.append(t_fused)
        if r == 0:
            print(f"  Clusters: {n_clust}, Edges: {n_edges}")
    avg_c = np.mean(times_compiled)
    print(f"  Fused kernel: {[f'{t:.4f}s' for t in times_compiled]}")
    print(f"  Average:      {avg_c:.4f}s")
    print()

    # Benchmark: JIT (recompiles each call)
    print("--- JIT (cutoff in source, recompiled each call) ---")
    times_jit = []
    for r in range(args.rounds):
        t_fused, t_butina, n_clust, n_edges = run_pipeline(fp_u32, args.cutoff, N, compiled=False)
        times_jit.append(t_fused)
        if r == 0:
            print(f"  Clusters: {n_clust}, Edges: {n_edges}")
    avg_j = np.mean(times_jit)
    print(f"  Fused kernel: {[f'{t:.4f}s' for t in times_jit]}")
    print(f"  Average:      {avg_j:.4f}s")
    print()

    # JIT with different cutoff (forces recompilation)
    print("--- JIT with varying cutoff (worst case: always recompiles) ---")
    times_jit_vary = []
    for r in range(args.rounds):
        cutoff_vary = args.cutoff + r * 1e-7
        t_fused, _, _, _ = run_pipeline(fp_u32, cutoff_vary, N, compiled=False)
        times_jit_vary.append(t_fused)
    avg_jv = np.mean(times_jit_vary)
    print(f"  Fused kernel: {[f'{t:.4f}s' for t in times_jit_vary]}")
    print(f"  Average:      {avg_jv:.4f}s")
    print()

    print("=== Summary ===")
    print(f"  Compiled (cached):    {avg_c:.4f}s")
    print(f"  JIT (same cutoff):    {avg_j:.4f}s")
    print(f"  JIT (varying cutoff): {avg_jv:.4f}s")
    if avg_c > 0:
        print(f"  Compiled vs JIT-vary: {avg_jv / avg_c:.1f}x faster")


if __name__ == "__main__":
    main()
