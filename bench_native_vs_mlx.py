#!/usr/bin/env python3
"""
Benchmark: Native Metal (.metallib + ctypes) vs Python MLX (metal_kernel).

Compares three pipelines:
  1. Native Metal: pre-compiled .metallib, direct Metal API dispatch via ctypes
  2. MLX Compiled: mx.fast.metal_kernel with cached kernel objects
  3. MLX JIT: mx.fast.metal_kernel with cutoff embedded in source

All three produce identical CSR neighbor lists → same Butina clusters.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import mlx.core as mx

from mlxmolkit.native_metal import fused_neighbor_list_native
from mlxmolkit.fp_uint32 import fp_uint8_to_uint32
from mlxmolkit.fused_tanimoto_nlist import fused_neighbor_list_metal
from mlxmolkit.butina import butina_from_neighbor_list_csr


def bench_native(fp_u32_np, cutoff, N, rounds):
    """Native Metal (.metallib + ctypes)."""
    times = []
    for r in range(rounds):
        t0 = time.time()
        offsets, indices, gpu_ms = fused_neighbor_list_native(fp_u32_np, cutoff)
        t_total = time.time() - t0
        times.append(t_total)

        if r == 0:
            t0b = time.time()
            result = butina_from_neighbor_list_csr(offsets, indices, N, cutoff)
            t_butina = time.time() - t0b
            edges = int(np.diff(offsets).sum())
            print(f"  Clusters: {len(result.clusters)}, Edges: {edges}")
            print(f"  GPU time (Metal internal): {gpu_ms:.1f}ms")
            print(f"  Butina (CPU): {t_butina*1000:.1f}ms")
    return times


def bench_mlx(fp_u32_mx, cutoff, N, rounds, compiled):
    """MLX metal_kernel (compiled or JIT)."""
    times = []
    for r in range(rounds):
        t0 = time.time()
        offsets, indices = fused_neighbor_list_metal(fp_u32_mx, cutoff, compiled=compiled)
        t_total = time.time() - t0
        times.append(t_total)

        if r == 0:
            t0b = time.time()
            result = butina_from_neighbor_list_csr(offsets, indices, N, cutoff)
            t_butina = time.time() - t0b
            edges = int(np.diff(offsets).sum())
            print(f"  Clusters: {len(result.clusters)}, Edges: {edges}")
            print(f"  Butina (CPU): {t_butina*1000:.1f}ms")
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-max", type=int, default=5000)
    p.add_argument("--cutoff", type=float, default=0.4)
    p.add_argument("--nbits", type=int, default=1024)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()

    N = args.n_max
    nbytes = args.nbits // 8
    nwords = args.nbits // 32

    np.random.seed(42)
    fp_u8 = np.random.randint(0, 256, (N, nbytes), dtype=np.uint8)
    fp_u32_np = fp_u8.view(np.uint32).reshape(N, nwords)
    fp_u32_mx = fp_uint8_to_uint32(mx.array(fp_u8))

    print(f"N={N}, nbits={args.nbits}, cutoff={args.cutoff}, rounds={args.rounds}")
    print()

    # Warmup
    print("Warming up...")
    fused_neighbor_list_native(fp_u32_np[:100], args.cutoff)
    fused_neighbor_list_metal(fp_u32_mx[:100], args.cutoff, compiled=True)
    fused_neighbor_list_metal(fp_u32_mx[:100], args.cutoff, compiled=False)
    print()

    # 1. Native Metal
    print("=" * 60)
    print("1. NATIVE METAL (.metallib + ctypes)")
    print("=" * 60)
    t_native = bench_native(fp_u32_np, args.cutoff, N, args.rounds)
    avg_native = np.mean(t_native)
    print(f"  Fused times: {[f'{t*1000:.1f}ms' for t in t_native]}")
    print(f"  Average:     {avg_native*1000:.1f}ms")
    print()

    # 2. MLX Compiled
    print("=" * 60)
    print("2. MLX metal_kernel (compiled/cached)")
    print("=" * 60)
    t_mlx_c = bench_mlx(fp_u32_mx, args.cutoff, N, args.rounds, compiled=True)
    avg_mlx_c = np.mean(t_mlx_c)
    print(f"  Fused times: {[f'{t*1000:.1f}ms' for t in t_mlx_c]}")
    print(f"  Average:     {avg_mlx_c*1000:.1f}ms")
    print()

    # 3. MLX JIT
    print("=" * 60)
    print("3. MLX metal_kernel (JIT, cutoff in source)")
    print("=" * 60)
    t_mlx_j = bench_mlx(fp_u32_mx, args.cutoff, N, args.rounds, compiled=False)
    avg_mlx_j = np.mean(t_mlx_j)
    print(f"  Fused times: {[f'{t*1000:.1f}ms' for t in t_mlx_j]}")
    print(f"  Average:     {avg_mlx_j*1000:.1f}ms")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY (fused Tanimoto + CSR, excl. Butina)")
    print("=" * 60)
    print(f"  Native Metal:   {avg_native*1000:7.1f}ms")
    print(f"  MLX compiled:   {avg_mlx_c*1000:7.1f}ms")
    print(f"  MLX JIT:        {avg_mlx_j*1000:7.1f}ms")
    print()
    if avg_native > 0:
        print(f"  MLX compiled / Native: {avg_mlx_c/avg_native:.2f}x")
        print(f"  MLX JIT / Native:      {avg_mlx_j/avg_native:.2f}x")


if __name__ == "__main__":
    main()
