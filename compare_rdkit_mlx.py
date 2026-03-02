#!/usr/bin/env python3
"""
Compare clustering workflow: pure RDKit (Python) vs MLX/Metal.

Same 3-step pipeline as the nvMolKit blog:
  https://greglandrum.github.io/rdkit-blog/posts/2026-02-28-nvmolkit-clustering.html

Usage:
  python compare_rdkit_mlx.py --n-max 2000
  python compare_rdkit_mlx.py --smiles-file /path/to/file.cxsmiles.bz2 --n-max 20000
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import List

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from rdkit import rdBase
    rdBase.DisableLog("rdApp.*")
except Exception:
    pass

ENAMINE_REAL_10_4M = "/Users/guillaume-osmo/Downloads/2025.02_Enamine_REAL_DB_10.4M.cxsmiles.bz2"


def load_smiles(smiles_file: str | None, n_max: int) -> List[str]:
    if smiles_file and not os.path.isfile(smiles_file):
        raise FileNotFoundError(f"SMILES file not found: {smiles_file}")
    if smiles_file:
        import pandas as pd
        df = pd.read_csv(smiles_file, nrows=n_max + 100, sep="\t", header=0, on_bad_lines="skip")
        col = df["smiles"] if "smiles" in df.columns else df.iloc[:, 0]
        return col.dropna().astype(str).str.strip().tolist()[:n_max]
    raise FileNotFoundError("No SMILES file provided.")


def rdkit_fps_and_bytes(mols, fp_radius: int, fp_nbits: int, n_cpu_threads: int):
    """Generate fps ONCE: return both RDKit fps and packed uint8 bytes for Metal."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_nbits)
    fps = generator.GetFingerprints(mols, numThreads=n_cpu_threads)

    nbytes = (fp_nbits + 7) // 8
    out = np.zeros((len(mols), nbytes), dtype=np.uint8)
    for i, bv in enumerate(fps):
        bits = np.zeros((fp_nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, bits)
        out[i] = np.packbits(bits, bitorder="little")[:nbytes]

    return fps, out


def run_rdkit_workflow(fps, n_mols: int, distance_threshold: float):
    """Blog workflow: BulkTanimotoSimilarity → ClusterData."""
    from rdkit.DataStructs import BulkTanimotoSimilarity
    from rdkit.ML.Cluster.Butina import ClusterData

    t0 = time.time()
    distances = []
    for i in range(n_mols):
        distances.extend(BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True))
    t_sim = time.time() - t0

    t0 = time.time()
    clusters = ClusterData(
        np.array(distances), n_mols, distance_threshold,
        isDistData=True, distFunc=None, reordering=True,
    )
    t_clust = time.time() - t0

    return {
        "similarity": t_sim,
        "clustering": t_clust,
        "total": t_sim + t_clust,
        "n_clusters": len(clusters),
        "cluster_sizes": sorted([len(c) for c in clusters], reverse=True),
    }


def run_mlx_workflow(fp_bytes_np: np.ndarray, similarity_cutoff: float):
    """
    Full Metal pipeline (like nvMolKit):
      1. Fused Tanimoto+threshold → CSR neighbor list (GPU, no N×N matrix)
      2. Butina greedy (CPU on CSR)
    """
    import mlx.core as mx
    from mlxmolkit.fp_uint32 import fp_uint8_to_uint32
    from mlxmolkit.fused_tanimoto_nlist import fused_neighbor_list_metal
    from mlxmolkit.butina import butina_from_neighbor_list_csr

    n_mols = fp_bytes_np.shape[0]
    fp_mx = mx.array(fp_bytes_np)
    fp_u32 = fp_uint8_to_uint32(fp_mx)

    # Fused: Tanimoto + threshold + CSR neighbor list (all on Metal, no N×N matrix)
    t0 = time.time()
    offsets, indices = fused_neighbor_list_metal(fp_u32, similarity_cutoff)
    t_fused = time.time() - t0

    # Butina greedy on CPU (CSR)
    t0 = time.time()
    result = butina_from_neighbor_list_csr(offsets, indices, n_mols, similarity_cutoff)
    t_butina = time.time() - t0

    n_edges = int(np.diff(offsets).sum())
    mem_saved_mb = (n_mols * n_mols * 4) / 1e6

    return {
        "fused_tanimoto_nlist": t_fused,
        "butina": t_butina,
        "total": t_fused + t_butina,
        "n_clusters": len(result.clusters),
        "cluster_sizes": sorted([len(c) for c in result.clusters], reverse=True),
        "n_edges": n_edges,
        "mem_saved_mb": mem_saved_mb,
    }


def main():
    p = argparse.ArgumentParser(description="RDKit vs MLX clustering (nvMolKit blog)")
    p.add_argument("--n-max", type=int, default=2000)
    p.add_argument("--smiles-file", type=str, default=ENAMINE_REAL_10_4M)
    p.add_argument("--distance-threshold", type=float, default=0.6)
    p.add_argument("--fp-radius", type=int, default=3)
    p.add_argument("--fp-nbits", type=int, default=1024)
    p.add_argument("--cpu-threads", type=int, default=8)
    args = p.parse_args()

    smiles_path = args.smiles_file if args.smiles_file else None
    if smiles_path and not os.path.isfile(smiles_path):
        print(f"Note: {smiles_path} not found.")
        smiles_path = None
    smiles = load_smiles(smiles_path, args.n_max)

    from rdkit.Chem import MolFromSmiles
    mols = [m for m in (MolFromSmiles(s) for s in smiles) if m is not None]
    n_mols = len(mols)
    if n_mols == 0:
        raise SystemExit("No valid molecules.")

    similarity_cutoff = 1.0 - args.distance_threshold

    print(f"Molecules: {n_mols}")
    print(f"Params: radius={args.fp_radius}, nbits={args.fp_nbits}, "
          f"distance_threshold={args.distance_threshold} (sim_cutoff={similarity_cutoff:.2f})")
    print()

    # Fingerprinting (ONCE, shared)
    t0 = time.time()
    fps_rdkit, fp_bytes_np = rdkit_fps_and_bytes(mols, args.fp_radius, args.fp_nbits, args.cpu_threads)
    t_fp = time.time() - t0
    print(f"Fingerprinting (shared, GetMorganGenerator): {t_fp:.3f}s")
    print()

    # RDKit
    print("--- RDKit (BulkTanimotoSimilarity + ClusterData) ---")
    rdkit = run_rdkit_workflow(fps_rdkit, n_mols, args.distance_threshold)
    print(f"  Similarity:  {rdkit['similarity']:.3f}s")
    print(f"  Clustering:  {rdkit['clustering']:.3f}s")
    print(f"  Total:       {rdkit['total']:.3f}s  ->  {rdkit['n_clusters']} clusters "
          f"(largest: {rdkit['cluster_sizes'][0]})")
    print()

    # MLX/Metal
    print("--- MLX/Metal (Fused Tanimoto→CSR + Butina CPU) ---")
    mlx_res = run_mlx_workflow(fp_bytes_np, similarity_cutoff)
    print(f"  Fused sim→CSR: {mlx_res['fused_tanimoto_nlist']:.3f}s  (Metal, no N×N matrix)")
    print(f"  Butina:        {mlx_res['butina']:.3f}s  (CPU CSR greedy)")
    print(f"  Total:         {mlx_res['total']:.3f}s  ->  {mlx_res['n_clusters']} clusters "
          f"(largest: {mlx_res['cluster_sizes'][0]})")
    print(f"  Edges: {mlx_res['n_edges']:,}  |  Memory saved: {mlx_res['mem_saved_mb']:.0f} MB (no sim matrix)")
    print()

    speedup_total = rdkit["total"] / mlx_res["total"] if mlx_res["total"] > 0 else float("inf")
    print(f"Speedup total (sim+clustering): {speedup_total:.1f}x")
    delta = abs(rdkit["n_clusters"] - mlx_res["n_clusters"])
    pct = 100.0 * delta / rdkit["n_clusters"] if rdkit["n_clusters"] > 0 else 0
    print(f"Cluster delta: {delta} ({pct:.2f}% — float32 vs float64 precision)")


if __name__ == "__main__":
    main()
