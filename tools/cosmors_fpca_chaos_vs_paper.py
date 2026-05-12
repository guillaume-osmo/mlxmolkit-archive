#!/usr/bin/env python3
"""Functional PCA on our CHAOS+25a σ-potentials, mirroring the paper.

The RSC Adv 2026 paper's methods:
  "FPCA calculations were performed on the standardized values of the
  database matrix of σ-potentials … using scikit-fda… the first two
  functional principal components capture 99.5% variance."

We reproduce that with our CHAOS+25a μ-matrix:

Step 1: 1588 paper mols (matched to CHAOS by canonical SMILES → 1413
        rows). Compare our FPCA Dim1, Dim2 to the paper's columns 64,65.
Step 2: Full 53,091 CHAOS mols. Report variance structure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def standardize_columnwise(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (M - mean) / std_safe, mean, std_safe


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    da, db = a - a.mean(), b - b.mean()
    s = float(np.sqrt((da*da).sum() * (db*db).sum()))
    return float((da*db).sum() / s) if s > 0 else float("nan")


def fpca_fit(mu_matrix: np.ndarray, sigma_grid: np.ndarray, n_components: int = 5):
    """Standardize column-wise, then FPCA via scikit-fda."""
    from skfda import FDataGrid
    from skfda.preprocessing.dim_reduction import FPCA

    mu_std, _, _ = standardize_columnwise(mu_matrix)
    fd = FDataGrid(data_matrix=mu_std, grid_points=sigma_grid)
    fpca = FPCA(n_components=n_components)
    scores = fpca.fit_transform(fd)
    # Variance ratio from explained_variance_ratio_
    return scores, fpca, mu_std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=REPO_ROOT / "data" / "chaos_25a_mu_matrix.npz")
    parser.add_argument("--paper-xlsx", type=Path, default=REPO_ROOT / "data" / "d5ra08246c1.xlsx")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "paper_database_smiles_cache.json")
    parser.add_argument("--out-paper", type=Path, default=REPO_ROOT / "benchmarks" / "fpca_paper_subset.csv")
    parser.add_argument("--out-full", type=Path, default=REPO_ROOT / "benchmarks" / "fpca_chaos_full.npz")
    args = parser.parse_args()

    print(f"Loading μ-matrix from {args.matrix}…")
    d = np.load(args.matrix, allow_pickle=False)
    chaos_ids = d["chaos_ids"]
    canonical_smiles = d["canonical_smiles"]
    sigma_grid = d["sigma_grid_e_per_A2"]
    mu_full = d["mu_J_per_mol"]
    print(f"  shape={mu_full.shape}  σ-grid={len(sigma_grid)} pts in [{sigma_grid[0]:+.3f}, {sigma_grid[-1]:+.3f}]")

    # === Stage 1: paper-subset FPCA ===
    from rdkit import Chem

    canon_to_row = {}
    for i, smi in enumerate(canonical_smiles):
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m:
                canon_to_row[Chem.MolToSmiles(m)] = i

    cache = json.loads(args.cache.read_text())
    paper = pd.read_excel(args.paper_xlsx, sheet_name="Database", header=0)

    # paper rows that match into CHAOS
    paper_match: list[dict] = []
    for _, r in paper.iterrows():
        name = str(r["Name"]).strip() if pd.notna(r["Name"]) else ""
        cas = str(r["CAS"]).strip() if pd.notna(r["CAS"]) else ""
        if not name:
            continue
        smi = cache.get(f"{name}||{cas}", "")
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if not m:
            continue
        canon = Chem.MolToSmiles(m)
        row_idx = canon_to_row.get(canon)
        if row_idx is None:
            continue
        paper_match.append({
            "name": name, "cas": cas, "smiles": smi, "canon": canon,
            "chaos_row": row_idx,
            "cluster": r["Cluster"] if pd.notna(r["Cluster"]) else None,
            "paper_dim1": float(r.iloc[64]) if pd.notna(r.iloc[64]) else None,
            "paper_dim2": float(r.iloc[65]) if pd.notna(r.iloc[65]) else None,
        })
    print(f"\n=== Stage 1: paper subset ===")
    print(f"Matched {len(paper_match)} of {len(paper)} paper rows to CHAOS")

    paper_idx = np.asarray([p["chaos_row"] for p in paper_match], dtype=int)
    mu_paper_sub = mu_full[paper_idx]                # (N_match, 61)
    print(f"σ-potential matrix shape: {mu_paper_sub.shape}")

    print("Running FPCA…")
    t0 = time.perf_counter()
    scores_p, fpca_p, _ = fpca_fit(mu_paper_sub, sigma_grid, n_components=5)
    wall = time.perf_counter() - t0
    var = fpca_p.explained_variance_ratio_
    cum = np.cumsum(var)
    print(f"  done in {wall:.2f}s")
    print(f"  Explained variance ratio (first 5 PCs):")
    for i, (v, c) in enumerate(zip(var, cum)):
        print(f"    PC{i+1}: {100*v:>6.2f}%   cumulative: {100*c:>6.2f}%")

    # Compare our Dim1/Dim2 to paper's
    paper_d1 = np.asarray([p["paper_dim1"] for p in paper_match], dtype=float)
    paper_d2 = np.asarray([p["paper_dim2"] for p in paper_match], dtype=float)
    mask = np.isfinite(paper_d1) & np.isfinite(paper_d2)
    print(f"\n  Paper Dim1/Dim2 available for {int(mask.sum())} of {len(paper_match)} matched mols")
    if mask.sum() >= 5:
        our_d1 = scores_p[mask, 0]
        our_d2 = scores_p[mask, 1]
        # FPCA signs are arbitrary; test both orientations
        r_11 = pearson(our_d1, paper_d1[mask])
        r_22 = pearson(our_d2, paper_d2[mask])
        r_12 = pearson(our_d1, paper_d2[mask])
        r_21 = pearson(our_d2, paper_d1[mask])
        print(f"  r(ours_PC1, paper_Dim1) = {r_11:+.4f}     r(ours_PC1, paper_Dim2) = {r_12:+.4f}")
        print(f"  r(ours_PC2, paper_Dim1) = {r_21:+.4f}     r(ours_PC2, paper_Dim2) = {r_22:+.4f}")
        best_axis_1 = max(abs(r_11), abs(r_12))
        best_axis_2 = max(abs(r_22), abs(r_21))
        print(f"  best |r| on each axis:  Dim1↔PC: {best_axis_1:.4f},  Dim2↔PC: {best_axis_2:.4f}")

    # Save paper-subset CSV
    args.out_paper.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame([{
        "name": p["name"], "cas": p["cas"], "smiles": p["smiles"], "cluster": p["cluster"],
        "chaos_id": chaos_ids[p["chaos_row"]],
        "paper_dim1": p["paper_dim1"], "paper_dim2": p["paper_dim2"],
        "ours_pc1": float(scores_p[i, 0]), "ours_pc2": float(scores_p[i, 1]),
        "ours_pc3": float(scores_p[i, 2]),
    } for i, p in enumerate(paper_match)])
    df_out.to_csv(args.out_paper, index=False)
    print(f"\nWrote {args.out_paper}")

    # === Stage 2: full 53k FPCA ===
    print(f"\n=== Stage 2: full 53k FPCA ===")
    print("Running FPCA on the full 53k matrix…")
    t0 = time.perf_counter()
    scores_f, fpca_f, _ = fpca_fit(mu_full, sigma_grid, n_components=5)
    wall = time.perf_counter() - t0
    var_f = fpca_f.explained_variance_ratio_
    cum_f = np.cumsum(var_f)
    print(f"  done in {wall:.2f}s")
    print(f"  Explained variance ratio (first 5 PCs):")
    for i, (v, c) in enumerate(zip(var_f, cum_f)):
        print(f"    PC{i+1}: {100*v:>6.2f}%   cumulative: {100*c:>6.2f}%")

    # Save full FPCA result
    args.out_full.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_full,
        chaos_ids=chaos_ids,
        canonical_smiles=canonical_smiles,
        scores=scores_f,
        explained_variance_ratio=var_f,
        sigma_grid_e_per_A2=sigma_grid,
    )
    print(f"Wrote {args.out_full}  shape={scores_f.shape}")


if __name__ == "__main__":
    main()
