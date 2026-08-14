#!/usr/bin/env python3
"""Merge CHAOS+25a + our pipeline fills → unified 1564-mol paper μ matrix,
then re-run FPCA and compare to paper's Dim1/Dim2.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    da, db = a - a.mean(), b - b.mean()
    s = float(np.sqrt((da*da).sum() * (db*db).sum()))
    return float((da*db).sum() / s) if s > 0 else float("nan")


def main() -> None:
    from rdkit import Chem
    from skfda import FDataGrid
    from skfda.preprocessing.dim_reduction import FPCA

    print("Loading CHAOS+25a matrix…")
    d = np.load(REPO_ROOT / "data" / "chaos_25a_mu_matrix.npz", allow_pickle=False)
    chaos_canon: dict[str, int] = {}
    chaos_canon_no_stereo: dict[str, int] = {}
    for i, smi in enumerate(d["canonical_smiles"]):
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m:
                chaos_canon[Chem.MolToSmiles(m)] = i
                chaos_canon_no_stereo.setdefault(Chem.MolToSmiles(m, isomericSmiles=False), i)

    print("Loading pipeline fills…")
    fill = np.load(REPO_ROOT / "data" / "paper_fill_25a_mu.npz", allow_pickle=False)
    fill_canon: dict[str, int] = {}
    for i, smi in enumerate(fill["canonical_smiles"]):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m:
            fill_canon[Chem.MolToSmiles(m)] = i

    sigma_grid = d["sigma_grid_e_per_A2"]
    mu_chaos = d["mu_J_per_mol"]
    mu_fill = fill["mu_J_per_mol"]

    # Match paper Database
    cache = json.loads((REPO_ROOT / "data" / "paper_database_smiles_cache.json").read_text())
    paper = pd.read_excel(REPO_ROOT / "data" / "d5ra08246c1.xlsx", sheet_name="Database", header=0)

    rows = []
    src_counts = {"chaos": 0, "chaos_no_stereo": 0, "ours": 0, "miss": 0}
    for _, r in paper.iterrows():
        name = str(r["Name"]).strip() if pd.notna(r["Name"]) else ""
        cas = str(r["CAS"]).strip() if pd.notna(r["CAS"]) else ""
        if not name:
            continue
        smi = cache.get(f"{name}||{cas}", "")
        if not smi:
            src_counts["miss"] += 1
            continue
        m = Chem.MolFromSmiles(smi)
        if not m:
            src_counts["miss"] += 1
            continue
        canon = Chem.MolToSmiles(m)
        canon_ns = Chem.MolToSmiles(m, isomericSmiles=False)

        mu_row = None; src = None
        if canon in chaos_canon:
            mu_row = mu_chaos[chaos_canon[canon]]; src = "chaos"
        elif canon_ns in chaos_canon_no_stereo:
            mu_row = mu_chaos[chaos_canon_no_stereo[canon_ns]]; src = "chaos_no_stereo"
        elif canon in fill_canon:
            mu_row = mu_fill[fill_canon[canon]]; src = "ours"
        else:
            src_counts["miss"] += 1
            continue
        src_counts[src] += 1
        rows.append({
            "name": name, "cas": cas, "smiles": smi,
            "cluster": r["Cluster"] if pd.notna(r["Cluster"]) else None,
            "paper_dim1": float(r.iloc[64]) if pd.notna(r.iloc[64]) else None,
            "paper_dim2": float(r.iloc[65]) if pd.notna(r.iloc[65]) else None,
            "source": src,
            "mu": mu_row,
        })
    print(f"Coverage: {len(rows)} / {len(paper)} ({100*len(rows)/len(paper):.1f}%)")
    print(f"  source breakdown: {src_counts}")

    mu_mat = np.asarray([r["mu"] for r in rows], dtype=np.float64)
    paper_d1 = np.asarray([r["paper_dim1"] for r in rows], dtype=float)
    paper_d2 = np.asarray([r["paper_dim2"] for r in rows], dtype=float)

    # Standardize column-wise + FPCA
    mu_std = (mu_mat - mu_mat.mean(axis=0)) / np.where(mu_mat.std(axis=0) == 0, 1.0, mu_mat.std(axis=0))
    fd = FDataGrid(data_matrix=mu_std, grid_points=sigma_grid)
    fpca = FPCA(n_components=5)
    scores = fpca.fit_transform(fd)

    var = fpca.explained_variance_ratio_
    cum = np.cumsum(var)
    print("\nFPCA on merged 1564-mol matrix:")
    for i, (v, c) in enumerate(zip(var, cum)):
        print(f"  PC{i+1}: {100*v:>6.2f}%   cum: {100*c:>6.2f}%")

    mask = np.isfinite(paper_d1) & np.isfinite(paper_d2)
    if mask.sum() >= 5:
        r_11 = pearson(scores[mask, 0], paper_d1[mask])
        r_22 = pearson(scores[mask, 1], paper_d2[mask])
        r_12 = pearson(scores[mask, 0], paper_d2[mask])
        r_21 = pearson(scores[mask, 1], paper_d1[mask])
        print(f"\nVs paper Dim1/Dim2 ({int(mask.sum())} mols with both):")
        print(f"  r(ours_PC1, paper_Dim1) = {r_11:+.4f}     r(ours_PC1, paper_Dim2) = {r_12:+.4f}")
        print(f"  r(ours_PC2, paper_Dim1) = {r_21:+.4f}     r(ours_PC2, paper_Dim2) = {r_22:+.4f}")
        print(f"  best |r|:  Dim1↔PC1 = {max(abs(r_11), abs(r_12)):.4f},  Dim2↔PC2 = {max(abs(r_22), abs(r_21)):.4f}")

    # Save merged
    out_path = REPO_ROOT / "benchmarks" / "fpca_paper_merged.csv"
    out_df = pd.DataFrame([{
        "name": r["name"], "cas": r["cas"], "smiles": r["smiles"], "cluster": r["cluster"],
        "source": r["source"],
        "paper_dim1": r["paper_dim1"], "paper_dim2": r["paper_dim2"],
        "ours_pc1": float(scores[i, 0]),
        "ours_pc2": float(scores[i, 1]),
        "ours_pc3": float(scores[i, 2]),
    } for i, r in enumerate(rows)])
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
