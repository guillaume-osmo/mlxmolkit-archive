#!/usr/bin/env python3
"""Cross-check our σ-potentials against CHAOS (53k DFT-COSMO reference).

For each of our N benchmarked mols (matched to CHAOS by canonical SMILES),
compute three σ-potentials on the paper's 61-bin grid:

  Ours_25a    : already in benchmarks/cosmors_100_auto.csv (col mu_ours
                isn't there — we re-derive at run-time from cosmo_sigma)
  CHAOS_Klamt : sigma_potential_from_arrays(Sigma_total bins,
                 use_sigma_orth=False) — Klamt 1995 kernel on CHAOS's
                 51-bin σ-profile.
  Paper       : the RSC Adv 2026 paper's reference μ(σ).

Reports per-mol and mean Pearson r:

  A: r(CHAOS_Klamt, Paper)   sanity — does CHAOS itself match the paper?
  B: r(Ours_25a, Paper)      our pipeline result
  C: r(Ours_25a, CHAOS_Klamt) does our σ-profile reproduce CHAOS DFT?

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_chaos_crosscheck.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


PAPER_GRID = np.round(np.arange(-0.030, 0.0301, 0.001), 6)
CHAOS_GRID = np.round(np.arange(-0.025, 0.0251, 0.001), 6)  # 51 bins


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    da = a - a.mean(); db = b - b.mean()
    denom = float(np.sqrt((da*da).sum() * (db*db).sum()))
    return float((da*db).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("/Users/guillaume-osmo/Github/data/CHAOS.zip"))
    parser.add_argument("--chaos-index", type=Path, default=REPO_ROOT / "data" / "chaos_index.csv")
    parser.add_argument("--benchmark-csv", type=Path, default=REPO_ROOT / "benchmarks" / "cosmors_100_auto.csv")
    parser.add_argument("--paper-xlsx", type=Path, default=REPO_ROOT / "data" / "d5ra08246c1.xlsx")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "cosmors_chaos_crosscheck.csv")
    args = parser.parse_args()

    from mlxmolkit.xtb import (
        cosmors_sigma_potential_auto,
        sigma_potential_from_arrays,
    )
    from rdkit import Chem

    chaos = pd.read_csv(args.chaos_index)
    chaos["chaos_id"] = chaos["chaos_id"].astype(str)
    chaos["canon"] = [
        Chem.MolToSmiles(Chem.MolFromSmiles(s)) if isinstance(s, str) and s else ""
        for s in chaos["canonical_smiles"]
    ]
    chaos_lookup = {c: cid for cid, c in zip(chaos["chaos_id"], chaos["canon"]) if c}

    ours = pd.read_csv(args.benchmark_csv)
    ours["canon"] = [
        Chem.MolToSmiles(Chem.MolFromSmiles(s)) if isinstance(s, str) and s else ""
        for s in ours["smiles"]
    ]
    paper = pd.read_excel(args.paper_xlsx, sheet_name="Database", header=0)
    paper_mu = {
        str(row["Name"]).strip().lower(): row.iloc[3:64].to_numpy(dtype=np.float64)
        for _, row in paper.iterrows() if pd.notna(row["Name"])
    }

    rows = []
    with zipfile.ZipFile(args.zip, "r") as zf:
        for _, r in ours.iterrows():
            name = r["name"]
            canon = r["canon"]
            cid = chaos_lookup.get(canon)
            if cid is None:
                rows.append({"name": name, "smiles": r["smiles"], "chaos_id": "",
                             "error": "no_chaos_match"})
                continue
            try:
                with zf.open(f"{cid}.json") as f:
                    data = json.loads(f.read())
                sig_total = np.asarray(data["solvation"]["Sigma_total"], dtype=np.float64)
            except Exception as e:
                rows.append({"name": name, "smiles": r["smiles"], "chaos_id": cid,
                             "error": f"chaos_read_failed: {e}"})
                continue

            # CHAOS_Klamt: feed Sigma_total (area per bin) + bin centers as
            # σ_avg values, use_sigma_orth=False (no σ_corr in CHAOS).
            _, mu_chaos = sigma_potential_from_arrays(
                CHAOS_GRID, sig_total,
                use_sigma_orth=False,
                sigma_grid_e_per_A2=PAPER_GRID,
            )

            # Ours_25a: re-derive via the auto pipeline (cache hits for
            # mols we've already done — fast).
            try:
                out = cosmors_sigma_potential_auto(r["smiles"])
                mu_ours = np.asarray(out["mu_S_J_per_mol"])
            except Exception as e:
                rows.append({"name": name, "smiles": r["smiles"], "chaos_id": cid,
                             "error": f"ours_failed: {e}"})
                continue

            mu_paper = paper_mu.get(str(name).strip().lower())
            rec = {
                "name": name, "smiles": r["smiles"], "chaos_id": cid,
                "cluster": r.get("cluster"),
                "r_chaos_vs_paper": pearson(mu_chaos, mu_paper) if mu_paper is not None else float("nan"),
                "r_ours_vs_paper":  pearson(mu_ours,  mu_paper) if mu_paper is not None else float("nan"),
                "r_ours_vs_chaos":  pearson(mu_ours,  mu_chaos),
                "error": None,
            }
            rows.append(rec)
            print(f"  {name:<22} cluster={rec['cluster']}  "
                  f"r(CHAOS,paper)={rec['r_chaos_vs_paper']:+.4f}  "
                  f"r(ours,paper)={rec['r_ours_vs_paper']:+.4f}  "
                  f"r(ours,CHAOS)={rec['r_ours_vs_chaos']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    ok = df[df["error"].isna()] if "error" in df.columns else df
    print(f"\nSummary over {len(ok)} mols (errors: {(df['error'].notna()).sum() if 'error' in df else 0}):")
    for col, label in [
        ("r_chaos_vs_paper", "r(CHAOS_Klamt, Paper)"),
        ("r_ours_vs_paper",  "r(Ours_25a,    Paper)"),
        ("r_ours_vs_chaos",  "r(Ours_25a,    CHAOS_Klamt)"),
    ]:
        arr = ok[col].dropna().to_numpy()
        if len(arr) == 0:
            continue
        print(f"  {label}: mean={arr.mean():+.4f}  median={np.median(arr):+.4f}  "
              f"min={arr.min():+.4f}  ≥0.9: {(arr>=0.9).sum()}/{len(arr)}")


if __name__ == "__main__":
    main()
