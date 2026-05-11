#!/usr/bin/env python3
"""Run auto-mode pipeline on cluster-7 outliers, compare to paper + CHAOS.

The 4 cluster-7 outliers (alcohols/diols where CHAOS-derived μ scores
poorly vs paper):
  - cis-9-octadecenoicacid (oleic acid)         CHAOS r=-0.87
  - 2,2-dimethyl-4-hydroxymethyl-1,3-dioxolane  CHAOS r=+0.71  (solketal)
  - 2-hydroxypropanoicacidethylester (ethyl lactate) CHAOS r=+0.82
  - oleyl-alcohol                               CHAOS r=+0.85

If our pipeline beats CHAOS on these by ≥0.1 r, that's a real win.
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


PAPER_GRID = np.round(np.arange(-0.030, 0.0301, 0.001), 6)
CHAOS_GRID = np.round(np.arange(-0.025, 0.0251, 0.001), 6)


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    da = a - a.mean(); db = b - b.mean()
    denom = float(np.sqrt((da*da).sum() * (db*db).sum()))
    return float((da*db).sum() / denom) if denom > 0 else float("nan")


def _worker(task: dict) -> dict:
    from mlxmolkit.xtb import cosmors_sigma_potential_auto
    name = task["name"]
    smi = task["smiles"]
    t0 = time.perf_counter()
    try:
        out = cosmors_sigma_potential_auto(smi, sigma_grid_e_per_A2=PAPER_GRID)
    except Exception as e:
        return {**task, "error": f"{type(e).__name__}: {e}"}
    wall = time.perf_counter() - t0
    return {
        **task,
        "mu_ours": np.asarray(out["mu_S_J_per_mol"]).tolist(),
        "auto_mode": out["mode"],
        "auto_reason": out["reason"],
        "n_kept": out["n_kept"],
        "wall_s": wall,
        "error": None,
    }


def main() -> None:
    from mlxmolkit.xtb import sigma_potential_from_arrays

    # Cluster-7 outliers
    targets = [
        {"name": "cis-9-octadecenoicacid",                  "smiles": r"C(CCCCCCC\C=C/CCCCCCCC)(=O)O"},
        {"name": "2,2-dimethyl-4-hydroxymethyl-1,3-dioxolane", "smiles": "CC1(OCC(O1)CO)C"},
        {"name": "2-hydroxypropanoicacidethylester",        "smiles": "CCOC(=O)C(C)O"},
        {"name": "oleyl-alcohol",                           "smiles": r"C(CCCCCCC\C=C/CCCCCCCC)O"},
        # Controls (CHAOS already nails these — verify our pipeline matches)
        {"name": "1-butanol",                               "smiles": "C(CCC)O"},
        {"name": "2-propanol",                              "smiles": "CC(C)O"},
        {"name": "cyclohexanol",                            "smiles": "C1(CCCCC1)O"},
    ]
    # Pre-load paper μ + CHAOS-derived μ for each
    paper = pd.read_excel(REPO_ROOT / "data" / "d5ra08246c1.xlsx", sheet_name="Database", header=0)
    paper_mu: dict[str, np.ndarray] = {}
    for _, r in paper.iterrows():
        nm = str(r["Name"]).strip().lower() if pd.notna(r["Name"]) else ""
        if nm:
            paper_mu[nm] = r.iloc[3:64].to_numpy(dtype=np.float64)

    chaos_index = pd.read_csv(REPO_ROOT / "data" / "chaos_index.csv")
    from rdkit import Chem
    chaos_index["canon"] = [
        Chem.MolToSmiles(Chem.MolFromSmiles(s)) if isinstance(s, str) and s else ""
        for s in chaos_index["canonical_smiles"]
    ]
    chaos_lookup = {c: str(cid) for cid, c in zip(chaos_index["chaos_id"], chaos_index["canon"]) if c}

    chaos_path = Path("/Users/guillaume-osmo/Github/data/CHAOS.zip")
    chaos_mu: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(chaos_path) as zf:
        for t in targets:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(t["smiles"]))
            cid = chaos_lookup.get(canon)
            if cid is None:
                continue
            try:
                with zf.open(f"{cid}.json") as f:
                    data = json.loads(f.read())
                sig_total = np.asarray(data["solvation"]["Sigma_total"], dtype=np.float64)
                _, mu_c = sigma_potential_from_arrays(
                    CHAOS_GRID, sig_total, use_sigma_orth=False,
                    sigma_grid_e_per_A2=PAPER_GRID,
                )
                chaos_mu[t["name"].lower()] = mu_c
            except Exception:
                pass

    # Run pipeline in parallel (auto-mode dispatches to single vs deep)
    print(f"Running auto-mode pipeline on {len(targets)} mols (workers=4)...\n")
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_worker, t): t for t in targets}
        for fut in as_completed(futures):
            results.append(fut.result())
    elapsed = time.perf_counter() - t0

    # Report
    print(f"{'name':<40} {'mode':<6} {'r_ours':>8} {'r_chaos':>9} {'Δours-chaos':>13} {'n_kept':>7} {'wall':>7}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["name"]):
        nm = r["name"].lower()
        mu_p = paper_mu.get(nm)
        if mu_p is None:
            print(f"  {r['name']:<38}  no paper data")
            continue
        if r.get("error"):
            print(f"  {r['name']:<38}  FAILED: {r['error']}")
            continue
        r_ours = pearson(np.asarray(r["mu_ours"]), mu_p)
        r_chaos = pearson(chaos_mu.get(nm), mu_p) if nm in chaos_mu else float("nan")
        delta = r_ours - r_chaos if not np.isnan(r_chaos) else float("nan")
        print(f"{r['name']:<40} {r['auto_mode']:<6} {r_ours:+8.4f} {r_chaos:+9.4f} {delta:+13.4f} {r['n_kept']:>7d} {r['wall_s']:>6.1f}s")
    print(f"\nTotal wall: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
