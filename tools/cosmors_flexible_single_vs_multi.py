#!/usr/bin/env python3
"""Single-conformer vs multi-conformer head-to-head on flexible mols.

For each (Name, SMILES) pair: find the row in the paper Database,
run both single-conformer and multi-conformer σ-potential, and report
per-molecule Pearson r vs the paper.

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_flexible_single_vs_multi.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


PAPER_SIGMA_E_PER_A2 = np.round(np.arange(-0.030, 0.0301, 0.001), 6)

# Flexible-mol test set: name in paper Database → SMILES. All have
# explicit rotational freedom (chain alcohols, polyols, glycol ethers).
FLEXIBLE_MOLS: list[tuple[str, str]] = [
    ("methanol",                    "CO"),
    ("glycol",                      "OCCO"),
    ("propyleneglycol",             "CC(O)CO"),
    ("glycerol",                    "OCC(O)CO"),
    ("diethyleneglycol",            "OCCOCCO"),
    ("triethyleneglycol",           "OCCOCCOCCO"),
    ("2-furanmethanol",             "OCc1ccco1"),
    ("1-butanol",                   "CCCCO"),
    ("2-butanol",                   "CCC(O)C"),
    ("benzylalcohol",               "OCc1ccccc1"),
]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    da = a - a.mean(); db = b - b.mean()
    denom = float(np.sqrt((da*da).sum() * (db*db).sum()))
    return float((da*db).sum() / denom) if denom > 0 else float("nan")


def _worker(task: dict) -> dict:
    """Worker: runs both single and multi conformer for one mol."""
    smi = task["smiles"]
    out: dict = {"name": task["name"], "smiles": smi, "error": None}
    try:
        from mlxmolkit.xtb import (
            cosmosegments_from_orcacosmo,
            sigma_potential,
            sigma_potential_ensemble,
            tiered_gxtb_orca_cosmors_from_smiles,
            tiered_multiconformer_gxtb_orca,
        )

        # Single-conformer
        t0 = time.perf_counter()
        sp = tiered_gxtb_orca_cosmors_from_smiles(smi, seed=42, solvent="water")
        cs = cosmosegments_from_orcacosmo(sp["orcacosmo_path"])
        _, mu_single = sigma_potential(cs, sigma_grid_e_per_A2=PAPER_SIGMA_E_PER_A2)
        out["wall_single_s"] = time.perf_counter() - t0
        out["mu_single"] = mu_single.tolist()

        # Multi-conformer
        t0 = time.perf_counter()
        mp = tiered_multiconformer_gxtb_orca(
            smi, n_conformers=10, n_keep=3, seed=42, solvent="water",
        )
        _, mu_multi, weights = sigma_potential_ensemble(
            mp["cosmos"], mp["energies_gxtb_hartree"],
            sigma_grid_e_per_A2=PAPER_SIGMA_E_PER_A2,
        )
        out["wall_multi_s"] = time.perf_counter() - t0
        out["mu_multi"] = mu_multi.tolist()
        out["n_kept"] = mp["n_kept"]
        out["weights"] = list(map(float, weights))
        return out
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def main() -> None:
    xlsx = REPO_ROOT / "data" / "d5ra08246c1.xlsx"
    df = pd.read_excel(xlsx, sheet_name="Database", header=0)

    # Resolve each (name, smi) to a paper row by exact name match
    tasks: list[dict] = []
    for name, smi in FLEXIBLE_MOLS:
        match = df[df["Name"].str.lower() == name.lower()]
        if match.empty:
            print(f"  skip {name!r}: not in paper Database")
            continue
        row = match.iloc[0]
        mu_paper = row.iloc[3:64].to_numpy(dtype=np.float64).tolist()
        tasks.append({
            "name": name, "smiles": smi,
            "cluster": row["Cluster"] if pd.notna(row["Cluster"]) else None,
            "mu_paper": mu_paper,
        })
    print(f"Resolved {len(tasks)} flexible mols\n")

    results: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            res = fut.result()
            results.append({**task, **res})
            if res.get("error"):
                print(f"  [fail] {task['name']}: {res['error']}", flush=True)
                continue
            mp_a = np.array(task["mu_paper"])
            r_s = pearson(mp_a, np.array(res["mu_single"]))
            r_m = pearson(mp_a, np.array(res["mu_multi"]))
            print(f"  {task['name']:<22} n_kept={res['n_kept']:>2}  "
                  f"single r={r_s:+.4f} ({res['wall_single_s']:.1f}s)  "
                  f"multi r={r_m:+.4f} ({res['wall_multi_s']:.1f}s)  "
                  f"Δ={r_m-r_s:+.4f}", flush=True)
    print(f"\nTotal wall: {time.perf_counter()-t0:.1f} s")

    valid = [r for r in results if not r.get("error")]
    rs = np.array([pearson(np.array(r["mu_paper"]), np.array(r["mu_single"])) for r in valid])
    rm = np.array([pearson(np.array(r["mu_paper"]), np.array(r["mu_multi"])) for r in valid])
    print("\n--- Summary ---")
    print(f"  Single-conformer: mean r={rs.mean():+.4f}  median={np.median(rs):+.4f}")
    print(f"  Multi-conformer:  mean r={rm.mean():+.4f}  median={np.median(rm):+.4f}")
    print(f"  Δr_raw: mean={(rm-rs).mean():+.4f}  positive: {(rm-rs > 0).sum()}/{len(rs)}  negative: {(rm-rs < 0).sum()}/{len(rs)}")
    print(f"  Largest single-conf gains from going multi:")
    delta = rm - rs
    order = np.argsort(-delta)
    for i in order[:5]:
        v = valid[i]
        print(f"    {v['name']:<22} Δ={delta[i]:+.4f}  ({pearson(np.array(v['mu_paper']), np.array(v['mu_single'])):+.4f} → "
              f"{pearson(np.array(v['mu_paper']), np.array(v['mu_multi'])):+.4f})  n_kept={v['n_kept']}")

    # Save
    out_csv = REPO_ROOT / "benchmarks" / "cosmors_flexible_single_vs_multi.csv"
    pd.DataFrame([{
        "name": r["name"], "smiles": r["smiles"], "cluster": r.get("cluster"),
        "wall_single_s": r.get("wall_single_s"), "wall_multi_s": r.get("wall_multi_s"),
        "n_kept": r.get("n_kept"),
        "r_single": pearson(np.array(r["mu_paper"]), np.array(r["mu_single"])),
        "r_multi": pearson(np.array(r["mu_paper"]), np.array(r["mu_multi"])),
    } for r in valid]).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
