#!/usr/bin/env python3
"""CHAOS-derived σ-potential for the FULL RSC Adv 2026 Database (1588 mols).

For every mol in the paper Database that has a clusterless or clusterful
row, match by canonical SMILES to CHAOS (53,057 indexed mols), compute
σ-potential from CHAOS's pre-computed DFT-COSMO Sigma_total via our
openCOSMORS25a kernel (Klamt 1995, since CHAOS doesn't expose σ_corr),
and compare to the paper's reference σ-potential.

This is fast (~minutes, no ORCA) because the QM work is already done
in CHAOS. We just need to read Sigma_total + apply the kernel.

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_chaos_full_paper.py
"""

from __future__ import annotations

import argparse
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


def _process_batch(args: tuple[str, list[tuple[str, str]]]) -> list[dict]:
    """Worker: open zip once, process N (chaos_id, ...) records."""
    zip_path, tasks = args
    from mlxmolkit.xtb import sigma_potential_from_arrays

    out: list[dict] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for task in tasks:
            try:
                with zf.open(f"{task['chaos_id']}.json") as f:
                    data = json.loads(f.read())
                sig_total = np.asarray(data["solvation"]["Sigma_total"], dtype=np.float64)
            except Exception as e:
                out.append({**task, "error": f"chaos_read: {e}"})
                continue
            try:
                _, mu_chaos = sigma_potential_from_arrays(
                    CHAOS_GRID, sig_total,
                    use_sigma_orth=False,
                    sigma_grid_e_per_A2=PAPER_GRID,
                )
            except Exception as e:
                out.append({**task, "error": f"kernel: {e}"})
                continue
            mu_paper = np.asarray(task["mu_paper"])
            r = pearson(mu_chaos, mu_paper)
            out.append({**task, "mu_chaos": mu_chaos.tolist(),
                        "r_chaos_vs_paper": r, "error": None})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("/Users/guillaume-osmo/Github/data/CHAOS.zip"))
    parser.add_argument("--chaos-index", type=Path, default=REPO_ROOT / "data" / "chaos_index.csv")
    parser.add_argument("--paper-xlsx", type=Path, default=REPO_ROOT / "data" / "d5ra08246c1.xlsx")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "paper_database_smiles_cache.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "cosmors_chaos_full_paper.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from rdkit import Chem

    chaos = pd.read_csv(args.chaos_index)
    chaos["chaos_id"] = chaos["chaos_id"].astype(str)
    chaos["canon"] = [
        Chem.MolToSmiles(Chem.MolFromSmiles(s)) if isinstance(s, str) and s else ""
        for s in chaos["canonical_smiles"]
    ]
    chaos_lookup = {c: cid for cid, c in zip(chaos["chaos_id"], chaos["canon"]) if c}
    print(f"CHAOS index: {len(chaos)} mols, {len(chaos_lookup)} with non-empty canonical SMILES")

    # Load paper Database
    paper = pd.read_excel(args.paper_xlsx, sheet_name="Database", header=0)
    print(f"Paper Database: {len(paper)} mols total, {paper['Cluster'].notna().sum()} with cluster assigned")

    # We need SMILES for paper mols. The cache has the ones we've looked up;
    # use it if available. (PubChem lookups are slow — skip mols not in cache.)
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    print(f"SMILES cache: {len(cache)} entries (PubChem lookups from prior runs)")

    # Build tasks: (paper_row, chaos_id, mu_paper) for mols where SMILES → canonical SMILES matches CHAOS
    tasks: list[dict] = []
    no_smiles, no_chaos = 0, 0
    for _, row in paper.iterrows():
        name = str(row["Name"]).strip() if pd.notna(row["Name"]) else ""
        cas = str(row["CAS"]).strip() if pd.notna(row["CAS"]) else ""
        key = f"{name}||{cas}"
        smi = cache.get(key, "")
        if not smi:
            # Try CAS-only key fallback
            no_smiles += 1
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            no_smiles += 1
            continue
        canon = Chem.MolToSmiles(m)
        cid = chaos_lookup.get(canon)
        if cid is None:
            no_chaos += 1
            continue
        mu_paper = row.iloc[3:64].to_numpy(dtype=np.float64).tolist()
        tasks.append({
            "name": name, "cas": cas, "smiles": smi, "canon": canon,
            "cluster": row["Cluster"] if pd.notna(row["Cluster"]) else None,
            "chaos_id": cid, "mu_paper": mu_paper,
        })
    print(f"Matched {len(tasks)} paper rows to CHAOS via cached SMILES")
    print(f"  Missing SMILES (no PubChem cache): {no_smiles}")
    print(f"  In cache but not in CHAOS:         {no_chaos}")

    if not tasks:
        print("No tasks; exiting.")
        return

    # Batch + parallel
    batch_size = max(50, len(tasks) // (args.workers * 4))
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    print(f"\nProcessing {len(tasks)} mols in {len(batches)} batches × ≤{batch_size}, {args.workers} workers...")

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_batch, (str(args.zip), b)): b for b in batches}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            rows.extend(res)
            done += len(res)
            elapsed = time.perf_counter() - t0
            rate = done / max(elapsed, 1e-6)
            print(f"  {done:>6d} / {len(tasks)}  ({rate:.0f} mol/s)", flush=True)
    print(f"\nDone in {time.perf_counter()-t0:.1f} s")

    # Save
    df = pd.DataFrame([{
        "name": r["name"], "cas": r["cas"], "smiles": r["smiles"],
        "cluster": r.get("cluster"), "chaos_id": r["chaos_id"],
        "r_chaos_vs_paper": r.get("r_chaos_vs_paper"),
        "error": r.get("error"),
    } for r in rows])
    df.to_csv(args.out, index=False)

    ok = df[df["error"].isna()]
    if len(ok) == 0:
        print("No successful rows.")
        return
    r_arr = ok["r_chaos_vs_paper"].to_numpy()
    print(f"\nSummary over {len(ok)} mols (errors: {(df['error'].notna()).sum()}):")
    print(f"  r(CHAOS_Klamt, Paper): mean={r_arr.mean():+.4f}  median={np.median(r_arr):+.4f}  "
          f"min={r_arr.min():+.4f}  max={r_arr.max():+.4f}")
    print(f"  ≥0.9: {(r_arr>=0.9).sum()}/{len(r_arr)}   ≥0.95: {(r_arr>=0.95).sum()}/{len(r_arr)}")
    if ok["cluster"].notna().any():
        print("\n  by cluster:")
        for c, sub in ok[ok["cluster"].notna()].groupby("cluster"):
            print(f"    cluster {int(c):>2}: n={len(sub):>4}  r_chaos_vs_paper mean={sub['r_chaos_vs_paper'].mean():+.4f}  "
                  f"≥0.9: {(sub['r_chaos_vs_paper']>=0.9).sum()}/{len(sub)}")


if __name__ == "__main__":
    main()
