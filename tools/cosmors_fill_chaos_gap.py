#!/usr/bin/env python3
"""Run our auto-mode pipeline on the paper mols missing from CHAOS,
merge results into the chaos_25a_mu_matrix to give 100% coverage of the
paper's 1588 mols, then re-run FPCA.
"""

from __future__ import annotations

import argparse
import json
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


PAPER_GRID = np.round(np.arange(-0.030, 0.0301, 0.001), 6)


def _worker(task: dict) -> dict:
    from mlxmolkit.xtb import cosmors_sigma_potential_auto
    smi = task["smiles"]
    t0 = time.perf_counter()
    try:
        out = cosmors_sigma_potential_auto(smi, sigma_grid_e_per_A2=PAPER_GRID)
    except Exception as e:
        return {**task, "error": f"{type(e).__name__}: {e}"}
    return {**task, "mu": np.asarray(out["mu_S_J_per_mol"]).tolist(),
            "mode": out["mode"], "wall_s": time.perf_counter() - t0,
            "error": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=REPO_ROOT / "data" / "chaos_25a_mu_matrix.npz")
    parser.add_argument("--paper-xlsx", type=Path, default=REPO_ROOT / "data" / "d5ra08246c1.xlsx")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "paper_database_smiles_cache.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "paper_fill_25a_mu.npz")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from rdkit import Chem

    print(f"Loading {args.matrix}…")
    d = np.load(args.matrix, allow_pickle=False)
    canon_to_row: dict[str, int] = {}
    canon_no_stereo_to_row: dict[str, int] = {}
    for i, smi in enumerate(d["canonical_smiles"]):
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m:
                canon_to_row[Chem.MolToSmiles(m)] = i
                canon_no_stereo_to_row.setdefault(Chem.MolToSmiles(m, isomericSmiles=False), i)

    cache = json.loads(args.cache.read_text())
    paper = pd.read_excel(args.paper_xlsx, sheet_name="Database", header=0)

    missing: list[dict] = []
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
        if canon in canon_to_row:
            continue
        if Chem.MolToSmiles(m, isomericSmiles=False) in canon_no_stereo_to_row:
            continue
        missing.append({"name": name, "cas": cas, "smiles": smi, "canon": canon,
                        "cluster": r["Cluster"] if pd.notna(r["Cluster"]) else None})
    print(f"Missing from CHAOS: {len(missing)} paper mols to compute via our pipeline.")

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, m): m for m in missing}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            rows.append(res)
            done += 1
            if res.get("error"):
                print(f"  [{done}/{len(missing)}] {res['name']:<35}  FAIL: {res['error']}", flush=True)
            else:
                print(f"  [{done}/{len(missing)}] {res['name']:<35}  mode={res['mode']:<6}  {res['wall_s']:.1f}s", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")

    ok = [r for r in rows if not r.get("error")]
    print(f"Successful: {len(ok)} / {len(missing)}")

    if ok:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            names=np.asarray([r["name"] for r in ok]),
            cas=np.asarray([r["cas"] for r in ok]),
            canonical_smiles=np.asarray([r["canon"] for r in ok]),
            sigma_grid_e_per_A2=PAPER_GRID,
            mu_J_per_mol=np.asarray([r["mu"] for r in ok], dtype=np.float64),
        )
        print(f"Wrote {args.out}  shape={(len(ok), 61)}")


if __name__ == "__main__":
    main()
