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


def _canonical_smiles(smiles: str, *, isomeric: bool = True) -> str:
    from rdkit import Chem

    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return ""
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=isomeric)


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


def _build_chaos_indices(matrix_path: Path) -> tuple[dict[str, int], dict[str, int], np.lib.npyio.NpzFile]:
    d = np.load(matrix_path, allow_pickle=False)
    canon_to_row: dict[str, int] = {}
    canon_no_stereo_to_row: dict[str, int] = {}
    for i, smi in enumerate(d["canonical_smiles"]):
        if not smi:
            continue
        canon = _canonical_smiles(str(smi), isomeric=True)
        if canon:
            canon_to_row[canon] = i
        canon_no_stereo = _canonical_smiles(str(smi), isomeric=False)
        if canon_no_stereo:
            canon_no_stereo_to_row.setdefault(canon_no_stereo, i)
    return canon_to_row, canon_no_stereo_to_row, d


def _missing_from_paper(
    args: argparse.Namespace,
    canon_to_row: dict[str, int],
    canon_no_stereo_to_row: dict[str, int],
) -> list[dict]:
    cache = json.loads(args.cache.read_text())
    paper = pd.read_excel(args.paper_xlsx, sheet_name="Database", header=0)

    missing: list[dict] = []
    for row_index, r in paper.iterrows():
        name = str(r["Name"]).strip() if pd.notna(r["Name"]) else ""
        cas = str(r["CAS"]).strip() if pd.notna(r["CAS"]) else ""
        if not name:
            continue
        smi = cache.get(f"{name}||{cas}", "")
        if not smi:
            continue
        canon = _canonical_smiles(smi, isomeric=True)
        if not canon:
            continue
        if canon in canon_to_row:
            continue
        if _canonical_smiles(smi, isomeric=False) in canon_no_stereo_to_row:
            continue
        missing.append({
            "row_index": int(row_index),
            "name": name,
            "cas": cas,
            "smiles": smi,
            "canon": canon,
            "cluster": r["Cluster"] if pd.notna(r["Cluster"]) else None,
            "target": np.nan,
        })
    return missing


def _missing_from_csv(
    args: argparse.Namespace,
    canon_to_row: dict[str, int],
    canon_no_stereo_to_row: dict[str, int],
) -> list[dict]:
    df = pd.read_csv(args.source_csv)
    if args.smiles_col not in df.columns:
        raise KeyError(f"{args.smiles_col!r} not found in {args.source_csv}")

    missing: list[dict] = []
    for row_index, r in df.iterrows():
        raw = r[args.smiles_col]
        if pd.isna(raw):
            continue
        smi = str(raw).strip()
        canon = _canonical_smiles(smi, isomeric=True)
        if not canon:
            continue
        if canon in canon_to_row:
            continue
        if _canonical_smiles(smi, isomeric=False) in canon_no_stereo_to_row:
            continue

        name = f"row{int(row_index):05d}"
        if args.name_col and args.name_col in df.columns and pd.notna(r[args.name_col]):
            name = str(r[args.name_col]).strip()
        cas = ""
        if args.cas_col and args.cas_col in df.columns and pd.notna(r[args.cas_col]):
            cas = str(r[args.cas_col]).strip()
        target = np.nan
        if args.target_col and args.target_col in df.columns and pd.notna(r[args.target_col]):
            target = float(r[args.target_col])
        missing.append({
            "row_index": int(row_index),
            "name": name,
            "cas": cas,
            "smiles": smi,
            "canon": canon,
            "cluster": None,
            "target": target,
        })
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=REPO_ROOT / "data" / "chaos_25a_mu_matrix.npz")
    parser.add_argument("--paper-xlsx", type=Path, default=REPO_ROOT / "data" / "d5ra08246c1.xlsx")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "paper_database_smiles_cache.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "paper_fill_25a_mu.npz")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-csv", type=Path, default=None,
                        help="Optional generic CSV source, e.g. AutoVap Database-Global.csv.")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--name-col", default=None)
    parser.add_argument("--cas-col", default="CAS")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap generated missing molecules after matching. Use 0 for matching-only.")
    args = parser.parse_args()

    print(f"Loading {args.matrix}…")
    canon_to_row, canon_no_stereo_to_row, _matrix = _build_chaos_indices(args.matrix)

    if args.source_csv is None:
        missing = _missing_from_paper(args, canon_to_row, canon_no_stereo_to_row)
        source_label = "paper"
    else:
        missing = _missing_from_csv(args, canon_to_row, canon_no_stereo_to_row)
        source_label = args.source_csv.stem
    print(f"Missing from CHAOS: {len(missing)} {source_label} mols to compute via our pipeline.")

    manifest = args.out.with_suffix(".missing.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing).to_csv(manifest, index=False)
    print(f"Wrote missing manifest: {manifest}")

    if args.limit is not None:
        missing = missing[: max(0, int(args.limit))]
        print(f"Generation list after --limit: {len(missing)}")
    if not missing:
        return

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, m): m for m in missing}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            rows.append(res)
            done += 1
            label = str(res.get("name") or res.get("row_index") or res["canon"])
            if res.get("error"):
                print(f"  [{done}/{len(missing)}] {label:<35}  FAIL: {res['error']}", flush=True)
            else:
                print(f"  [{done}/{len(missing)}] {label:<35}  mode={res['mode']:<6}  {res['wall_s']:.1f}s", flush=True)
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
            row_index=np.asarray([r["row_index"] for r in ok], dtype=np.int64),
            smiles=np.asarray([r["smiles"] for r in ok]),
            canonical_smiles=np.asarray([r["canon"] for r in ok]),
            target=np.asarray([r.get("target", np.nan) for r in ok], dtype=np.float64),
            sigma_grid_e_per_A2=PAPER_GRID,
            mu_J_per_mol=np.asarray([r["mu"] for r in ok], dtype=np.float64),
        )
        print(f"Wrote {args.out}  shape={(len(ok), 61)}")


if __name__ == "__main__":
    main()
