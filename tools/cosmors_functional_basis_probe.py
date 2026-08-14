#!/usr/bin/env python3
"""Probe whether functional/basis swaps fix the 3 outliers.

Targets: anisole, aniline, benzylalcohol  (r_raw ≈ 0.77–0.87 with BP86/TZVP)
Controls: propanone, butanone           (r_raw ≈ 0.997 with BP86/TZVP)

Sweep:
    BP86/def2-TZVP     (baseline)
    BP86/def2-TZVPD    (the FACCTS canonical SP basis — extra diffuse)
    BLYP/def2-TZVPD    (different correlation functional)
    BPW91/def2-TZVPD   (different correlation, closer to TURBOMOLE BP-like)

Run from repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_functional_basis_probe.py
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


PAPER_GRID = np.round(np.arange(-0.030, 0.0301, 0.001), 6)

# (name, SMILES); name must match the paper Database row exactly
TARGETS: list[tuple[str, str]] = [
    ("anisole",       "COc1ccccc1"),
    ("aniline",       "Nc1ccccc1"),
    ("benzylalcohol", "OCc1ccccc1"),
    ("propanone",     "CC(=O)C"),    # control
    ("butanone",      "CCC(=O)C"),   # control
]

SWEEPS: list[tuple[str, str, str]] = [
    # (label, method, basis)
    ("BP86/TZVP",     "BP86",   "def2-TZVP"),
    ("BP86/TZVPD",    "BP86",   "def2-TZVPD"),
    ("BLYP/TZVPD",    "BLYP",   "def2-TZVPD"),
    ("BPW91/TZVPD",   "BPW91",  "def2-TZVPD"),
]


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    da = a - a.mean(); db = b - b.mean()
    denom = float(np.sqrt((da*da).sum() * (db*db).sum()))
    return float((da*db).sum() / denom) if denom > 0 else float("nan")


def _worker(task: dict) -> dict:
    smi = task["smiles"]; method = task["method"]; basis = task["basis"]
    try:
        from mlxmolkit.xtb import (
            cosmosegments_from_orcacosmo,
            gxtb_optimize_geometry,
            orca_cosmors_singlepoint,
            sigma_potential,
            generate_rdkit_conformers,
        )
        # Single-conformer for the probe (we already know multi doesn't help on
        # these three; the issue is QM-side).
        confs = generate_rdkit_conformers(smi, n_conformers=1, seed=42)
        atoms, coords = confs[0]
        opt_coords, e_g = gxtb_optimize_geometry(atoms, coords)
        t0 = time.perf_counter()
        path = orca_cosmors_singlepoint(
            atoms, opt_coords,
            method=method, basis=basis, solvent="water",
        )
        wall = time.perf_counter() - t0
        cs = cosmosegments_from_orcacosmo(path)
        _, mu = sigma_potential(cs, sigma_grid_e_per_A2=PAPER_GRID)
        return {**task, "mu": mu.tolist(), "wall_orca_s": wall, "error": None}
    except Exception as exc:
        return {**task, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    df = pd.read_excel(REPO_ROOT / "data" / "d5ra08246c1.xlsx",
                       sheet_name="Database", header=0)
    paper_mu = {}
    for name, smi in TARGETS:
        row = df[df["Name"].str.lower() == name.lower()]
        if row.empty:
            print(f"  {name}: not in Database, skipping")
            continue
        paper_mu[name] = row.iloc[0, 3:64].to_numpy(dtype=np.float64)

    tasks = [
        {"name": name, "smiles": smi, "method": method, "basis": basis, "label": label}
        for (label, method, basis) in SWEEPS
        for (name, smi) in TARGETS if name in paper_mu
    ]
    print(f"Sweep: {len(TARGETS)} mols × {len(SWEEPS)} configs = {len(tasks)} ORCA calls\n")

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            rows.append(res)
            if res.get("error"):
                print(f"  [fail] {res['name']:<14} {res['label']:<14}: {res['error']}")
                continue
            r = pearson(paper_mu[res["name"]], np.array(res["mu"]))
            print(f"  {res['name']:<14} {res['label']:<14}  r_raw={r:+.4f}  ({res['wall_orca_s']:.1f}s)", flush=True)
    print(f"\nTotal wall: {time.perf_counter()-t0:.1f}s")

    # Build comparison table
    print(f"\n{'mol':<14} | " + " | ".join(f"{lbl:<12}" for lbl,_,_ in SWEEPS))
    print("-" * (14 + len(SWEEPS) * 15))
    grid: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.get("error"): continue
        grid.setdefault(r["name"], {})[r["label"]] = pearson(paper_mu[r["name"]], np.array(r["mu"]))
    for name, _ in TARGETS:
        if name not in grid: continue
        line = f"{name:<14} | " + " | ".join(
            f"{grid[name].get(lbl, float('nan')):+.4f}      " for lbl,_,_ in SWEEPS
        )
        print(line)

    # Save
    pd.DataFrame([{
        "name": r["name"], "smiles": r["smiles"], "label": r["label"],
        "method": r["method"], "basis": r["basis"],
        "wall_orca_s": r.get("wall_orca_s"),
        "r_raw": pearson(paper_mu[r["name"]], np.array(r["mu"])) if not r.get("error") else float("nan"),
    } for r in rows]).to_csv(REPO_ROOT / "benchmarks" / "cosmors_functional_basis_probe.csv", index=False)


if __name__ == "__main__":
    main()
