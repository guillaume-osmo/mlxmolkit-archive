#!/usr/bin/env python3
"""Compare our σ-potential μ_S(σ) against the RSC Adv 2026 paper dataset.

The Database sheet's 61 columns from σ=-3.0 to +3.0 e/nm² are σ-POTENTIALS
μ_S(σ), produced by COSMOtherm 2024 at B88-PW86/TZVP COSMO. FPCA in the
paper is performed on STANDARDIZED values (column-wise z-score), so we
compare shapes after the same standardization.

This script lets the user choose the σ-profile backend per molecule:

  --backend tmcosmo   :  hybrid g-xTB --opt + GFN2 --tmcosmo inf (cheap)
  --backend orca      :  tiered g-xTB --opt + ORCA BP86/TZVP COSMORS (DFT)

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_sigma_potential_vs_paper.py --n 20 --backend tmcosmo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
ALLOWED_Z = {1, 6, 7, 8, 9, 16, 17, 35}
SMILES_CACHE_PATH = REPO_ROOT / "data" / "paper_database_smiles_cache.json"


def load_cache() -> dict[str, str]:
    if SMILES_CACHE_PATH.exists():
        return json.loads(SMILES_CACHE_PATH.read_text())
    return {}


def save_cache(c: dict[str, str]) -> None:
    SMILES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMILES_CACHE_PATH.write_text(json.dumps(c, indent=2, sort_keys=True))


def resolve_smiles(name: str, cas: str, cache: dict[str, str]) -> str | None:
    key = f"{name}||{cas}"
    if key in cache:
        return cache[key] or None
    import pubchempy as pcp
    for q in (cas, name):
        if not isinstance(q, str) or not q:
            continue
        try:
            cs = pcp.get_compounds(q, "name")
            if cs:
                smi = cs[0].isomeric_smiles or cs[0].canonical_smiles
                if smi:
                    cache[key] = smi
                    return smi
        except Exception:
            pass
    cache[key] = ""
    return None


def smiles_passes(smi: str, max_heavy: int) -> bool:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None or len(Chem.GetMolFrags(m)) != 1:
        return False
    if Chem.GetFormalCharge(m) != 0:
        return False
    if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
        return False
    if any(a.GetAtomicNum() not in ALLOWED_Z for a in m.GetAtoms()):
        return False
    return m.GetNumHeavyAtoms() <= max_heavy


def sigma_pot_tmcosmo(smi: str) -> np.ndarray:
    """μ_S on PAPER_SIGMA_E_PER_A2 from GFN2-tmcosmo σ-profile."""
    from mlxmolkit.xtb import hybrid_gxtb_gfn2_cosmo_from_smiles, sigma_potential
    out = hybrid_gxtb_gfn2_cosmo_from_smiles(smi, seed=42, solvent="inf")
    _, mu = sigma_potential(out["cosmo"], sigma_grid_e_per_A2=PAPER_SIGMA_E_PER_A2)
    return mu  # J/mol


def sigma_pot_orca(smi: str, n_cores: int = 1) -> np.ndarray:
    """μ_S from g-xTB opt + ORCA BP86/def2-TZVP COSMORS (single conformer)."""
    from mlxmolkit.xtb import (
        cosmosegments_from_orcacosmo,
        sigma_potential,
        tiered_gxtb_orca_cosmors_from_smiles,
    )
    out = tiered_gxtb_orca_cosmors_from_smiles(smi, seed=42, solvent="water", n_cores=n_cores)
    cs = cosmosegments_from_orcacosmo(out["orcacosmo_path"])
    _, mu = sigma_potential(cs, sigma_grid_e_per_A2=PAPER_SIGMA_E_PER_A2)
    return mu


def sigma_pot_orca_multiconformer(smi: str, *, n_conformers: int = 10, n_keep: int = 3,
                                    n_cores: int = 1) -> tuple[np.ndarray, int, list[float]]:
    """Multi-conformer μ_S via RDKit→g-xTB-screen→ORCA→Boltzmann-weighted."""
    from mlxmolkit.xtb import sigma_potential_ensemble, tiered_multiconformer_gxtb_orca
    out = tiered_multiconformer_gxtb_orca(
        smi, n_conformers=n_conformers, n_keep=n_keep,
        seed=42, solvent="water", orca_cores=n_cores,
    )
    _, mu, weights = sigma_potential_ensemble(
        out["cosmos"], out["energies_gxtb_hartree"],
        sigma_grid_e_per_A2=PAPER_SIGMA_E_PER_A2,
    )
    return mu, out["n_kept"], list(weights)


def sigma_pot_orca_auto(smi: str, *, n_cores: int = 1) -> tuple[np.ndarray, int, list[float], str, str]:
    """Auto-mode: detects complex cases, dispatches to single or deep-multi."""
    from mlxmolkit.xtb import cosmors_sigma_potential_auto
    out = cosmors_sigma_potential_auto(
        smi, sigma_grid_e_per_A2=PAPER_SIGMA_E_PER_A2,
        n_cores=n_cores,
    )
    return (np.asarray(out["mu_S_J_per_mol"]),
            int(out["n_kept"]), list(out["weights"]),
            str(out["mode"]), str(out["reason"]))


def standardize_columnwise(M: np.ndarray) -> np.ndarray:
    """Z-score per column (the paper's FPCA pre-processing)."""
    M = np.asarray(M, dtype=np.float64)
    mean = M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (M - mean) / std


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    da = a - a.mean(); db = b - b.mean()
    denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum() / denom) if denom > 0 else float("nan")


def _worker(task: dict) -> dict | None:
    """Per-molecule worker for ProcessPoolExecutor."""
    name = task["name"]; smi = task["smiles"]; backend = task["backend"]
    try:
        t0 = time.perf_counter()
        meta_extra: dict = {}
        if backend == "tmcosmo":
            mu_ours = sigma_pot_tmcosmo(smi)
        elif backend == "orca":
            mu_ours = sigma_pot_orca(smi, n_cores=task.get("orca_cores", 1))
        elif backend == "orca-multi":
            mu_ours, n_kept, weights = sigma_pot_orca_multiconformer(
                smi,
                n_conformers=task.get("n_conformers", 10),
                n_keep=task.get("n_keep", 3),
                n_cores=task.get("orca_cores", 1),
            )
            meta_extra = {"n_kept": n_kept, "boltzmann_weights": weights}
        elif backend == "orca-auto":
            mu_ours, n_kept, weights, mode, reason = sigma_pot_orca_auto(
                smi, n_cores=task.get("orca_cores", 1),
            )
            meta_extra = {"n_kept": n_kept, "boltzmann_weights": weights,
                          "auto_mode": mode, "auto_reason": reason}
        else:
            raise ValueError(f"unknown backend {backend!r}")
        wall = time.perf_counter() - t0
        return {
            "name": name, "cas": task["cas"], "smiles": smi,
            "cluster": task["cluster"], "n_heavy": task["n_heavy"],
            "mu_ours": mu_ours.tolist(),
            "wall_s": wall, "error": None,
            **meta_extra,
        }
    except Exception as exc:
        return {"name": name, "smiles": smi, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=REPO_ROOT / "data" / "d5ra08246c1.xlsx")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-heavy", type=int, default=10)
    parser.add_argument("--backend", choices=["tmcosmo", "orca", "orca-multi", "orca-auto"], default="tmcosmo",
                        help="orca-multi: RDKit→g-xTB-screen→ORCA on each survivor, Boltzmann-weighted. "
                             "orca-auto: detects complex cases and switches between single and deep-multi.")
    parser.add_argument("--n-conformers", type=int, default=10,
                        help="(orca-multi only) total RDKit conformers to generate per molecule")
    parser.add_argument("--n-keep", type=int, default=3,
                        help="(orca-multi only) lowest-energy g-xTB conformers kept for ORCA")
    parser.add_argument("--require-cluster", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "cosmors_sigma_potential_vs_paper.csv")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of parallel σ-potential workers (1 = serial)")
    parser.add_argument("--orca-cores", type=int, default=1,
                        help="ORCA --pal nprocs per worker (multiply by --workers for total)")
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="Database", header=0)
    if args.require_cluster:
        df = df[df["Cluster"].notna()].copy()
    cache = load_cache()

    # Resolve SMILES + filter serially first (PubChem rate-limited, fast).
    tasks: list[dict] = []
    from rdkit import Chem
    for _, row in df.iterrows():
        if len(tasks) >= args.n:
            break
        name = str(row["Name"]).strip() if pd.notna(row["Name"]) else ""
        cas = str(row["CAS"]).strip() if pd.notna(row["CAS"]) else ""
        if not name:
            continue
        smi = resolve_smiles(name, cas, cache)
        if smi is None:
            continue
        try:
            if not smiles_passes(smi, args.max_heavy):
                continue
        except Exception:
            continue
        m = Chem.MolFromSmiles(smi)
        mu_paper = row.iloc[3:64].to_numpy(dtype=np.float64).tolist()
        tasks.append({
            "name": name, "cas": cas, "smiles": smi,
            "cluster": row["Cluster"] if pd.notna(row["Cluster"]) else None,
            "n_heavy": m.GetNumHeavyAtoms(),
            "mu_paper": mu_paper,
            "backend": args.backend,
            "orca_cores": args.orca_cores,
            "n_conformers": args.n_conformers,
            "n_keep": args.n_keep,
        })
    save_cache(cache)
    print(f"Prepared {len(tasks)} tasks (backend={args.backend}, workers={args.workers}, orca_cores={args.orca_cores})", flush=True)

    results: list[dict] = []
    t_total = time.perf_counter()
    if args.workers <= 1:
        for task in tasks:
            res = _worker(task)
            results.append(res)
            if res.get("error"):
                print(f"  [fail] {task['name']!r}: {res['error']}", flush=True)
            else:
                rr = pearson(np.array(task["mu_paper"]), np.array(res["mu_ours"]))
                print(f"  [{len(results):>3}/{len(tasks)}] {task['name']:<22} "
                      f"r_raw={rr:+.4f}  ({res['wall_s']:.1f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, task): task for task in tasks}
            done = 0
            for fut in as_completed(futures):
                done += 1
                task = futures[fut]
                res = fut.result()
                results.append(res)
                if res.get("error"):
                    print(f"  [fail {done}/{len(tasks)}] {task['name']!r}: {res['error']}", flush=True)
                else:
                    rr = pearson(np.array(task["mu_paper"]), np.array(res["mu_ours"]))
                    print(f"  [{done}/{len(tasks)}] {task['name']:<22} nat={task['n_heavy']:>2} "
                          f"cluster={task['cluster']}  r_raw={rr:+.4f}  ({res['wall_s']:.1f}s)", flush=True)
    elapsed = time.perf_counter() - t_total

    # Aggregate
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print("\nno successful rows")
        return
    paper_mat = np.array([r["mu_paper"] if "mu_paper" in r else
                          next(t["mu_paper"] for t in tasks if t["name"] == r["name"])
                          for r in valid], dtype=np.float64)
    # Re-attach mu_paper from tasks (worker stripped it)
    name_to_paper = {t["name"]: np.asarray(t["mu_paper"], dtype=np.float64) for t in tasks}
    paper_mat = np.array([name_to_paper[r["name"]] for r in valid], dtype=np.float64)
    ours_mat = np.array([r["mu_ours"] for r in valid], dtype=np.float64)

    paper_std = standardize_columnwise(paper_mat)
    ours_std = standardize_columnwise(ours_mat)
    r_raw = np.array([pearson(paper_mat[i], ours_mat[i]) for i in range(len(valid))])
    r_std = np.array([pearson(paper_std[i], ours_std[i]) for i in range(len(valid))])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dfo = pd.DataFrame([{
        "name": r["name"], "cas": r.get("cas", ""), "smiles": r["smiles"],
        "cluster": r.get("cluster"), "n_heavy": r.get("n_heavy"),
        "wall_s": r["wall_s"],
    } for r in valid])
    dfo["pearson_r_raw"] = r_raw
    dfo["pearson_r_standardized"] = r_std
    dfo.to_csv(args.out, index=False)

    failed = [r for r in results if r.get("error")]
    print(f"\nWrote {len(dfo)} rows to {args.out}")
    print(f"Wall: {elapsed:.1f} s  ({elapsed/len(valid):.1f} s/mol effective; speedup vs serial ≈ {sum(r['wall_s'] for r in valid)/elapsed:.1f}×)")
    if failed:
        print(f"Failed: {len(failed)} (names: {', '.join(r['name'] for r in failed[:5])}{'...' if len(failed)>5 else ''})")
    print(f"\nSummary (backend={args.backend}, {len(dfo)} molecules):")
    print(f"  Pearson r raw          : mean={r_raw.mean():+.4f}  median={np.median(r_raw):+.4f}  "
          f"min={r_raw.min():+.4f}  max={r_raw.max():+.4f}  ≥0.9: {(r_raw>=0.9).sum()}/{len(r_raw)}")
    print(f"  Pearson r standardized : mean={r_std.mean():+.4f}  median={np.median(r_std):+.4f}  "
          f"min={r_std.min():+.4f}  max={r_std.max():+.4f}  ≥0.5: {(r_std>=0.5).sum()}/{len(r_std)}")
    if dfo["cluster"].notna().any():
        print("\n  by cluster:")
        for c, sub in dfo[dfo["cluster"].notna()].groupby("cluster"):
            print(f"    cluster {int(c):>2}: n={len(sub):>3}  r_raw mean={sub['pearson_r_raw'].mean():+.4f}  "
                  f"r_std mean={sub['pearson_r_standardized'].mean():+.4f}")


if __name__ == "__main__":
    main()
