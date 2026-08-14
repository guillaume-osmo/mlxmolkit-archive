#!/usr/bin/env python3
"""Compare our hybrid σ-profile against the 'true COSMO-RS' dataset.

Reference: d5ra08246c1.xlsx (RSC Advances 2026 paper). The ``Database``
sheet contains 1588 molecules with reference σ-profiles produced by full
COSMO-RS at the DFT level. Columns:

    [0] Name, [1] CAS, [2] Cluster (1-10, ~224 of 1588 assigned),
    [3..63] σ-profile p(σ) values on bins σ ∈ [-3.0, +3.0] e/nm² in
              0.1 e/nm² steps (61 bins) ≡ σ ∈ [-0.03, +0.03] e/Å²,
    [64] Dim1, [65] Dim2 (PCA projections).

For each selected molecule we:
  1. Resolve Name/CAS → SMILES via PubChem (pubchempy), cached locally.
  2. Run our hybrid pipeline (g-xTB --opt + GFN2 --tmcosmo inf).
  3. Compute the σ-profile on the same 61-bin reference grid.
  4. Compare to the paper profile: Pearson r, L1, area-normalized L1.

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_vs_paper_database.py --n 100 --max-heavy 12
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


PAPER_BINS = np.round(np.arange(-3.0, 3.01, 0.1), 4)  # e/nm²
PAPER_EDGES = np.concatenate([PAPER_BINS - 0.05, [PAPER_BINS[-1] + 0.05]])  # bin edges (62)
ALLOWED_Z = {1, 6, 7, 8, 9, 16, 17, 35}  # CHNOF + S, Cl, Br
SMILES_CACHE_PATH = REPO_ROOT / "data" / "paper_database_smiles_cache.json"


def load_smiles_cache() -> dict[str, str]:
    if SMILES_CACHE_PATH.exists():
        return json.loads(SMILES_CACHE_PATH.read_text())
    return {}


def save_smiles_cache(cache: dict[str, str]) -> None:
    SMILES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMILES_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def resolve_smiles(name: str, cas: str, cache: dict[str, str]) -> str | None:
    key = f"{name}||{cas}"
    if key in cache:
        return cache[key] or None  # empty string == known failure
    import pubchempy as pcp

    # 1) by CAS / name → PubChem
    for query in (cas, name):
        if not isinstance(query, str) or not query:
            continue
        # cas may be "67-64-1" or "75-15-0"
        try:
            cs = pcp.get_compounds(query, "name")
            if cs:
                smi = cs[0].isomeric_smiles or cs[0].canonical_smiles
                if smi:
                    cache[key] = smi
                    return smi
        except Exception:
            pass
    cache[key] = ""  # mark unresolvable
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
    if m.GetNumHeavyAtoms() > max_heavy:
        return False
    return True


def project_to_paper_bins(cosmo, *, use_klamt: bool, variant: str = "mullins") -> np.ndarray:
    """Histogram our σ values in e/Å² onto the paper's e/nm² grid.

    Paper bins are e/nm² (= e/Å² × 100). Our σ in e/Å² ranges roughly
    [-0.025, +0.025] → on the paper grid [-2.5, +2.5].
    """

    from mlxmolkit.xtb import klamt_average_sigmas

    if use_klamt:
        sigma = klamt_average_sigmas(cosmo, variant=variant)
    else:
        sigma = np.asarray(cosmo.segments_sigma, dtype=np.float64)
    sigma_e_per_nm2 = sigma * 100.0
    weights = np.asarray(cosmo.segments_area, dtype=np.float64)
    p, _ = np.histogram(sigma_e_per_nm2, bins=PAPER_EDGES, weights=weights)
    return p


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
    if denom == 0:
        return float("nan")
    return float((da * db).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=REPO_ROOT / "data" / "d5ra08246c1.xlsx")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-heavy", type=int, default=12)
    parser.add_argument("--require-cluster", action="store_true",
                        help="only include rows where Cluster is assigned")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "cosmors_vs_paper.csv")
    parser.add_argument("--klamt-variant", choices=["mullins", "hsieh", "raw"], default="mullins")
    args = parser.parse_args()

    print(f"Loading {args.xlsx}...")
    df = pd.read_excel(args.xlsx, sheet_name="Database", header=0)
    cand = df.copy()
    if args.require_cluster:
        cand = cand[cand["Cluster"].notna()]
    print(f"  candidates after cluster filter: {len(cand)}")

    cache = load_smiles_cache()
    print(f"  smiles cache: {len(cache)} entries")

    rows_out = []
    success = 0
    for i, row in cand.iterrows():
        if success >= args.n:
            break
        name = str(row["Name"]).strip() if pd.notna(row["Name"]) else ""
        cas = str(row["CAS"]).strip() if pd.notna(row["CAS"]) else ""
        cluster = row["Cluster"] if pd.notna(row["Cluster"]) else None

        smi = resolve_smiles(name, cas, cache)
        if smi is None:
            continue
        from rdkit import Chem
        try:
            if not smiles_passes(smi, args.max_heavy):
                continue
        except Exception:
            continue

        # save cache periodically
        if (i % 10) == 0:
            save_smiles_cache(cache)

        # generate hybrid σ-profile
        try:
            from mlxmolkit.xtb import hybrid_gxtb_gfn2_cosmo_from_smiles
            t0 = time.perf_counter()
            out = hybrid_gxtb_gfn2_cosmo_from_smiles(smi, seed=42, solvent="inf")
            wall = time.perf_counter() - t0
        except Exception as exc:
            print(f"  [skip] {name!r}: pipeline failed: {exc}")
            continue

        cosmo = out["cosmo"]
        use_klamt = args.klamt_variant != "raw"
        ours_p = project_to_paper_bins(cosmo, use_klamt=use_klamt, variant=args.klamt_variant if use_klamt else "mullins")
        paper_p = row.iloc[3:64].to_numpy(dtype=np.float64)

        # area-normalize both before comparison (the paper's normalization is unknown)
        s_ours = ours_p.sum()
        s_paper = paper_p.sum()
        if abs(s_paper) < 1e-12 or abs(s_ours) < 1e-12:
            continue
        ours_norm = ours_p / s_ours
        paper_norm = paper_p / s_paper

        r = pearson(ours_p, paper_p)
        l1_norm = float(np.abs(ours_norm - paper_norm).sum())
        l2_norm = float(np.sqrt(((ours_norm - paper_norm) ** 2).sum()))

        success += 1
        msg = (f"[{success:3d}] {name:<20} nat={Chem.MolFromSmiles(smi).GetNumHeavyAtoms():>2} "
               f"cluster={cluster}  r={r:+.4f}  L1n={l1_norm:.3f}  ({wall:.2f}s)")
        print(msg, flush=True)
        rows_out.append({
            "name": name, "cas": cas, "smiles": smi, "cluster": cluster,
            "n_heavy": Chem.MolFromSmiles(smi).GetNumHeavyAtoms(),
            "pearson_r": r, "L1_normalized": l1_norm, "L2_normalized": l2_norm,
            "ours_area_total": float(cosmo.area),
            "ours_p_sum": float(s_ours), "paper_p_sum": float(s_paper),
            "wall_s": wall,
        })

    save_smiles_cache(cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.out, index=False)
    print(f"\nSaved {len(rows_out)} rows to {args.out}")
    if rows_out:
        r_arr = np.array([row["pearson_r"] for row in rows_out])
        l1_arr = np.array([row["L1_normalized"] for row in rows_out])
        print(f"\nSummary over {len(rows_out)} molecules:")
        print(f"  Pearson r: mean={r_arr.mean():.4f}  median={np.median(r_arr):.4f}  "
              f"min={r_arr.min():.4f}  max={r_arr.max():.4f}")
        print(f"  L1 (norm): mean={l1_arr.mean():.4f}  median={np.median(l1_arr):.4f}")
        # by cluster
        dfo = pd.DataFrame(rows_out)
        if dfo["cluster"].notna().any():
            print("\n  by cluster:")
            for c, sub in dfo[dfo["cluster"].notna()].groupby("cluster"):
                print(f"    cluster {int(c):>2}: n={len(sub):>3}  mean r={sub['pearson_r'].mean():.4f}  "
                      f"mean L1n={sub['L1_normalized'].mean():.4f}")


if __name__ == "__main__":
    main()
