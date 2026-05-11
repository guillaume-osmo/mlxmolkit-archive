#!/usr/bin/env python3
"""Apply our 25a σ-orth kernel to CHAOS raw segments (no ORCA).

Three-way test on cluster-7 outliers + simple-alcohol controls:

  A. CHAOS Sigma_total + Klamt-1995  (= original CHAOS comparison)
  B. CHAOS SegmentList + our σ-orth  (NEW: CHAOS QM, our 25a physics)
  C. Our pipeline (ORCA + σ-orth)    (= ran earlier; for context)

Hypothesis: B ≈ C if our σ-orth physics is the dominant source of
parity, regardless of QM (CHAOS uses ωB97X-D, we use BP86). If so,
CHAOS becomes a free QM source for 25a-quality σ-potentials.
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
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


def chaos_25a_mu(zf, chaos_id, sigma_grid):
    """Read CHAOS SegmentList, build CosmoSegments, apply 25a σ-orth kernel."""
    from mlxmolkit.xtb import sigma_potential
    from mlxmolkit.xtb.cosmo_sigma import CosmoSegments

    with zf.open(f"{chaos_id}.json") as f:
        data = json.loads(f.read())
    sl = np.asarray(data["solvation"]["SegmentList"], dtype=np.float64)
    # cols: [idx, atom_idx, x_Bohr, y_Bohr, z_Bohr, charge_e, area_Å², sigma_e/Å², col8?]
    cs = CosmoSegments(
        epsilon=float("inf"), fepsi=1.0,
        area=float(data["solvation"]["CavArea"]),
        volume=float(data["solvation"]["CavVolume"]),
        total_screening_charge=0.0,
        total_energy_hartree=float("nan"),
        dielectric_energy_hartree=float("nan"),
        atom_radii=np.zeros(1),                # not needed for σ-potential
        atom_coords_bohr=np.zeros((1, 3)),     # not needed
        atom_z=[0],
        segments_atom=sl[:, 1].astype(np.intp),
        segments_xyz_bohr=sl[:, 2:5].copy(),
        segments_charge=sl[:, 5].copy(),
        segments_area=sl[:, 6].copy(),
        segments_sigma=sl[:, 7].copy(),
        segments_potential=sl[:, 8].copy(),
        cosmo_text="",
    )
    _, mu = sigma_potential(cs, sigma_grid_e_per_A2=sigma_grid)
    return mu


def chaos_basic_mu(zf, chaos_id, sigma_grid):
    """Original strategy: Sigma_total + Klamt 1995 (no σ-orth)."""
    from mlxmolkit.xtb import sigma_potential_from_arrays
    with zf.open(f"{chaos_id}.json") as f:
        data = json.loads(f.read())
    sig_total = np.asarray(data["solvation"]["Sigma_total"], dtype=np.float64)
    _, mu = sigma_potential_from_arrays(
        CHAOS_GRID, sig_total,
        use_sigma_orth=False, sigma_grid_e_per_A2=sigma_grid,
    )
    return mu


def main() -> None:
    from rdkit import Chem

    targets = [
        ("cis-9-octadecenoicacid",                  r"C(CCCCCCC\C=C/CCCCCCCC)(=O)O",  -0.8748,  0.9880),
        ("2,2-dimethyl-4-hydroxymethyl-1,3-dioxolane", "CC1(OCC(O1)CO)C",            +0.7110,  0.9154),
        ("2-hydroxypropanoicacidethylester",        "CCOC(=O)C(C)O",                +0.8159,  0.9924),
        ("oleyl-alcohol",                           r"C(CCCCCCC\C=C/CCCCCCCC)O",     +0.8538,  0.9912),
        ("1-butanol",                               "C(CCC)O",                       +0.9888,  0.9931),
        ("2-propanol",                              "CC(C)O",                        +0.9960,  0.9914),
        ("cyclohexanol",                            "C1(CCCCC1)O",                   +0.9962,  0.9901),
    ]

    paper = pd.read_excel(REPO_ROOT / "data" / "d5ra08246c1.xlsx", sheet_name="Database", header=0)
    paper_mu = {str(r["Name"]).strip().lower(): r.iloc[3:64].to_numpy(dtype=np.float64)
                for _, r in paper.iterrows() if pd.notna(r["Name"])}

    chaos_idx = pd.read_csv(REPO_ROOT / "data" / "chaos_index.csv")
    chaos_idx["canon"] = [
        Chem.MolToSmiles(Chem.MolFromSmiles(s)) if isinstance(s, str) and s else ""
        for s in chaos_idx["canonical_smiles"]
    ]
    chaos_lookup = {c: str(cid) for cid, c in zip(chaos_idx["chaos_id"], chaos_idx["canon"]) if c}

    print(f"{'mol':<42} {'A: CHAOS+Klamt1995':>20} {'B: CHAOS+25a':>14} {'C: ours+25a':>14} {'B-A':>6} {'B-C':>7}")
    print("-" * 110)
    with zipfile.ZipFile("/Users/guillaume-osmo/Github/data/CHAOS.zip") as zf:
        for name, smi, r_A_known, r_C_known in targets:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            cid = chaos_lookup.get(canon)
            if cid is None:
                print(f"  {name:<40}  no CHAOS match")
                continue
            mu_p = paper_mu.get(name.lower())
            if mu_p is None:
                continue
            mu_A = chaos_basic_mu(zf, cid, PAPER_GRID)
            mu_B = chaos_25a_mu(zf, cid, PAPER_GRID)
            r_A = pearson(mu_A, mu_p)
            r_B = pearson(mu_B, mu_p)
            print(f"{name:<42}  {r_A:>+18.4f}   {r_B:>+12.4f}   {r_C_known:>+12.4f}   {r_B-r_A:>+5.3f}  {r_B-r_C_known:>+6.3f}")


if __name__ == "__main__":
    main()
