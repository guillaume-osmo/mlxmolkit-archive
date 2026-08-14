#!/usr/bin/env python3
"""Tiered COSMO-RS paracetamol solubility: mlxmolkit cheap → ORCA DFT only.

Pipeline:
  1. SMILES → RDKit embed → g-xTB --opt   (mlxmolkit, cheap, ~1 s/mol)
  2. ORCA !BP86 def2-TZVP COSMORS(water)  (DFT, ~1 min/mol for 20 atoms)
     → produces <job>.solute.orcacosmo, the DFT-level σ-profile that
       openCOSMORS24a was parameterized against.
  3. opencosmorspy.COSMORS(par=openCOSMORS24a()) → ln(γ_∞).
  4. SLE: ln(x_sat) = -ln(γ) - (ΔH_fus/R)(1/T - 1/T_fus).

Target: paracetamol experimental solubility (FACCTS notebook reference).
  water:   x_exp = 0.001773
  ethanol: x_exp = 0.075860

Compare to the GFN2-tmcosmo result from cosmors_paracetamol_solubility.py:
  water:   x_pred = 0.118 (67× over)
  ethanol: x_pred = 0.038 (0.50× under)

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_paracetamol_tiered.py
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


PARACETAMOL_DH_FUS_J = 27.1e3
PARACETAMOL_T_FUS_K = 442.0
T_K = 298.15
R_GAS = 8.314462618


def build_orcacosmo(smiles: str, name: str, workdir: Path, *, solvent: str = "water", n_cores: int = 4) -> Path:
    from mlxmolkit.xtb import tiered_gxtb_orca_cosmors_from_smiles

    t0 = time.perf_counter()
    res = tiered_gxtb_orca_cosmors_from_smiles(
        smiles,
        seed=42,
        solvent=solvent,
        workdir=workdir / name,
        keep_workdir=True,
        n_cores=n_cores,
    )
    wall = time.perf_counter() - t0
    print(f"  {name:<12}  E_gxtb={res['gxtb_energy_hartree']:+.4f} Ha  ({wall:.1f}s)")
    print(f"    → {res['orcacosmo_path']}")
    return res["orcacosmo_path"]


def cosmors_lng(solute: Path, solvent: Path) -> float:
    from opencosmorspy.cosmors import COSMORS
    from opencosmorspy.parameterization import openCOSMORS24a

    crs = COSMORS(par=openCOSMORS24a())
    crs.add_molecule([str(solute)])
    crs.add_molecule([str(solvent)])
    x_inf = np.array([1.0e-8, 1.0 - 1.0e-8])
    crs.add_job(x=x_inf, T=T_K, refst="pure_component")
    res = crs.calculate()
    return float(res["tot"]["lng"][0][0])


def ideal_solubility() -> float:
    ln_x_id = -(PARACETAMOL_DH_FUS_J / R_GAS) * (1.0 / T_K - 1.0 / PARACETAMOL_T_FUS_K)
    return math.exp(ln_x_id)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tiered_pcm_") as td:
        wd = Path(td)
        print("Tier 1+2: building DFT-level σ-profiles (g-xTB --opt → ORCA BP86/def2-TZVP COSMORS)...")
        paracetamol_p = build_orcacosmo("CC(=O)NC1=CC=C(C=C1)O", "paracetamol", wd, solvent="water")
        water_p = build_orcacosmo("O", "water", wd, solvent="water")
        ethanol_p = build_orcacosmo("CCO", "ethanol", wd, solvent="water")

        print("\nTier 3: openCOSMORS24a (parameterized for BP86/def2-TZVP COSMO σ-profiles)...")
        ln_g_w = cosmors_lng(paracetamol_p, water_p)
        ln_g_e = cosmors_lng(paracetamol_p, ethanol_p)
        print(f"  ln γ_pcm(in water)   = {ln_g_w:+.4f}   γ_∞ = {math.exp(ln_g_w):.3f}")
        print(f"  ln γ_pcm(in ethanol) = {ln_g_e:+.4f}   γ_∞ = {math.exp(ln_g_e):.3f}")

        x_id = ideal_solubility()
        x_pred_w = math.exp(math.log(x_id) - ln_g_w)
        x_pred_e = math.exp(math.log(x_id) - ln_g_e)
        print(f"\n  Ideal x_id = {x_id:.4f} (γ=1 limit at 298.15 K)")
        print(f"  {'solvent':<10} {'x_pred':>10} {'x_exp':>10} {'pred/exp':>10}")
        print(f"  {'-' * 44}")
        for solvent, x_pred, x_exp in [("water", x_pred_w, 0.001773),
                                         ("ethanol", x_pred_e, 0.075860)]:
            print(f"  {solvent:<10} {x_pred:>10.5f} {x_exp:>10.5f} {x_pred / x_exp:>10.2f}")

        # Reference (GFN2-tmcosmo baseline from cosmors_paracetamol_solubility.py)
        print()
        print("  --- Reference (GFN2-tmcosmo σ-profiles, prior commit) ---")
        print(f"  {'water':<10} {0.118:>10.5f} {0.001773:>10.5f} {0.118/0.001773:>10.2f}")
        print(f"  {'ethanol':<10} {0.038:>10.5f} {0.075860:>10.5f} {0.038/0.075860:>10.2f}")


if __name__ == "__main__":
    main()
