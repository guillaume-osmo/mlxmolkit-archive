#!/usr/bin/env python3
"""End-to-end COSMO-RS paracetamol solubility from the binary-backed pipeline.

Reproduces the FACCTS OPI openCOSMO-RS notebook target on top of
``mlxmolkit.xtb.cosmo_sigma``:

  1. Build hybrid g-xTB(opt) + GFN2(tmcosmo) σ-profiles for paracetamol,
     water, and ethanol.
  2. Run openCOSMO-RS_py's ``COSMORS(par=openCOSMORS24a())`` to get
     ln(γ_paracetamol) at infinite dilution in each solvent at 298.15 K.
  3. Convert to predicted mole-fraction solubility via the standard
     SLE relation:

       ln(x_sat) = -ln(γ) - (ΔH_fus/R)(1/T - 1/T_fus)

  4. Compare to literature (cited by the FACCTS notebook):

       water:   x_exp = 0.001773
       ethanol: x_exp = 0.075860

Note: the FACCTS notebook uses DFT-level σ-profiles; we use GFN2 +
ddCOSMO, so quantitative agreement isn't expected — the goal is to
demonstrate the full pipeline runs and yields physically sensible
numbers in the right ballpark.

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_paracetamol_solubility.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_ADDONS_SRC = Path.home() / "Github" / "mlx-addons" / "src"
for p in (REPO_ROOT, MLX_ADDONS_SRC):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# Paracetamol thermodynamic constants
PARACETAMOL_DH_FUS_J = 27.1e3  # J/mol  (Granberg & Rasmuson, 1999; lit. range 26-28 kJ/mol)
PARACETAMOL_T_FUS_K = 442.0    # K       (169 °C)
T_K = 298.15
R_GAS = 8.314462618             # J/(mol·K)


def sigma_profile_cosmo_path(smiles: str, name: str, workdir: Path, *, solvent: str = "inf") -> Path:
    """Run the hybrid pipeline and write an openCOSMO-RS-readable .cosmo.

    The COSMO-RS convention is to use ``epsilon=infinity`` (ideal-conductor)
    σ-profiles regardless of the actual target solvent — the dielectric
    response of each solvent enters via its own σ-profile, not via the QM
    surface charges.
    """

    from mlxmolkit.xtb import hybrid_gxtb_gfn2_cosmo_from_smiles, write_cosmo_file

    out = hybrid_gxtb_gfn2_cosmo_from_smiles(smiles, seed=42, solvent=solvent)
    path = workdir / f"{name}.cosmo"
    write_cosmo_file(out["cosmo"], path)
    return path


def ideal_solubility_correction(*, dh_fus_J: float, T_fus_K: float, T_K: float) -> float:
    """ln(x_id) = -(ΔH_fus/R)(1/T - 1/T_fus). Standard SLE ideal-solubility limit."""

    return -(dh_fus_J / R_GAS) * (1.0 / T_K - 1.0 / T_fus_K)


def predict_solubility_infinite_dilution(ln_gamma_inf: float) -> float:
    """x_sat ≈ exp(-ln γ_∞ + ln x_id). Cheap closed-form, valid as long as the
    actual saturated x is small enough that γ ≈ γ_∞ at saturation."""

    ln_x_id = ideal_solubility_correction(
        dh_fus_J=PARACETAMOL_DH_FUS_J,
        T_fus_K=PARACETAMOL_T_FUS_K,
        T_K=T_K,
    )
    return math.exp(ln_x_id - ln_gamma_inf)


def cosmors_lng(solute_cosmo: Path, solvent_cosmo: Path) -> float:
    """Compute ln(γ_solute) at infinite dilution in solvent at T = 298.15 K."""

    from opencosmorspy.cosmors import COSMORS
    from opencosmorspy.parameterization import openCOSMORS24a

    crs = COSMORS(par=openCOSMORS24a())
    crs.add_molecule([str(solute_cosmo)])
    crs.add_molecule([str(solvent_cosmo)])

    x_inf = np.array([1.0e-8, 1.0 - 1.0e-8])  # solute infinitely dilute
    crs.add_job(x=x_inf, T=T_K, refst="pure_component")
    res = crs.calculate()
    return float(res["tot"]["lng"][0][0])  # solute = component 0


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cosmors_pcm_") as td:
        workdir = Path(td)
        print("Generating σ-profiles (g-xTB --opt → GFN2 --tmcosmo)...")
        paracetamol_path = sigma_profile_cosmo_path(
            "CC(=O)NC1=CC=C(C=C1)O", "paracetamol", workdir
        )
        water_path = sigma_profile_cosmo_path("O", "water", workdir)
        ethanol_path = sigma_profile_cosmo_path("CCO", "ethanol", workdir)
        print(f"  paracetamol: {paracetamol_path.stat().st_size / 1024:.1f} KB")
        print(f"  water:       {water_path.stat().st_size / 1024:.1f} KB")
        print(f"  ethanol:     {ethanol_path.stat().st_size / 1024:.1f} KB")

        print("\nRunning openCOSMO-RS_py (par=openCOSMORS24a)...")
        ln_g_water = cosmors_lng(paracetamol_path, water_path)
        ln_g_ethanol = cosmors_lng(paracetamol_path, ethanol_path)

        print()
        print(f"  ln γ_paracetamol(in water, x→0)   = {ln_g_water:+.4f}   γ_∞ = {math.exp(ln_g_water):.3f}")
        print(f"  ln γ_paracetamol(in ethanol, x→0) = {ln_g_ethanol:+.4f}   γ_∞ = {math.exp(ln_g_ethanol):.3f}")

        x_pred_water = predict_solubility_infinite_dilution(ln_g_water)
        x_pred_ethanol = predict_solubility_infinite_dilution(ln_g_ethanol)

        ln_x_id = ideal_solubility_correction(
            dh_fus_J=PARACETAMOL_DH_FUS_J, T_fus_K=PARACETAMOL_T_FUS_K, T_K=T_K
        )
        x_id = math.exp(ln_x_id)
        print(f"\n  Ideal solubility (γ=1) at 298.15 K: x_id = {x_id:.4f}")
        print(f"  ΔH_fus = {PARACETAMOL_DH_FUS_J/1000:.1f} kJ/mol,  T_fus = {PARACETAMOL_T_FUS_K:.1f} K\n")

        print(f"  {'solvent':<10} {'x_pred (γ_∞)':>14} {'x_exp':>12} {'pred/exp':>12}")
        print(f"  {'-' * 50}")
        for solvent, x_pred, x_exp in [
            ("water",   x_pred_water,   0.001773),
            ("ethanol", x_pred_ethanol, 0.075860),
        ]:
            print(f"  {solvent:<10} {x_pred:>14.5f} {x_exp:>12.5f} {x_pred / x_exp:>12.2f}")


if __name__ == "__main__":
    main()
