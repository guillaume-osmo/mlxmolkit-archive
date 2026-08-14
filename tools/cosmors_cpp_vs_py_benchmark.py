#!/usr/bin/env python3
"""Speed + accuracy benchmark: openCOSMO-RS_py vs openCOSMO-RS_cpp.

Same QM inputs (our hybrid g-xTB --opt + GFN2 --tmcosmo inf σ-profiles
for paracetamol, water, ethanol). Compute ln(γ) at paracetamol→0
dilution in water and in ethanol via both backends and time each.

The Python backend uses ``openCOSMORS24a`` (BP86/def2-TZVPD-trained).
The C++ backend uses the parameter set from the Chem Eng Sci 2025
paper (the latest, polarizability-projection variant) — what users
informally call openCOSMORS25.

Run from the repo root:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src:/tmp/openCOSMO-RS_cpp/build \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/cosmors_cpp_vs_py_benchmark.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, Path.home() / "Github/mlx-addons/src", Path("/tmp/openCOSMO-RS_cpp/build")):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


T_K = 298.15


CPP_OPTIONS = {
    "sw_skip_COSMOSPACE_errors": 0,
    "sw_SR_COSMOfiles_type": "Turbomole_COSMO_TZVP",  # our xtb tmcosmo writes TM format
    "sw_SR_combTerm": 1,
    "sw_SR_alwaysReloadSigmaProfiles": 1,
    "sw_SR_alwaysCalculateSizeRelatedParameters": 1,
    "sw_SR_useSegmentReferenceStateForInteractionMatrix": 0,
    "sw_SR_calculateContactStatisticsAndAdditionalProperties": 0,
    "sw_SR_partialInteractionMatrices": [],
    "sw_SR_atomicNumber": 1,
    "sw_SR_misfit": 2,
    "sw_SR_differentiateHydrogens": 0,
    "sw_SR_differentiateMoleculeGroups": 0,
    # openCOSMORS25a polarizability projections require per-atom
    # polarizabilities (written by ORCA's COSMO but NOT by xtb --tmcosmo);
    # set 0 to skip — speed-only swap vs Python24a, accuracy gain from
    # polarizabilities is unavailable until we add an ORCA backend.
    "sw_SR_polarizabilities": 0,
}

# openCOSMORS25 latest params (Chem Eng Sci 2025; doi:10.1016/j.ces.2025.122170).
CPP_PARAMS = {
    "Aeff": 4.90825,
    "alpha": 7876000.0,
    "CHB": 49318000.0,
    "CHBT": 1.5,
    "SigmaHB": 0.009953,
    "Rav": 0.5,
    "RavCorr": 1,
    "fCorr": 2.4,
    "comb_SG_z_coord": 0.0,
    "comb_SG_A_std": 1.0,
    "comb_modSG_exp": 2.0 / 3.0,
    "comb_lambda0": 0.463,
    "comb_lambda1": 0.42,
    "comb_lambda2": 0.065,
    "comb_SGG_lambda": 0.773,
    "comb_SGG_beta": 0.778,
    "m_vdW": 29.567,
    "E_F_corr": 346.82,
    "radii": {},
    "exp": {},
}


def make_sigma_profiles(workdir: Path) -> dict[str, Path]:
    from mlxmolkit.xtb import hybrid_gxtb_gfn2_cosmo_from_smiles, write_cosmo_file

    smiles_map = {
        "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
        "water": "O",
        "ethanol": "CCO",
    }
    out: dict[str, Path] = {}
    for name, smi in smiles_map.items():
        t0 = time.perf_counter()
        res = hybrid_gxtb_gfn2_cosmo_from_smiles(smi, seed=42, solvent="inf")
        wall = time.perf_counter() - t0
        path = workdir / f"{name}.cosmo"
        write_cosmo_file(res["cosmo"], path)
        print(f"  σ-profile {name:<12} ({len(res['cosmo'].segments_sigma)} segs, {wall:.2f}s)")
        out[name] = path
    return out


def cosmors_py_lng(solute_path: Path, solvent_path: Path) -> tuple[float, float]:
    """Returns ``(ln γ_solute_at_infinite_dilution, wall_seconds)``."""

    from opencosmorspy.cosmors import COSMORS
    from opencosmorspy.parameterization import openCOSMORS24a

    crs = COSMORS(par=openCOSMORS24a())
    crs.add_molecule([str(solute_path)])
    crs.add_molecule([str(solvent_path)])
    x_inf = np.array([1.0e-8, 1.0 - 1.0e-8])
    crs.add_job(x=x_inf, T=T_K, refst="pure_component")

    t0 = time.perf_counter()
    res = crs.calculate()
    wall = time.perf_counter() - t0
    return float(res["tot"]["lng"][0][0]), wall


def cosmors_cpp_lng(solute_path: Path, solvent_path: Path) -> tuple[float, float]:
    """Returns ``(ln γ_solute_total_at_infinite_dilution, wall_seconds)``."""

    import openCOSMORS

    openCOSMORS.initialize()  # required between successive runs

    components = [str(solute_path), str(solvent_path)]
    x_inf = np.array([[1.0e-8, 1.0 - 1.0e-8]])

    calc_inf = {
        "concentrations": x_inf,
        "temperatures": np.array([T_K]),
        "components": components,
        "reference_state_types": np.array([3]),  # 3 == pure_component
        "reference_state_concentrations": np.array([[]]),
        "component_indices": [0, 1],
        "ln_gamma_x_SR_combinatorial_calc": np.zeros_like(x_inf),
        "ln_gamma_x_SR_residual_calc": np.zeros_like(x_inf),
        "ln_gamma_x_SR_calc": np.zeros_like(x_inf),
        "index": 0,
    }
    calculations = [calc_inf]

    t0 = time.perf_counter()
    openCOSMORS.loadMolecules(CPP_OPTIONS, CPP_PARAMS, components)
    openCOSMORS.loadCalculations(calculations)
    calculations = openCOSMORS.calculate(CPP_PARAMS, calculations, False)
    wall = time.perf_counter() - t0

    res = calculations[0]["ln_gamma_x_SR_residual_calc"][0, 0]
    res += calculations[0]["ln_gamma_x_SR_combinatorial_calc"][0, 0]
    return float(res), wall


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cosmors_bench_") as td:
        wd = Path(td)
        print("Generating σ-profiles (hybrid g-xTB --opt → GFN2 --tmcosmo inf)...")
        cosmo_paths = make_sigma_profiles(wd)

        print("\nPython openCOSMO-RS_py (par=openCOSMORS24a):")
        py_w_lng, py_w_t = cosmors_py_lng(cosmo_paths["paracetamol"], cosmo_paths["water"])
        py_e_lng, py_e_t = cosmors_py_lng(cosmo_paths["paracetamol"], cosmo_paths["ethanol"])
        print(f"  water:    ln γ = {py_w_lng:+.4f}   ({py_w_t:.4f} s)")
        print(f"  ethanol:  ln γ = {py_e_lng:+.4f}   ({py_e_t:.4f} s)")

        print("\nC++ openCOSMO-RS_cpp (par = Chem Eng Sci 2025 / openCOSMORS25):")
        cpp_w_lng, cpp_w_t = cosmors_cpp_lng(cosmo_paths["paracetamol"], cosmo_paths["water"])
        cpp_e_lng, cpp_e_t = cosmors_cpp_lng(cosmo_paths["paracetamol"], cosmo_paths["ethanol"])
        print(f"  water:    ln γ = {cpp_w_lng:+.4f}   ({cpp_w_t:.4f} s)")
        print(f"  ethanol:  ln γ = {cpp_e_lng:+.4f}   ({cpp_e_t:.4f} s)")

        print("\n--- speedup (cpp ÷ py wall) ---")
        print(f"  water:    py {py_w_t*1000:.1f} ms  →  cpp {cpp_w_t*1000:.1f} ms   speedup ×{py_w_t / max(cpp_w_t, 1e-9):.1f}")
        print(f"  ethanol:  py {py_e_t*1000:.1f} ms  →  cpp {cpp_e_t*1000:.1f} ms   speedup ×{py_e_t / max(cpp_e_t, 1e-9):.1f}")


if __name__ == "__main__":
    main()
