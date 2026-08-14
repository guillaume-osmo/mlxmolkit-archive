#!/usr/bin/env python3
"""End-to-end benchmark: mlxmolkit (our GFN2 + ALPB(water)) vs tblite.

Mirrors the openCOSMO-RS conformer pipeline's inner loop: many small
molecules, GFN2-xTB + ALPB(water) singlepoint each. Shows that the
mlxmolkit path is (a) functionally correct vs the canonical xtb/tblite
reference and (b) quantifies the current cost ratio.

Run: ``python benchmarks/gfn2_alpb_pipeline.py`` from a checkout where
``mlxmolkit`` and ``mlx-addons`` are importable.
"""
import sys
# Repo-relative imports — adjust paths if running from elsewhere.
sys.path.insert(0, '/Users/guillaume-osmo/Github/mlxmolkit')
sys.path.insert(0, '/Users/guillaume-osmo/Github/mlx-addons/src')

import time
import numpy as np
from tblite.interface import Calculator
from mlxmolkit.xtb.scf_gfn2 import gfn2_energy
from mlxmolkit.xtb.solvation_alpb import alpb_water_correction

ANG_TO_BOHR = 1.8897259886
KCAL = 627.5094740631


CONFORMERS = {
    "H2O":   ([8, 1, 1], np.array([
        [0.0, 0.0, 0.117], [0.0, 0.755, -0.470], [0.0, -0.755, -0.470]])),
    "NH3":   ([7, 1, 1, 1], np.array([
        [0.0, 0.0, 0.0],
        [0.939, 0.0, -0.34],
        [-0.4695, 0.813, -0.34],
        [-0.4695, -0.813, -0.34]])),
    "CH4":   ([6, 1, 1, 1, 1], np.array([
        [0.0, 0.0, 0.0],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629]])),
    "CH3OH": ([6, 8, 1, 1, 1, 1], np.array([
        [-0.748, -0.015, 0.024],
        [0.626,  0.310, 0.026],
        [-1.293, 0.949, 0.06],
        [-1.022,-0.580,-0.876],
        [-1.018,-0.626, 0.882],
        [1.114, -0.554, -0.043]])),
    "ethanol": ([6, 6, 8, 1, 1, 1, 1, 1, 1], np.array([
        [-1.137, -0.387, 0.0],
        [ 0.0,    0.554, 0.0],
        [ 1.183, -0.231, 0.0],
        [-1.137, -1.034, 0.880],
        [-1.137, -1.034,-0.880],
        [-2.018,  0.255, 0.0],
        [ 0.0,    1.198,-0.880],
        [ 0.0,    1.198, 0.880],
        [ 1.945,  0.350, 0.0]])),
    "formamide": ([7, 6, 8, 1, 1, 1], np.array([
        [-0.745,  0.392, 0.0],
        [ 0.430, -0.232, 0.0],
        [ 1.547,  0.247, 0.0],
        [-1.610, -0.099, 0.0],
        [-0.755,  1.387, 0.0],
        [ 0.388, -1.328, 0.0]])),
    "acetone":  ([6, 6, 6, 8, 1, 1, 1, 1, 1, 1], np.array([
        [-1.290, -0.060,  0.0],
        [ 0.0,    0.685,  0.0],
        [ 1.290, -0.060,  0.0],
        [ 0.0,    1.918,  0.0],
        [-2.158,  0.595,  0.0],
        [-1.295, -0.687,  0.882],
        [-1.295, -0.687, -0.882],
        [ 2.158,  0.595,  0.0],
        [ 1.295, -0.687,  0.882],
        [ 1.295, -0.687, -0.882]])),
    "benzene":  ([6]*6 + [1]*6, np.array([
        [ 1.396,  0.0,    0.0],
        [ 0.698,  1.209,  0.0],
        [-0.698,  1.209,  0.0],
        [-1.396,  0.0,    0.0],
        [-0.698, -1.209,  0.0],
        [ 0.698, -1.209,  0.0],
        [ 2.481,  0.0,    0.0],
        [ 1.241,  2.149,  0.0],
        [-1.241,  2.149,  0.0],
        [-2.481,  0.0,    0.0],
        [-1.241, -2.149,  0.0],
        [ 1.241, -2.149,  0.0]])),
}


def time_call(fn, n=3):
    """Best-of-n median timing."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, sorted(ts)[n // 2]


def tblite_singlepoint(atoms, coords_ang, alpb=True):
    pos_b = np.asarray(coords_ang) * ANG_TO_BOHR
    c = Calculator("GFN2-xTB", np.asarray(atoms), pos_b)
    c.set("verbosity", 0)
    if alpb:
        c.add("alpb-solvation", "water")
    res = c.singlepoint()
    return float(res.get("energy"))


print(f"{'mol':<10s}{'tblite (Eh)':>16s}{'mlxmol (Eh)':>16s}{'Δ (kcal)':>14s}{'tblite (s)':>12s}{'mlxmol (s)':>12s}")

total_ref = 0.0
total_ours = 0.0
total_t_tblite = 0.0
total_t_ours = 0.0

for name, (atoms, coords) in CONFORMERS.items():
    # tblite reference (GFN2 + ALPB(water))
    e_ref, t_tb = time_call(lambda: tblite_singlepoint(atoms, coords, alpb=True))
    # mlxmolkit (our GFN2 + tblite-backed ALPB correction)
    def mlx_calc():
        r = gfn2_energy(atoms, coords, conv_tol=1e-7)
        return r["energy_hartree"] + alpb_water_correction(atoms, coords)
    e_ours, t_us = time_call(mlx_calc)
    delta = (e_ours - e_ref) * KCAL
    print(f"{name:<10s}{e_ref:>16.6f}{e_ours:>16.6f}{delta:>14.3f}{t_tb:>12.3f}{t_us:>12.3f}")
    total_ref += e_ref
    total_ours += e_ours
    total_t_tblite += t_tb
    total_t_ours += t_us

print()
print(f"{'TOTAL':<10s}{total_ref:>16.6f}{total_ours:>16.6f}{(total_ours - total_ref) * KCAL:>14.3f}{total_t_tblite:>12.3f}{total_t_ours:>12.3f}")
print(f"\nmlxmolkit / tblite cost ratio: {total_t_ours / total_t_tblite:.2f}x")
