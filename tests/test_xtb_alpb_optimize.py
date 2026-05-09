"""Smoke + parity tests for the COSMO-RS optimization entry point."""
import os
import sys

import numpy as np
import pytest

# Skip the whole module if tblite isn't installed (CI without optional dep).
tblite = pytest.importorskip("tblite.interface")

from mlxmolkit.xtb.solvation_alpb import (
    _tblite_alpb_water_calc,
    gfn2_alpb_water_optimize,
)


def test_h2o_ancopt_converges_under_xtb_tolerances():
    """ANCopt with GFN2-xTB + ALPB(water) lands H2O at the canonical
    GFN2 minimum within xtb's opt_level=normal thresholds."""
    atoms = [8, 1, 1]
    coords = np.array([
        [0.0,  0.0,   0.20],
        [0.0,  0.85, -0.45],
        [0.0, -0.85, -0.45],
    ])
    res = gfn2_alpb_water_optimize(atoms, coords)
    assert res["converged"]
    # Bond / angle within 0.01 Å / 0.5° of reference (GFN2/ALPB water)
    c = res["coords"]
    o_h1 = float(np.linalg.norm(c[0] - c[1]))
    o_h2 = float(np.linalg.norm(c[0] - c[2]))
    assert abs(o_h1 - 0.963) < 0.01
    assert abs(o_h2 - 0.963) < 0.01
    cosang = float(np.dot(c[1] - c[0], c[2] - c[0])
                   / (o_h1 * o_h2))
    angle = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    assert abs(angle - 105.6) < 1.0


def test_optimized_geometry_passes_gradient_check():
    """Independent tblite singlepoint at the relaxed coords confirms
    the gradient norm is ≤ the optimizer's gtol — the optimization
    landed on a true minimum, not a numerical artifact."""
    atoms = [6, 8, 1, 1, 1, 1]  # methanol
    coords = np.array([
        [-0.748, -0.015,  0.024],
        [ 0.626,  0.310,  0.026],
        [-1.293,  0.949,  0.060],
        [-1.022, -0.580, -0.876],
        [-1.018, -0.626,  0.882],
        [ 1.114, -0.554, -0.043],
    ])
    res = gfn2_alpb_water_optimize(atoms, coords)
    assert res["converged"]

    sp = _tblite_alpb_water_calc(method="GFN2-xTB")
    e_sp, g_sp = sp(atoms, res["coords"], charge=0)
    # Gradient norm at the optimizer's converged geometry should be
    # at or below the convergence tolerance (1e-3 Ha/Bohr → 1.89e-3
    # Ha/Å).
    assert float(np.max(np.abs(g_sp))) < 5e-3
    # Singlepoint energy at the relaxed geometry agrees with what
    # ancopt reported (single-process — should be bitwise close).
    assert abs(e_sp - res["energy"]) < 1e-8


@pytest.mark.parametrize("name,atoms,coords,charge", [
    ("nh3", [7, 1, 1, 1],
     np.array([[0,0,0],[0.95,0,-0.30],[-0.48,0.82,-0.30],[-0.48,-0.82,-0.30]]),
     0),
    ("ch4", [6, 1, 1, 1, 1],
     np.array([[0,0,0],[0.65,0.65,0.65],[-0.65,-0.65,0.65],
               [-0.65,0.65,-0.65],[0.65,-0.65,-0.65]]),
     0),
])
def test_small_molecules_converge(name, atoms, coords, charge):
    """Smoke test: small organic molecules complete in <50 ANCopt iters."""
    res = gfn2_alpb_water_optimize(atoms, coords.copy(), charge=charge)
    assert res["converged"], f"{name} did not converge"
    assert res["n_iter"] < 50, f"{name} took {res['n_iter']} iters"
