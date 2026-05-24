"""End-to-end API smoke tests for the supported semi-empirical methods.

Locks in that `nddo_energy(method=...)` runs without crash and produces
reasonable heats of formation for the 7 registered methods.

Reference values are frozen from the current implementation — these are
regression tests, not literature comparisons. The literature comparison
for PM6_D is in tests/test_pm6_d_native.py (charges vs PYSEQM/MOPAC).
"""
import numpy as np
import pytest

from mlxmolkit.rm1 import nddo_energy, METHOD_PARAMS, pm6_d3h4_correction


H2O_ATOMS = [8, 1, 1]
H2O_COORDS = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]


@pytest.mark.parametrize("method,expected_hof_kcal,tol", [
    # CHNO-coverage methods on H2O — frozen references from this build.
    ("RM1",      -57.81, 0.5),
    ("AM1",      -59.22, 0.5),
    ("PM3",      -53.19, 0.5),
    ("PM6",      -54.19, 0.5),
    ("PM6_SP",   -54.19, 0.5),
    ("AM1_STAR", -53.71, 0.5),
    ("RM1_STAR", -54.47, 0.5),
])
def test_h2o_hof_each_method(method, expected_hof_kcal, tol):
    """nddo_energy(method) on H2O must converge and give the frozen HoF."""
    r = nddo_energy(H2O_ATOMS, H2O_COORDS, method=method)
    assert r["converged"], f"{method}: SCF did not converge"
    diff = abs(r["heat_of_formation_kcal"] - expected_hof_kcal)
    assert diff < tol, (
        f"{method}: HoF = {r['heat_of_formation_kcal']:.2f} kcal/mol, "
        f"expected {expected_hof_kcal} +/- {tol}"
    )


@pytest.mark.parametrize("method", ["RM1", "AM1", "PM3", "PM6", "PM6_SP",
                                     "AM1_STAR", "RM1_STAR", "PM6_D"])
def test_method_returns_expected_keys(method):
    """All methods must return the same dict shape."""
    if method == "AM1" and 16 in METHOD_PARAMS.get("AM1", {}):
        pytest.skip("AM1 lacks S parameters")
    r = nddo_energy(H2O_ATOMS, H2O_COORDS, method=method)
    for k in ("energy_eV", "energy_kcal", "electronic_eV", "nuclear_eV",
              "heat_of_formation_eV", "heat_of_formation_kcal", "converged"):
        assert k in r, f"{method}: missing key {k!r} in result"


def test_pm6_d_runs_on_d_orbital_molecules():
    """PM6_D must run on molecules with d-orbital atoms (S, Cl, Br) — was
    previously delegating to PYSEQM and crashing if PYSEQM was missing."""
    for label, atoms, coords in [
        ("H2S", [16, 1, 1], [[0, 0, 0], [0.97, 0, 0.93], [-0.97, 0, 0.93]]),
        ("HCl", [17, 1], [[0, 0, 0], [1.275, 0, 0]]),
        ("CH3Cl", [6, 17, 1, 1, 1],
         [[0, 0, 0], [1.785, 0, 0], [-0.366, 0.515, 0.892],
          [-0.366, 0.515, -0.892], [-0.366, -1.029, 0]]),
    ]:
        r = nddo_energy(atoms, coords, method="PM6_D")
        assert r["converged"], f"PM6_D on {label}: SCF did not converge"
        assert np.isfinite(r["heat_of_formation_kcal"]), (
            f"PM6_D on {label}: non-finite HoF"
        )


def test_pm6_d3h4_correction_dict_shape():
    """pm6_d3h4_correction returns a dict with the four expected keys."""
    corr = pm6_d3h4_correction(H2O_ATOMS, np.asarray(H2O_COORDS, dtype=np.float64))
    assert isinstance(corr, dict)
    for k in ("e_disp", "e_hb", "e_hh", "e_total"):
        assert k in corr
        assert np.isfinite(corr[k])
    # e_total must be the sum of components
    assert abs(corr["e_total"] - (corr["e_disp"] + corr["e_hb"] + corr["e_hh"])) < 1e-9


def test_pm6_d3h4_dimer_has_hbond_contribution():
    """A water dimer should have a non-zero (negative) H4 contribution."""
    atoms = [8, 1, 1, 8, 1, 1]
    coords = np.asarray([
        [0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0],
        [3.0, 0, 0], [3.96, 0, 0], [2.76, 0.93, 0],
    ], dtype=np.float64)
    corr = pm6_d3h4_correction(atoms, coords)
    assert corr["e_hb"] < 0.0, f"H-bond correction should be negative, got {corr['e_hb']}"
