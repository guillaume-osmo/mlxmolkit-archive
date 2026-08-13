"""D3 with Becke-Johnson damping, and the halogen-bond (X) correction.

D3(BJ) is the dispersion variant PM6-ML uses (Nováček & Řezáč, JCTC 2025).
It is a different damping *function* from the zero-damping form already in
pm6_d3h4.py, not a reparameterisation: BJ takes its damping radius from the
dispersion coefficients, R0 = sqrt(C8/C6), rather than from the tabulated
r0ab cutoffs, and it goes to a finite constant at R -> 0 instead of zero.

The D3(BJ) energies are checked against Grimme's own simple-dftd3, frozen
into tests/_d3bj_ref_generated.py by tools/gen_d3bj_ref.py.

The X correction is a port of MOPAC src/corrections/disp_DnX.F90 (Apache-2.0).
No independent reference implementation was available to check it numerically,
so the tests below pin the parameter tables against the Fortran source and
assert structural properties, not third-party numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.nddo.pm6_d3h4 import (
    AU_TO_KCAL, D3BJ_WB97M, PM6_D3H4_DISP, X_D3H4X, X_DH2X,
    d3_energy, d3bj_energy, x_energy, pm6_d3h4_correction,
    pm6_d3h4x_correction,
)

from ._d3bj_ref_generated import D3BJ_REF


# ---------------------------------------------------------------------------
# D3(BJ) against simple-dftd3
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(D3BJ_REF))
def test_d3bj_matches_simple_dftd3(name):
    case = D3BJ_REF[name]
    got = d3bj_energy(case["atoms"], np.array(case["coords"]), rthr=60.0)
    got_hartree = got["e_disp"] / AU_TO_KCAL
    assert got_hartree == pytest.approx(case["energy_hartree"], rel=1e-7)


def test_d3bj_parameters_are_the_wb97m_values():
    """PM6-ML uses these unmodified; a typo here is silent and systematic."""
    assert D3BJ_WB97M == dict(s6=1.0, s8=0.3908, a1=0.566, a2=3.128)


def test_d3bj_is_not_the_zero_damping_variant():
    """Guard against anyone 'simplifying' the two paths into one."""
    case = D3BJ_REF["CCO"]
    atoms, coords = case["atoms"], np.array(case["coords"])
    bj = d3bj_energy(atoms, coords)["e_disp"]
    zero = d3_energy(atoms, coords, params=PM6_D3H4_DISP)["e_disp"]
    assert bj != pytest.approx(zero, rel=0.1)
    # Zero-damping runs with s8=0 for PM6-D3H4; BJ carries a real E8 term.
    assert d3bj_energy(atoms, coords)["e8"] != 0.0
    assert d3_energy(atoms, coords, params=PM6_D3H4_DISP)["e8"] == 0.0


def test_d3bj_is_attractive_and_grows_with_size():
    energies = []
    for name in ("O", "CCO", "CCCCCCO", "CCCCCCCCCC=O"):
        case = D3BJ_REF[name]
        energies.append(d3bj_energy(case["atoms"], np.array(case["coords"]))["e_disp"])
    assert all(e < 0 for e in energies), "dispersion must be attractive"
    assert energies == sorted(energies, reverse=True), \
        f"dispersion should deepen with molecule size: {energies}"


def test_d3bj_monomer_is_zero_for_a_single_atom():
    assert d3bj_energy([6], np.zeros((1, 3)))["e_disp"] == 0.0


def test_d3bj_decays_with_separation():
    """Two neon atoms pulled apart: attractive, monotonically weakening."""
    prev = None
    for r in (3.0, 4.0, 6.0, 10.0):
        e = d3bj_energy([10, 10], np.array([[0.0, 0, 0], [r, 0, 0]]),
                        rthr=60.0)["e_disp"]
        assert e < 0.0
        if prev is not None:
            assert e > prev, "must weaken with distance"
        prev = e


# ---------------------------------------------------------------------------
# Halogen-bond (X) correction
# ---------------------------------------------------------------------------

def test_x_parameters_match_the_mopac_source():
    """Pinned against MOPAC disp_DnX.F90 (Brahmkshatriya 2013, Table 2)."""
    assert X_D3H4X == {
        (17, 7): (1.049e12, -9.95), (35, 7): (5.560e4, -3.04),
        (53, 7): (5.237e8, -6.77),
        (17, 8): (1.871e9, -7.44), (35, 8): (2.160e4, -3.30),
        (53, 8): (2.436e6, -4.71),
        (53, 16): (1.051e6, -3.82),
    }


def test_dh2x_parameters_match_the_mopac_source():
    """The older Řezáč & Hobza 2011 set MOPAC uses for PM6-DH2X."""
    assert X_DH2X == {
        (17, 7): (1.0489e12, -9.946), (35, 7): (1.0226e5, -3.236),
        (53, 7): (1.2751e12, -9.534),
        (17, 8): (4.6783e8, -6.867), (35, 8): (9.6021e3, -2.900),
        (53, 8): (6.0912e5, -4.154),
    }


def test_sulfur_pairs_only_with_iodine():
    """MOPAC: `if (k /= 53 .and. nat(j) == 16) cycle`."""
    assert (53, 16) in X_D3H4X
    assert (17, 16) not in X_D3H4X
    assert (35, 16) not in X_D3H4X
    # ...and DH2X has no sulfur entry at all.
    assert not any(acceptor == 16 for _, acceptor in X_DH2X)

    geom = np.array([[0.0, 0, 0], [3.0, 0, 0]])
    assert x_energy([17, 16], geom) == 0.0
    assert x_energy([35, 16], geom) == 0.0
    assert x_energy([53, 16], geom) > 0.0


def test_x_is_zero_without_a_halogen_or_without_an_acceptor():
    case = D3BJ_REF["CCO"]                      # O present, no halogen
    assert x_energy(case["atoms"], np.array(case["coords"])) == 0.0
    case = D3BJ_REF["ClCCl"]                    # halogen present, no acceptor
    assert x_energy(case["atoms"], np.array(case["coords"])) == 0.0


def test_x_is_a_decaying_short_range_term():
    """a > 0, b < 0 — repulsive at contact, negligible past ~4 A."""
    e = [x_energy([17, 7], np.array([[0.0, 0, 0], [r, 0, 0]]))
         for r in (2.5, 3.0, 3.5, 4.0, 5.0)]
    assert all(x > 0 for x in e), "the term is positive as implemented"
    assert e == sorted(e, reverse=True), "must decay monotonically"
    assert e[0] > 10.0, "should be a real wall at 2.5 A"
    assert e[-1] < 1e-4, "should be negligible by 5 A"


def test_x_counts_each_halogen_acceptor_pair_once():
    """Halogens and acceptors are disjoint sets, so no double counting."""
    single = x_energy([17, 7], np.array([[0.0, 0, 0], [3.0, 0, 0]]))
    # Two independent, well-separated Cl...N pairs.
    both = x_energy(
        [17, 7, 17, 7],
        np.array([[0.0, 0, 0], [3.0, 0, 0], [0.0, 60, 0], [3.0, 60, 0]]),
    )
    assert both == pytest.approx(2 * single, rel=1e-9)


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def test_d3h4x_reduces_to_d3h4_without_halogens():
    case = D3BJ_REF["CC(=O)NC"]
    atoms, coords = case["atoms"], np.array(case["coords"])
    base = pm6_d3h4_correction(atoms, coords)
    withx = pm6_d3h4x_correction(atoms, coords)
    assert withx["e_x"] == 0.0
    assert withx["e_total"] == pytest.approx(base["e_total"], rel=1e-12)


def test_d3h4x_adds_the_x_term_when_halogens_are_present():
    case = D3BJ_REF["OCCCl"]
    atoms, coords = case["atoms"], np.array(case["coords"])
    base = pm6_d3h4_correction(atoms, coords)
    withx = pm6_d3h4x_correction(atoms, coords)
    assert withx["e_x"] > 0.0
    assert withx["e_total"] == pytest.approx(base["e_total"] + withx["e_x"],
                                             rel=1e-12)


def test_corrections_stay_density_independent():
    """Every term here is a function of geometry only.

    This is what lets them be applied post-SCF, and it is also why they cannot
    change Mulliken charges or COSMO sigma profiles.
    """
    import inspect
    from mlxmolkit.nddo import pm6_d3h4 as mod
    for fn in (mod.d3bj_energy, mod.x_energy, mod.pm6_d3h4x_correction):
        params = set(inspect.signature(fn).parameters)
        assert "density" not in params and "P" not in params
