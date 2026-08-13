"""Tests for the recovered mlxmolkit.cosmo module.

The module was reconstructed from the surviving `__pycache__/*.cpython-311.pyc`
after the source was lost. Every constant here was read back out of the
bytecode, not refitted — see docs/cosmo_recovery/RECOVERY.md.

Three groups of tests:

1. Constants — pin the recovered values, especially the per-method MF_ALPHA_*
   misfit prefactors, which represent fitting work that cannot be redone from
   anything left on disk.
2. Numerics — quadrature, harmonics, sigma profiles, COSMOspace.
3. Known defects — behaviour that is wrong in the original and was preserved
   verbatim by the recovery. These assert the *broken* behaviour so that a
   future fix trips the test deliberately rather than silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.cosmo import params
from mlxmolkit.cosmo import lebedev
from mlxmolkit.cosmo import spherical_harmonics as sh
from mlxmolkit.cosmo import sigma as sigma_mod
from mlxmolkit.cosmo import cosmors


# ---------------------------------------------------------------------------
# 1. Recovered constants
# ---------------------------------------------------------------------------

def test_mf_alpha_per_method_values():
    """Per-method misfit prefactors — the irreplaceable part of the recovery."""
    assert params.MF_ALPHA_DFT == 7579075.0
    assert params.MF_ALPHA_SIMPLE == 75790750.0
    assert params.MF_ALPHA_DDCOSMO == 100000000.0
    assert params.MF_ALPHA_PM6 == 32000000.0
    assert params.MF_ALPHA_SH4 == 5000000.0
    assert params.MF_ALPHA_SH6 == 1500000.0
    # The active default is the ddCOSMO value.
    assert params.MF_ALPHA == params.MF_ALPHA_DDCOSMO


def test_segment_and_hb_constants():
    assert params.A_EFF == 6.226
    assert params.R_AV == 0.5
    assert params.CAVITY_SCALING == 1.2
    assert params.MF_F_CORR == 2.4
    assert params.MF_R_AV_CORR == 1.0
    assert params.HB_C == 27488747.0
    assert params.HB_C_T == 1.5
    assert params.HB_SIGMA_THRESH == 0.007686
    assert params.COMB_SG_Z_COORD == 10.0
    assert params.COMB_SG_A_STD == 47.999
    assert params.EPSILON_WATER == 78.39
    assert params.R_GAS == 8.314462


def test_sigma_grid_shape():
    g = params.SIGMA_GRID
    assert len(g) == 301
    assert g[0] == pytest.approx(-0.15)
    assert g[-1] == pytest.approx(0.15)
    assert np.diff(g) == pytest.approx(0.001)


def test_vdw_radii_recovered():
    # Spot-check the elements that matter for organics.
    assert params.VDW_RADII[1] == 1.3
    assert params.VDW_RADII[6] == 2.0
    assert params.VDW_RADII[7] == 1.83
    assert params.VDW_RADII[8] == 1.72
    assert params.VDW_RADII[16] == 2.16
    assert len(params.VDW_RADII) == 31


def test_hb_element_sets():
    assert params.HB_DONOR_ELEMENTS == {1}
    assert params.HB_ACCEPTOR_ELEMENTS == {7, 8, 9, 16}


# ---------------------------------------------------------------------------
# 2. Numerics
# ---------------------------------------------------------------------------

def test_lebedev_110_weights_are_the_verified_ones():
    """The five weights that were cross-checked against the raw bytecode."""
    import inspect
    src = inspect.getsource(lebedev.lebedev_110)
    for w in ("0.015313081979748", "0.039174950050656", "0.1851156353447362",
              "0.032846949132764", "0.6904210483822922"):
        assert w in src


def test_lebedev_194_is_unit_vectors_summing_to_4pi():
    xyz, w = lebedev.get_lebedev_grid(194)
    assert xyz.shape == (194, 3)
    assert np.linalg.norm(xyz, axis=1) == pytest.approx(1.0, abs=1e-12)
    assert w.sum() == pytest.approx(4 * np.pi)


def test_get_lebedev_grid_falls_back_to_194():
    xyz, _ = lebedev.get_lebedev_grid(50)
    assert len(xyz) == 194


def test_spherical_harmonics_near_orthonormal_on_194_grid():
    xyz, w = lebedev.get_lebedev_grid(194)
    Y = sh.real_spherical_harmonics(4, xyz)
    assert Y.shape == (25, 194)
    gram = (Y * w) @ Y.T
    # The 194-point grid is a Fibonacci spiral, not a true Lebedev rule, so
    # orthonormality holds only approximately (see test_known_defects below).
    assert np.abs(gram - np.eye(25)).max() < 0.02


def test_harmonic_projection_roundtrip():
    xyz, w = lebedev.get_lebedev_grid(194)
    Y = sh.real_spherical_harmonics(4, xyz)
    # A function that lives exactly in the l<=4 space.
    coeffs = np.zeros(25)
    coeffs[[0, 3, 7]] = [1.0, -0.5, 0.25]
    f = sh.expand_from_harmonics(coeffs, Y)
    back = sh.project_to_harmonics(f, w, Y)
    assert back == pytest.approx(coeffs, abs=0.05)


def test_associated_legendre_matches_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    x = np.linspace(-0.95, 0.95, 11)
    for l in range(4):
        for m in range(l + 1):
            got = sh._associated_legendre(l, m, x)
            want = scipy_special.lpmv(m, l, x)
            assert got == pytest.approx(want, rel=1e-10, abs=1e-12)


def test_associated_legendre_zero_above_l():
    x = np.linspace(-0.9, 0.9, 5)
    assert sh._associated_legendre(2, 3, x) == pytest.approx(0.0)


def test_sigma_profile_conserves_area():
    rng = np.random.default_rng(0)
    n = 400
    seg_area = rng.uniform(0.05, 0.4, n)
    seg_sigma = rng.normal(0.0, 0.01, n)
    grid, profile = sigma_mod.compute_sigma_profile(seg_area, seg_sigma)
    assert len(profile) == len(params.SIGMA_GRID)
    # Linear interpolation splits each segment across two bins but preserves
    # the total.
    assert profile.sum() == pytest.approx(seg_area.sum(), rel=1e-12)


def test_average_sigma_preserves_sign_and_smooths():
    rng = np.random.default_rng(1)
    pos = rng.normal(0, 3.0, (150, 3))
    area = np.full(150, 0.2)
    raw = rng.normal(0.0, 0.02, 150)
    av = sigma_mod.average_sigma(pos, area, raw)
    assert av.shape == raw.shape
    # Averaging is a weighted mean, so it must not amplify the range.
    assert np.abs(av).max() <= np.abs(raw).max() + 1e-12


def test_classify_segments_donor_acceptor():
    atoms = [8, 1, 1, 6]
    seg_atom = np.array([0, 0, 1, 1, 3, 3])
    seg_sigma = np.array([0.02, -0.02, -0.02, 0.02, 0.02, -0.02])
    hb, elem = sigma_mod.classify_segments(atoms, seg_sigma, seg_atom)
    assert list(elem) == [8, 8, 1, 1, 6, 6]
    # O with positive sigma -> acceptor(2); H with negative sigma -> donor(1)
    assert list(hb) == [2, 0, 1, 0, 0, 0]


def test_cosmospace_ideal_mixture_gives_unit_gamma():
    """With no interactions (tau == 1) every Gamma must be 1."""
    n = 12
    X = np.full(n, 1.0 / n)
    tau = np.ones((n, n))
    Gamma, n_iter = cosmors.cosmospace(X, tau)
    assert Gamma == pytest.approx(1.0, abs=1e-6)
    assert n_iter >= 1


def test_cosmospace_converges_on_a_real_tau():
    rng = np.random.default_rng(2)
    n = 25
    X = rng.uniform(0, 1, n)
    X /= X.sum()
    A = rng.normal(0, 800.0, (n, n))
    A = 0.5 * (A + A.T)
    tau = np.exp(-A / (params.R_GAS * 298.15))
    Gamma, n_iter = cosmors.cosmospace(X, tau, max_iter=5000, conv_thresh=1e-10)
    assert n_iter < 5000, "COSMOspace did not converge"
    # Self-consistency of the fixed point: Gamma = 1 / ((X*Gamma) @ tau.T)
    assert Gamma == pytest.approx(1.0 / ((X * Gamma) @ tau.T), rel=1e-6)


def test_interaction_matrices_shapes_and_symmetry():
    sig = np.linspace(-0.02, 0.02, 9)
    hb = np.zeros(9, dtype=np.int32)
    A_mf, A_hb = cosmors._compute_interaction_matrices(sig, hb, 298.15)
    assert A_mf.shape == (9, 9)
    assert A_mf == pytest.approx(A_mf.T)
    # Misfit vanishes for σ + σ' == 0 (the anti-diagonal of a symmetric grid).
    assert np.diag(np.fliplr(A_mf)) == pytest.approx(0.0)
    # No HB types present -> no HB contribution.
    assert A_hb == pytest.approx(0.0)


def test_activity_coefficients_pure_component_reference_is_zero():
    """A one-component 'mixture' at x=1 has ln(gamma) = 0 by definition."""
    grid = params.SIGMA_GRID
    prof = np.exp(-0.5 * (grid / 0.006) ** 2) * 3.0
    mol = {'sigma_grid': grid, 'sigma_profile': prof,
           'total_area': float(prof.sum())}
    lng = cosmors.activity_coefficients([mol], np.array([1.0]), T=298.15)
    assert lng[0] == pytest.approx(0.0, abs=1e-6)


def test_activity_coefficients_identical_components_are_ideal():
    """Two copies of the same molecule must behave as an ideal mixture."""
    grid = params.SIGMA_GRID
    prof = np.exp(-0.5 * (grid / 0.006) ** 2) * 3.0
    mol = {'sigma_grid': grid, 'sigma_profile': prof,
           'total_area': float(prof.sum())}
    lng = cosmors.activity_coefficients([mol, dict(mol)], np.array([0.5, 0.5]))
    assert lng == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Defects preserved from the original source
# ---------------------------------------------------------------------------

def test_lebedev_110_is_a_valid_degree_17_rule():
    """110 distinct points, weights summing to 4π, exact through l=17.

    `_gen_oh(code=5, ...)` used to build the 48-point (a,b,c) class instead of
    Laikov's 24-point (a,b,0) class, so the fifth shell was over-generated:
    134 points whose weights summed to 15.49, integrating a constant 23% high.
    """
    xyz, w = lebedev.lebedev_110()
    assert len(xyz) == 110
    assert len(np.unique(np.round(xyz, 10), axis=0)) == 110, "duplicate points"
    assert np.linalg.norm(xyz, axis=1) == pytest.approx(1.0, abs=1e-12)
    assert w.sum() == pytest.approx(4 * np.pi, rel=1e-10)


def test_lebedev_110_integrates_harmonics_exactly_through_l17():
    xyz, w = lebedev.lebedev_110()
    Y = sh.real_spherical_harmonics(17, xyz)
    integrals = Y @ w
    # l=0 gives sqrt(4*pi); every other harmonic must integrate to zero.
    assert integrals[0] == pytest.approx(np.sqrt(4 * np.pi), rel=1e-10)
    assert np.abs(integrals[1:]).max() < 1e-9


def test_gen_oh_class_sizes():
    sizes = {code: len(lebedev._gen_oh(code, a=a, v=1.0))
             for code, a in [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.185), (5, 0.478)]}
    assert sizes == {1: 6, 2: 12, 3: 8, 4: 24, 5: 24}


def test_known_defect_lebedev_194_is_a_fibonacci_grid():
    """`lebedev_194` is a golden-spiral grid with uniform weights.

    It is a sane quasi-uniform rule (weights do sum to 4π) but it is not the
    degree-23 Lebedev rule its name and docstring claim, so it is not exact
    for l <= 23. Left as recovered: it is the default for every cavity, and
    the fitted MF_ALPHA_* values were fitted against it.
    """
    _, w = lebedev.lebedev_194()
    assert np.allclose(w, w[0]), "weights are uniform -> not a Lebedev rule"
    assert w[0] == pytest.approx(4 * np.pi / 194)


def test_known_defect_the_default_194_grid_is_worse_than_the_110_rule():
    """The default grid integrates harmonics ~10 orders of magnitude worse.

    Now that lebedev_110 is a real degree-17 rule, the 194-point Fibonacci
    default is by far the less accurate of the two despite being denser.
    Switching the default would change every cavity and invalidate the fitted
    MF_ALPHA_* values, so it is left alone and recorded here instead.
    """
    x110, w110 = lebedev.lebedev_110()
    x194, w194 = lebedev.lebedev_194()
    err110 = np.abs(sh.real_spherical_harmonics(17, x110) @ w110)[1:].max()
    err194 = np.abs(sh.real_spherical_harmonics(17, x194) @ w194)[1:].max()
    assert err110 < 1e-9
    assert err194 > 1e-3
    assert err194 > err110 * 1e6
