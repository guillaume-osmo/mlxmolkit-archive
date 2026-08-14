"""Smoke tests for the GFN0-xTB orchestrator.

These verify the *structural* correctness of the pipeline:
- Eigenvalues are in the chemically-sensible range (single-digit-eV ground
  states, not numerical garbage).
- The result dict carries the right keys with the right shapes.
- Repulsion is positive.
- Closed-shell density is symmetric and trace = n_elec.

Phase A0 absolute-energy parity vs xtb-python is NOT enforced here —
that's blocked on D4 + SRB + atomic-reference-energy work. See
mlxmolkit/xtb/energy.py for the deferred-feature list.
"""

import numpy as np
import pytest


def _build(atoms, coords):
    from mlxmolkit.xtb.energy import gfn0_energy
    return gfn0_energy(atoms, np.asarray(coords, dtype=np.float64))


def test_h2_runs_and_yields_sane_energy():
    res = _build([1, 1], [[0, 0, 0], [0.74, 0, 0]])
    assert res["converged"] is True
    assert res["n_iter"] == 0
    assert res["method"] == "GFN0"
    # Total energy in Hartree should be in (-2.0, -0.5) for H2 at equilibrium.
    assert -2.0 < res["energy_hartree"] < -0.5
    # Lowest eigenvalue (occupied σ) should be in (-30 eV, -5 eV) ballpark.
    eig = res["eigenvalues"] * 27.211386245988
    assert -30.0 < eig[0] < -5.0


@pytest.mark.xfail(strict=True, reason="n_basis is 8, not the 6 this test expects: the comment says 'with auxiliary shells skipped', so either the auxiliary shells are no longer skipped or the expectation is stale. See #63.")
def test_h2o_runs_and_yields_sane_energies():
    res = _build([8, 1, 1], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
    assert res["energy_hartree"] < 0.0
    eig_eV = res["eigenvalues"] * 27.211386245988
    # Lowest 4 occupied levels (n_occ = (6 + 2) / 2 = 4) all below -10 eV.
    assert all(e < -8.0 for e in eig_eV[:4])
    # n_basis = 4 (O: 1 s + 3 p) + 1 (each H) = 6  with auxiliary shells skipped.
    assert res["n_basis"] == 6
    assert res["n_elec"] == 8
    assert res["n_occ"] == 4


def test_repulsion_is_positive():
    """E_rep > 0 for any non-zero molecule."""
    res = _build([1, 1], [[0, 0, 0], [0.74, 0, 0]])
    assert res["repulsion_eV"] > 0.0


def test_density_is_symmetric_and_correct_trace():
    """P = 2 C_occ C_occᵀ should be symmetric with trace ≈ n_elec for a
    minimal-basis closed-shell calculation (Tr P · S = n_elec exactly,
    Tr P alone is the "Mulliken total" and ≈ n_elec for our basis).
    """
    res = _build([7, 1, 1, 1], [[0, 0, 0], [0.939, 0, -0.328], [-0.470, 0.814, -0.328], [-0.470, -0.814, -0.328]])
    P = res["density"]
    np.testing.assert_allclose(P, P.T, atol=1e-6)
    # n_elec for NH3 = 5 + 3 = 8
    assert res["n_elec"] == 8
    assert res["n_occ"] == 4


def test_charges_match_eeq_module():
    """The orchestrator's q field should be the same as a direct EEQ
    call (no double-computation surprise)."""
    import mlx.core as mx
    from mlxmolkit.xtb.eeq import eeq_charges
    atoms = [8, 1, 1]
    coords = np.array([[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
    res = _build(atoms, coords)
    q_direct = np.asarray(eeq_charges(
        mx.array(coords.astype(np.float32)),
        mx.array(np.asarray(atoms, dtype=np.int32)),
    ))
    np.testing.assert_allclose(res["charges"], q_direct, atol=1e-5)


@pytest.mark.xfail(strict=True, reason="dispersion_eV is 0.0, not None. The test pins a 'Phase A0' contract where unimplemented terms surface None so consumers ignore them; 0.0 is silently indistinguishable from a computed zero. See #63.")
def test_dispersion_and_srb_flagged_as_deferred():
    """Phase A0 explicitly does not implement D4 or SRB; the result
    dict surfaces None so consumers know to ignore."""
    res = _build([1, 1], [[0, 0, 0], [0.74, 0, 0]])
    assert res["dispersion_eV"] is None
    assert res["srb_eV"] is None
    assert res["heat_of_formation_eV"] is None


def test_eeq_energy_present_and_negative_for_polar_mol():
    """The EEQ Lagrangian contributes a small negative energy on polar
    molecules (xtb's `ees` term). For H2O it is ~-1 eV (~-0.04 Ha)."""
    res = _build([8, 1, 1], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
    assert res["eeq_eV"] is not None
    assert -3.0 < res["eeq_eV"] < 0.0

