"""The misfit prefactor must follow the source of the surface charges.

alpha' absorbs the systematic scale of whatever produced the charges, so it is
not a universal constant — the calibrated values in `params` span a factor of
67, from sh6 at 1.5e6 to ddcosmo at 1.0e8.

All of them were present and none was ever selected: `_compute_interaction_
matrices` read the bare `MF_ALPHA`, which is ddCOSMO's, so COSMO-RS on PM6
charges ran with a misfit prefactor 3.12x too large. `MF_ALPHA_PM6` and
`MF_ALPHA_DFT` were referenced only by tests asserting their values, never by
anything that computed with them — which is the shape of a constant that looks
maintained and does nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.cosmo import params as P
from mlxmolkit.cosmo.cosmors import (
    CHARGE_SOURCES,
    _compute_interaction_matrices,
    misfit_alpha,
)


def test_every_calibrated_alpha_is_reachable():
    """Each MF_ALPHA_* in params must be selectable by name.

    A new calibration added to params and not added here is one nobody can
    use, which is the state this test exists to end.
    """
    declared = {name for name in dir(P)
                if name.startswith("MF_ALPHA_")}
    reachable = set(CHARGE_SOURCES.values())
    assert declared == reachable, (
        f"calibrations with no way to select them: {sorted(declared - reachable)}; "
        f"names pointing at nothing: {sorted(reachable - declared)}"
    )


def test_the_default_is_unchanged():
    """Existing callers and every recorded benchmark must be reproducible."""
    assert misfit_alpha() == P.MF_ALPHA
    assert misfit_alpha(None) == P.MF_ALPHA


@pytest.mark.parametrize("source,expected", sorted(
    (k, getattr(P, v)) for k, v in CHARGE_SOURCES.items()))
def test_each_source_resolves_to_its_own_constant(source, expected):
    assert misfit_alpha(source) == expected


def test_an_unknown_source_is_rejected_rather_than_defaulted():
    """Silently falling back would reintroduce the original bug."""
    with pytest.raises(ValueError, match="unknown charge_source"):
        misfit_alpha("B3LYP")


def test_pm6_and_ddcosmo_give_measurably_different_misfit():
    """The point of the fix: the choice has to change the numbers.

    PM6 against the default is a factor of 3.12 in alpha', and the misfit
    matrix is linear in it.
    """
    sigma = np.linspace(-0.03, 0.03, 51)
    hb = np.zeros(51, dtype=np.int32)

    A_default, _ = _compute_interaction_matrices(sigma, hb, 298.15)
    A_pm6, _ = _compute_interaction_matrices(sigma, hb, 298.15, charge_source="PM6")
    A_dft, _ = _compute_interaction_matrices(sigma, hb, 298.15, charge_source="DFT")

    ratio = P.MF_ALPHA / P.MF_ALPHA_PM6
    assert ratio == pytest.approx(3.125, rel=1e-3)

    nonzero = np.abs(A_default) > 0
    assert np.allclose(A_default[nonzero] / A_pm6[nonzero], ratio)
    assert np.max(np.abs(A_pm6)) < np.max(np.abs(A_default))
    assert np.max(np.abs(A_dft)) < np.max(np.abs(A_pm6))


def test_the_hydrogen_bond_matrix_ignores_the_charge_source():
    """alpha' scales misfit only; HB has its own constant and must not move."""
    sigma = np.linspace(-0.03, 0.03, 51)
    hb = np.zeros(51, dtype=np.int32)
    hb[sigma < -0.01] = 1          # donors
    hb[sigma > 0.01] = 2           # acceptors

    _, hb_default = _compute_interaction_matrices(sigma, hb, 298.15)
    _, hb_pm6 = _compute_interaction_matrices(sigma, hb, 298.15, charge_source="PM6")

    assert np.max(np.abs(hb_default)) > 0, "the HB term produced nothing to compare"
    assert np.array_equal(hb_default, hb_pm6)
