"""A hyphenated method name must select the same physics as an underscored one.

`get_params` normalises internally, so `nddo_energy(method="PM6-D")` picks up
PM6's parameters. But the *raw* string was what reached the core-core selector,
so "PM6-D" missed `PM6_CORE_CORE_METHODS`, silently fell through to the
AM1-style core-core, and produced an energy 185 kcal/mol from the underscored
spelling of the same method.

Silently: no exception, converged SCF, plausible-looking charges.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.nddo.scf import nddo_energy

# methanol, a fixed geometry — the point is the two spellings, not the molecule
ATOMS = [6, 8, 1, 1, 1, 1]
COORDS = np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 0.0], [-0.36, 1.02, 0.0],
                   [-0.36, -0.51, 0.88], [-0.36, -0.51, -0.88], [1.76, 0.89, 0.0]])


@pytest.mark.parametrize("hyphen,underscore", [("PM6-D", "PM6_D")])
def test_hyphen_and_underscore_spellings_agree(hyphen, underscore):
    a = nddo_energy(ATOMS, COORDS, method=hyphen, max_iter=400, conv_tol=1e-9)
    b = nddo_energy(ATOMS, COORDS, method=underscore, max_iter=400, conv_tol=1e-9)
    assert a["energy_eV"] == pytest.approx(b["energy_eV"], abs=1e-9)
    assert a["heat_of_formation_kcal"] == pytest.approx(
        b["heat_of_formation_kcal"], abs=1e-9)


def test_normalize_method_matches_what_get_params_accepts():
    """The helper must agree with get_params' own normalisation, or the two
    will disagree about which method is being run."""
    from mlxmolkit.nddo.methods import get_params
    from mlxmolkit.nddo.pwcct import normalize_method

    for spelling in ("PM6", "pm6", "PM6-D", "pm6_d", "AM1*", "am1-star"):
        canonical = normalize_method(spelling)
        get_params(spelling)          # must not raise
        assert get_params(spelling) is get_params(canonical)


def test_a_pm6_variant_gets_pm6s_core_core_not_am1s():
    """The concrete symptom: PM6-D must reproduce PM6, since PM6_D is an alias.

    If the core-core selector ever compares raw strings again, this diverges by
    ~185 kcal/mol rather than failing subtly.
    """
    pm6 = nddo_energy(ATOMS, COORDS, method="PM6", max_iter=400, conv_tol=1e-9)
    hyphen = nddo_energy(ATOMS, COORDS, method="PM6-D", max_iter=400, conv_tol=1e-9)
    assert hyphen["heat_of_formation_kcal"] == pytest.approx(
        pm6["heat_of_formation_kcal"], abs=1e-6)
