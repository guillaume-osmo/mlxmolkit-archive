"""eisol and eheat must match openMOPAC, for every element and every method.

These two constants never touch the SCF, the density or the charges — they enter
only the heat-of-formation assembly:

    HoF = (E_total - sum eisol) * 23.06 + sum eheat

So a wrong value is invisible to almost every other test: energies agree, charges
agree, nothing raises, and only the heat of formation is wrong. Two real bugs hid
there — F and P had wrong eheat, and thirty elements carried a placeholder 0.0
for both, which put a single silicon 108 kcal/mol out.

A table comparison is the only thing that catches this class of error, so this
file is that table comparison.
"""
from __future__ import annotations

import pytest

from mlxmolkit.nddo.atomic_heats import EHEAT_MOPAC
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.params import _compute_eisol

METHODS = ["RM1", "AM1", "PM3", "PM6", "PM6_D", "AM1_STAR", "RM1_STAR"]


@pytest.mark.parametrize("method", METHODS)
def test_eheat_matches_mopacs_table(method):
    """Transcribed from MOPAC src/models/parameters_C.F90 `data eheat(Z)`."""
    params = get_params(method)
    wrong = {z: (p.eheat, EHEAT_MOPAC[z]) for z, p in params.items()
             if z in EHEAT_MOPAC and abs(p.eheat - EHEAT_MOPAC[z]) > 1e-6}
    assert not wrong, f"{method}: eheat differs from MOPAC for {wrong}"


@pytest.mark.parametrize("method", METHODS)
def test_eisol_matches_the_calpar_formula(method):
    """MOPAC computes eisol rather than tabulating it, so this recomputes it.

    src/models/calpar.F90:

        eisol = uss*ios + upp*iop + udd*iod
              + gss*gssc + gpp*gppc + gsp*gspc + gp2*gp2c + hsp*hspc
    """
    params = get_params(method)
    wrong = {z: (p.eisol, _compute_eisol(p)) for z, p in params.items()
             if abs(p.eisol - _compute_eisol(p)) > 1e-6}
    assert not wrong, f"{method}: eisol differs from calpar for {wrong}"


@pytest.mark.parametrize("method", METHODS)
def test_no_element_carries_a_placeholder_zero(method):
    """A zero here is silently wrong rather than loudly missing.

    Every element the toolkit exposes should have both constants, so that a
    molecule containing it gets a heat of formation rather than a plausible
    looking number that is hundreds of kcal/mol out.
    """
    params = get_params(method)
    NOBLE = {2, 10, 18, 36, 54, 86}          # eheat is genuinely 0 for these
    missing = [z for z, p in params.items()
               if (p.eisol == 0.0 or p.eheat == 0.0) and z not in NOBLE]
    assert not missing, f"{method}: placeholder constants for Z={missing}"


def test_the_ten_hand_calibrated_elements_are_unchanged():
    """The derivation replaced a hand-written coefficient table; these ten are
    the values it had, and it must reproduce every one of them exactly."""
    params = get_params("PM6")
    expected = {1: -11.24696, 6: -115.20159, 7: -174.95141, 8: -287.12722,
                9: -468.15754, 15: -139.66795, 16: -171.74302,
                17: -252.90936, 35: -226.96098, 53: -248.12865}
    for z, value in expected.items():
        assert params[z].eisol == pytest.approx(value, abs=1e-5), f"Z={z}"
