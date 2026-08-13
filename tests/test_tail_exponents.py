"""Every d-bearing PM6 element needs its one-centre Slater exponents.

PM6 gives a d-bearing atom two sets of exponents: `zeta_s/p/d` for the basis,
and a separate set (MOPAC's `zsn`/`zpn`/`zdn`) for the one-centre two-electron
integrals. They are not close — silicon's are 8.388 against a basis 1.753.

A missing entry is silent: the code falls back to the basis exponents, the SCF
converges, the charges look fine, and only the heat of formation is wrong — by
528 kcal/mol for one silicon. So the table's *completeness* is the thing worth
testing, not any particular value.
"""
from __future__ import annotations

import pytest

from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.tetci_multipole_pyseqm import PM6_TAIL_EXPONENTS


def test_every_d_bearing_pm6_element_has_tail_exponents():
    """The invariant. Break it and heats of formation go silently wrong."""
    params = get_params("PM6")
    d_elements = [z for z, p in params.items() if p.n_basis == 9]
    assert d_elements, "no d-bearing elements found — the check would be vacuous"
    missing = sorted(z for z in d_elements if z not in PM6_TAIL_EXPONENTS)
    assert not missing, (
        f"Z={missing} carry d orbitals but have no one-centre exponents; they "
        f"will silently fall back to zeta_s/p/d and their heats of formation "
        f"will be wrong by hundreds of kcal/mol"
    )


def test_the_five_hand_listed_elements_are_unchanged():
    """P, S, Cl, Br and I were hardcoded and correct; loading from the CSV must
    reproduce them exactly, or the fix traded three broken elements for five."""
    hand_written = {
        15: (6.04271, 2.37647, 7.14775),
        16: (0.479722, 1.015507, 4.31747),
        17: (0.9563, 2.46407, 6.41033),
        35: (3.09478, 3.06576, 2.82),
        53: (9.13524, 6.88819, 3.79152),
    }
    for z, expected in hand_written.items():
        assert PM6_TAIL_EXPONENTS[z] == pytest.approx(expected, abs=1e-9), f"Z={z}"


@pytest.mark.parametrize("z,expected", [
    (13, (4.74234, 4.66963, 7.13114)),    # Al — was falling back, -107 kcal/mol
    (14, (8.38811, 1.84305, 0.70860)),    # Si — was falling back, -528 kcal/mol
    (33, (2.00654, 3.31683, 4.65344)),    # As — was falling back, -16 kcal/mol
])
def test_the_previously_missing_elements_now_have_mopacs_values(z, expected):
    """Checked against openMOPAC's zsn6/zpn6/zdn6 in parameters_for_PM6_C.F90."""
    assert PM6_TAIL_EXPONENTS[z] == pytest.approx(expected, abs=1e-5)


def test_no_d_bearing_element_has_a_zero_exponent():
    """A zero exponent on a real d element is a parse failure, not a physical
    value, and would blow up the AIJL scaling rather than merely bias it.

    Scoped to elements PM6 actually parameterises: the CSV also carries MOPAC's
    capped-bond pseudo-atoms at Z=99 and Z=100, which legitimately have no d
    shell and are never looked up.
    """
    params = get_params("PM6")
    zero = {z: v for z, v in PM6_TAIL_EXPONENTS.items()
            if z in params and params[z].n_basis == 9 and any(x == 0.0 for x in v)}
    assert not zero, f"zero exponents for {zero}"
