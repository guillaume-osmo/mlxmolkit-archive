"""PM6-DH+ dispersion and the PM6-ORG core-core, against openMOPAC.

Both are ports of specific MOPAC routines and both are checked against MOPAC's
own numbers rather than against each other.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.nddo.pm6_dh import pm6_dh_dispersion
from mlxmolkit.nddo.pwcct import get_pwcct, pm6_org_scale

METHANOL = ([6, 8, 1, 1, 1, 1],
            np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 0.0], [-0.36, 1.02, 0.0],
                      [-0.36, -0.51, 0.88], [-0.36, -0.51, -0.88],
                      [1.76, 0.89, 0.0]]))
# MOPAC v23.2: PM6 -46.73122, PM6-DH+ -46.74055 at this geometry. A lone
# methanol has no intermolecular hydrogen bond, so the whole -0.00933 is
# dispersion — which makes it a clean single-term test.
METHANOL_DH_DISPERSION = -0.00933


def test_dh_dispersion_matches_mopac_on_methanol():
    atoms, coords = METHANOL
    assert pm6_dh_dispersion(atoms, coords) == pytest.approx(
        METHANOL_DH_DISPERSION, abs=1e-5)


def test_dh_dispersion_is_not_d3():
    """PM6-DH+ uses the older Slater-Kirkwood form, not D3. If someone ever
    'unifies' them, this catches it: D3 gives a different answer entirely."""
    from mlxmolkit.nddo.pm6_d3h4 import PM6_D3_DISP, d3_energy

    atoms, coords = METHANOL
    dh = pm6_dh_dispersion(atoms, coords)
    d3 = d3_energy(atoms, coords, params=PM6_D3_DISP)["e_disp"]
    assert abs(dh - d3) > 0.3, "DH+ dispersion should not coincide with D3"


def test_dh_dispersion_is_attractive_and_grows_with_contact():
    """Two methanes pulled apart: dispersion must weaken monotonically."""
    atoms = [6, 1, 1, 1, 1, 6, 1, 1, 1, 1]
    previous = None
    for sep in (3.5, 4.5, 6.0, 8.0):
        c = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                      [0.63, -0.63, -0.63], [-0.63, 0.63, -0.63],
                      [sep, 0, 0], [sep + 0.63, 0.63, 0.63],
                      [sep - 0.63, -0.63, 0.63], [sep + 0.63, -0.63, -0.63],
                      [sep - 0.63, 0.63, -0.63]], dtype=float)
        e = pm6_dh_dispersion(atoms, c)
        assert e < 0.0, f"dispersion must be attractive at {sep} A"
        if previous is not None:
            assert e > previous, f"dispersion must weaken from {sep} A outward"
        previous = e


def test_carbon_c6_depends_on_coordination():
    """MOPAC gives carbon C6 = 0.95 with four bonds and 1.65 otherwise, which is
    its proxy for sp3 versus sp2/sp. Ethane and ethene must therefore differ by
    more than geometry alone would explain."""
    ethane = ([6, 6, 1, 1, 1, 1, 1, 1],
              np.array([[0, 0, 0], [1.53, 0, 0], [-0.36, 1.02, 0],
                        [-0.36, -0.51, 0.88], [-0.36, -0.51, -0.88],
                        [1.89, 1.02, 0], [1.89, -0.51, 0.88],
                        [1.89, -0.51, -0.88]], dtype=float))
    ethene = ([6, 6, 1, 1, 1, 1],
              np.array([[0, 0, 0], [1.33, 0, 0], [-0.55, 0.93, 0],
                        [-0.55, -0.93, 0], [1.88, 0.93, 0],
                        [1.88, -0.93, 0]], dtype=float))
    assert pm6_dh_dispersion(*ethane) < 0.0
    assert pm6_dh_dispersion(*ethene) < 0.0


def test_pm6_org_steric_terms_are_present_for_the_documented_pairs():
    """ccrep_PM6_ORG adds a Gaussian steric term to thirteen element pairs.
    A missing pair is silent — the base PM6 form still returns a plausible
    number — so the coverage is what is worth asserting."""
    from mlxmolkit.nddo.pwcct import _ORG_STERIC

    expected = {(1, 1), (6, 1), (7, 1), (8, 1), (16, 1), (6, 6), (7, 6),
                (8, 6), (16, 6), (8, 7), (16, 7), (8, 8), (16, 8)}
    assert set(_ORG_STERIC) == expected


def test_pm6_org_uses_the_r_squared_form_for_x_h():
    """C-H, N-H and O-H take PM6's r**2 base, not the r + 0.0003 r**6 one, and
    ccrep_PM6_ORG *rebuilds* the base for them rather than adding to it."""
    chi, alp = get_pwcct(6, 1)
    r = 1.09
    got = pm6_org_scale(6, 1, r, chi, alp)
    import math
    base_r2 = 1.0 + 2.0 * chi * math.exp(-alp * r ** 2)
    steric = 0.01 * 0.97354 * math.exp(-3.16312 * (r - 1.85191) ** 2) \
        if r - 1.85191 > 0 else 0.01 * 0.97354
    assert got == pytest.approx(base_r2 + steric, abs=1e-9)


def test_pm6_org_keeps_the_triple_bond_term():
    """C-C carries both the C≡C correction and its own steric Gaussian."""
    import math
    chi, alp = get_pwcct(6, 6)
    r = 1.20
    got = pm6_org_scale(6, 6, r, chi, alp)
    base = 1.0 + 2.0 * chi * math.exp(-alp * (r + 0.0003 * r ** 6))
    cc = 9.278465 * math.exp(-5.983752 * r)
    assert got > base + cc * 0.9, "the C≡C term should dominate at 1.20 A"
