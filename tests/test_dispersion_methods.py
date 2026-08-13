"""PM6-D3, PM6-D3H4 and PM6-D3H4X against openMOPAC.

These are plain PM6 electronically — same parameters, same SCF, same density and
charges — and differ only by a post-SCF function of geometry added to the heat
of formation. So the whole test is: does that function reproduce MOPAC.

The references are MOPAC v23.2 at frozen RDKit/MMFF geometries.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from mlxmolkit.nddo.pm6_d3h4 import PM6_D3_DISP, d3_energy, pm6_d3h4_correction
from mlxmolkit.nddo.pipeline import _smiles_to_3d
from mlxmolkit.nddo.scf import nddo_energy

# MOPAC v23.2, 1SCF, at the methanol geometry below.
METHANOL = ([6, 8, 1, 1, 1, 1],
            np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 0.0], [-0.36, 1.02, 0.0],
                      [-0.36, -0.51, 0.88], [-0.36, -0.51, -0.88],
                      [1.76, 0.89, 0.0]]))
MOPAC_METHANOL = {"PM6": -46.73122, "PM6-D3": -47.09141,
                  "PM6-D3H4": -40.46396, "PM6-D3H4X": -40.46396}


def test_the_r0ab_table_is_in_bohr():
    """The bug this file exists for.

    Grimme's r0ab is tabulated in Angstrom — r0ab(C,C) = 2.9103 A — and `edisp`
    compares it against Bohr distances. Read as Bohr it looks like a plausible
    C-C bond length, which is why it survived inspection, but it leaves the
    r^-8 damping far too weak at bonded separations and put methanol's
    dispersion at -44.8 kcal/mol instead of -0.360.
    """
    from mlxmolkit.nddo.pm6_d3h4 import BOHR, _load_r0ab

    r0ab = _load_r0ab()
    assert r0ab[5, 5] == pytest.approx(2.9103 / BOHR, abs=1e-4), (
        "r0ab(C,C) should be Grimme's 2.9103 A expressed in Bohr (5.4997 a0)")
    # A cutoff radius must exceed the bond length it screens.
    assert r0ab[5, 5] * BOHR > 2.0, "r0ab(C,C) is smaller than a C-C bond"


def test_pm6_d3_dispersion_matches_mopac_exactly():
    """MOPAC's PM6-D3 is zero-damping with hard-wired parameters, not
    Becke-Johnson. Its dispersion on methanol is -0.36019 kcal/mol."""
    atoms, coords = METHANOL
    e = d3_energy(atoms, coords, params=PM6_D3_DISP)["e_disp"]
    assert e == pytest.approx(-0.36019, abs=1e-4)


def test_pm6_d3h4_correction_matches_mopac_exactly():
    """MOPAC's D3H4 correction on methanol is +6.26726 kcal/mol — net
    *repulsive*, because the H-H term outweighs a small dispersion."""
    atoms, coords = METHANOL
    c = pm6_d3h4_correction(atoms, coords)
    assert c["e_total"] == pytest.approx(6.26726, abs=1e-4)
    assert c["e_disp"] == pytest.approx(-0.0642, abs=1e-3)
    assert c["e_hh"] > 0.0, "the H-H term is a repulsion"


@pytest.mark.parametrize("method", sorted(MOPAC_METHANOL))
def test_methanol_heat_of_formation_matches_mopac(method):
    """All four agree to the PM6 baseline offset, so the corrections are exact."""
    atoms, coords = METHANOL
    got = nddo_energy(atoms, coords, method=method,
                      max_iter=400, conv_tol=1e-9)["heat_of_formation_kcal"]
    assert got == pytest.approx(MOPAC_METHANOL[method], abs=0.1)


@pytest.mark.parametrize("smiles,reference", [
    ("O", -49.2547), ("CC(C)(C)C", -13.2926), ("Clc1ccccc1", 12.9373),
    ("Brc1ccccc1", 24.9171), ("c1ccccc1", 23.6808), ("CCO", -48.4505),
    ("CC(=O)N", -45.7292), ("Ic1ccccc1", 39.4126),
])
def test_pm6_d3h4x_across_molecules(smiles, reference):
    """Chosen to exercise each term separately: neopentane for H-H repulsion,
    water and acetamide for H4, the halobenzenes for the X term."""
    result = _smiles_to_3d(smiles, seed=1)
    if result is None:
        pytest.skip(f"could not embed {smiles}")
    got = nddo_energy(result[0], result[1], method="PM6-D3H4X",
                      max_iter=500, conv_tol=1e-9)["heat_of_formation_kcal"]
    assert got == pytest.approx(reference, abs=0.15)


def test_the_variants_share_pm6s_density():
    """They differ only after the SCF, so charges must be bit-identical."""
    atoms, coords = METHANOL
    base = nddo_energy(atoms, coords, method="PM6", max_iter=400, conv_tol=1e-9)
    for method in ("PM6-D3", "PM6-D3H4", "PM6-D3H4X"):
        other = nddo_energy(atoms, coords, method=method,
                            max_iter=400, conv_tol=1e-9)
        assert np.array_equal(np.asarray(base["charges"]),
                              np.asarray(other["charges"])), method
        assert other["energy_eV"] == base["energy_eV"], method
