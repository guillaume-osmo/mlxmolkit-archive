"""The C≡C heat-of-formation defect, pinned. See issue #33.

Every molecule with a carbon-carbon triple bond is ~11.8 kcal/mol below MOPAC's
PM6 heat of formation, by an amount that does not grow with the molecule.
Everything without one agrees to ~0.1-0.5 kcal/mol.

The SCF itself is *correct*: on the C7H12 case the orbitals match MOPAC to four
decimals (HOMO -10.0750 vs -10.075, LUMO 2.0522 vs 2.052), the Mulliken charges
agree to 7e-5 e, and both codes report 20 filled levels. So the whole defect is
in the total-energy assembly at very short C-C range, downstream of a density
that is right.

These tests are xfail rather than deleted, so the day the core-core term is
fixed they turn XPASS and say so, and until then nothing silently drifts.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from mlxmolkit.nddo.pipeline import _smiles_to_3d
from mlxmolkit.nddo.scf import nddo_energy

# MOPAC v23.2, PM6 1SCF, RDKit ETKDG randomSeed=1 + MMFF94 geometry.
MOPAC_HF = {
    "C#C": 57.381,           # acetylene
    "C#CC": 46.218,          # propyne
    "CC#CC": 36.094,         # 2-butyne
    "CC#CCC": 31.890,        # 2-pentyne
    "CCC#CCC": 27.717,       # 3-hexyne
    "CCC#CC(C)C": 21.913,    # the molecule that surfaced this, in #18
}
# Same provenance, no triple bond — these must keep passing.
MOPAC_HF_OK = {
    "CC=CC": -3.139,         # 2-butene, C=C at 1.342 A
    "CCCC": -25.577,         # butane
    "C1CC1": 12.300,         # cyclopropane, strained C-C
    "CC(C)(C)C": -34.913,    # neopentane
}


def hof(smiles):
    result = _smiles_to_3d(smiles, seed=1)
    if result is None:
        pytest.skip(f"could not embed {smiles}")
    atoms, coords = result[0], result[1]
    return nddo_energy(atoms, coords, method="PM6",
                       max_iter=400, conv_tol=1e-10)["heat_of_formation_kcal"]


@pytest.mark.parametrize("smiles,reference", sorted(MOPAC_HF_OK.items()))
def test_molecules_without_a_triple_bond_match_mopac(smiles, reference):
    """The control. If these ever break, the defect is not specific to C≡C."""
    assert hof(smiles) == pytest.approx(reference, abs=0.5)


@pytest.mark.parametrize("smiles,reference", sorted(MOPAC_HF.items()))
def test_triple_bonded_molecules_match_mopac(smiles, reference):
    assert hof(smiles) == pytest.approx(reference, abs=0.5)


def test_the_correction_is_a_switch_not_a_decaying_function():
    """The property that identified the bug, kept as the property that pins it.

    MOPAC counts a C-C bond as acetylenic below 1.21 A, tapers to zero at
    1.33 A through a quintic-sextic switch, and multiplies the count by 12
    kcal/mol. It is therefore essentially a step: full weight at an alkyne's
    1.20 A, none at an alkene's 1.34 A. No smooth function of R can imitate
    that, which is why fitting an exponential or an R^-12 to those two
    distances failed.
    """
    from mlxmolkit.nddo.pwcct import c_triple_bond_correction

    def correction(smiles):
        result = _smiles_to_3d(smiles, seed=1)
        if result is None:
            pytest.skip(f"could not embed {smiles}")
        return c_triple_bond_correction(result[0], result[1])

    assert correction("C#C") == pytest.approx(12.0)
    assert correction("CCC#CC(C)C") == pytest.approx(12.0)   # size-independent
    assert correction("CC=CC") == 0.0
    assert correction("CCCC") == 0.0
    # C≡N is not C-C, and nitriles were always accurate without it
    assert correction("CC#N") == 0.0
    assert correction("N#Cc1ccccc1") == 0.0

    # The switch itself, evaluated directly on a stretched acetylene.
    import numpy as _np
    for r, expected in ((1.15, 12.0), (1.20, 12.0), (1.40, 0.0), (1.50, 0.0)):
        c = c_triple_bond_correction([6, 6], _np.array([[0.0, 0, 0], [r, 0, 0]]))
        assert c == pytest.approx(expected), f"at R={r}"
    mid = c_triple_bond_correction([6, 6], _np.array([[0.0, 0, 0], [1.27, 0, 0]]))
    assert 0.0 < mid < 12.0, "the 1.21-1.33 taper should be partial, not a hard step"


def test_the_density_is_right_even_where_the_energy_is_not():
    """The reason this is an energy-assembly bug and not an SCF bug.

    MOPAC on this geometry: HOMO -10.075 eV, LUMO 2.052 eV, 20 filled levels.
    If these ever stop matching, the diagnosis in #33 no longer holds and the
    problem is somewhere else entirely.
    """
    result = _smiles_to_3d("CCC#CC(C)C", seed=1)
    if result is None:
        pytest.skip("could not embed")
    r = nddo_energy(result[0], result[1], method="PM6",
                    max_iter=400, conv_tol=1e-10)
    assert r["n_filled_levels"] == 20
    assert r["homo_eV"] == pytest.approx(-10.075, abs=0.002)
    assert r["lumo_eV"] == pytest.approx(2.052, abs=0.002)
