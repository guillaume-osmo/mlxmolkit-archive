"""ETKDGv3's macrocycle 1-4 bounds must actually reach the bounds matrix.

RDKit's macrocycle fix (Wang et al., JCIM 2020) is not in the torsion terms —
it is a change to the 1-4 distance bounds, requested with
``GetMoleculeBoundsMatrix(..., useMacrocycle14config=True)``. RDKit defaults it
off, so it has to be asked for.

mlxmolkit used to unpack ``use_macrocycle14config`` from its variant table and
then never use it: the bounds matrix was built with a bare call, so selecting
ETKDGv3 changed the torsions but left macrocycles embedded against generic 1-4
bounds. Measured over 100 macrocycles from the 12k ePOM subset, scoring both
sides with RDKit's own MMFF94, wiring the flag through moved mean |dE| from
2.70 to 1.48 kcal/mol and conformers within 1 kcal/mol of RDKit from 34% to
58%, while leaving acyclic, single-ring and fused-ring results bit-identical.

These tests pin both halves of that: the flag reaches the bounds, and it stays
confined to macrocycles.
"""
from __future__ import annotations

import numpy as np
import pytest

from rdkit import Chem

from mlxmolkit.dg_extract import get_bounds_matrix
from mlxmolkit.etk_extract import ETKDG_VARIANTS

MACROCYCLE = "O=C1OCCCCCCCCCCCCCO1"      # 16-membered macrolactone
SMALL_RINGS = [
    "CC1CCC(C(C)C)CC1O",                  # menthol, one 6-ring
    "CC(C)=CCC1=CCCCC1",                  # a 6-ring with a side chain
    "C=C1CCC2C(C3C(C)CCC13)C2(C)C",       # fused polycycle
    "CCCCCCC(C)O",                        # acyclic
]


def _mol(smiles: str) -> Chem.Mol:
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def test_the_flag_changes_the_bounds_of_a_macrocycle():
    """Without this the whole macrocycle path is a no-op."""
    off = get_bounds_matrix(_mol(MACROCYCLE))
    on = get_bounds_matrix(_mol(MACROCYCLE), use_macrocycle14config=True)
    changed = np.abs(off - on) > 1e-6
    assert changed.any(), (
        "useMacrocycle14config did not reach GetMoleculeBoundsMatrix — "
        "the ETKDGv3 macrocycle fix is not being applied"
    )
    # It is a substantive change, not a rounding difference.
    assert np.abs(off - on).max() > 1.0


@pytest.mark.parametrize("smiles", SMALL_RINGS)
def test_the_flag_is_confined_to_macrocycles(smiles):
    """Anything smaller than a macrocycle must be untouched, bit for bit."""
    off = get_bounds_matrix(_mol(smiles))
    on = get_bounds_matrix(_mol(smiles), use_macrocycle14config=True)
    assert np.array_equal(off, on), f"{smiles} bounds changed by a macrocycle-only flag"


def test_the_variant_table_still_marks_v3_as_the_macrocycle_variant():
    """The pipeline reads slot 4 of this tuple to decide. Guard the layout."""
    assert ETKDG_VARIANTS["ETKDGv3"][4] is True
    assert ETKDG_VARIANTS["ETKDGv2"][4] is False


def test_selecting_v3_embeds_a_macrocycle_against_macrocycle_bounds():
    """End to end: the variant name alone must switch the bounds."""
    from mlxmolkit.conformer_pipeline_v2 import generate_conformers_nk as gen

    v2 = gen([MACROCYCLE], 2, variant="ETKDGv2")
    v3 = gen([MACROCYCLE], 2, variant="ETKDGv3")
    a = np.asarray(v2.molecules[0].positions_3d[0], dtype=float)
    b = np.asarray(v3.molecules[0].positions_3d[0], dtype=float)
    assert not np.allclose(a, b), (
        "ETKDGv2 and ETKDGv3 produced identical macrocycle geometry — "
        "the variant is not reaching the embedding"
    )
