"""Tests for mlxmolkit.erg_features."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlxmolkit.erg_features import erg_fp_from_mols, erg_fp_from_smiles


# Druglike SMILES — small molecules like CCO give a zero ERG vector, so use
# something pharmacophore-rich.
_DRUGLIKE = [
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",        # ibuprofen
    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",      # naproxen
    "CC(=O)Nc1ccc(O)cc1",                # paracetamol
    "CC(=O)Oc1ccccc1C(=O)O",             # aspirin
]


class TestErgFromSmiles:
    def test_shape_315(self):
        fp, idx = erg_fp_from_smiles(_DRUGLIKE)
        mx.eval(fp)
        assert fp.shape == (len(_DRUGLIKE), 315)
        assert idx == list(range(len(_DRUGLIKE)))

    def test_invalid_smiles_dropped(self):
        smis = ["CCO", "not_a_smiles_xx", _DRUGLIKE[0]]
        fp, idx = erg_fp_from_smiles(smis)
        mx.eval(fp)
        # bad_smiles dropped; CCO valid (but gives a near-zero ERG); ibuprofen valid
        assert idx == [0, 2]
        assert fp.shape == (2, 315)

    def test_empty_input(self):
        fp, idx = erg_fp_from_smiles([])
        mx.eval(fp)
        assert fp.shape == (0, 315)
        assert idx == []

    def test_all_invalid(self):
        fp, idx = erg_fp_from_smiles(["not_a", "still_not"])
        mx.eval(fp)
        assert fp.shape == (0, 315)
        assert idx == []

    def test_druglike_nonzero(self):
        fp, _ = erg_fp_from_smiles([_DRUGLIKE[0]])
        mx.eval(fp)
        assert float(mx.sum(fp != 0)) > 0


class TestErgFromMols:
    def test_passes_through(self):
        from rdkit.Chem import MolFromSmiles
        mols = [MolFromSmiles(s) for s in _DRUGLIKE]
        fp = erg_fp_from_mols(mols)
        mx.eval(fp)
        assert fp.shape == (len(_DRUGLIKE), 315)

    def test_none_mol_yields_zero_row(self):
        from rdkit.Chem import MolFromSmiles
        mols = [MolFromSmiles(_DRUGLIKE[0]), None, MolFromSmiles(_DRUGLIKE[1])]
        fp = erg_fp_from_mols(mols)
        mx.eval(fp)
        arr = np.array(fp)
        # Row 1 corresponds to the None mol — should be all zeros.
        assert np.all(arr[1] == 0.0)
        # The other two should have nonzero entries.
        assert np.any(arr[0] != 0.0)
        assert np.any(arr[2] != 0.0)
