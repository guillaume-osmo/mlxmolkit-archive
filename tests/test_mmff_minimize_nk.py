import numpy as np
import mlx.core as mx
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from mlxmolkit.mmff_energy_mlx import mmff_energy_batch, params_to_mlx
from mlxmolkit.mmff_minimize import mmff_minimize_nk
from mlxmolkit.mmff_params import extract_mmff_params


def _optimized_mol(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    AllChem.EmbedMolecule(mol, params)
    rdForceFieldHelpers.MMFFOptimizeMolecule(mol, maxIters=200)
    return mol


def _positions(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.asarray(conf.GetPositions(), dtype=np.float32)


def test_mmff_minimize_nk_initial_energy_matches_mlx_energy():
    mol = _optimized_mol("c1ccccc1")
    pos = _positions(mol)
    mmff_params = extract_mmff_params(mol)

    expected = mmff_energy_batch(params_to_mlx(mmff_params), mx.array(pos[None, :, :]))
    _, got, _ = mmff_minimize_nk(
        [mmff_params],
        [1],
        pos.reshape(-1),
        max_iters=0,
        use_lbfgs=False,
    )
    mx.eval(expected)

    np.testing.assert_allclose(got, np.asarray(expected), atol=1.0e-3)


def test_mmff_minimize_nk_aspirin_remains_in_rdkit_energy_range():
    mol = _optimized_mol("CC(=O)OC1=CC=CC=C1C(=O)O")
    pos = _positions(mol)
    mmff_params = extract_mmff_params(mol)

    out_pos, got, converged = mmff_minimize_nk(
        [mmff_params],
        [1],
        pos.reshape(-1),
        max_iters=100,
        use_lbfgs=False,
    )
    assert converged.shape == (1,)
    assert np.isfinite(got[0])
    assert got[0] < 50.0
    assert out_pos.shape == pos.reshape(-1).shape
