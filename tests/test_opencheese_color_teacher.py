import numpy as np

import mlx.core as mx

from opencheese.descriptors import cheese_batch
from tools.compute_opencheese_color_teacher import (
    _feature_factory,
    acceptor_donor_masks_from_smiles,
    color_similarity_block,
)
from mlxmolkit.charge_model import bond_matrix_from_rdkit_mol


def test_acceptor_donor_masks_follow_rdkit_feature_families():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CC(=O)N")
    atoms = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
    bonds = bond_matrix_from_rdkit_mol(mol)

    acceptor, donor = acceptor_donor_masks_from_smiles("CC(=O)N", atoms, bonds, factory=_feature_factory())

    assert acceptor.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert donor.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_binary_color_similarity_penalizes_acceptor_donor_mismatch():
    atoms = [np.asarray([6, 8, 7], dtype=np.int32)] * 2
    coords = [
        np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]], dtype=np.float32),
    ]
    charges = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    batch = cheese_batch(atoms, coords, charges)
    owner_colors = mx.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=mx.float32)
    empty = mx.zeros_like(owner_colors)

    _, acceptor, _, color, combined = color_similarity_block(
        batch,
        batch,
        owner_colors,
        owner_colors,
        empty,
        empty,
        shape_metric="carbo",
        shape_weight=1.0,
        acceptor_weight=1.0,
        donor_weight=0.0,
        gaussian_alpha=2.7,
        vdw_scale=1.0,
        default_radius=1.8,
    )
    mx.eval(acceptor, color, combined)
    acceptor_np = np.asarray(acceptor)
    combined_np = np.asarray(combined)

    assert acceptor_np[0, 0] > 0.99
    assert acceptor_np[1, 1] > 0.99
    assert acceptor_np[0, 1] < 0.2
    assert combined_np[0, 1] < combined_np[0, 0]
