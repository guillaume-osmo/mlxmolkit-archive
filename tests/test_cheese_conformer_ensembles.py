import numpy as np

from tools.prepare_cheese_conformer_ensembles import (
    conformer_pairwise_rmsd,
    select_conformer_ensemble,
)


def _propane_with_manual_conformers():
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromSmiles("CCC")
    coords = [
        np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.5, 0.0]], dtype=float),
    ]
    for conf_id, xyz in enumerate(coords):
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetId(conf_id)
        for atom_index, point in enumerate(xyz):
            conf.SetAtomPosition(atom_index, Point3D(float(point[0]), float(point[1]), float(point[2])))
        mol.AddConformer(conf, assignId=False)
    return mol


def test_conformer_diversity_selector_drops_post_optimization_duplicates():
    mol = _propane_with_manual_conformers()
    conf_ids = [0, 1, 2]
    energies = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    converged = np.asarray([True, True, True])

    selected, selected_energies, selected_converged, stats = select_conformer_ensemble(
        mol,
        conf_ids,
        energies,
        converged,
        n_keep=3,
        selection_mode="energy_diverse",
        rms_thresh=0.2,
        energy_window=10.0,
        atom_mode="heavy",
        fill_to_n=False,
    )

    assert selected == [0, 2]
    np.testing.assert_allclose(selected_energies, [0.0, 0.2])
    assert selected_converged.tolist() == [True, True]
    assert stats["n_selected"] == 2
    assert stats["rejected_for_rms"] == 1
    assert stats["min_pair_rmsd"] >= 0.2


def test_conformer_diversity_selector_can_fill_fixed_size_when_requested():
    mol = _propane_with_manual_conformers()
    selected, _, _, stats = select_conformer_ensemble(
        mol,
        [0, 1, 2],
        np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        np.asarray([True, True, True]),
        n_keep=3,
        selection_mode="energy_diverse",
        rms_thresh=0.2,
        energy_window=10.0,
        atom_mode="heavy",
        fill_to_n=True,
    )

    assert selected == [0, 2, 1]
    assert stats["n_selected"] == 3
    rmsd = conformer_pairwise_rmsd(mol, selected, atom_mode="heavy")
    assert rmsd.shape == (3, 3)
