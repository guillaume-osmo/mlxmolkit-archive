import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.dg_extract import get_bounds_matrix
from mlxmolkit.etk_extract import ETKDG_VARIANTS, ETKParams, batch_etk_params, extract_etk_params
from mlxmolkit.etk_minimize_metal import etk_minimize_batch
from mlxmolkit.etk_metal import etk_minimize_shared
from mlxmolkit.shared_batch import SharedConstraintBatch


def _distance_only_etk_params() -> ETKParams:
    return ETKParams(
        n_atoms=3,
        torsion_idx=np.zeros((0, 4), dtype=np.int32),
        torsion_V=np.zeros((0, 6), dtype=np.float32),
        torsion_signs=np.zeros((0, 6), dtype=np.int32),
        improper_idx=np.zeros((0, 4), dtype=np.int32),
        improper_weight=np.zeros(0, dtype=np.float32),
        dist12_idx1=np.array([0], dtype=np.int32),
        dist12_idx2=np.array([1], dtype=np.int32),
        dist12_lb=np.array([0.95], dtype=np.float32),
        dist12_ub=np.array([1.05], dtype=np.float32),
        dist12_weight=np.array([100.0], dtype=np.float32),
        dist13_idx1=np.array([0], dtype=np.int32),
        dist13_idx2=np.array([2], dtype=np.int32),
        dist13_lb=np.array([1.45], dtype=np.float32),
        dist13_ub=np.array([1.55], dtype=np.float32),
        dist13_weight=np.array([100.0], dtype=np.float32),
        dist14_idx1=np.zeros(0, dtype=np.int32),
        dist14_idx2=np.zeros(0, dtype=np.int32),
        dist14_lb=np.zeros(0, dtype=np.float32),
        dist14_ub=np.zeros(0, dtype=np.float32),
        dist14_weight=np.zeros(0, dtype=np.float32),
    )


def _distance_only_shared_batch(params: ETKParams) -> SharedConstraintBatch:
    return SharedConstraintBatch(
        n_mols=1,
        n_confs_total=1,
        n_confs_per_mol=[1],
        dim=3,
        conf_atom_starts=np.array([0, 3], dtype=np.int32),
        conf_to_mol=np.array([0], dtype=np.int32),
        mol_n_atoms=np.array([3], dtype=np.int32),
        dist_idx1=np.zeros(0, dtype=np.int32),
        dist_idx2=np.zeros(0, dtype=np.int32),
        dist_lb2=np.zeros(0, dtype=np.float32),
        dist_ub2=np.zeros(0, dtype=np.float32),
        dist_weight=np.zeros(0, dtype=np.float32),
        dist_term_starts=np.array([0, 0], dtype=np.int32),
        chiral_idx1=np.zeros(0, dtype=np.int32),
        chiral_idx2=np.zeros(0, dtype=np.int32),
        chiral_idx3=np.zeros(0, dtype=np.int32),
        chiral_idx4=np.zeros(0, dtype=np.int32),
        chiral_vol_lower=np.zeros(0, dtype=np.float32),
        chiral_vol_upper=np.zeros(0, dtype=np.float32),
        chiral_term_starts=np.array([0, 0], dtype=np.int32),
        fourth_idx=np.zeros(0, dtype=np.int32),
        fourth_term_starts=np.array([0, 0], dtype=np.int32),
        etk_torsion_idx=params.torsion_idx,
        etk_torsion_V=params.torsion_V,
        etk_torsion_signs=params.torsion_signs,
        etk_torsion_term_starts=np.array([0, 0], dtype=np.int32),
        etk_improper_idx=params.improper_idx,
        etk_improper_weight=params.improper_weight,
        etk_improper_term_starts=np.array([0, 0], dtype=np.int32),
        etk_dist12_idx1=params.dist12_idx1,
        etk_dist12_idx2=params.dist12_idx2,
        etk_dist12_lb=params.dist12_lb,
        etk_dist12_ub=params.dist12_ub,
        etk_dist12_weight=params.dist12_weight,
        etk_dist12_term_starts=np.array([0, 1], dtype=np.int32),
        etk_dist13_idx1=params.dist13_idx1,
        etk_dist13_idx2=params.dist13_idx2,
        etk_dist13_lb=params.dist13_lb,
        etk_dist13_ub=params.dist13_ub,
        etk_dist13_weight=params.dist13_weight,
        etk_dist13_term_starts=np.array([0, 1], dtype=np.int32),
        etk_dist14_idx1=params.dist14_idx1,
        etk_dist14_idx2=params.dist14_idx2,
        etk_dist14_lb=params.dist14_lb,
        etk_dist14_ub=params.dist14_ub,
        etk_dist14_weight=params.dist14_weight,
        etk_dist14_term_starts=np.array([0, 0], dtype=np.int32),
    )


def test_shared_etk_returned_energy_matches_recomputed_objective_with_d12_d13():
    params = _distance_only_etk_params()
    batch = _distance_only_shared_batch(params)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape(-1)

    out_pos, energy, _ = etk_minimize_shared(batch, positions, max_iters=1)
    _, recomputed, _ = etk_minimize_shared(batch, out_pos, max_iters=0)

    np.testing.assert_allclose(energy, recomputed, atol=1.0e-4)


def test_legacy_etk_returned_energy_matches_recomputed_objective_with_d12_d13():
    params = _distance_only_etk_params()
    system = batch_etk_params([params], np.array([0, 3], dtype=np.int32))
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape(-1)

    result = etk_minimize_batch(system, positions, max_iters=1)
    recomputed = etk_minimize_batch(system, result.positions, max_iters=0)

    np.testing.assert_allclose(result.energies, recomputed.energies, atol=1.0e-4)


def test_etkdg_variant_shortcuts_match_rdkit_embed_parameter_flags():
    expected_attrs = {
        "useExpTorsionAnglePrefs": 0,
        "useBasicKnowledge": 1,
        "useSmallRingTorsions": 2,
        "useMacrocycleTorsions": 3,
        "useMacrocycle14config": 4,
        "ETversion": 5,
    }
    for variant, expected in ETKDG_VARIANTS.items():
        if variant == "DG":
            continue
        params = getattr(rdDistGeom, variant)()
        for attr, idx in expected_attrs.items():
            assert getattr(params, attr) == expected[idx], f"{variant}.{attr}"


def test_etk_variant_term_families_are_not_mixed():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC1=CC=CC=C1"))
    bounds = get_bounds_matrix(mol)

    dg = extract_etk_params(mol, bounds, variant="DG")
    etdg = extract_etk_params(mol, bounds, variant="ETDG")
    kdg = extract_etk_params(mol, bounds, variant="KDG")
    etkdg = extract_etk_params(mol, bounds, variant="ETKDGv3")

    assert len(dg.torsion_idx) == 0
    assert len(dg.improper_idx) == 0
    assert len(dg.dist12_idx1) == 0
    assert len(dg.dist13_idx1) == 0

    assert len(etdg.torsion_idx) > 0
    assert len(etdg.improper_idx) == 0
    assert len(etdg.dist12_idx1) == 0
    assert len(etdg.dist13_idx1) == 0

    assert len(kdg.dist12_idx1) > 0
    assert len(kdg.dist13_idx1) > 0
    assert len(kdg.improper_idx) > 0

    assert len(etkdg.torsion_idx) > 0
    assert len(etkdg.dist12_idx1) > 0
    assert len(etkdg.dist13_idx1) > 0
