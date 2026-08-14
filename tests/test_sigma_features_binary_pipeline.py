from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem", reason="RDKit required")

from mlxmolkit.xtb.cosmo_sigma import hybrid_gxtb_gfn2_cosmo_from_smiles
from mlxmolkit.xtb.sigma_features import sigma_feature_tensors


XTB = Path("/tmp/gxtb-v2-macos/bin/xtb")


CASES = {
    "water": {
        "smiles": "O",
        "n_atoms": 3,
        "n_segments": 225,
        "area": 60.08567498499907,
        "v_pot_sum": 338658.00520516076,
        "v_pot_l2": 59346.43331726638,
        "v_pot_first5": [
            15799.822058602273,
            14631.603561471142,
            13516.079858041197,
            12452.967737055013,
            11441.94297084962,
        ],
        "atom_area": [40.085490652, 9.890266046999999, 10.109918285],
        "mean_atom_potential": [
            -0.006954503883857849,
            0.05905848795483563,
            0.059231980247696825,
        ],
        "edge_shape": (2, 4),
        "angle_shape": (3, 1),
    },
    "methanol": {
        "smiles": "CO",
        "n_atoms": 6,
        "n_segments": 355,
        "area": 113.54529790894206,
        "v_pot_sum": 428557.9733473867,
        "v_pot_l2": 73178.45938587846,
        "v_pot_first5": [
            20372.241128019272,
            19018.451505957586,
            17714.72701735651,
            16460.663218736143,
            15255.834782919927,
        ],
        "atom_area": [
            53.92823509799999,
            28.684745998999997,
            7.5759537009999995,
            7.070379700999999,
            7.447606179999999,
            8.838377224999999,
        ],
        "mean_atom_potential": [
            0.002239675119054331,
            -0.025420644843358754,
            0.022038379464077617,
            0.0003466089977668093,
            0.022371440286547667,
            0.05828204797477137,
        ],
        "edge_shape": (2, 10),
        "angle_shape": (3, 7),
    },
    "ethanol": {
        "smiles": "CCO",
        "n_atoms": 9,
        "n_segments": 487,
        "area": 161.14952672946984,
        "v_pot_sum": 458256.6141923636,
        "v_pot_l2": 78305.67020136662,
        "v_pot_first5": [
            21128.983496783487,
            19760.274883939845,
            18439.592673351584,
            17166.457300917235,
            15940.391616018525,
        ],
        "atom_area": [
            48.25847013299999,
            43.39568934799999,
            26.708237436999994,
            7.076769291,
            6.413400622999999,
            7.097665128999999,
            6.921150762,
            6.58991499,
            8.688229007999999,
        ],
        "mean_atom_potential": [
            0.015042364757867428,
            -0.005569188028084325,
            -0.031185297808124544,
            0.022696302710987677,
            0.004415065000422163,
            0.03129053306353993,
            -0.008010951107223918,
            0.013517603783244755,
            0.05405128031383581,
        ],
        "edge_shape": (2, 16),
        "angle_shape": (3, 13),
    },
    "methyl_iodide": {
        "smiles": "CI",
        "n_atoms": 5,
        "n_segments": 310,
        "area": 128.9441147313891,
        "v_pot_sum": 487967.2577255029,
        "v_pot_l2": 83925.4495014969,
        "v_pot_first5": [
            23231.230400662254,
            21694.009212646517,
            20210.55340232264,
            18780.753304651196,
            17404.49478554263,
        ],
        "atom_area": [
            50.85246768499999,
            56.585168096000004,
            7.416687948,
            7.102223897,
            6.9875671,
        ],
        "mean_atom_potential": [
            0.027179859703564995,
            -0.026139536361980756,
            0.04325590089363356,
            0.044241033074752155,
            0.04363440910937806,
        ],
        "edge_shape": (2, 8),
        "angle_shape": (3, 6),
    },
}


@pytest.mark.slow
@pytest.mark.skipif(not XTB.exists(), reason=f"official g-xTB binary not found at {XTB}")
@pytest.mark.parametrize("name", CASES)
def test_binary_cosmo_sigma_features_match_golden_fingerprints(name: str):
    """Prove the binary-backed COSMO -> sigma tensor path is reproducible."""

    expected = CASES[name]
    out = hybrid_gxtb_gfn2_cosmo_from_smiles(
        expected["smiles"],
        solvent="inf",
        seed=42,
        acc=0.2,
        xtb_path=XTB,
    )
    cosmo = out["cosmo"]
    mol = Chem.AddHs(Chem.MolFromSmiles(expected["smiles"]))
    features = sigma_feature_tensors(
        cosmo,
        {"method": "binary-gxtb-geometry-plus-gfn2-tmcosmo"},
        mol,
        include_angles=True,
    )

    assert len(cosmo.atom_z) == expected["n_atoms"]
    assert cosmo.segments_area.size == expected["n_segments"]
    assert features["X_atom"].shape == (expected["n_atoms"], 213)
    assert features["v_pot"].shape == (61,)
    assert features["v_profile"].shape == (61,)
    assert tuple(features["edge_index"].shape) == expected["edge_shape"]
    assert tuple(features["angle_index"].shape) == expected["angle_shape"]

    assert np.isfinite(features["X_atom"]).all()
    assert np.isfinite(features["v_pot"]).all()
    assert np.isfinite(features["v_profile"]).all()
    assert abs(cosmo.total_screening_charge) < 1.0e-8

    np.testing.assert_allclose(cosmo.area, expected["area"], rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(np.sum(features["v_profile"]), cosmo.area, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        np.sum(features["y_mol"]["sigma_profile_corr"]),
        cosmo.area,
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(features["y_atom"]["surface_area"], expected["atom_area"], atol=1.0e-8)
    np.testing.assert_allclose(
        np.sum(features["y_atom"]["surface_area"]),
        cosmo.area,
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        features["y_atom"]["mean_segment_potential"],
        expected["mean_atom_potential"],
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(np.sum(features["v_pot"]), expected["v_pot_sum"], atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(features["v_pot"]), expected["v_pot_l2"], atol=1.0e-6)
    np.testing.assert_allclose(features["v_pot"][:5], expected["v_pot_first5"], atol=1.0e-8)

    primary = slice(*features["meta"]["feature_slices"]["sigma_primary_bins"])
    corr = slice(*features["meta"]["feature_slices"]["sigma_corr_bins"])
    pot = slice(*features["meta"]["feature_slices"]["sigma_potential_bins"])
    np.testing.assert_allclose(features["X_atom"][:, primary], features["y_atom"]["profile_primary"])
    np.testing.assert_allclose(features["X_atom"][:, corr], features["y_atom"]["profile_corr"])
    np.testing.assert_allclose(features["X_atom"][:, pot], features["y_atom"]["profile_potential"])
