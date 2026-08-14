from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rdkit", reason="RDKit required")

from mlxmolkit.xtb.cosmo_sigma import CosmoSegments
from mlxmolkit.xtb.sigma_features import sigma_feature_tensors


def _toy_water_cosmo() -> CosmoSegments:
    atom_coords_bohr = np.array(
        [
            [0.0, 0.0, 0.2226],
            [0.0, 1.4271, -0.8902],
            [0.0, -1.4271, -0.8902],
        ],
        dtype=np.float64,
    )
    segments_atom = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
    segments_xyz_bohr = np.array(
        [
            [0.30, 0.00, 0.45],
            [-0.30, 0.00, 0.45],
            [0.10, 1.70, -0.70],
            [-0.10, 1.70, -0.70],
            [0.10, -1.70, -0.70],
            [-0.10, -1.70, -0.70],
        ],
        dtype=np.float64,
    )
    segments_area = np.array([2.1, 1.9, 1.2, 1.1, 1.2, 1.1], dtype=np.float64)
    segments_charge = np.array([-0.030, -0.027, 0.012, 0.010, 0.012, 0.010], dtype=np.float64)
    segments_sigma = segments_charge / segments_area
    segments_potential = np.array([-0.80, -0.72, 0.18, 0.14, 0.18, 0.14], dtype=np.float64)
    return CosmoSegments(
        epsilon=float("inf"),
        fepsi=1.0,
        area=float(np.sum(segments_area)),
        volume=25.0,
        total_screening_charge=float(np.sum(segments_charge)),
        total_energy_hartree=-76.0,
        dielectric_energy_hartree=-0.5,
        atom_radii=np.array([1.7, 1.2, 1.2], dtype=np.float64),
        atom_coords_bohr=atom_coords_bohr,
        atom_z=[8, 1, 1],
        segments_atom=segments_atom,
        segments_xyz_bohr=segments_xyz_bohr,
        segments_charge=segments_charge,
        segments_area=segments_area,
        segments_sigma=segments_sigma,
        segments_potential=segments_potential,
        cosmo_text="",
    )


def _toy_gxtb_res() -> dict[str, object]:
    shell_atom = np.array([0, 0, 1, 2], dtype=np.int64)
    shell_charges = np.array([-0.42, -0.18, 0.30, 0.30], dtype=np.float64)
    return {
        "method": "g-xTB-reconstructed",
        "atom_charges": np.array([-0.60, 0.30, 0.30], dtype=np.float64),
        "shell_charges": shell_charges,
        "coordination_number": np.array([1.95, 0.88, 0.88], dtype=np.float64),
        "eeqbc_charges": np.array([-0.52, 0.26, 0.26], dtype=np.float64),
        "basis": SimpleNamespace(shell_atom=shell_atom),
        "energy_hartree": -5.123,
        "repulsion_hartree": 0.234,
    }


def test_sigma_feature_tensors_shapes_and_targets():
    from rdkit import Chem

    cosmo = _toy_water_cosmo()
    gxtb_res = _toy_gxtb_res()
    mol = Chem.MolFromSmiles("O")

    out = sigma_feature_tensors(cosmo, gxtb_res, mol, include_angles=True)

    assert out["sigma_grid_e_per_A2"].shape == (61,)
    assert out["v_pot"].shape == (61,)
    assert out["v_profile"].shape == (61,)
    assert out["X_atom"].shape == (3, 213)
    assert out["edge_index"].shape == (2, 4)
    assert out["edge_attr"].shape == (4, 5)
    assert out["angle_index"].shape == (3, 1)
    assert out["angle_attr"].shape == (1, 3)

    np.testing.assert_allclose(out["y_atom"]["atom_charges"], [-0.60, 0.30, 0.30])
    np.testing.assert_allclose(out["y_atom"]["shell_charge_sum"], [-0.60, 0.30, 0.30])
    np.testing.assert_allclose(out["y_atom"]["shell_charge_abs_sum"], [0.60, 0.30, 0.30])
    np.testing.assert_allclose(out["y_atom"]["surface_area"], [4.0, 2.3, 2.3])

    primary_slice = slice(*out["meta"]["feature_slices"]["sigma_primary_bins"])
    corr_slice = slice(*out["meta"]["feature_slices"]["sigma_corr_bins"])
    pot_slice = slice(*out["meta"]["feature_slices"]["sigma_potential_bins"])
    scalar_slice = slice(*out["meta"]["feature_slices"]["local_scalars"])
    chem_slice = slice(*out["meta"]["feature_slices"]["chemical_features"])

    np.testing.assert_allclose(out["X_atom"][:, primary_slice], out["y_atom"]["profile_primary"])
    np.testing.assert_allclose(out["X_atom"][:, corr_slice], out["y_atom"]["profile_corr"])
    np.testing.assert_allclose(out["X_atom"][:, pot_slice], out["y_atom"]["profile_potential"])
    np.testing.assert_allclose(out["X_atom"][:, scalar_slice.start], out["y_atom"]["surface_area"])
    np.testing.assert_allclose(out["X_atom"][:, scalar_slice.start + 6], out["y_atom"]["mean_segment_potential"])
    np.testing.assert_allclose(out["X_atom"][:, chem_slice.start], [8.0, 1.0, 1.0])

    assert out["y_mol"]["energy_hartree"] == pytest.approx(-5.123)
    assert out["y_mol"]["repulsion_hartree"] == pytest.approx(0.234)
    assert out["y_mol"]["sigma_potential"].shape == (61,)
    assert out["y_mol"]["sigma_profile"].shape == (61,)
    assert len(out["meta"]["atom_feature_names"]) == out["X_atom"].shape[1]

    np.testing.assert_allclose(np.sum(out["v_profile"]), cosmo.area)
