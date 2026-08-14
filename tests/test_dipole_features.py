from __future__ import annotations

import csv
import json

import numpy as np
import pytest

pytest.importorskip("rdkit", reason="RDKit required")

from mlxmolkit.dipole_features import (
    ATOM_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    MOLECULE_FEATURE_NAMES,
    dipole_atom_feature_tensors,
    export_dipole_atom_feature_dataset,
)


@pytest.mark.parametrize(
    ("smiles", "n_heavy", "n_explicit", "n_edges"),
    [
        ("O", 1, 3, 0),
        ("CO", 2, 6, 2),
        ("CCO", 3, 9, 4),
        ("CI", 2, 5, 2),
    ],
)
def test_dipole_atom_features_default_to_chemprop_heavy_atom_rows(
    smiles: str,
    n_heavy: int,
    n_explicit: int,
    n_edges: int,
):
    tensors = dipole_atom_feature_tensors(smiles, seed=42)

    assert tensors.atom_mode == "implicit_h_heavy_atoms"
    assert tensors.atom_features.shape == (n_heavy, len(ATOM_FEATURE_NAMES))
    assert tensors.molecule_features.shape == (len(MOLECULE_FEATURE_NAMES),)
    assert tensors.edge_index.shape == (2, n_edges)
    assert tensors.edge_attr.shape == (n_edges, len(EDGE_FEATURE_NAMES))
    assert tensors.atom_coords_ang.shape == (n_heavy, 3)
    assert tensors.molecule_features[0] == pytest.approx(n_heavy)
    assert tensors.molecule_features[1] == pytest.approx(n_explicit)
    assert np.isfinite(tensors.atom_features).all()
    assert np.isfinite(tensors.molecule_features).all()
    assert np.isfinite(tensors.edge_attr).all()

    gasteiger_proxy = tensors.molecule_features[
        tensors.molecule_feature_names.index("gasteiger_dipole_proxy_debye")
    ]
    assert gasteiger_proxy >= 0.0


def test_dipole_atom_features_can_emit_explicit_h_rows():
    tensors = dipole_atom_feature_tensors("CO", seed=42, include_h=True)

    assert tensors.atom_mode == "explicit_h_atoms"
    assert tensors.atom_features.shape == (6, len(ATOM_FEATURE_NAMES))
    assert tensors.edge_index.shape == (2, 10)
    assert tensors.atom_coords_ang.shape == (6, 3)


def test_export_dipole_atom_feature_dataset_writes_chemprop_npz(tmp_path):
    input_csv = tmp_path / "dipoles.csv"
    with input_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "smiles", "dipole_debye"])
        writer.writeheader()
        writer.writerows(
            [
                {"name": "water", "smiles": "O", "dipole_debye": "1.8546"},
                {"name": "methanol", "smiles": "CO", "dipole_debye": "1.70"},
                {"name": "methyl_iodide", "smiles": "CI", "dipole_debye": "1.64"},
            ]
        )

    out_dir = tmp_path / "features"
    meta = export_dipole_atom_feature_dataset(
        input_csv,
        out_dir,
        target_cols=["dipole_debye"],
        seed=42,
        strict=True,
    )

    assert meta["n_exported"] == 3
    assert meta["n_failed"] == 0
    assert meta["atom_mode"] == "implicit_h_heavy_atoms"
    assert meta["target_cols"] == ["dipole_debye"]

    atom_npz = np.load(out_dir / "atom_features.npz")
    assert atom_npz["arr_0"].shape == (1, len(ATOM_FEATURE_NAMES))
    assert atom_npz["arr_1"].shape == (2, len(ATOM_FEATURE_NAMES))
    assert atom_npz["arr_2"].shape == (2, len(ATOM_FEATURE_NAMES))

    mol_npz = np.load(out_dir / "molecule_features.npz")
    assert mol_npz["arr_0"].shape == (len(MOLECULE_FEATURE_NAMES),)

    graph_npz = np.load(out_dir / "graph_tensors.npz")
    assert graph_npz["edge_index_0"].shape == (2, 0)
    assert graph_npz["edge_index_1"].shape == (2, 2)
    assert graph_npz["edge_attr_1"].shape == (2, len(EDGE_FEATURE_NAMES))

    rows = list(csv.DictReader((out_dir / "dipole_chemprop.csv").open()))
    assert [row["smiles"] for row in rows] == ["O", "CO", "CI"]
    assert [row["feature_index"] for row in rows] == ["0", "1", "2"]
    assert [row["dipole_debye"] for row in rows] == ["1.8546", "1.70", "1.64"]

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["atom_feature_names"] == list(ATOM_FEATURE_NAMES)
    assert metadata["molecule_feature_names"] == list(MOLECULE_FEATURE_NAMES)
