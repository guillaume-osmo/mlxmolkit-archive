from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EANG_TO_DEBYE = 4.803204712570263


ATOM_FEATURE_NAMES = [
    "atomic_number",
    "atomic_mass",
    "covalent_radius",
    "vdw_radius",
    "default_valence",
    "outer_electrons",
    "degree",
    "total_degree_with_h",
    "total_valence",
    "formal_charge",
    "total_num_h",
    "is_aromatic",
    "is_in_ring",
    "ring_count",
    "hybridization_code",
    "chiral_tag_code",
    "radical_electrons",
    "r_geom_centroid_ang",
    "r_mass_centroid_ang",
    "r_heavy_centroid_ang",
    "nearest_heavy_distance_ang",
    "mean_heavy_bond_distance_ang",
    "min_heavy_bond_distance_ang",
    "max_heavy_bond_distance_ang",
    "sum_neighbor_inverse_distance",
    "attached_h_count",
    "attached_h_mean_distance_ang",
    "attached_h_max_distance_ang",
    "gasteiger_charge",
    "gasteiger_charge_with_attached_h",
    "gasteiger_attached_h_charge_sum",
    "gasteiger_abs_charge_with_attached_h",
    "gasteiger_q_r_mass_centroid",
    "gasteiger_abs_q_r_mass_centroid",
    "gasteiger_neighbor_charge_sum",
    "gasteiger_local_bond_polarity",
    "mmff_charge",
    "mmff_charge_with_attached_h",
    "mmff_attached_h_charge_sum",
    "mmff_abs_charge_with_attached_h",
    "mmff_q_r_mass_centroid",
    "mmff_abs_q_r_mass_centroid",
    "mmff_neighbor_charge_sum",
    "mmff_local_bond_polarity",
]


MOLECULE_FEATURE_NAMES = [
    "n_feature_atoms",
    "n_explicit_atoms",
    "n_bonds",
    "molecular_weight",
    "formal_charge",
    "radius_gyration_ang",
    "gasteiger_dipole_proxy_debye",
    "mmff_dipole_proxy_debye",
    "gasteiger_total_abs_charge",
    "mmff_total_abs_charge",
]


EDGE_FEATURE_NAMES = [
    "distance_ang",
    "bond_order",
    "is_aromatic",
    "is_conjugated",
    "is_in_ring",
    "gasteiger_abs_dq",
    "mmff_abs_dq",
]


@dataclass(frozen=True)
class DipoleFeatureTensors:
    smiles: str
    atom_features: np.ndarray
    atom_feature_names: tuple[str, ...]
    molecule_features: np.ndarray
    molecule_feature_names: tuple[str, ...]
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_feature_names: tuple[str, ...]
    atom_coords_ang: np.ndarray
    atom_mode: str


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _embed_explicit_h_mol(smiles: str, *, seed: int = 42, max_iters: int = 300):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")

    mol_h = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        params.useRandomCoords = False
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            # Last-resort deterministic fallback for unusual valence/stereo
            # cases. It keeps graph rows aligned with the target dataset; the
            # coordinate-derived fields become planar but remain finite.
            if AllChem.Compute2DCoords(mol_h) < 0:
                raise RuntimeError(f"RDKit embedding failed for {smiles!r}")
            return mol, mol_h
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=max_iters)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=max_iters)
    except Exception:
        pass
    return mol, mol_h


def _coords_ang(mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.asarray(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=np.float64,
    )


def _gasteiger_charges(mol_h) -> np.ndarray:
    from rdkit.Chem import rdPartialCharges

    rdPartialCharges.ComputeGasteigerCharges(mol_h, nIter=12, throwOnParamFailure=False)
    charges = []
    for atom in mol_h.GetAtoms():
        charges.append(_finite_float(atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0))
    return np.asarray(charges, dtype=np.float64)


def _mmff_charges(mol_h) -> np.ndarray:
    from rdkit.Chem import AllChem

    props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
    if props is None:
        return np.zeros(mol_h.GetNumAtoms(), dtype=np.float64)
    return np.asarray(
        [_finite_float(props.GetMMFFPartialCharge(i)) for i in range(mol_h.GetNumAtoms())],
        dtype=np.float64,
    )


def _attached_h_indices(mol_h, n_heavy: int) -> list[list[int]]:
    attached = [[] for _ in range(n_heavy)]
    for atom in mol_h.GetAtoms():
        idx = int(atom.GetIdx())
        if int(atom.GetAtomicNum()) != 1:
            continue
        nbrs = atom.GetNeighbors()
        if len(nbrs) == 1:
            heavy_idx = int(nbrs[0].GetIdx())
            if heavy_idx < n_heavy:
                attached[heavy_idx].append(idx)
    return attached


def _centroids(coords: np.ndarray, masses: np.ndarray, heavy_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom_centroid = np.mean(coords, axis=0)
    total_mass = float(np.sum(masses))
    mass_centroid = np.sum(coords * masses[:, None], axis=0) / total_mass if total_mass > 0.0 else geom_centroid
    heavy_centroid = np.mean(coords[heavy_mask], axis=0) if np.any(heavy_mask) else geom_centroid
    return geom_centroid, mass_centroid, heavy_centroid


def _dipole_proxy(charges: np.ndarray, coords: np.ndarray, origin: np.ndarray) -> float:
    vec_e_ang = np.sum(charges[:, None] * (coords - origin[None, :]), axis=0)
    return float(np.linalg.norm(vec_e_ang) * EANG_TO_DEBYE)


def _heavy_bond_distances(mol, coords: np.ndarray, atom_idx: int) -> list[float]:
    out = []
    atom = mol.GetAtomWithIdx(atom_idx)
    for nbr in atom.GetNeighbors():
        j = int(nbr.GetIdx())
        out.append(float(np.linalg.norm(coords[atom_idx] - coords[j])))
    return out


def _local_charge_terms(mol, coords: np.ndarray, charges: np.ndarray, atom_idx: int) -> tuple[float, float]:
    qsum = 0.0
    polarity = 0.0
    atom = mol.GetAtomWithIdx(atom_idx)
    qi = float(charges[atom_idx])
    for nbr in atom.GetNeighbors():
        j = int(nbr.GetIdx())
        d = float(np.linalg.norm(coords[atom_idx] - coords[j]))
        if d <= 1.0e-12:
            continue
        qj = float(charges[j])
        qsum += qj
        polarity += abs(qi - qj) / d
    return qsum, polarity


def _base_atom_features(
    mol,
    mol_h,
    coords_h: np.ndarray,
    gasteiger: np.ndarray,
    mmff: np.ndarray,
    *,
    include_h: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    n_heavy = mol.GetNumAtoms()
    atoms_h = list(mol_h.GetAtoms())
    heavy_mask = np.asarray([int(a.GetAtomicNum()) > 1 for a in atoms_h], dtype=bool)
    masses_h = np.asarray([float(a.GetMass()) for a in atoms_h], dtype=np.float64)
    geom_centroid, mass_centroid, heavy_centroid = _centroids(coords_h, masses_h, heavy_mask)
    attached_h = _attached_h_indices(mol_h, n_heavy)

    if include_h:
        atom_indices = list(range(mol_h.GetNumAtoms()))
        feature_mol = mol_h
        atom_mode = "explicit_h_atoms"
    else:
        atom_indices = list(range(n_heavy))
        feature_mol = mol
        atom_mode = "implicit_h_heavy_atoms"

    features: list[list[float]] = []
    coords_out: list[np.ndarray] = []
    for local_idx, h_idx in enumerate(atom_indices):
        atom_h = mol_h.GetAtomWithIdx(h_idx)
        atom = feature_mol.GetAtomWithIdx(local_idx) if not include_h else atom_h
        z = int(atom_h.GetAtomicNum())
        coords_i = coords_h[h_idx]
        coords_out.append(coords_i)

        if include_h or h_idx >= n_heavy:
            h_list: list[int] = []
            qg_with_h = float(gasteiger[h_idx])
            qm_with_h = float(mmff[h_idx])
            qg_h_sum = 0.0
            qm_h_sum = 0.0
        else:
            h_list = attached_h[h_idx]
            qg_h_sum = float(np.sum(gasteiger[h_list])) if h_list else 0.0
            qm_h_sum = float(np.sum(mmff[h_list])) if h_list else 0.0
            qg_with_h = float(gasteiger[h_idx] + qg_h_sum)
            qm_with_h = float(mmff[h_idx] + qm_h_sum)

        if h_list:
            h_dist = np.linalg.norm(coords_h[h_list] - coords_i[None, :], axis=1)
            h_mean = float(np.mean(h_dist))
            h_max = float(np.max(h_dist))
        else:
            h_mean = 0.0
            h_max = 0.0

        if include_h:
            bond_dist = _heavy_bond_distances(mol_h, coords_h, h_idx)
            g_nbr_sum, g_pol = _local_charge_terms(mol_h, coords_h, gasteiger, h_idx)
            m_nbr_sum, m_pol = _local_charge_terms(mol_h, coords_h, mmff, h_idx)
        else:
            bond_dist = _heavy_bond_distances(mol, coords_h[:n_heavy], h_idx)
            g_nbr_sum, g_pol = _local_charge_terms(mol, coords_h[:n_heavy], gasteiger[:n_heavy], h_idx)
            m_nbr_sum, m_pol = _local_charge_terms(mol, coords_h[:n_heavy], mmff[:n_heavy], h_idx)

        if bond_dist:
            nearest = float(np.min(bond_dist))
            mean_bond = float(np.mean(bond_dist))
            min_bond = nearest
            max_bond = float(np.max(bond_dist))
            inv_sum = float(np.sum(1.0 / np.maximum(bond_dist, 1.0e-12)))
        else:
            nearest = mean_bond = min_bond = max_bond = inv_sum = 0.0

        r_geom = float(np.linalg.norm(coords_i - geom_centroid))
        r_mass = float(np.linalg.norm(coords_i - mass_centroid))
        r_heavy = float(np.linalg.norm(coords_i - heavy_centroid))

        features.append(
            [
                float(z),
                float(atom_h.GetMass()),
                float(pt.GetRcovalent(z)),
                float(pt.GetRvdw(z)),
                float(pt.GetDefaultValence(z)),
                float(pt.GetNOuterElecs(z)),
                float(atom.GetDegree()),
                float(atom_h.GetDegree()),
                float(atom.GetTotalValence()),
                float(atom.GetFormalCharge()),
                float(atom.GetTotalNumHs()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                float(atom.GetOwningMol().GetRingInfo().NumAtomRings(atom.GetIdx())),
                float(int(atom.GetHybridization())),
                float(int(atom.GetChiralTag())),
                float(atom.GetNumRadicalElectrons()),
                r_geom,
                r_mass,
                r_heavy,
                nearest,
                mean_bond,
                min_bond,
                max_bond,
                inv_sum,
                float(len(h_list)),
                h_mean,
                h_max,
                float(gasteiger[h_idx]),
                qg_with_h,
                qg_h_sum,
                abs(qg_with_h),
                qg_with_h * r_mass,
                abs(qg_with_h) * r_mass,
                g_nbr_sum,
                g_pol,
                float(mmff[h_idx]),
                qm_with_h,
                qm_h_sum,
                abs(qm_with_h),
                qm_with_h * r_mass,
                abs(qm_with_h) * r_mass,
                m_nbr_sum,
                m_pol,
            ]
        )

    return np.asarray(features, dtype=np.float64), np.asarray(coords_out, dtype=np.float64), atom_mode


def _edge_tensors(
    mol,
    coords: np.ndarray,
    gasteiger: np.ndarray,
    mmff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    senders: list[int] = []
    receivers: list[int] = []
    attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        d = float(np.linalg.norm(coords[i] - coords[j]))
        feat = [
            d,
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsAromatic()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            abs(float(gasteiger[i] - gasteiger[j])),
            abs(float(mmff[i] - mmff[j])),
        ]
        senders.extend([i, j])
        receivers.extend([j, i])
        attrs.extend([feat, feat])
    edge_index = np.asarray([senders, receivers], dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float64) if attrs else np.zeros((0, len(EDGE_FEATURE_NAMES)))
    return edge_index, edge_attr


def dipole_atom_feature_tensors(
    smiles: str,
    *,
    seed: int = 42,
    include_h: bool = False,
) -> DipoleFeatureTensors:
    """Build atom-level descriptors for dipole-moment learning.

    By default the rows are heavy atoms only, matching Chemprop's usual
    implicit-H graph for a standard SMILES string. Hydrogen charge and geometry
    are still embedded and folded onto the attached heavy atom.
    """

    mol, mol_h = _embed_explicit_h_mol(smiles, seed=seed)
    coords_h = _coords_ang(mol_h)
    gasteiger = _gasteiger_charges(mol_h)
    mmff = _mmff_charges(mol_h)
    atom_features, coords_out, atom_mode = _base_atom_features(
        mol,
        mol_h,
        coords_h,
        gasteiger,
        mmff,
        include_h=include_h,
    )

    masses_h = np.asarray([float(a.GetMass()) for a in mol_h.GetAtoms()], dtype=np.float64)
    heavy_mask = np.asarray([int(a.GetAtomicNum()) > 1 for a in mol_h.GetAtoms()], dtype=bool)
    _, mass_centroid, _ = _centroids(coords_h, masses_h, heavy_mask)
    centered = coords_h - mass_centroid[None, :]
    radius_gyration = float(np.sqrt(np.sum(masses_h * np.sum(centered * centered, axis=1)) / np.sum(masses_h)))
    mol_features = np.asarray(
        [
            atom_features.shape[0],
            mol_h.GetNumAtoms(),
            mol.GetNumBonds() if not include_h else mol_h.GetNumBonds(),
            sum(float(a.GetMass()) for a in mol_h.GetAtoms()),
            sum(float(a.GetFormalCharge()) for a in mol.GetAtoms()),
            radius_gyration,
            _dipole_proxy(gasteiger, coords_h, mass_centroid),
            _dipole_proxy(mmff, coords_h, mass_centroid),
            float(np.sum(np.abs(gasteiger))),
            float(np.sum(np.abs(mmff))),
        ],
        dtype=np.float64,
    )

    if include_h:
        edge_mol = mol_h
        edge_coords = coords_h
        edge_gasteiger = gasteiger
        edge_mmff = mmff
    else:
        n_heavy = mol.GetNumAtoms()
        edge_mol = mol
        edge_coords = coords_h[:n_heavy]
        edge_gasteiger = gasteiger[:n_heavy]
        edge_mmff = mmff[:n_heavy]
    edge_index, edge_attr = _edge_tensors(edge_mol, edge_coords, edge_gasteiger, edge_mmff)

    return DipoleFeatureTensors(
        smiles=smiles,
        atom_features=atom_features,
        atom_feature_names=tuple(ATOM_FEATURE_NAMES),
        molecule_features=mol_features,
        molecule_feature_names=tuple(MOLECULE_FEATURE_NAMES),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_feature_names=tuple(EDGE_FEATURE_NAMES),
        atom_coords_ang=coords_out,
        atom_mode=atom_mode,
    )


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _write_csv_rows(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_dipole_atom_feature_dataset(
    input_csv: str | Path,
    out_dir: str | Path,
    *,
    smiles_col: str = "smiles",
    target_cols: list[str] | None = None,
    seed: int = 42,
    include_h: bool = False,
    max_rows: int | None = None,
    strict: bool = False,
    write_graph_tensors: bool = True,
    progress_every: int = 0,
) -> dict[str, object]:
    """Export Chemprop-style atom feature arrays from a dipole CSV.

    The main artifact is ``atom_features.npz``, saved with ``np.savez(...,
    *arrays)`` so that each array is in the same order as the output CSV.
    """

    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = _read_csv_rows(input_csv)
    if smiles_col not in fieldnames:
        raise ValueError(f"{smiles_col!r} column not found in {input_csv}")
    if max_rows is not None:
        rows = rows[: int(max_rows)]

    atom_arrays: list[np.ndarray] = []
    mol_arrays: list[np.ndarray] = []
    output_rows: list[dict[str, object]] = []
    graph_kwargs: dict[str, np.ndarray] = {}
    failures: list[dict[str, object]] = []
    t0 = time.perf_counter()

    for idx, row in enumerate(rows):
        smiles = row.get(smiles_col, "")
        try:
            tensors = dipole_atom_feature_tensors(smiles, seed=seed, include_h=include_h)
        except Exception as exc:
            failure = {"row": idx, "smiles": smiles, "error": str(exc)}
            failures.append(failure)
            if strict:
                raise RuntimeError(f"failed at row {idx} smiles={smiles!r}: {exc}") from exc
            continue

        feature_idx = len(atom_arrays)
        atom_arrays.append(tensors.atom_features)
        mol_arrays.append(tensors.molecule_features)
        if write_graph_tensors:
            graph_kwargs[f"edge_index_{feature_idx}"] = tensors.edge_index
            graph_kwargs[f"edge_attr_{feature_idx}"] = tensors.edge_attr
            graph_kwargs[f"atom_coords_ang_{feature_idx}"] = tensors.atom_coords_ang

        out_row = dict(row)
        out_row["feature_index"] = feature_idx
        out_row["n_atom_features"] = tensors.atom_features.shape[0]
        output_rows.append(out_row)

        if progress_every and (idx + 1) % int(progress_every) == 0:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / elapsed if elapsed > 0.0 else 0.0
            print(
                f"[dipole-features] processed {idx + 1}/{len(rows)} "
                f"exported={len(atom_arrays)} failed={len(failures)} "
                f"rate={rate:.1f} mol/s",
                flush=True,
            )

    np.savez(out_dir / "atom_features.npz", *atom_arrays)
    np.savez(out_dir / "molecule_features.npz", *mol_arrays)
    if write_graph_tensors:
        np.savez(out_dir / "graph_tensors.npz", **graph_kwargs)

    out_fields = list(fieldnames)
    for extra in ("feature_index", "n_atom_features"):
        if extra not in out_fields:
            out_fields.append(extra)
    _write_csv_rows(out_dir / "dipole_chemprop.csv", output_rows, out_fields)

    metadata = {
        "input_csv": str(input_csv),
        "smiles_col": smiles_col,
        "target_cols": target_cols or [],
        "seed": int(seed),
        "include_h": bool(include_h),
        "write_graph_tensors": bool(write_graph_tensors),
        "atom_mode": "explicit_h_atoms" if include_h else "implicit_h_heavy_atoms",
        "n_input": len(rows),
        "n_exported": len(atom_arrays),
        "n_failed": len(failures),
        "failures": failures,
        "atom_feature_names": list(ATOM_FEATURE_NAMES),
        "molecule_feature_names": list(MOLECULE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "files": {
            "chemprop_csv": "dipole_chemprop.csv",
            "atom_features": "atom_features.npz",
            "molecule_features": "molecule_features.npz",
            "graph_tensors": "graph_tensors.npz" if write_graph_tensors else None,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
