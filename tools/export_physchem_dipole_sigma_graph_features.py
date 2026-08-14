#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from mlxmolkit.dipole_features import EANG_TO_DEBYE


PAPER_GRID = np.round(np.arange(-0.030, 0.0301, 0.001), 6)
CHAOS_PROFILE_GRID = np.round(np.arange(-0.025, 0.0251, 0.001), 6)
BOHR_TO_ANG = 0.5291772108


ATOM_FEATURE_NAMES = (
    [
        "atomic_number",
        "atomic_mass",
        "covalent_radius",
        "vdw_radius",
        "default_valence",
        "outer_electrons",
        "x_centered_ang",
        "y_centered_ang",
        "z_centered_ang",
        "r_geom_centroid_ang",
        "r_mass_centroid_ang",
        "mulliken_charge",
        "apt_charge",
        "cosmo_atom_charge",
        "cosmo_atom_area",
        "cosmo_atom_sigma",
        "segment_area_sum",
        "segment_charge_sum",
        "segment_sigma_mean",
        "segment_abs_sigma_mean",
        "segment_sigma_var",
        "segment_positive_area_fraction",
        "segment_negative_area_fraction",
        "segment_potential_mean",
        "segment_centroid_norm_ang",
        "segment_mean_radius_ang",
        "segment_rms_radius_ang",
        "segment_cov_eig1",
        "segment_cov_eig2",
        "segment_cov_eig3",
    ]
    + [f"atom_sigma_profile_{i:02d}_A2" for i in range(PAPER_GRID.size)]
    + [f"atom_sigma_potential_weighted_{i:02d}" for i in range(PAPER_GRID.size)]
)

MOLECULE_FEATURE_NAMES = [
    "n_atoms_explicit",
    "n_heavy_atoms",
    "n_inferred_bonds",
    "molecular_weight",
    "formal_charge",
    "radius_gyration_ang",
    "chaos_dipole_debye",
    "mulliken_dipole_proxy_debye",
    "apt_dipole_proxy_debye",
    "cosmo_charge_abs_sum",
    "cosmo_area_A2",
    "cosmo_volume_A3",
]

EDGE_FEATURE_NAMES = [
    "distance_ang",
    "covalent_ratio",
    "is_hydrogen_edge",
    "abs_d_mulliken_charge",
    "abs_d_apt_charge",
    "abs_d_cosmo_charge",
]

ANGLE_FEATURE_NAMES = ["angle_rad", "cos_angle", "sin_angle"]


def _canonical_smiles(smiles: str, *, isomeric: bool = True) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def _load_chaos_index(path: Path) -> tuple[dict[str, int], dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    chaos_ids = np.asarray(data["chaos_ids"])
    smiles = np.asarray(data["canonical_smiles"])
    sigma_grid = np.asarray(data["sigma_grid_e_per_A2"], dtype=np.float64)
    mu = np.asarray(data["mu_J_per_mol"], dtype=np.float64)
    iso_index = {str(s): i for i, s in enumerate(smiles)}
    no_stereo_index: dict[str, int] = {}
    for i, smi in enumerate(smiles):
        key = _canonical_smiles(str(smi), isomeric=False)
        if key and key not in no_stereo_index:
            no_stereo_index[key] = i
    return iso_index, no_stereo_index, chaos_ids, sigma_grid, mu


def _match_chaos_idx(
    canonical_iso: str,
    canonical_no_stereo: str,
    iso_index: dict[str, int],
    no_stereo_index: dict[str, int],
) -> tuple[int, str]:
    if canonical_iso in iso_index:
        return iso_index[canonical_iso], "isomeric"
    if canonical_no_stereo in no_stereo_index:
        return no_stereo_index[canonical_no_stereo], "no_stereo"
    return -1, "none"


def _chaos_profile_from_zip(zf: zipfile.ZipFile, chaos_id: str, sigma_grid: np.ndarray) -> np.ndarray:
    with zf.open(f"{chaos_id}.json") as f:
        data = json.loads(f.read())
    sig_total = np.asarray(data["solvation"]["Sigma_total"], dtype=np.float64)
    return np.interp(sigma_grid, CHAOS_PROFILE_GRID, sig_total, left=0.0, right=0.0)


def _regular_grid_edges(grid: np.ndarray) -> np.ndarray:
    dx = float(grid[1] - grid[0])
    edges = np.empty(grid.size + 1, dtype=np.float64)
    edges[0] = grid[0] - 0.5 * dx
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[-1] = grid[-1] + 0.5 * dx
    return edges


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0.0 else 0.0


def _weighted_var(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    total = float(np.sum(weights))
    return float(np.sum(weights * (values - mean) ** 2) / total) if total > 0.0 else 0.0


def _periodic_props(atomic_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    mass = np.asarray([float(pt.GetAtomicWeight(int(z))) for z in atomic_numbers], dtype=np.float64)
    rcov = np.asarray([float(pt.GetRcovalent(int(z))) for z in atomic_numbers], dtype=np.float64)
    rvdw = np.asarray([float(pt.GetRvdw(int(z))) for z in atomic_numbers], dtype=np.float64)
    val_outer = np.asarray(
        [[float(pt.GetDefaultValence(int(z))), float(pt.GetNOuterElecs(int(z)))] for z in atomic_numbers],
        dtype=np.float64,
    )
    return mass, rcov, rvdw, val_outer


def _dipole_proxy(charges: np.ndarray, coords: np.ndarray, origin: np.ndarray) -> float:
    vec = np.sum(charges[:, None] * (coords - origin[None, :]), axis=0)
    return float(np.linalg.norm(vec) * EANG_TO_DEBYE)


def _infer_edges(atomic_numbers: np.ndarray, coords_ang: np.ndarray, rcov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    senders: list[int] = []
    receivers: list[int] = []
    attrs: list[list[float]] = []
    n = len(atomic_numbers)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords_ang[i] - coords_ang[j]))
            cutoff = 1.25 * float(rcov[i] + rcov[j]) + 0.15
            if d <= max(cutoff, 0.45):
                cov_ratio = d / max(float(rcov[i] + rcov[j]), 1.0e-12)
                is_h = float(atomic_numbers[i] == 1 or atomic_numbers[j] == 1)
                senders.extend([i, j])
                receivers.extend([j, i])
                attrs.extend([[d, cov_ratio, is_h], [d, cov_ratio, is_h]])
    return np.asarray([senders, receivers], dtype=np.int64), np.asarray(attrs, dtype=np.float64)


def _angle_tensors(edge_index: np.ndarray, coords_ang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = coords_ang.shape[0]
    neighbors: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in edge_index.T:
        neighbors[int(i)].add(int(j))
    triplets: list[list[int]] = []
    attrs: list[list[float]] = []
    for j in range(n):
        nbrs = sorted(neighbors[j])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i = nbrs[a]
                k = nbrs[b]
                v1 = coords_ang[i] - coords_ang[j]
                v2 = coords_ang[k] - coords_ang[j]
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 <= 1.0e-12 or n2 <= 1.0e-12:
                    angle = 0.0
                else:
                    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    angle = float(np.arccos(c))
                triplets.append([i, j, k])
                attrs.append([angle, float(np.cos(angle)), float(np.sin(angle))])
    if not triplets:
        return np.zeros((3, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float64)
    return np.asarray(triplets, dtype=np.int64).T, np.asarray(attrs, dtype=np.float64)


def _chaos_3d_tensors(data: dict, sigma_grid: np.ndarray) -> dict[str, np.ndarray | float]:
    atom_list = data["general"]["AtomList"]
    atomic_numbers = np.asarray([int(a["atomic_number"]) for a in atom_list], dtype=np.int64)
    coord_source = "Coordinates"
    coords_raw = data["structural"].get("Coordinates")
    if coords_raw is None:
        coords_raw = data["structural"].get("Coordinates_Input")
        coord_source = "Coordinates_Input"
    coords_ang = np.asarray(coords_raw, dtype=np.float64)
    n_atoms = len(atomic_numbers)
    mass, rcov, rvdw, val_outer = _periodic_props(atomic_numbers)

    geom_centroid = np.mean(coords_ang, axis=0)
    mass_centroid = np.sum(coords_ang * mass[:, None], axis=0) / float(np.sum(mass))
    centered = coords_ang - mass_centroid[None, :]
    rg = float(np.sqrt(np.sum(mass * np.sum(centered * centered, axis=1)) / np.sum(mass)))

    mull = np.asarray(data.get("electronic", {}).get("PartChargeMulliken", np.zeros(n_atoms)), dtype=np.float64)
    apt = np.asarray(data.get("electronic", {}).get("PartChargeAPT", np.zeros(n_atoms)), dtype=np.float64)
    if mull.shape != (n_atoms,):
        mull = np.zeros(n_atoms, dtype=np.float64)
    if apt.shape != (n_atoms,):
        apt = np.zeros(n_atoms, dtype=np.float64)

    atom_cosmo = data.get("solvation", {}).get("AtomCOSMOCharge", [])
    cosmo_area = np.zeros(n_atoms, dtype=np.float64)
    cosmo_charge = np.zeros(n_atoms, dtype=np.float64)
    cosmo_sigma = np.zeros(n_atoms, dtype=np.float64)
    for i, row in enumerate(atom_cosmo[:n_atoms]):
        cosmo_area[i] = float(row.get("area", 0.0))
        cosmo_charge[i] = float(row.get("charge", 0.0))
        cosmo_sigma[i] = float(row.get("sigma", 0.0))

    sl = np.asarray(data["solvation"]["SegmentList"], dtype=np.float64)
    seg_atom = sl[:, 1].astype(np.int64) - 1
    seg_xyz_ang = sl[:, 2:5] * BOHR_TO_ANG
    seg_charge = sl[:, 5]
    seg_area = sl[:, 6]
    seg_sigma = sl[:, 7]
    seg_pot = sl[:, 8] if sl.shape[1] > 8 else np.zeros(sl.shape[0], dtype=np.float64)
    edges = _regular_grid_edges(sigma_grid)

    scalar = np.zeros((n_atoms, 14), dtype=np.float64)
    prof = np.zeros((n_atoms, sigma_grid.size), dtype=np.float64)
    pot_prof = np.zeros((n_atoms, sigma_grid.size), dtype=np.float64)
    for i in range(n_atoms):
        mask = seg_atom == i
        area_i = seg_area[mask]
        sigma_i = seg_sigma[mask]
        q_i = seg_charge[mask]
        pot_i = seg_pot[mask]
        rel = seg_xyz_ang[mask] - coords_ang[i][None, :]
        area_total = float(np.sum(area_i))
        sigma_mean = _weighted_mean(sigma_i, area_i)
        sigma_abs = _weighted_mean(np.abs(sigma_i), area_i)
        sigma_var = _weighted_var(sigma_i, area_i, sigma_mean)
        pot_mean = _weighted_mean(pot_i, area_i)
        pos = float(np.sum(area_i[sigma_i > 0.0]) / area_total) if area_total > 0.0 else 0.0
        neg = float(np.sum(area_i[sigma_i < 0.0]) / area_total) if area_total > 0.0 else 0.0
        if rel.size and area_total > 0.0:
            w = area_i / area_total
            centroid = np.sum(w[:, None] * rel, axis=0)
            r = np.linalg.norm(rel, axis=1)
            mean_r = float(np.sum(w * r))
            rms_r = float(np.sqrt(np.sum(w * r * r)))
            cov = (rel - centroid[None, :]).T @ ((rel - centroid[None, :]) * w[:, None])
            eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        else:
            centroid = np.zeros(3)
            mean_r = rms_r = 0.0
            eig = np.zeros(3)
        scalar[i] = [
            area_total,
            float(np.sum(q_i)),
            sigma_mean,
            sigma_abs,
            sigma_var,
            pos,
            neg,
            pot_mean,
            float(np.linalg.norm(centroid)),
            mean_r,
            rms_r,
            float(eig[0]),
            float(eig[1]),
            float(eig[2]),
        ]
        prof[i], _ = np.histogram(sigma_i, bins=edges, weights=area_i)
        pot_prof[i], _ = np.histogram(sigma_i, bins=edges, weights=area_i * pot_i)

    edge_index, edge_base = _infer_edges(atomic_numbers, coords_ang, rcov)
    edge_extra = []
    for i, j in edge_index.T:
        edge_extra.append(
            [
                abs(float(mull[i] - mull[j])),
                abs(float(apt[i] - apt[j])),
                abs(float(cosmo_charge[i] - cosmo_charge[j])),
            ]
        )
    edge_attr = (
        np.concatenate([edge_base, np.asarray(edge_extra, dtype=np.float64)], axis=1)
        if edge_base.size
        else np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float64)
    )
    angle_index, angle_attr = _angle_tensors(edge_index, coords_ang)

    base = np.column_stack(
        [
            atomic_numbers.astype(np.float64),
            mass,
            rcov,
            rvdw,
            val_outer,
            centered,
            np.linalg.norm(coords_ang - geom_centroid[None, :], axis=1),
            np.linalg.norm(coords_ang - mass_centroid[None, :], axis=1),
            mull,
            apt,
            cosmo_charge,
            cosmo_area,
            cosmo_sigma,
            scalar,
        ]
    )
    atom_features = np.concatenate([base, prof, pot_prof], axis=1)
    mol_features = np.asarray(
        [
            n_atoms,
            int(np.sum(atomic_numbers > 1)),
            edge_index.shape[1] // 2,
            float(data["general"].get("MolecularMass", np.sum(mass))),
            float(data.get("electronic", {}).get("Charge", 0)),
            rg,
            float(data.get("electronic", {}).get("DipoleMoment", np.nan)),
            _dipole_proxy(mull, coords_ang, mass_centroid),
            _dipole_proxy(apt, coords_ang, mass_centroid),
            float(np.sum(np.abs(cosmo_charge))),
            float(data["solvation"].get("CavArea", np.nan)),
            float(data["solvation"].get("CavVolume", np.nan)),
        ],
        dtype=np.float64,
    )
    return {
        "atom_features": atom_features,
        "molecule_features": mol_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "angle_index": angle_index,
        "angle_attr": angle_attr,
        "atom_coords_ang": coords_ang,
        "atomic_numbers": atomic_numbers,
        "coordinate_source": coord_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export graph atom features for the calcphyschemprop DipoleMoment "
            "dataset, joined to the local 53k CHAOS sigma-potential matrix."
        )
    )
    parser.add_argument(
        "--dipole-csv",
        type=Path,
        default=Path("/Users/guillaume-osmo/Github/data/cascade_v3.4/preds/DipoleMoment_predictions.csv"),
    )
    parser.add_argument(
        "--chaos-npz",
        type=Path,
        default=Path("data/chaos_25a_mu_matrix.npz"),
    )
    parser.add_argument(
        "--chaos-zip",
        type=Path,
        default=Path("/Users/guillaume-osmo/Github/data/CHAOS.zip"),
        help="optional CHAOS.zip source for P(sigma) rows",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/dipole_physchem_chaos3d_graph"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-uncovered", action="store_true", help="keep rows without CHAOS sigma coverage")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dip = pd.read_csv(args.dipole_csv)
    if args.max_rows is not None:
        dip = dip.iloc[: args.max_rows].copy()
    required = {"smiles", "y_true", "y_pred_final"}
    missing = required - set(dip.columns)
    if missing:
        raise ValueError(f"{args.dipole_csv} missing required columns: {sorted(missing)}")

    chaos_iso_index, chaos_no_stereo_index, chaos_ids, sigma_grid, chaos_mu = _load_chaos_index(args.chaos_npz)
    dip["canonical_smiles"] = [_canonical_smiles(s, isomeric=True) for s in dip["smiles"]]
    dip["canonical_smiles_no_stereo"] = [_canonical_smiles(s, isomeric=False) for s in dip["smiles"]]
    matches = [
        _match_chaos_idx(iso, no, chaos_iso_index, chaos_no_stereo_index)
        for iso, no in zip(dip["canonical_smiles"], dip["canonical_smiles_no_stereo"])
    ]
    dip["chaos_idx"] = [idx for idx, _ in matches]
    dip["chaos_match_mode"] = [mode for _, mode in matches]
    dip["has_chaos_sigma"] = dip["chaos_idx"] >= 0
    if not args.keep_uncovered:
        dip = dip[dip["has_chaos_sigma"]].copy()

    atom_arrays: list[np.ndarray] = []
    mol_arrays: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    profile_rows: list[np.ndarray] = []
    edge_kwargs: dict[str, np.ndarray] = {}
    output_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    zf = zipfile.ZipFile(args.chaos_zip) if args.chaos_zip.exists() else None
    for row_idx, row in dip.reset_index(drop=False).iterrows():
        smiles = str(row["smiles"])
        chaos_idx = int(row["chaos_idx"])
        try:
            if zf is None or chaos_idx < 0:
                raise RuntimeError("CHAOS JSON is required for 3D atom features")
            data = json.loads(zf.read(f"{chaos_ids[chaos_idx]}.json"))
            tensors = _chaos_3d_tensors(data, sigma_grid)
        except Exception as exc:
            failures.append(
                {
                    "input_index": int(row["index"]),
                    "smiles": smiles,
                    "canonical_smiles": str(row["canonical_smiles"]),
                    "error": str(exc),
                }
            )
            continue

        feature_idx = len(atom_arrays)
        atom_arrays.append(tensors["atom_features"])
        mol_arrays.append(tensors["molecule_features"])
        edge_kwargs[f"edge_index_{feature_idx}"] = tensors["edge_index"]
        edge_kwargs[f"edge_attr_{feature_idx}"] = tensors["edge_attr"]
        edge_kwargs[f"angle_index_{feature_idx}"] = tensors["angle_index"]
        edge_kwargs[f"angle_attr_{feature_idx}"] = tensors["angle_attr"]
        edge_kwargs[f"atom_coords_ang_{feature_idx}"] = tensors["atom_coords_ang"]
        edge_kwargs[f"atomic_numbers_{feature_idx}"] = tensors["atomic_numbers"]

        sigma_rows.append(chaos_mu[chaos_idx] if chaos_idx >= 0 else np.full(sigma_grid.shape, np.nan))
        if zf is not None and chaos_idx >= 0:
            profile_rows.append(_chaos_profile_from_zip(zf, str(chaos_ids[chaos_idx]), sigma_grid))
        else:
            profile_rows.append(np.full(sigma_grid.shape, np.nan))

        output_rows.append(
            {
                "feature_index": feature_idx,
                "input_index": int(row["index"]),
                "smiles": smiles,
                "canonical_smiles": str(row["canonical_smiles"]),
                "canonical_smiles_no_stereo": str(row["canonical_smiles_no_stereo"]),
                "chaos_id": str(chaos_ids[chaos_idx]) if chaos_idx >= 0 else "",
                "chaos_match_mode": str(row["chaos_match_mode"]),
                "has_chaos_sigma": bool(chaos_idx >= 0),
                "dipole_debye": float(row["y_true"]),
                "calcphyschemprop_pred_debye": float(row["y_pred_final"]),
                "n_atom_features": int(tensors["atom_features"].shape[0]),
                "chaos_coordinate_source": str(tensors["coordinate_source"]),
            }
        )

        if args.progress_every and (row_idx + 1) % int(args.progress_every) == 0:
            print(
                f"[physchem-dipole] rows={row_idx + 1}/{len(dip)} "
                f"exported={len(atom_arrays)} failed={len(failures)}",
                flush=True,
            )

    if zf is not None:
        zf.close()

    np.savez(out_dir / "atom_features.npz", *atom_arrays)
    np.savez(out_dir / "molecule_features.npz", *mol_arrays)
    np.savez(out_dir / "graph_tensors.npz", **edge_kwargs)
    np.savez(
        out_dir / "sigma_features.npz",
        sigma_grid_e_per_A2=sigma_grid,
        mu_J_per_mol=np.asarray(sigma_rows, dtype=np.float64),
        profile_area_A2=np.asarray(profile_rows, dtype=np.float64),
    )

    csv_path = out_dir / "dipole_sigma_chemprop.csv"
    fieldnames = [
        "feature_index",
        "input_index",
        "smiles",
        "canonical_smiles",
        "canonical_smiles_no_stereo",
        "chaos_id",
        "chaos_match_mode",
        "has_chaos_sigma",
        "dipole_debye",
        "calcphyschemprop_pred_debye",
        "n_atom_features",
        "chaos_coordinate_source",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "dipole_csv": str(args.dipole_csv),
        "chaos_npz": str(args.chaos_npz),
        "chaos_zip": str(args.chaos_zip) if args.chaos_zip.exists() else None,
        "seed": int(args.seed),
        "keep_uncovered": bool(args.keep_uncovered),
        "n_dipole_input": int(len(pd.read_csv(args.dipole_csv))),
        "n_after_sigma_filter": int(len(dip)),
        "n_exported": int(len(atom_arrays)),
        "n_failed": int(len(failures)),
        "failures": failures,
        "atom_mode": "chaos_explicit_atoms_3d",
        "atom_order": "CHAOS general.AtomList order; explicit hydrogens included",
        "atom_feature_names": list(ATOM_FEATURE_NAMES),
        "molecule_feature_names": list(MOLECULE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "angle_feature_names": list(ANGLE_FEATURE_NAMES),
        "sigma_feature_names": [f"mu_sigma_{i:02d}_J_per_mol" for i in range(sigma_grid.size)],
        "sigma_profile_feature_names": [f"profile_sigma_{i:02d}_A2" for i in range(sigma_grid.size)],
        "files": {
            "chemprop_csv": "dipole_sigma_chemprop.csv",
            "atom_features": "atom_features.npz",
            "molecule_features": "molecule_features.npz",
            "graph_tensors": "graph_tensors.npz",
            "sigma_features": "sigma_features.npz",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Exported {len(atom_arrays)} molecules to {out_dir}")
    print(f"  covered rows before feature failures: {len(dip)}")
    print(f"  failures: {len(failures)}")
    print(f"  CSV: {csv_path}")
    print(f"  atom features: {out_dir / 'atom_features.npz'}")
    print(f"  sigma features: {out_dir / 'sigma_features.npz'}")


if __name__ == "__main__":
    main()
