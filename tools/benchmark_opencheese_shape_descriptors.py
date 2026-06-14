#!/usr/bin/env python
"""Benchmark alignment-free shape descriptors against an openCHEESE teacher.

The first descriptor implemented here is RDKit USR/USRCAT. Descriptors are
computed for every cached conformer, molecule-pair scores keep the best
conformer-pair score, then the resulting matrix is compared to the shape
channel of an existing teacher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from tools.prepare_cheese_conformer_ensembles import atom_order_mapping_from_dataset_to_rdkit
from tools.train_cheese_projection import _rankdata_np, _retrieval_metrics_np


DEFAULT_ENSEMBLES = Path("outputs/cheese_projection/opencheese_old1k_plus_medchem4k_ensembles_k10_q_resp_cached.npz")
DEFAULT_TEACHER = Path("outputs/cheese_projection/opencheese_old1k_plus_medchem4k_teacher_k10_bestpair_principal_carbo.npz")
DEFAULT_OUT = Path("outputs/cheese_projection/opencheese_usrcat_vs_shape_teacher_5k.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembles", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--descriptor", choices=["usrcat", "usr", "surface_usr"], default="usrcat")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--descriptor-cache", type=Path, default=None)
    parser.add_argument("--reuse-descriptor-cache", action="store_true")
    parser.add_argument("--surface-method", choices=["shell", "ses"], default="shell")
    parser.add_argument("--surface-points", type=int, default=256)
    parser.add_argument("--surface-point-density", type=float, default=0.35)
    parser.add_argument("--surface-grid-spacing", type=float, default=0.40)
    parser.add_argument("--surface-probe-radius", type=float, default=1.4)
    parser.add_argument(
        "--skip-map-failures",
        action="store_true",
        help="Drop rows whose explicit-H SMILES cannot be mapped to the cached atom order.",
    )
    return parser.parse_args()


def _load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata_json" not in data.files:
        return {}
    return json.loads(str(data["metadata_json"][0]))


def _bond_matrix(data: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    atom0, atom1 = int(data["atom_offsets"][index]), int(data["atom_offsets"][index + 1])
    n_atoms = atom1 - atom0
    out = np.zeros((n_atoms, n_atoms), dtype=np.int32)
    bond0, bond1 = int(data["bond_offsets"][index]), int(data["bond_offsets"][index + 1])
    i = data["bond_i"][bond0:bond1]
    j = data["bond_j"][bond0:bond1]
    state = data["bond_state"][bond0:bond1]
    out[i, j] = state
    return out


def _rdkit_mol_for_cached_coords(
    smiles: str,
    atoms: np.ndarray,
    bonds: np.ndarray,
    coords: np.ndarray,
) -> object:
    """Return an RDKit molecule with ``coords`` attached in cached atom order.

    The cached conformer coordinates are stored in the charge-dataset atom
    order. The explicit-H SMILES carries the better chemistry annotations
    (formal charges, aromaticity, donor/acceptor context), so we map dataset
    atom indices back to the parsed RDKit atom indices before calling USRCAT.
    """

    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")
    Chem.SanitizeMol(mol, catchErrors=True)
    atom_map = atom_order_mapping_from_dataset_to_rdkit(atoms, bonds, mol)
    if mol.GetNumAtoms() != len(atoms):
        raise ValueError(f"mapped molecule has {mol.GetNumAtoms()} atoms, expected {len(atoms)}")

    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetId(0)
    for dataset_index, rdkit_index in enumerate(atom_map):
        x, y, z = coords[int(dataset_index)]
        conf.SetAtomPosition(int(rdkit_index), Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=False)
    return mol


def _descriptor_for_mol(mol: object, descriptor: str) -> np.ndarray:
    from rdkit.Chem import rdMolDescriptors

    if descriptor == "usrcat":
        values = rdMolDescriptors.GetUSRCAT(mol, confId=0)
    elif descriptor == "usr":
        values = rdMolDescriptors.GetUSR(mol, confId=0)
    else:  # pragma: no cover - argparse keeps this unreachable.
        raise ValueError(f"unknown descriptor {descriptor!r}")
    return np.asarray(values, dtype=np.float32)


def _usr_descriptor_from_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError("surface points must have shape (n_points, 3)")
    centroid = pts.mean(axis=0)
    d_centroid = np.linalg.norm(pts - centroid[None, :], axis=1)
    closest = pts[int(np.argmin(d_centroid))]
    farthest = pts[int(np.argmax(d_centroid))]
    d_farthest = np.linalg.norm(pts - farthest[None, :], axis=1)
    farthest_from_farthest = pts[int(np.argmax(d_farthest))]
    refs = np.stack([centroid, closest, farthest, farthest_from_farthest], axis=0)
    out = []
    for ref in refs:
        distances = np.linalg.norm(pts - ref[None, :], axis=1).astype(np.float64)
        mean = float(np.mean(distances))
        centered = distances - mean
        variance = float(np.mean(centered * centered))
        third = float(np.mean(centered * centered * centered))
        out.extend([mean, float(np.sqrt(max(variance, 0.0))), float(np.cbrt(third))])
    return np.asarray(out, dtype=np.float32)


def _surface_usr_descriptor(
    atoms: np.ndarray,
    coords: np.ndarray,
    *,
    method: str,
    points_num: int,
    point_density: float,
    grid_spacing: float,
    probe_radius: float,
    random_seed: int,
) -> np.ndarray:
    from mlxmolkit.connolly import connolly_ses_surface_from_atoms_mlx
    from mlxmolkit.esp_resp import connolly_surface_grid
    from opencheese.surface import surface_sample_indices

    atom_array = np.asarray(atoms, dtype=np.int32)
    coord_array = np.asarray(coords, dtype=np.float32)
    if method == "shell":
        raw_points = connolly_surface_grid(
            atom_array,
            coord_array,
            point_density=float(point_density),
            min_points_per_shell=16,
            max_points_per_shell=128,
        ).astype(np.float32)
    elif method == "ses":
        surface = connolly_ses_surface_from_atoms_mlx(
            atom_array,
            coord_array,
            probe_radius=float(probe_radius),
            grid_spacing=float(grid_spacing),
        )
        raw_points = np.asarray(surface.vertices, dtype=np.float32)
    else:  # pragma: no cover - argparse keeps this unreachable.
        raise ValueError(f"unknown surface method {method!r}")
    indices = surface_sample_indices(
        raw_points,
        int(points_num),
        mode="farthest",
        random_seed=int(random_seed),
    )
    return _usr_descriptor_from_points(raw_points[indices])


def compute_descriptor_cache(
    ensembles_path: Path,
    *,
    descriptor: str,
    limit: int,
    skip_map_failures: bool,
    surface_method: str,
    surface_points: int,
    surface_point_density: float,
    surface_grid_spacing: float,
    surface_probe_radius: float,
) -> dict[str, np.ndarray | dict[str, object]]:
    data = np.load(ensembles_path, allow_pickle=False)
    eligible = np.flatnonzero(data["ok"].astype(bool) & (data["n_conformers"].astype(np.int32) > 0))
    if limit > 0:
        eligible = eligible[: int(limit)]

    descriptors: list[np.ndarray] = []
    source_indices: list[int] = []
    ids: list[str] = []
    smiles_out: list[str] = []
    n_conformers: list[int] = []
    desc_offsets = [0]
    failed_rows: list[int] = []
    failed_errors: list[str] = []

    ids_all = data["ids"].astype(str)
    smiles_all = data["smiles"].astype(str)
    start_time = time.perf_counter()
    print(
        f"Computing {descriptor.upper()} descriptors for {len(eligible)} molecules from {ensembles_path}",
        flush=True,
    )
    for out_index, cache_index in enumerate(eligible):
        cache_index = int(cache_index)
        atom0, atom1 = int(data["atom_offsets"][cache_index]), int(data["atom_offsets"][cache_index + 1])
        coord0, coord1 = int(data["coord_offsets"][cache_index]), int(data["coord_offsets"][cache_index + 1])
        atoms = data["atomic_numbers"][atom0:atom1].astype(np.int32)
        n_atoms = int(len(atoms))
        n_conf = int(data["n_conformers"][cache_index])
        coords = data["coords"][coord0:coord1].reshape(n_conf, n_atoms, 3).astype(np.float32)
        bonds = _bond_matrix(data, cache_index)
        mol_desc: list[np.ndarray] = []
        try:
            for conf_index, conf_coords in enumerate(coords):
                if descriptor == "surface_usr":
                    mol_desc.append(
                        _surface_usr_descriptor(
                            atoms,
                            conf_coords,
                            method=surface_method,
                            points_num=int(surface_points),
                            point_density=float(surface_point_density),
                            grid_spacing=float(surface_grid_spacing),
                            probe_radius=float(surface_probe_radius),
                            random_seed=cache_index * 1000 + conf_index,
                        )
                    )
                else:
                    mol = _rdkit_mol_for_cached_coords(smiles_all[cache_index], atoms, bonds, conf_coords)
                    mol_desc.append(_descriptor_for_mol(mol, descriptor))
        except Exception as exc:  # noqa: BLE001
            if not skip_map_failures:
                raise
            failed_rows.append(cache_index)
            failed_errors.append(f"{type(exc).__name__}: {exc}")
            continue

        if not mol_desc:
            if not skip_map_failures:
                raise ValueError(f"no descriptors produced for cache row {cache_index}")
            failed_rows.append(cache_index)
            failed_errors.append("no descriptors produced")
            continue
        block = np.stack(mol_desc, axis=0).astype(np.float32)
        descriptors.append(block)
        desc_offsets.append(desc_offsets[-1] + int(block.shape[0]))
        source_indices.append(int(data["source_indices"][cache_index]))
        ids.append(str(ids_all[cache_index]))
        smiles_out.append(str(smiles_all[cache_index]))
        n_conformers.append(int(block.shape[0]))
        if out_index % 250 == 0:
            print(f"  descriptors {out_index:05d}/{len(eligible):05d}", flush=True)

    if descriptors:
        descriptor_array = np.concatenate(descriptors, axis=0).astype(np.float32)
    else:
        dim = 60 if descriptor == "usrcat" else 12
        descriptor_array = np.empty((0, dim), dtype=np.float32)
    elapsed = time.perf_counter() - start_time
    metadata = {
        "format": "opencheese.alignment_free_shape_descriptor_cache",
        "format_version": 1,
        "ensembles": str(ensembles_path),
        "descriptor": descriptor,
        "surface_method": surface_method if descriptor == "surface_usr" else None,
        "surface_points": int(surface_points) if descriptor == "surface_usr" else None,
        "surface_point_density": float(surface_point_density) if descriptor == "surface_usr" else None,
        "surface_grid_spacing": float(surface_grid_spacing) if descriptor == "surface_usr" else None,
        "surface_probe_radius": float(surface_probe_radius) if descriptor == "surface_usr" else None,
        "limit": int(limit),
        "n_input_eligible": int(len(eligible)),
        "n_molecules": int(len(source_indices)),
        "n_conformer_descriptors": int(descriptor_array.shape[0]),
        "n_failed": int(len(failed_rows)),
        "seconds": elapsed,
    }
    return {
        "metadata": metadata,
        "source_indices": np.asarray(source_indices, dtype=np.int64),
        "ids": np.asarray(ids, dtype=str),
        "smiles": np.asarray(smiles_out, dtype=str),
        "n_conformers": np.asarray(n_conformers, dtype=np.int32),
        "descriptor_offsets": np.asarray(desc_offsets, dtype=np.int64),
        "descriptors": descriptor_array,
        "failed_rows": np.asarray(failed_rows, dtype=np.int64),
        "failed_errors": np.asarray(failed_errors, dtype=str),
    }


def save_descriptor_cache(path: Path, cache: dict[str, np.ndarray | dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array([json.dumps(cache["metadata"], sort_keys=True)], dtype=str),
        source_indices=cache["source_indices"],
        ids=cache["ids"],
        smiles=cache["smiles"],
        n_conformers=cache["n_conformers"],
        descriptor_offsets=cache["descriptor_offsets"],
        descriptors=cache["descriptors"],
        failed_rows=cache["failed_rows"],
        failed_errors=cache["failed_errors"],
    )


def load_descriptor_cache(path: Path) -> dict[str, np.ndarray | dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    return {
        "metadata": _load_metadata(data),
        "source_indices": data["source_indices"].astype(np.int64),
        "ids": data["ids"].astype(str),
        "smiles": data["smiles"].astype(str),
        "n_conformers": data["n_conformers"].astype(np.int32),
        "descriptor_offsets": data["descriptor_offsets"].astype(np.int64),
        "descriptors": data["descriptors"].astype(np.float32),
        "failed_rows": data["failed_rows"].astype(np.int64),
        "failed_errors": data["failed_errors"].astype(str),
    }


def _usr_score_matrix(row_desc: np.ndarray, col_desc: np.ndarray) -> np.ndarray:
    # RDKit GetUSRScore divides the L1 descriptor difference by 12 even for
    # 60-D USRCAT descriptors. This vectorized form matches RDKit exactly.
    l1 = np.sum(np.abs(row_desc[:, None, :] - col_desc[None, :, :]), axis=2, dtype=np.float32)
    return (1.0 / (1.0 + l1 / 12.0)).astype(np.float32)


def _best_owner_scores(
    scores: np.ndarray,
    row_owner: np.ndarray,
    col_owner: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    out = np.full((n_rows, n_cols), -np.inf, dtype=np.float32)
    for i in range(n_rows):
        row_mask = row_owner == i
        if not np.any(row_mask):
            continue
        for j in range(n_cols):
            block = scores[np.ix_(row_mask, col_owner == j)]
            if block.size:
                out[i, j] = float(np.max(block))
    return out


def best_conformer_similarity_matrix(cache: dict[str, np.ndarray | dict[str, object]], *, tile_size: int) -> np.ndarray:
    descriptors = np.asarray(cache["descriptors"], dtype=np.float32)
    offsets = np.asarray(cache["descriptor_offsets"], dtype=np.int64)
    n = int(len(offsets) - 1)
    out = np.zeros((n, n), dtype=np.float32)
    start_time = time.perf_counter()
    print(f"Computing best-conformer USR similarity matrix n={n} tile={tile_size}", flush=True)
    for row0 in range(0, n, tile_size):
        row1 = min(n, row0 + tile_size)
        row_desc = descriptors[int(offsets[row0]) : int(offsets[row1])]
        row_owner = np.concatenate(
            [np.full((int(offsets[i + 1] - offsets[i]),), i - row0, dtype=np.int32) for i in range(row0, row1)]
        )
        for col0 in range(row0, n, tile_size):
            col1 = min(n, col0 + tile_size)
            col_desc = descriptors[int(offsets[col0]) : int(offsets[col1])]
            col_owner = np.concatenate(
                [np.full((int(offsets[j + 1] - offsets[j]),), j - col0, dtype=np.int32) for j in range(col0, col1)]
            )
            score = _usr_score_matrix(row_desc, col_desc)
            block = _best_owner_scores(score, row_owner, col_owner, row1 - row0, col1 - col0)
            out[row0:row1, col0:col1] = block
            if col0 != row0:
                out[col0:col1, row0:row1] = block.T
        print(f"  matrix rows {row0:05d}:{row1:05d}", flush=True)
    elapsed = time.perf_counter() - start_time
    print(f"Computed descriptor matrix in {elapsed:.2f}s", flush=True)
    return out


def _offdiag_values(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return np.asarray(matrix[mask], dtype=np.float64)


def compare_to_teacher(
    descriptor_sim: np.ndarray,
    cache: dict[str, np.ndarray | dict[str, object]],
    teacher_path: Path,
) -> tuple[np.ndarray, dict[str, float]]:
    teacher = np.load(teacher_path, allow_pickle=False)
    source_indices = np.asarray(cache["source_indices"], dtype=np.int64)
    teacher_source = teacher["source_indices"].astype(np.int64)
    teacher_ids = teacher["ids"].astype(str)
    teacher_smiles = teacher["smiles"].astype(str) if "smiles" in teacher.files else None

    source_to_teacher = {int(src): i for i, src in enumerate(teacher_source)}
    try:
        teacher_positions = np.asarray([source_to_teacher[int(src)] for src in source_indices], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"descriptor source index {exc.args[0]} is absent from teacher {teacher_path}") from exc
    if not np.array_equal(teacher_ids[teacher_positions], np.asarray(cache["ids"], dtype=str)):
        raise ValueError("descriptor IDs do not align with teacher IDs after source-index mapping")
    if teacher_smiles is not None and not np.array_equal(teacher_smiles[teacher_positions], np.asarray(cache["smiles"], dtype=str)):
        raise ValueError("descriptor SMILES do not align with teacher SMILES after source-index mapping")

    teacher_shape = teacher["shape"].astype(np.float32)[np.ix_(teacher_positions, teacher_positions)]
    x = _offdiag_values(descriptor_sim)
    y = _offdiag_values(teacher_shape)
    if x.size > 1 and np.std(x) > 1.0e-12 and np.std(y) > 1.0e-12:
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman_global = float(np.corrcoef(_rankdata_np(x), _rankdata_np(y))[0, 1])
    else:
        pearson = 0.0
        spearman_global = 0.0
    retrieval = _retrieval_metrics_np(1.0 - descriptor_sim, teacher_shape)
    metrics = {
        "pearson_offdiag": pearson,
        "spearman_offdiag": spearman_global,
        "mean_descriptor_similarity": float(np.mean(x)) if x.size else 0.0,
        "mean_teacher_shape": float(np.mean(y)) if y.size else 0.0,
        "min_descriptor_similarity": float(np.min(x)) if x.size else 0.0,
        "max_descriptor_similarity": float(np.max(x)) if x.size else 0.0,
        "retrieval_spearman": float(retrieval["spearman"]),
        "recall_at_5": float(retrieval["recall_at_5"]),
        "recall_at_10": float(retrieval["recall_at_10"]),
        "ndcg_at_5": float(retrieval["ndcg_at_5"]),
        "ndcg_at_10": float(retrieval["ndcg_at_10"]),
        "soft_kl": float(retrieval["soft_kl"]),
        "adaptive_recall": float(retrieval["adaptive_recall"]),
    }
    return teacher_shape, metrics


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")

    descriptor_cache_path = args.descriptor_cache
    if descriptor_cache_path is None:
        stem = args.out.with_suffix("")
        descriptor_cache_path = stem.parent / f"{stem.name}_{args.descriptor}_descriptors.npz"
    if args.reuse_descriptor_cache and descriptor_cache_path.exists():
        cache = load_descriptor_cache(descriptor_cache_path)
        print(f"Loaded descriptor cache {descriptor_cache_path}", flush=True)
    else:
        cache = compute_descriptor_cache(
            args.ensembles,
            descriptor=args.descriptor,
            limit=int(args.limit),
            skip_map_failures=bool(args.skip_map_failures),
            surface_method=args.surface_method,
            surface_points=int(args.surface_points),
            surface_point_density=float(args.surface_point_density),
            surface_grid_spacing=float(args.surface_grid_spacing),
            surface_probe_radius=float(args.surface_probe_radius),
        )
        save_descriptor_cache(descriptor_cache_path, cache)
        print(f"Wrote descriptor cache {descriptor_cache_path}", flush=True)

    descriptor_sim = best_conformer_similarity_matrix(cache, tile_size=int(args.tile_size))
    teacher_shape, metrics = compare_to_teacher(descriptor_sim, cache, args.teacher)
    metadata = {
        "format": "opencheese.alignment_free_shape_descriptor_benchmark",
        "format_version": 1,
        "ensembles": str(args.ensembles),
        "teacher": str(args.teacher),
        "descriptor_cache": str(descriptor_cache_path),
        "descriptor": args.descriptor,
        "surface_method": args.surface_method if args.descriptor == "surface_usr" else None,
        "surface_points": int(args.surface_points) if args.descriptor == "surface_usr" else None,
        "limit": int(args.limit),
        "tile_size": int(args.tile_size),
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)], dtype=str),
        source_indices=cache["source_indices"],
        ids=cache["ids"],
        smiles=cache["smiles"],
        descriptor_similarity=descriptor_sim.astype(np.float32),
        teacher_shape=teacher_shape.astype(np.float32),
    )
    print(f"Wrote {args.out}", flush=True)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
