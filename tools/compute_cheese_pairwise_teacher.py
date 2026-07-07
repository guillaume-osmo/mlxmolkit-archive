#!/usr/bin/env python
"""Compute tiled all-pairs openCHEESE teacher matrices from charge data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

import mlx.core as mx

from opencheese.descriptors import cheese_batch, cheese_similarity_matrix_mlx
from mlxmolkit.esp_resp import VDW_RADII
from tools.train_charge_model import ChargeTrainingDataset


DEFAULT_DATASET = Path(
    "data/espaloma_charge_zenodo_17308526/recalculated_charges/"
    "test_random1000_both_symmetrized_partial_bcc_fill/"
    "cheese_charge_training_am1bcc_resp.npz"
)
DEFAULT_OUT = Path(
    "outputs/cheese_projection/cheese_teacher_1000_q_resp_carbo_canonical.npz"
)


def canonicalize_coords(
    atomic_numbers: Sequence[int] | np.ndarray,
    coords: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Return centered/canonicalized coordinates for fast pairwise teachers."""

    xyz = np.asarray(coords, dtype=np.float64)
    if mode == "none":
        return xyz.astype(np.float32)

    atoms = np.asarray(atomic_numbers, dtype=np.int32)
    weights = np.asarray([VDW_RADII.get(int(z), 1.8) ** 3 for z in atoms], dtype=np.float64)
    weights = np.maximum(weights, 1.0e-6)
    centroid = np.sum(xyz * weights[:, None], axis=0) / np.sum(weights)
    centered = xyz - centroid[None, :]
    if mode == "center":
        return centered.astype(np.float32)
    if mode != "principal":
        raise ValueError("canonicalize must be 'none', 'center', or 'principal'")

    cov = (centered * weights[:, None]).T @ centered / np.sum(weights)
    values, vectors = np.linalg.eigh(cov)
    axes = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0

    # Fix eigenvector signs deterministically using weighted skewness. This is
    # not a replacement for full pairwise alignment, but it gives the fast
    # all-pairs teacher a stable common frame.
    projections = centered @ axes
    for axis in range(3):
        skew = float(np.sum(weights * projections[:, axis] ** 3))
        if abs(skew) < 1.0e-10:
            idx = int(np.argmax(np.abs(projections[:, axis])))
            skew = float(projections[idx, axis])
        if skew < 0:
            axes[:, axis] *= -1.0
            projections[:, axis] *= -1.0
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
        projections[:, -1] *= -1.0
    return projections.astype(np.float32)


def make_cheese_batch(
    dataset: ChargeTrainingDataset,
    indices: Sequence[int],
    target: str,
    *,
    canonicalize: str,
    pad_to: int,
):
    atoms = []
    coords = []
    charges = []
    ids = []
    for index in indices:
        z, xyz, _, _, q = dataset.molecule_arrays(int(index), target)
        atoms.append(z)
        coords.append(canonicalize_coords(z, xyz, mode=canonicalize))
        charges.append(q)
        ids.append(str(dataset.ids[int(index)]))
    return cheese_batch(atoms, coords, charges, ids=ids, pad_to=pad_to)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target", default="q_resp", choices=["q_reference", "q_esp", "q_resp"])
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--canonicalize", choices=["none", "center", "principal"], default="principal")
    parser.add_argument("--shape-metric", choices=["carbo", "tanimoto"], default="carbo")
    parser.add_argument("--electrostatic-metric", choices=["carbo", "tanimoto"], default="carbo")
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument("--electrostatic-weight", type=float, default=1.0)
    parser.add_argument("--no-map-electrostatic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")

    dataset = ChargeTrainingDataset(args.data)
    indices = dataset.finite_label_indices(args.target)
    n = int(len(indices))
    if n == 0:
        raise ValueError(f"no molecules with finite {args.target} labels")

    shape = np.zeros((n, n), dtype=np.float32)
    electrostatic = np.zeros((n, n), dtype=np.float32)
    combined = np.zeros((n, n), dtype=np.float32)
    start_time = time.perf_counter()

    print(
        f"Computing {n}x{n} openCHEESE teacher target={args.target} "
        f"tile={args.tile_size} canonicalize={args.canonicalize}",
        flush=True,
    )
    for row0 in range(0, n, args.tile_size):
        row1 = min(n, row0 + args.tile_size)
        row_batch = make_cheese_batch(
            dataset,
            indices[row0:row1],
            args.target,
            canonicalize=args.canonicalize,
            pad_to=dataset.max_atoms,
        )
        for col0 in range(0, n, args.tile_size):
            col1 = min(n, col0 + args.tile_size)
            col_batch = make_cheese_batch(
                dataset,
                indices[col0:col1],
                args.target,
                canonicalize=args.canonicalize,
                pad_to=dataset.max_atoms,
            )
            scores = cheese_similarity_matrix_mlx(
                row_batch,
                col_batch,
                shape_weight=args.shape_weight,
                electrostatic_weight=args.electrostatic_weight,
                map_electrostatic_to_unit=not args.no_map_electrostatic,
                electrostatic_metric=args.electrostatic_metric,
                shape_metric=args.shape_metric,
            )
            mx.eval(scores.shape, scores.electrostatic, scores.combined)
            shape[row0:row1, col0:col1] = np.asarray(scores.shape, dtype=np.float32)
            electrostatic[row0:row1, col0:col1] = np.asarray(scores.electrostatic, dtype=np.float32)
            combined[row0:row1, col0:col1] = np.asarray(scores.combined, dtype=np.float32)
        print(f"  rows {row0:04d}:{row1:04d}", flush=True)

    elapsed = time.perf_counter() - start_time
    metadata = {
        "format": "opencheese.pairwise_teacher",
        "format_version": 1,
        "dataset": str(dataset.path),
        "target": args.target,
        "n_molecules": n,
        "tile_size": args.tile_size,
        "canonicalize": args.canonicalize,
        "shape_metric": args.shape_metric,
        "electrostatic_metric": args.electrostatic_metric,
        "shape_weight": args.shape_weight,
        "electrostatic_weight": args.electrostatic_weight,
        "map_electrostatic_to_unit": not args.no_map_electrostatic,
        "seconds": elapsed,
        "note": "Fast teacher assumes canonicalized/common-frame conformers; exact pairwise Roshambo alignment is a second-stage teacher.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        format_version=np.array([1], dtype=np.int64),
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)], dtype=str),
        source_indices=indices.astype(np.int64),
        ids=dataset.ids[indices].astype(str),
        smiles=dataset.smiles[indices].astype(str),
        shape=shape,
        electrostatic=electrostatic,
        combined=combined,
    )
    print(f"Wrote {args.out} in {elapsed:.2f}s", flush=True)
    print(
        "combined stats: "
        f"min={combined.min():.4f} mean={combined.mean():.4f} max={combined.max():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
