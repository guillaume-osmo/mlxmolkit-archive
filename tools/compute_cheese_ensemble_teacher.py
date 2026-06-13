#!/usr/bin/env python
"""Compute best-conformer-pair openCHEESE teachers from cached ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

import mlx.core as mx

from opencheese.descriptors import cheese_batch, cheese_similarity_matrix_mlx
from tools.compute_cheese_pairwise_teacher import canonicalize_coords


DEFAULT_ENSEMBLES = Path("outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz")
DEFAULT_OUT = Path("outputs/cheese_projection/cheese_teacher_1000_q_resp_k10_bestpair.npz")


class EnsembleCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        data = np.load(self.path, allow_pickle=False)
        self.metadata = json.loads(str(data["metadata_json"][0])) if "metadata_json" in data.files else {}
        self.source_indices = data["source_indices"].astype(np.int64)
        self.ids = data["ids"].astype(str)
        self.smiles = data["smiles"].astype(str)
        self.ok = data["ok"].astype(bool)
        self.n_conformers = data["n_conformers"].astype(np.int32)
        self.atom_offsets = data["atom_offsets"].astype(np.int64)
        self.conformer_offsets = data["conformer_offsets"].astype(np.int64)
        self.coord_offsets = (
            data["coord_offsets"].astype(np.int64)
            if "coord_offsets" in data.files
            else self._legacy_coord_offsets()
        )
        self.atomic_numbers = data["atomic_numbers"].astype(np.int32)
        self.coords = data["coords"].astype(np.float32)
        self.charges = data["charges"].astype(np.float32)

    @property
    def n_molecules(self) -> int:
        return int(len(self.source_indices))

    @property
    def max_atoms(self) -> int:
        sizes = self.atom_offsets[1:] - self.atom_offsets[:-1]
        return int(np.max(sizes)) if len(sizes) else 0

    @property
    def max_conformers(self) -> int:
        return int(np.max(self.n_conformers)) if len(self.n_conformers) else 0

    def molecule(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        atom_start, atom_end = int(self.atom_offsets[index]), int(self.atom_offsets[index + 1])
        conf_start, conf_end = int(self.conformer_offsets[index]), int(self.conformer_offsets[index + 1])
        coord_start, coord_end = int(self.coord_offsets[index]), int(self.coord_offsets[index + 1])
        atoms = self.atomic_numbers[atom_start:atom_end]
        charges = self.charges[atom_start:atom_end]
        n_atoms = len(atoms)
        n_conf = conf_end - conf_start
        coords = self.coords[coord_start:coord_end].reshape(n_conf, n_atoms, 3)
        return atoms, coords, charges

    def _legacy_coord_offsets(self) -> np.ndarray:
        out = np.zeros((len(self.n_conformers) + 1,), dtype=np.int64)
        for i, n_conf in enumerate(self.n_conformers):
            n_atoms = int(self.atom_offsets[i + 1] - self.atom_offsets[i])
            out[i + 1] = out[i] + int(n_conf) * n_atoms
        return out


def conformer_batch(cache: EnsembleCache, indices: Sequence[int], *, canonicalize: str):
    atoms = []
    coords = []
    charges = []
    owners = []
    for local_index, mol_index in enumerate(indices):
        z, xyz, q = cache.molecule(int(mol_index))
        for conf in xyz:
            atoms.append(z)
            coords.append(canonicalize_coords(z, conf, mode=canonicalize))
            charges.append(q)
            owners.append(local_index)
    return cheese_batch(atoms, coords, charges, pad_to=cache.max_atoms), np.asarray(owners, dtype=np.int32)


def best_pair_scores(
    scores: np.ndarray,
    row_owner: np.ndarray,
    col_owner: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    out = np.full((n_rows, n_cols), -np.inf, dtype=np.float32)
    for i in range(n_rows):
        row_mask = row_owner == i
        for j in range(n_cols):
            block = scores[np.ix_(row_mask, col_owner == j)]
            if block.size:
                out[i, j] = float(np.max(block))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembles", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--shape-metric", choices=["carbo", "tanimoto"], default="carbo")
    parser.add_argument("--electrostatic-metric", choices=["carbo", "tanimoto"], default="carbo")
    parser.add_argument("--canonicalize", choices=["none", "center", "principal"], default="principal")
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument("--electrostatic-weight", type=float, default=1.0)
    parser.add_argument("--no-map-electrostatic", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")
    cache = EnsembleCache(args.ensembles)
    eligible = np.flatnonzero(cache.ok & (cache.n_conformers > 0))
    if args.limit > 0:
        eligible = eligible[: int(args.limit)]
    n = int(len(eligible))
    if n == 0:
        raise ValueError("no successful ensemble rows")

    shape = np.zeros((n, n), dtype=np.float32)
    electrostatic = np.zeros((n, n), dtype=np.float32)
    combined = np.zeros((n, n), dtype=np.float32)
    start_time = time.perf_counter()

    print(
        f"Computing best-pair ensemble teacher n={n} k<= {cache.max_conformers} tile={args.tile_size}",
        flush=True,
    )
    for row0 in range(0, n, args.tile_size):
        row1 = min(n, row0 + args.tile_size)
        row_indices = eligible[row0:row1]
        row_batch, row_owner = conformer_batch(cache, row_indices, canonicalize=args.canonicalize)
        for col0 in range(0, n, args.tile_size):
            col1 = min(n, col0 + args.tile_size)
            col_indices = eligible[col0:col1]
            col_batch, col_owner = conformer_batch(cache, col_indices, canonicalize=args.canonicalize)
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
            shape[row0:row1, col0:col1] = best_pair_scores(
                np.asarray(scores.shape, dtype=np.float32),
                row_owner,
                col_owner,
                row1 - row0,
                col1 - col0,
            )
            electrostatic[row0:row1, col0:col1] = best_pair_scores(
                np.asarray(scores.electrostatic, dtype=np.float32),
                row_owner,
                col_owner,
                row1 - row0,
                col1 - col0,
            )
            combined[row0:row1, col0:col1] = best_pair_scores(
                np.asarray(scores.combined, dtype=np.float32),
                row_owner,
                col_owner,
                row1 - row0,
                col1 - col0,
            )
        print(f"  rows {row0:04d}:{row1:04d}", flush=True)

    elapsed = time.perf_counter() - start_time
    metadata = {
        "format": "opencheese.ensemble_pairwise_teacher",
        "format_version": 1,
        "ensembles": str(cache.path),
        "n_molecules": n,
        "max_conformers": cache.max_conformers,
        "canonicalize": args.canonicalize,
        "shape_metric": args.shape_metric,
        "electrostatic_metric": args.electrostatic_metric,
        "shape_weight": args.shape_weight,
        "electrostatic_weight": args.electrostatic_weight,
        "map_electrostatic_to_unit": not args.no_map_electrostatic,
        "tile_size": args.tile_size,
        "seconds": elapsed,
        "note": "Best score over all conformer pairs in each molecule pair tile.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        format_version=np.array([1], dtype=np.int64),
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)], dtype=str),
        source_indices=cache.source_indices[eligible].astype(np.int64),
        ids=cache.ids[eligible].astype(str),
        smiles=cache.smiles[eligible].astype(str),
        shape=shape,
        electrostatic=electrostatic,
        combined=combined,
    )
    print(f"Wrote {args.out} in {elapsed:.2f}s", flush=True)
    print(
        f"combined stats min={combined.min():.4f} mean={combined.mean():.4f} max={combined.max():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
