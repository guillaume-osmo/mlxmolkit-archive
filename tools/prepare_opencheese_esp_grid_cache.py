#!/usr/bin/env python
"""Cache conformer ESP grids for openCHEESE electrostatic teachers.

This is a preprocessing bridge between charge/conformer generation and
field-based teacher scoring. For each selected conformer it builds a
Connolly/MK-style surface grid, then regenerates point-charge electrostatic
potential values on the GPU via the openCHEESE/MLX Metal path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from opencheese.descriptors import electrostatic_potential_on_grid_metal
from mlxmolkit.esp_resp import connolly_surface_grid
from tools.compute_cheese_ensemble_teacher import EnsembleCache


DEFAULT_ENSEMBLES = Path("outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz")
DEFAULT_OUT = Path("outputs/cheese_projection/opencheese_esp_grids_1000_k10_q_resp.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembles", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-conformers", type=int, default=10)
    parser.add_argument("--point-density", type=float, default=0.35)
    parser.add_argument("--min-points-per-shell", type=int, default=16)
    parser.add_argument("--max-points-per-shell", type=int, default=96)
    parser.add_argument("--shell-factors", default="1.4,1.6,1.8,2.0")
    parser.add_argument("--min-distance", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shell_factors = tuple(float(x) for x in args.shell_factors.split(",") if x.strip())
    cache = EnsembleCache(args.ensembles)
    eligible = np.flatnonzero(cache.ok & (cache.n_conformers > 0))
    if args.limit > 0:
        eligible = eligible[: int(args.limit)]

    grid_offsets = [0]
    owner_mol = []
    owner_conf = []
    all_grids = []
    all_esp = []
    start_time = time.perf_counter()

    print(f"Caching ESP grids for {len(eligible)} molecules", flush=True)
    for out_mol_index, mol_index in enumerate(eligible):
        atoms, coords, charges = cache.molecule(int(mol_index))
        n_conf = min(int(len(coords)), int(args.max_conformers) if args.max_conformers > 0 else int(len(coords)))
        for conf_index in range(n_conf):
            conf = coords[conf_index]
            grid = connolly_surface_grid(
                atoms,
                conf,
                shell_factors=shell_factors,
                point_density=float(args.point_density),
                min_points_per_shell=int(args.min_points_per_shell),
                max_points_per_shell=int(args.max_points_per_shell),
            ).astype(np.float32)
            if grid.size == 0:
                esp = np.empty((0,), dtype=np.float32)
            else:
                esp_mx = electrostatic_potential_on_grid_metal(
                    conf[None, :, :],
                    charges[None, :],
                    grid,
                    min_distance=float(args.min_distance),
                )
                mx.eval(esp_mx)
                esp = np.asarray(esp_mx[0], dtype=np.float32)
            owner_mol.append(out_mol_index)
            owner_conf.append(conf_index)
            all_grids.append(grid)
            all_esp.append(esp)
            grid_offsets.append(grid_offsets[-1] + int(len(grid)))
        if out_mol_index == 0 or (out_mol_index + 1) % 25 == 0 or out_mol_index + 1 == len(eligible):
            print(f"  molecules {out_mol_index + 1}/{len(eligible)}", flush=True)

    grids = np.vstack(all_grids).astype(np.float32) if all_grids else np.empty((0, 3), dtype=np.float32)
    esp_values = np.concatenate(all_esp).astype(np.float32) if all_esp else np.empty((0,), dtype=np.float32)
    elapsed = time.perf_counter() - start_time
    metadata = {
        "format": "opencheese.esp_grid_cache",
        "format_version": 1,
        "ensembles": str(cache.path),
        "n_molecules": int(len(eligible)),
        "n_conformer_grids": int(len(owner_mol)),
        "max_conformers": int(args.max_conformers),
        "point_density": float(args.point_density),
        "min_points_per_shell": int(args.min_points_per_shell),
        "max_points_per_shell": int(args.max_points_per_shell),
        "shell_factors": list(shell_factors),
        "min_distance": float(args.min_distance),
        "seconds": elapsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        format_version=np.array([1], dtype=np.int64),
        metadata_json=np.array([json.dumps(metadata, sort_keys=True)], dtype=str),
        source_indices=cache.source_indices[eligible].astype(np.int64),
        ids=cache.ids[eligible].astype(str),
        smiles=cache.smiles[eligible].astype(str),
        owner_mol=np.asarray(owner_mol, dtype=np.int32),
        owner_conf=np.asarray(owner_conf, dtype=np.int32),
        grid_offsets=np.asarray(grid_offsets, dtype=np.int64),
        grid_coords=grids,
        esp_values=esp_values,
    )
    print(f"Wrote {args.out} in {elapsed:.2f}s with {len(esp_values)} grid values", flush=True)


if __name__ == "__main__":
    main()
