#!/usr/bin/env python
"""Cache LG-Mol-style surface ESP clouds from openCHEESE conformer ensembles.

Each conformer becomes one ``(n_points, 4)`` cloud with columns
``x, y, z, ESP``. The default path uses the MLX Connolly SES surface plus the
Metal point-charge ESP kernel, avoiding Amber/Antechamber/APBS/MSMS while
keeping LG-Mol's data contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from opencheese.surface import lgmol_surface_cloud_from_atoms
from tools.compute_cheese_ensemble_teacher import EnsembleCache


DEFAULT_ENSEMBLES = Path("outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz")
DEFAULT_OUT = Path("outputs/cheese_projection/opencheese_surface_clouds_1000_k10_q_resp.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembles", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-conformers", type=int, default=10)
    parser.add_argument("--method", choices=["ses", "shell"], default="ses")
    parser.add_argument("--points-num", type=int, default=200)
    parser.add_argument("--sampling", choices=["none", "first", "random", "farthest"], default="farthest")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--min-distance", type=float, default=1.0e-4)
    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument("--grid-spacing", type=float, default=0.35)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--keep-cavities", action="store_true")
    parser.add_argument("--shell-factors", default="1.4,1.6,1.8,2.0")
    parser.add_argument("--point-density", type=float, default=0.35)
    parser.add_argument("--min-points-per-shell", type=int, default=16)
    parser.add_argument("--max-points-per-shell", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points_num < 0:
        raise ValueError("--points-num must be non-negative")
    if args.max_conformers < 0:
        raise ValueError("--max-conformers must be non-negative")
    shell_factors = tuple(float(x) for x in args.shell_factors.split(",") if x.strip())
    cache = EnsembleCache(args.ensembles)
    eligible = np.flatnonzero(cache.ok & (cache.n_conformers > 0))
    if args.limit > 0:
        eligible = eligible[: int(args.limit)]

    cloud_offsets = [0]
    owner_mol: list[int] = []
    owner_conf: list[int] = []
    cloud_ok: list[bool] = []
    error_messages: list[str] = []
    all_clouds: list[np.ndarray] = []
    surface_area: list[float] = []
    surface_volume: list[float] = []
    euler: list[int] = []
    n_surface_vertices: list[int] = []
    gpu_field_seconds: list[float] = []

    start_time = time.perf_counter()
    print(
        f"Caching openCHEESE surface clouds for {len(eligible)} molecules "
        f"method={args.method} points={args.points_num}",
        flush=True,
    )
    for out_mol_index, mol_index in enumerate(eligible):
        atoms, coords, charges = cache.molecule(int(mol_index))
        max_conf = int(args.max_conformers) if args.max_conformers > 0 else int(len(coords))
        n_conf = min(int(len(coords)), max_conf)
        for conf_index in range(n_conf):
            owner_mol.append(out_mol_index)
            owner_conf.append(conf_index)
            try:
                result = lgmol_surface_cloud_from_atoms(
                    atoms,
                    coords[conf_index],
                    charges,
                    method=args.method,
                    points_num=int(args.points_num) if args.points_num > 0 else None,
                    sampling=args.sampling,
                    random_seed=int(args.random_seed) + int(mol_index) * 997 + conf_index,
                    center=bool(args.center),
                    min_distance=float(args.min_distance),
                    probe_radius=float(args.probe_radius),
                    grid_spacing=float(args.grid_spacing),
                    margin=float(args.margin),
                    chunk_size=int(args.chunk_size),
                    remove_cavities=not args.keep_cavities,
                    shell_factors=shell_factors,
                    point_density=float(args.point_density),
                    min_points_per_shell=int(args.min_points_per_shell),
                    max_points_per_shell=int(args.max_points_per_shell),
                )
                cloud = result.cloud.astype(np.float32, copy=False)
                cloud_ok.append(True)
                error_messages.append("")
                surface_area.append(float(result.surface_area))
                surface_volume.append(float(result.surface_volume))
                euler.append(int(result.euler_characteristic))
                n_surface_vertices.append(int(result.n_surface_vertices))
                gpu_field_seconds.append(float(result.gpu_field_seconds))
            except Exception as exc:
                cloud = np.empty((0, 4), dtype=np.float32)
                cloud_ok.append(False)
                error_messages.append(f"{exc.__class__.__name__}: {exc}")
                surface_area.append(float("nan"))
                surface_volume.append(float("nan"))
                euler.append(0)
                n_surface_vertices.append(0)
                gpu_field_seconds.append(0.0)
            all_clouds.append(cloud)
            cloud_offsets.append(cloud_offsets[-1] + int(len(cloud)))

        if out_mol_index == 0 or (out_mol_index + 1) % 25 == 0 or out_mol_index + 1 == len(eligible):
            ok_count = int(np.sum(np.asarray(cloud_ok, dtype=bool)))
            print(f"  molecules {out_mol_index + 1}/{len(eligible)} clouds_ok={ok_count}/{len(cloud_ok)}", flush=True)

    clouds = np.vstack(all_clouds).astype(np.float32) if all_clouds else np.empty((0, 4), dtype=np.float32)
    elapsed = time.perf_counter() - start_time
    metadata = {
        "format": "opencheese.surface_cloud_cache",
        "format_version": 1,
        "layout": "LG-Mol compatible [x,y,z,ESP] rows",
        "ensembles": str(cache.path),
        "n_molecules": int(len(eligible)),
        "n_clouds": int(len(owner_mol)),
        "method": args.method,
        "points_num": int(args.points_num),
        "sampling": args.sampling,
        "center": bool(args.center),
        "max_conformers": int(args.max_conformers),
        "probe_radius": float(args.probe_radius),
        "grid_spacing": float(args.grid_spacing),
        "shell_factors": list(shell_factors),
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
        cloud_offsets=np.asarray(cloud_offsets, dtype=np.int64),
        clouds=clouds,
        cloud_ok=np.asarray(cloud_ok, dtype=bool),
        error_messages=np.asarray(error_messages, dtype=str),
        surface_area=np.asarray(surface_area, dtype=np.float32),
        surface_volume=np.asarray(surface_volume, dtype=np.float32),
        euler_characteristic=np.asarray(euler, dtype=np.int32),
        n_surface_vertices=np.asarray(n_surface_vertices, dtype=np.int32),
        gpu_field_seconds=np.asarray(gpu_field_seconds, dtype=np.float32),
    )
    print(
        f"Wrote {args.out} in {elapsed:.2f}s with {clouds.shape[0]} point rows "
        f"and {int(np.sum(np.asarray(cloud_ok, dtype=bool)))}/{len(cloud_ok)} ok clouds",
        flush=True,
    )


if __name__ == "__main__":
    main()
