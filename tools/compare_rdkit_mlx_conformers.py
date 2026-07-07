#!/usr/bin/env python
"""Compare RDKit and mlxmolkit conformer ensembles with openCHEESE scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

import mlx.core as mx

from opencheese.descriptors import (
    CheeseBatch,
    cheese_similarity_pairs_mlx,
    horn_align_pairwise_mlx,
)
from mlxmolkit.conformer_pipeline_v2 import generate_conformers_nk
from tools.prepare_cheese_conformer_ensembles import (
    atom_order_mapping_from_dataset_to_rdkit,
    mol_from_smiles_with_conformers,
)
from tools.train_charge_model import ChargeTrainingDataset


DEFAULT_DATASET = Path(
    "data/espaloma_charge_zenodo_17308526/recalculated_charges/"
    "test_random1000_both_symmetrized_partial_bcc_fill/"
    "cheese_charge_training_am1bcc_resp.npz"
)


def _heavy_atom_weights(atoms: np.ndarray) -> np.ndarray:
    weights = np.where(np.asarray(atoms, dtype=np.int32) == 1, 0.0, 1.0).astype(np.float32)
    if float(np.sum(weights)) <= 0:
        weights[:] = 1.0
    return weights


def _rdkit_conformers_for_dataset_row(
    dataset: ChargeTrainingDataset,
    dataset_index: int,
    target: str,
    *,
    n_conformers: int,
    seed: int,
    optimize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    z, _, bonds, _, q_dataset = dataset.molecule_arrays(dataset_index, target)
    mol, conf_ids, _, _ = mol_from_smiles_with_conformers(
        str(dataset.smiles[dataset_index]),
        n_conformers=n_conformers,
        seed=seed + dataset_index,
        prune_rms_thresh=-1.0,
        max_embed_attempts=1000,
        optimize=optimize,
        mmff_variant="MMFF94",
        max_opt_iters=300,
    )
    dataset_to_rdkit = atom_order_mapping_from_dataset_to_rdkit(z, bonds, mol)
    rdkit_to_dataset = np.empty_like(dataset_to_rdkit)
    rdkit_to_dataset[dataset_to_rdkit] = np.arange(len(dataset_to_rdkit), dtype=np.int32)
    atoms_rdkit = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
    q_rdkit = np.asarray(q_dataset, dtype=np.float32)[rdkit_to_dataset]
    coords = [np.asarray(mol.GetConformer(int(conf_id)).GetPositions(), dtype=np.float32) for conf_id in conf_ids]
    return atoms_rdkit, q_rdkit, rdkit_to_dataset, coords


def _mlx_conformers(smiles: str, *, n_conformers: int, run_mmff: bool) -> list[np.ndarray]:
    result = generate_conformers_nk(
        [smiles],
        n_confs_per_mol=n_conformers,
        variant="ETKDGv3",
        run_mmff=run_mmff,
        mmff_variant="MMFF94",
    )
    return [np.asarray(x, dtype=np.float32) for x in result.molecules[0].positions_3d]


def _score_conformer_pairs_mlx(
    atoms: np.ndarray,
    charges: np.ndarray,
    mlx_coords: list[np.ndarray],
    rdkit_coords: list[np.ndarray],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return full MLX-vs-RDKit score matrices after batched GPU alignment."""

    probe_np = np.stack(mlx_coords, axis=0).astype(np.float32)
    ref_np = np.stack(rdkit_coords, axis=0).astype(np.float32)
    if probe_np.ndim != 3 or ref_np.ndim != 3 or probe_np.shape[1:] != ref_np.shape[1:]:
        raise ValueError(f"MLX coords shape {probe_np.shape} does not match RDKit {ref_np.shape}")

    n_mlx, n_atoms = probe_np.shape[0], probe_np.shape[1]
    n_rdkit = ref_np.shape[0]
    pair_count = n_mlx * n_rdkit

    aligned, rmsd, _ = horn_align_pairwise_mlx(
        mx.array(probe_np),
        mx.array(ref_np),
        weights=mx.array(weights.astype(np.float32)),
        power_iters=48,
    )

    atoms_mx = mx.array(np.asarray(atoms, dtype=np.int32))
    charges_mx = mx.array(np.asarray(charges, dtype=np.float32))
    atom_pairs = mx.broadcast_to(atoms_mx[None, :], (pair_count, n_atoms))
    charge_pairs = mx.broadcast_to(charges_mx[None, :], (pair_count, n_atoms))
    mask_pairs = mx.ones((pair_count, n_atoms), dtype=mx.float32)

    aligned_flat = mx.reshape(aligned, (pair_count, n_atoms, 3))
    ref_mx = mx.array(ref_np)
    ref_pairs = mx.broadcast_to(ref_mx[None, :, :, :], (n_mlx, n_rdkit, n_atoms, 3))
    ref_flat = mx.reshape(ref_pairs, (pair_count, n_atoms, 3))

    probe_batch = CheeseBatch(atom_pairs, aligned_flat, charge_pairs, mask_pairs)
    ref_batch = CheeseBatch(atom_pairs, ref_flat, charge_pairs, mask_pairs)
    scores = cheese_similarity_pairs_mlx(
        probe_batch,
        ref_batch,
        shape_metric="carbo",
        electrostatic_metric="carbo",
        map_electrostatic_to_unit=True,
    )
    mx.eval(scores.shape, scores.electrostatic, scores.combined, rmsd)
    shape = np.asarray(mx.reshape(scores.shape, (n_mlx, n_rdkit)))
    electrostatic = np.asarray(mx.reshape(scores.electrostatic, (n_mlx, n_rdkit)))
    combined = np.asarray(mx.reshape(scores.combined, (n_mlx, n_rdkit)))
    rmsd_np = np.asarray(rmsd)
    return shape, electrostatic, combined, rmsd_np


def compare_one(
    dataset: ChargeTrainingDataset,
    dataset_index: int,
    target: str,
    *,
    n_conformers: int,
    seed: int,
    optimize_rdkit: bool,
    run_mlx_mmff: bool,
) -> dict[str, object]:
    atoms, charges, _, rdkit_coords = _rdkit_conformers_for_dataset_row(
        dataset,
        dataset_index,
        target,
        n_conformers=n_conformers,
        seed=seed,
        optimize=optimize_rdkit,
    )
    mlx_coords = _mlx_conformers(
        str(dataset.smiles[dataset_index]),
        n_conformers=n_conformers,
        run_mmff=run_mlx_mmff,
    )
    n_rdkit = len(rdkit_coords)
    n_mlx = len(mlx_coords)
    weights = _heavy_atom_weights(atoms)

    shape, electrostatic, combined, rmsd = _score_conformer_pairs_mlx(
        atoms,
        charges,
        mlx_coords,
        rdkit_coords,
        weights,
    )

    best_flat = int(np.argmax(combined))
    best_i, best_j = np.unravel_index(best_flat, combined.shape)
    return {
        "source_index": int(dataset_index),
        "id": str(dataset.ids[dataset_index]),
        "smiles": str(dataset.smiles[dataset_index]),
        "n_atoms": int(len(atoms)),
        "n_rdkit": int(n_rdkit),
        "n_mlx": int(n_mlx),
        "best_mlx_conf": int(best_i),
        "best_rdkit_conf": int(best_j),
        "best_shape_carbo": float(shape[best_i, best_j]),
        "best_electrostatic_carbo": float(electrostatic[best_i, best_j]),
        "best_combined": float(combined[best_i, best_j]),
        "best_heavy_rmsd": float(rmsd[best_i, best_j]),
        "mean_best_mlx_to_rdkit_shape": float(np.mean(np.max(shape, axis=1))),
        "mean_best_mlx_to_rdkit_combined": float(np.mean(np.max(combined, axis=1))),
        "median_best_mlx_to_rdkit_rmsd": float(np.median(np.min(rmsd, axis=1))),
        "ok": True,
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=Path("outputs/cheese_projection/rdkit_vs_mlx_conformers.csv"))
    parser.add_argument("--target", choices=["q_reference", "q_esp", "q_resp"], default="q_resp")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--n-conformers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--no-rdkit-optimize", action="store_true")
    parser.add_argument("--no-mlx-mmff", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ChargeTrainingDataset(args.data)
    indices = dataset.finite_label_indices(args.target)
    if args.limit > 0:
        indices = indices[: int(args.limit)]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    start = time.perf_counter()
    for row_index, dataset_index in enumerate(indices):
        try:
            row = compare_one(
                dataset,
                int(dataset_index),
                args.target,
                n_conformers=args.n_conformers,
                seed=args.seed,
                optimize_rdkit=not args.no_rdkit_optimize,
                run_mlx_mmff=not args.no_mlx_mmff,
            )
        except Exception as exc:
            row = {
                "source_index": int(dataset_index),
                "id": str(dataset.ids[int(dataset_index)]),
                "smiles": str(dataset.smiles[int(dataset_index)]),
                "n_atoms": 0,
                "n_rdkit": 0,
                "n_mlx": 0,
                "best_mlx_conf": -1,
                "best_rdkit_conf": -1,
                "best_shape_carbo": np.nan,
                "best_electrostatic_carbo": np.nan,
                "best_combined": np.nan,
                "best_heavy_rmsd": np.nan,
                "mean_best_mlx_to_rdkit_shape": np.nan,
                "mean_best_mlx_to_rdkit_combined": np.nan,
                "median_best_mlx_to_rdkit_rmsd": np.nan,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(
            f"{row_index + 1}/{len(indices)} id={row['id']} ok={row['ok']} "
            f"best_shape={row['best_shape_carbo']} best_rmsd={row['best_heavy_rmsd']}",
            flush=True,
        )

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["ok"]]
    summary = {
        "n": len(rows),
        "n_ok": len(ok_rows),
        "seconds": time.perf_counter() - start,
        "mean_best_shape": float(np.nanmean([row["best_shape_carbo"] for row in rows])),
        "mean_best_combined": float(np.nanmean([row["best_combined"] for row in rows])),
        "median_best_rmsd": float(np.nanmedian([row["best_heavy_rmsd"] for row in rows])),
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
