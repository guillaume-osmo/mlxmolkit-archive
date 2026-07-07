#!/usr/bin/env python
"""Compute an openCHEESE binary donor/acceptor color teacher.

This is a charge-free pharmacophore-color variant: Gaussian shape overlap is
combined with binary H-bond acceptor and donor atom fields from RDKit feature
perception. The output keeps the same matrix contract as openCHEESE teachers:

``shape``
    Uncolored Gaussian shape Carbo/Tanimoto.
``acceptor`` / ``donor``
    Binary color-field similarities for HBA/HBD atoms.
``color``
    Weighted acceptor/donor aggregate.
``combined``
    Shape + color score, excluding color weights for pairs where neither side
    has that color.

For compatibility with the existing trainer, ``electrostatic`` is an alias for
``color`` in this artifact. It is not a partial-charge electrostatic channel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Sequence

import mlx.core as mx
import numpy as np

from opencheese.descriptors import (
    CheeseBatch,
    cheese_batch,
    shape_carbo_matrix_mlx,
    shape_tanimoto_matrix_mlx,
)
from tools.benchmark_opencheese_shape_descriptors import _bond_matrix
from tools.compute_cheese_ensemble_teacher import EnsembleCache, best_pair_scores, principal_sign_flip
from tools.compute_cheese_pairwise_teacher import canonicalize_coords
from tools.prepare_cheese_conformer_ensembles import atom_order_mapping_from_dataset_to_rdkit


DEFAULT_ENSEMBLES = Path("outputs/cheese_projection/opencheese_old1k_plus_medchem4k_ensembles_k10_q_resp_cached.npz")
DEFAULT_OUT = Path("outputs/cheese_projection/opencheese_color_teacher_hba_hbd_k10_bestpair_principal_carbo.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembles", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--canonicalize", choices=["none", "center", "principal"], default="principal")
    parser.add_argument(
        "--principal-sign-mode",
        choices=["canonical", "random_even"],
        default="canonical",
        help="Optional proper sign-flip nuisance for principal-axis canonicalized conformers.",
    )
    parser.add_argument("--principal-sign-seed", type=int, default=0)
    parser.add_argument("--shape-metric", choices=["carbo", "tanimoto"], default="carbo")
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument("--acceptor-weight", type=float, default=0.35)
    parser.add_argument("--donor-weight", type=float, default=0.35)
    parser.add_argument("--gaussian-alpha", type=float, default=2.7)
    parser.add_argument("--vdw-scale", type=float, default=1.0)
    parser.add_argument("--default-radius", type=float, default=1.80)
    parser.add_argument(
        "--skip-feature-failures",
        action="store_true",
        help="Drop rows whose explicit-H SMILES cannot be mapped for feature perception.",
    )
    return parser.parse_args()


def _feature_factory():
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(fdef)


def acceptor_donor_masks_from_smiles(
    smiles: str,
    atoms: np.ndarray,
    bonds: np.ndarray,
    *,
    factory,
) -> tuple[np.ndarray, np.ndarray]:
    """Return HBA/HBD masks in the cached dataset atom order."""

    from rdkit import Chem

    atom_array = np.asarray(atoms, dtype=np.int32)
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")
    Chem.SanitizeMol(mol, catchErrors=True)
    atom_map = atom_order_mapping_from_dataset_to_rdkit(atom_array, bonds, mol)
    rdkit_to_dataset = {int(rdkit_idx): dataset_idx for dataset_idx, rdkit_idx in enumerate(atom_map)}

    acceptor = np.zeros((len(atom_array),), dtype=np.float32)
    donor = np.zeros((len(atom_array),), dtype=np.float32)
    for feature in factory.GetFeaturesForMol(mol):
        family = str(feature.GetFamily())
        if family not in {"Acceptor", "Donor"}:
            continue
        target = acceptor if family == "Acceptor" else donor
        for rdkit_idx in feature.GetAtomIds():
            dataset_idx = rdkit_to_dataset.get(int(rdkit_idx))
            if dataset_idx is not None:
                target[int(dataset_idx)] = 1.0
    return acceptor, donor


def feature_masks_for_eligible(
    cache: EnsembleCache,
    eligible: np.ndarray,
    *,
    skip_feature_failures: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, str]]]:
    raw = np.load(cache.path, allow_pickle=False)
    factory = _feature_factory()
    acceptor_all = np.zeros((len(cache.atomic_numbers),), dtype=np.float32)
    donor_all = np.zeros((len(cache.atomic_numbers),), dtype=np.float32)
    kept: list[int] = []
    failures: list[tuple[int, str]] = []

    for out_index, cache_index in enumerate(np.asarray(eligible, dtype=np.int64)):
        cache_index = int(cache_index)
        atom0, atom1 = int(cache.atom_offsets[cache_index]), int(cache.atom_offsets[cache_index + 1])
        atoms = cache.atomic_numbers[atom0:atom1]
        bonds = _bond_matrix(raw, cache_index)
        try:
            acceptor, donor = acceptor_donor_masks_from_smiles(
                str(cache.smiles[cache_index]),
                atoms,
                bonds,
                factory=factory,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            if not skip_feature_failures:
                raise ValueError(f"feature perception failed for cache row {cache_index}: {message}") from exc
            failures.append((cache_index, message))
            continue
        acceptor_all[atom0:atom1] = acceptor
        donor_all[atom0:atom1] = donor
        kept.append(cache_index)
        if out_index % 500 == 0:
            print(f"  features {out_index:05d}/{len(eligible):05d}", flush=True)
    return acceptor_all, donor_all, np.asarray(kept, dtype=np.int64), failures


def conformer_color_batch(
    cache: EnsembleCache,
    indices: Sequence[int],
    *,
    acceptor_all: np.ndarray,
    donor_all: np.ndarray,
    canonicalize: str,
    principal_sign_mode: str = "canonical",
    principal_sign_seed: int = 0,
) -> tuple[CheeseBatch, np.ndarray, mx.array, mx.array]:
    atoms = []
    coords = []
    charges = []
    acceptor_masks = []
    donor_masks = []
    owners = []
    for local_index, mol_index in enumerate(indices):
        mol_index = int(mol_index)
        z, xyz, q = cache.molecule(mol_index)
        atom0, atom1 = int(cache.atom_offsets[mol_index]), int(cache.atom_offsets[mol_index + 1])
        acceptor = acceptor_all[atom0:atom1].astype(np.float32, copy=False)
        donor = donor_all[atom0:atom1].astype(np.float32, copy=False)
        for conf_index, conf in enumerate(xyz):
            conf_coords = canonicalize_coords(z, conf, mode=canonicalize)
            if principal_sign_mode == "random_even":
                conf_coords = conf_coords * principal_sign_flip(
                    int(mol_index),
                    int(conf_index),
                    seed=int(principal_sign_seed),
                )[None, :]
            elif principal_sign_mode != "canonical":
                raise ValueError("principal_sign_mode must be 'canonical' or 'random_even'")
            atoms.append(z)
            coords.append(conf_coords)
            charges.append(q)
            acceptor_masks.append(acceptor)
            donor_masks.append(donor)
            owners.append(local_index)
    batch = cheese_batch(atoms, coords, charges, pad_to=cache.max_atoms)
    acceptor_pad = np.zeros(tuple(batch.mask.shape), dtype=np.float32)
    donor_pad = np.zeros(tuple(batch.mask.shape), dtype=np.float32)
    for i, (acc, don) in enumerate(zip(acceptor_masks, donor_masks, strict=True)):
        acceptor_pad[i, : len(acc)] = acc
        donor_pad[i, : len(don)] = don
    return batch, np.asarray(owners, dtype=np.int32), mx.array(acceptor_pad), mx.array(donor_pad)


def _shape_matrix(
    probe: CheeseBatch,
    reference: CheeseBatch,
    *,
    metric: str,
    gaussian_alpha: float,
    vdw_scale: float,
    default_radius: float,
) -> mx.array:
    if metric == "carbo":
        return shape_carbo_matrix_mlx(
            probe,
            reference,
            gaussian_alpha=gaussian_alpha,
            vdw_scale=vdw_scale,
            default_radius=default_radius,
        )
    if metric == "tanimoto":
        return shape_tanimoto_matrix_mlx(
            probe,
            reference,
            gaussian_alpha=gaussian_alpha,
            vdw_scale=vdw_scale,
            default_radius=default_radius,
        )
    raise ValueError(f"unknown shape metric {metric!r}")


def color_shape_matrix(
    probe: CheeseBatch,
    reference: CheeseBatch,
    probe_color: mx.array,
    reference_color: mx.array,
    *,
    metric: str,
    gaussian_alpha: float,
    vdw_scale: float,
    default_radius: float,
) -> tuple[mx.array, mx.array]:
    """Return color similarity and active-pair mask for one binary feature."""

    probe_mask = probe.mask * probe_color
    reference_mask = reference.mask * reference_color
    probe_colored = CheeseBatch(probe.atomic_numbers, probe.coords, probe.charges, probe_mask, probe.ids)
    reference_colored = CheeseBatch(
        reference.atomic_numbers,
        reference.coords,
        reference.charges,
        reference_mask,
        reference.ids,
    )
    score = _shape_matrix(
        probe_colored,
        reference_colored,
        metric=metric,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    probe_has = mx.sum(probe_mask, axis=1) > 0.0
    reference_has = mx.sum(reference_mask, axis=1) > 0.0
    active = probe_has[:, None] | reference_has[None, :]
    both_absent = (~probe_has[:, None]) & (~reference_has[None, :])
    adjusted = mx.where(both_absent, mx.ones_like(score), score)
    return adjusted, active.astype(mx.float32)


def color_similarity_block(
    row_batch: CheeseBatch,
    col_batch: CheeseBatch,
    row_acceptor: mx.array,
    col_acceptor: mx.array,
    row_donor: mx.array,
    col_donor: mx.array,
    *,
    shape_metric: str,
    shape_weight: float,
    acceptor_weight: float,
    donor_weight: float,
    gaussian_alpha: float,
    vdw_scale: float,
    default_radius: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    shape = _shape_matrix(
        row_batch,
        col_batch,
        metric=shape_metric,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    acceptor, acceptor_active = color_shape_matrix(
        row_batch,
        col_batch,
        row_acceptor,
        col_acceptor,
        metric=shape_metric,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    donor, donor_active = color_shape_matrix(
        row_batch,
        col_batch,
        row_donor,
        col_donor,
        metric=shape_metric,
        gaussian_alpha=gaussian_alpha,
        vdw_scale=vdw_scale,
        default_radius=default_radius,
    )
    color_weight = float(acceptor_weight) * acceptor_active + float(donor_weight) * donor_active
    color_num = float(acceptor_weight) * acceptor * acceptor_active + float(donor_weight) * donor * donor_active
    color = mx.where(color_weight > 0.0, color_num / mx.maximum(color_weight, 1.0e-8), mx.ones_like(shape))
    denom = float(shape_weight) + color_weight
    combined = (float(shape_weight) * shape + color_num) / mx.maximum(denom, 1.0e-8)
    return shape, acceptor, donor, color, combined


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if args.shape_weight <= 0:
        raise ValueError("--shape-weight must be positive")
    if args.acceptor_weight < 0 or args.donor_weight < 0:
        raise ValueError("color weights must be non-negative")

    cache = EnsembleCache(args.ensembles)
    eligible = np.flatnonzero(cache.ok & (cache.n_conformers > 0))
    if args.limit > 0:
        eligible = eligible[: int(args.limit)]
    start_time = time.perf_counter()
    print(f"Computing HBA/HBD feature masks for {len(eligible)} molecules", flush=True)
    acceptor_all, donor_all, eligible, failures = feature_masks_for_eligible(
        cache,
        eligible,
        skip_feature_failures=bool(args.skip_feature_failures),
    )
    n = int(len(eligible))
    if n == 0:
        raise ValueError("no successful feature rows")

    shape = np.zeros((n, n), dtype=np.float32)
    acceptor = np.zeros((n, n), dtype=np.float32)
    donor = np.zeros((n, n), dtype=np.float32)
    color = np.zeros((n, n), dtype=np.float32)
    combined = np.zeros((n, n), dtype=np.float32)
    print(
        f"Computing color teacher n={n} k<= {cache.max_conformers} tile={args.tile_size} "
        f"metric={args.shape_metric}",
        flush=True,
    )
    for row0 in range(0, n, args.tile_size):
        row1 = min(n, row0 + args.tile_size)
        row_batch, row_owner, row_acc, row_don = conformer_color_batch(
            cache,
            eligible[row0:row1],
            acceptor_all=acceptor_all,
            donor_all=donor_all,
            canonicalize=args.canonicalize,
            principal_sign_mode=args.principal_sign_mode,
            principal_sign_seed=args.principal_sign_seed,
        )
        for col0 in range(row0, n, args.tile_size):
            col1 = min(n, col0 + args.tile_size)
            col_batch, col_owner, col_acc, col_don = conformer_color_batch(
                cache,
                eligible[col0:col1],
                acceptor_all=acceptor_all,
                donor_all=donor_all,
                canonicalize=args.canonicalize,
                principal_sign_mode=args.principal_sign_mode,
                principal_sign_seed=args.principal_sign_seed,
            )
            score_shape, score_acceptor, score_donor, score_color, score_combined = color_similarity_block(
                row_batch,
                col_batch,
                row_acc,
                col_acc,
                row_don,
                col_don,
                shape_metric=args.shape_metric,
                shape_weight=args.shape_weight,
                acceptor_weight=args.acceptor_weight,
                donor_weight=args.donor_weight,
                gaussian_alpha=args.gaussian_alpha,
                vdw_scale=args.vdw_scale,
                default_radius=args.default_radius,
            )
            mx.eval(score_shape, score_acceptor, score_donor, score_color, score_combined)
            blocks = [
                (shape, np.asarray(score_shape, dtype=np.float32)),
                (acceptor, np.asarray(score_acceptor, dtype=np.float32)),
                (donor, np.asarray(score_donor, dtype=np.float32)),
                (color, np.asarray(score_color, dtype=np.float32)),
                (combined, np.asarray(score_combined, dtype=np.float32)),
            ]
            for matrix, conf_scores in blocks:
                block = best_pair_scores(conf_scores, row_owner, col_owner, row1 - row0, col1 - col0)
                matrix[row0:row1, col0:col1] = block
                if col0 != row0:
                    matrix[col0:col1, row0:row1] = block.T
        print(f"  rows {row0:04d}:{row1:04d}", flush=True)

    elapsed = time.perf_counter() - start_time
    metadata = {
        "format": "opencheese.binary_pharmacophore_color_teacher",
        "format_version": 1,
        "ensembles": str(cache.path),
        "n_molecules": n,
        "max_conformers": cache.max_conformers,
        "canonicalize": args.canonicalize,
        "principal_sign_mode": args.principal_sign_mode,
        "principal_sign_seed": int(args.principal_sign_seed),
        "shape_metric": args.shape_metric,
        "shape_weight": float(args.shape_weight),
        "acceptor_weight": float(args.acceptor_weight),
        "donor_weight": float(args.donor_weight),
        "gaussian_alpha": float(args.gaussian_alpha),
        "vdw_scale": float(args.vdw_scale),
        "default_radius": float(args.default_radius),
        "electrostatic_alias": "color",
        "n_feature_failures": len(failures),
        "feature_failures": failures[:20],
        "seconds": elapsed,
        "note": "Charge-free teacher: binary RDKit HBA/HBD color fields plus Gaussian shape.",
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
        acceptor=acceptor,
        donor=donor,
        color=color,
        electrostatic=color,
        combined=combined,
    )
    print(f"Wrote {args.out} in {elapsed:.2f}s", flush=True)
    print(
        f"combined stats min={combined.min():.4f} mean={combined.mean():.4f} max={combined.max():.4f}; "
        f"color mean={color.mean():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
