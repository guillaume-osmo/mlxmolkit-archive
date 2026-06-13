#!/usr/bin/env python
"""Train geometry-aware MLX partial-charge models from CHEESE charge NPZ files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from typing import Iterator, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxmolkit.charge_model import (
    ChargeModelBatch,
    ChargeModelConfig,
    GeometricChargePredictor,
    charge_model_batch,
    charge_prediction_loss,
)


DEFAULT_DATASET = Path(
    "data/espaloma_charge_zenodo_17308526/recalculated_charges/"
    "test_random1000_both_symmetrized_partial_bcc_fill/"
    "cheese_charge_training_am1bcc_resp.npz"
)


class ChargeTrainingDataset:
    """In-memory variable-size charge-label dataset backed by one NPZ."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        data = np.load(self.path, allow_pickle=False)
        self.ids = data["ids"].astype(str)
        self.smiles = data["smiles"].astype(str)
        self.n_atoms = data["n_atoms"].astype(np.int32)
        self.total_charge = data["total_charge"].astype(np.float32)
        self.atom_offsets = data["atom_offsets"].astype(np.int64)
        self.atomic_numbers = data["atomic_numbers"].astype(np.int32)
        self.coords = data["coords"].astype(np.float32)
        self.bond_offsets = data["bond_offsets"].astype(np.int64)
        self.bond_i = data["bond_i"].astype(np.int32)
        self.bond_j = data["bond_j"].astype(np.int32)
        self.bond_state = data["bond_state"].astype(np.int32)
        self.ok = data["ok"].astype(bool)
        self.labels = {
            name: data[name].astype(np.float32)
            for name in ("q_reference", "q_esp", "q_resp")
            if name in data.files
        }

    @property
    def n_molecules(self) -> int:
        return int(len(self.n_atoms))

    @property
    def max_atoms(self) -> int:
        return int(np.max(self.n_atoms)) if len(self.n_atoms) else 0

    def finite_label_indices(self, target: str) -> np.ndarray:
        if target not in self.labels:
            raise ValueError(f"unknown target {target!r}; available: {sorted(self.labels)}")
        label = self.labels[target]
        good = self.ok.copy()
        for i in range(self.n_molecules):
            start, end = self.atom_range(i)
            good[i] = bool(good[i] and np.all(np.isfinite(label[start:end])))
        return np.flatnonzero(good)

    def atom_range(self, index: int) -> tuple[int, int]:
        return int(self.atom_offsets[index]), int(self.atom_offsets[index + 1])

    def bond_range(self, index: int) -> tuple[int, int]:
        return int(self.bond_offsets[index]), int(self.bond_offsets[index + 1])

    def molecule_arrays(self, index: int, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        atom_start, atom_end = self.atom_range(index)
        bond_start, bond_end = self.bond_range(index)
        n_atoms = atom_end - atom_start

        bonds = np.zeros((n_atoms, n_atoms), dtype=np.int32)
        rows = self.bond_i[bond_start:bond_end]
        cols = self.bond_j[bond_start:bond_end]
        states = self.bond_state[bond_start:bond_end]
        bonds[rows, cols] = states

        return (
            self.atomic_numbers[atom_start:atom_end],
            self.coords[atom_start:atom_end],
            bonds,
            float(self.total_charge[index]),
            self.labels[target][atom_start:atom_end],
        )

    def batch(self, indices: Sequence[int], target: str, *, pad_to: int | None = None) -> ChargeModelBatch:
        atoms: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        bonds: list[np.ndarray] = []
        total_charges: list[float] = []
        labels: list[np.ndarray] = []
        for index in indices:
            z, xyz, bond_matrix, charge, label = self.molecule_arrays(int(index), target)
            atoms.append(z)
            coords.append(xyz)
            bonds.append(bond_matrix)
            total_charges.append(charge)
            labels.append(label)
        return charge_model_batch(
            atoms,
            coords,
            bond_matrices=bonds,
            total_charges=total_charges,
            labels=labels,
            pad_to=pad_to,
        )


def split_indices(
    indices: np.ndarray,
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if valid_fraction <= 0 or len(indices) < 2:
        return indices, np.empty((0,), dtype=np.int64)

    n_valid = int(round(len(indices) * valid_fraction))
    n_valid = min(max(1, n_valid), len(indices) - 1)
    return indices[n_valid:], indices[:n_valid]


def iter_minibatches(
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def evaluate_model(
    model: GeometricChargePredictor,
    dataset: ChargeTrainingDataset,
    indices: np.ndarray,
    target: str,
    *,
    batch_size: int,
    pad_to: int | None,
) -> dict[str, float]:
    model.eval()
    n_atoms = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    sum_mol_charge_abs = 0.0
    n_mols = 0
    loss_sum = 0.0
    loss_weight = 0.0

    for batch_indices in iter_minibatches(indices, batch_size=batch_size, shuffle=False, seed=0):
        batch = dataset.batch(batch_indices, target, pad_to=pad_to)
        pred = model(batch.atomic_numbers, batch.coords, batch.bond_matrix, batch.mask, batch.total_charge)
        loss = charge_prediction_loss(pred, batch.labels, batch.mask)
        mx.eval(pred, loss)

        pred_np = np.asarray(pred, dtype=np.float64)
        label_np = np.asarray(batch.labels, dtype=np.float64)
        mask_np = np.asarray(batch.mask, dtype=np.float64)
        err = (pred_np - label_np) * mask_np
        abs_err = np.abs(err)
        batch_atoms = float(mask_np.sum())

        n_atoms += batch_atoms
        sum_abs += float(abs_err.sum())
        sum_sq += float((err * err).sum())
        max_abs = max(max_abs, float(abs_err.max(initial=0.0)))
        loss_sum += float(loss) * batch_atoms
        loss_weight += batch_atoms

        pred_charge = (pred_np * mask_np).sum(axis=1)
        target_charge = (label_np * mask_np).sum(axis=1)
        sum_mol_charge_abs += float(np.abs(pred_charge - target_charge).sum())
        n_mols += int(len(batch_indices))

    denom = max(n_atoms, 1.0)
    return {
        "loss": loss_sum / max(loss_weight, 1.0),
        "mae": sum_abs / denom,
        "rmse": float(np.sqrt(sum_sq / denom)),
        "max_abs": max_abs,
        "mean_mol_charge_abs_error": sum_mol_charge_abs / max(n_mols, 1),
        "n_molecules": float(n_mols),
        "n_atoms": float(n_atoms),
    }


def train_one_target(
    dataset: ChargeTrainingDataset,
    target: str,
    *,
    out_dir: Path,
    config: ChargeModelConfig,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    pad_to: int | None,
    eval_every: int,
) -> dict[str, object]:
    model = GeometricChargePredictor(config)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss_and_grad = nn.value_and_grad(model, _training_loss)

    target_dir = out_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = target_dir / "metrics.csv"
    best_path = target_dir / "best.safetensors"
    last_path = target_dir / "last.safetensors"
    config_path = target_dir / "config.json"

    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")

    best_metric = float("inf")
    best_epoch = 0
    rows: list[dict[str, float | int | str]] = []
    start_time = time.perf_counter()

    print(
        f"[{target}] train={len(train_indices)} valid={len(valid_indices)} "
        f"batch={batch_size} pad_to={pad_to or 'dynamic'} readout={config.readout}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_atoms = 0.0
        epoch_start = time.perf_counter()

        for batch_indices in iter_minibatches(
            train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            batch = dataset.batch(batch_indices, target, pad_to=pad_to)
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            n_batch_atoms = float(np.asarray(batch.mask).sum())
            train_loss_sum += float(loss) * n_batch_atoms
            train_atoms += n_batch_atoms

        train_metrics = evaluate_model(
            model,
            dataset,
            train_indices,
            target,
            batch_size=batch_size,
            pad_to=pad_to,
        )
        valid_metrics = (
            evaluate_model(
                model,
                dataset,
                valid_indices,
                target,
                batch_size=batch_size,
                pad_to=pad_to,
            )
            if len(valid_indices)
            else dict(train_metrics)
        )

        row: dict[str, float | int | str] = {
            "target": target,
            "epoch": epoch,
            "train_step_loss": train_loss_sum / max(train_atoms, 1.0),
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_max_abs": train_metrics["max_abs"],
            "valid_loss": valid_metrics["loss"],
            "valid_mae": valid_metrics["mae"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_max_abs": valid_metrics["max_abs"],
            "valid_mean_mol_charge_abs_error": valid_metrics["mean_mol_charge_abs_error"],
            "seconds": time.perf_counter() - epoch_start,
        }
        rows.append(row)

        if valid_metrics["mae"] < best_metric:
            best_metric = float(valid_metrics["mae"])
            best_epoch = epoch
            model.save_weights(str(best_path))

        if epoch == 1 or epoch == epochs or epoch % max(eval_every, 1) == 0:
            print(
                f"[{target}] epoch {epoch:04d}/{epochs} "
                f"train_mae={train_metrics['mae']:.6f} "
                f"valid_mae={valid_metrics['mae']:.6f} "
                f"valid_rmse={valid_metrics['rmse']:.6f} "
                f"best={best_metric:.6f}@{best_epoch}",
                flush=True,
            )

    model.save_weights(str(last_path))
    _write_metrics_csv(metrics_path, rows)

    summary = {
        "target": target,
        "best_epoch": best_epoch,
        "best_valid_mae": best_metric,
        "best_weights": str(best_path),
        "last_weights": str(last_path),
        "metrics": str(metrics_path),
        "config": str(config_path),
        "seconds": time.perf_counter() - start_time,
        "final": rows[-1] if rows else {},
    }
    (target_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _training_loss(model: GeometricChargePredictor, batch: ChargeModelBatch) -> mx.array:
    predicted = model(batch.atomic_numbers, batch.coords, batch.bond_matrix, batch.mask, batch.total_charge)
    return charge_prediction_loss(predicted, batch.labels, batch.mask)


def _write_metrics_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/charge_models/cheese_am1bcc_1000"))
    parser.add_argument("--targets", default="q_esp,q_resp", help="comma-separated NPZ label arrays")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bond-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--rbf", type=int, default=32)
    parser.add_argument("--embedding-rmax", type=float, default=8.0)
    parser.add_argument("--readout", choices=["direct", "qeq"], default="direct")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--dynamic-pad",
        action="store_true",
        help="pad each batch only to its local max atom count instead of the dataset max",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    if not targets:
        raise ValueError("at least one target is required")

    dataset = ChargeTrainingDataset(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = ChargeModelConfig(
        hidden_dim=args.hidden_dim,
        bond_dim=args.bond_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_rbf=args.rbf,
        rbf_max=args.embedding_rmax,
        readout=args.readout,
    )
    pad_to = None if args.dynamic_pad else dataset.max_atoms

    run_summary: dict[str, object] = {
        "dataset": str(dataset.path),
        "n_molecules": dataset.n_molecules,
        "max_atoms": dataset.max_atoms,
        "targets": targets,
        "config": asdict(config),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "valid_fraction": args.valid_fraction,
        "seed": args.seed,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "pad_to": pad_to,
        "target_summaries": [],
    }

    print(
        f"Loaded {dataset.n_molecules} molecules from {dataset.path} "
        f"(max_atoms={dataset.max_atoms})",
        flush=True,
    )

    for target_index, target in enumerate(targets):
        eligible = dataset.finite_label_indices(target)
        train_indices, valid_indices = split_indices(
            eligible,
            valid_fraction=args.valid_fraction,
            seed=args.seed + target_index,
        )
        summary = train_one_target(
            dataset,
            target,
            out_dir=args.out_dir,
            config=config,
            train_indices=train_indices,
            valid_indices=valid_indices,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + 1000 * target_index,
            pad_to=pad_to,
            eval_every=args.eval_every,
        )
        run_summary["target_summaries"].append(summary)  # type: ignore[index]

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote run summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
