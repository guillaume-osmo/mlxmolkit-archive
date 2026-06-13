#!/usr/bin/env python
"""Train or fine-tune openCHEESE projection embeddings from pairwise teachers."""

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

from opencheese.embedding import (
    CheeseEmbeddingBatch,
    CheeseEmbeddingConfig,
    CheeseGraphTransformer,
    atom_reconstruction_loss_mlx,
    cheese_embedding_batch,
    embedding_cosine_similarity_matrix_mlx,
    pair_distance_reconstruction_loss_mlx,
)
from tools.train_charge_model import ChargeTrainingDataset


DEFAULT_DATASET = Path(
    "data/espaloma_charge_zenodo_17308526/recalculated_charges/"
    "test_random1000_both_symmetrized_partial_bcc_fill/"
    "cheese_charge_training_am1bcc_resp.npz"
)
DEFAULT_TEACHER = Path("outputs/cheese_projection/cheese_teacher_1000_q_resp_carbo_canonical.npz")
DEFAULT_OUT = Path("outputs/cheese_projection/embedding_q_resp_carbo_h128_l4_e100")


class CheeseProjectionDataset:
    """Graph/3D batches aligned with a pairwise openCHEESE teacher matrix."""

    def __init__(
        self,
        dataset_path: str | Path,
        teacher_path: str | Path,
        *,
        target: str | None = None,
        teacher_channel: str = "combined",
    ):
        self.dataset = ChargeTrainingDataset(dataset_path)
        self.teacher_path = Path(teacher_path)
        if not self.teacher_path.exists():
            raise FileNotFoundError(self.teacher_path)
        teacher = np.load(self.teacher_path, allow_pickle=False)
        self.source_indices = teacher["source_indices"].astype(np.int64)
        self.ids = teacher["ids"].astype(str)
        if teacher_channel not in {"combined", "shape", "electrostatic"}:
            raise ValueError("teacher_channel must be 'combined', 'shape', or 'electrostatic'")
        self.teacher_channel = teacher_channel
        self.teacher = teacher[teacher_channel].astype(np.float32)
        self.shape_teacher = teacher["shape"].astype(np.float32)
        self.electrostatic_teacher = teacher["electrostatic"].astype(np.float32)
        if self.teacher.shape != (len(self.source_indices), len(self.source_indices)):
            raise ValueError("combined teacher matrix shape does not match source_indices")

        metadata = json.loads(str(teacher["metadata_json"][0])) if "metadata_json" in teacher.files else {}
        self.target = str(target or metadata.get("target", "q_resp"))
        if self.target not in self.dataset.labels:
            raise ValueError(f"target {self.target!r} is not available in dataset labels")

    @property
    def n_molecules(self) -> int:
        return int(len(self.source_indices))

    @property
    def max_atoms(self) -> int:
        return self.dataset.max_atoms

    def batch(self, positions: Sequence[int], *, pad_to: int | None = None) -> CheeseEmbeddingBatch:
        atoms = []
        coords = []
        bonds = []
        charges = []
        ids = []
        for position in positions:
            dataset_index = int(self.source_indices[int(position)])
            z, xyz, bond_matrix, _, q = self.dataset.molecule_arrays(dataset_index, self.target)
            atoms.append(z)
            coords.append(xyz)
            bonds.append(bond_matrix)
            charges.append(q)
            ids.append(str(self.dataset.ids[dataset_index]))
        return cheese_embedding_batch(
            atoms,
            coords,
            bonds,
            charges,
            ids=ids,
            pad_to=pad_to,
        )


def split_positions(n: int, *, valid_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if valid_fraction <= 0 or n < 2:
        return indices, np.empty((0,), dtype=np.int64)
    n_valid = int(round(n * valid_fraction))
    n_valid = min(max(1, n_valid), n - 1)
    return indices[n_valid:], indices[:n_valid]


def iter_batches(indices: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def projection_metric_loss(
    row_embeddings: mx.array,
    col_embeddings: mx.array,
    target_similarity: mx.array,
    *,
    loss_mode: str,
) -> mx.array:
    if loss_mode == "cosine_similarity":
        pred = embedding_cosine_similarity_matrix_mlx(row_embeddings, col_embeddings)
        target = 2.0 * target_similarity.astype(mx.float32) - 1.0
        return mx.mean((pred - target) ** 2)
    if loss_mode == "euclidean_dissimilarity":
        diff = row_embeddings[:, None, :] - col_embeddings[None, :, :]
        pred = mx.sqrt(mx.sum(diff * diff, axis=-1) + mx.array(1.0e-8, dtype=row_embeddings.dtype))
        target = 1.0 - target_similarity.astype(mx.float32)
        return mx.mean((pred - target) ** 2)
    raise ValueError("loss_mode must be 'cosine_similarity' or 'euclidean_dissimilarity'")


def train_loss(
    model: CheeseGraphTransformer,
    row_batch: CheeseEmbeddingBatch,
    col_batch: CheeseEmbeddingBatch,
    target_similarity: mx.array,
    *,
    metric_weight: float,
    atom_weight: float,
    distance_weight: float,
    loss_mode: str,
) -> mx.array:
    row_output = model(
        row_batch.atomic_numbers,
        row_batch.coords,
        row_batch.bond_matrix,
        row_batch.mask,
        row_batch.charges,
        row_batch.chiral_features,
    )
    col_output = model(
        col_batch.atomic_numbers,
        col_batch.coords,
        col_batch.bond_matrix,
        col_batch.mask,
        col_batch.charges,
        col_batch.chiral_features,
    )
    metric = projection_metric_loss(
        row_output.embedding,
        col_output.embedding,
        target_similarity,
        loss_mode=loss_mode,
    )
    atom = 0.5 * (
        atom_reconstruction_loss_mlx(row_output.atom_logits, row_batch.atomic_numbers, row_batch.mask)
        + atom_reconstruction_loss_mlx(col_output.atom_logits, col_batch.atomic_numbers, col_batch.mask)
    )
    distance = 0.5 * (
        pair_distance_reconstruction_loss_mlx(row_output.reconstructed_coords, row_batch.coords, row_batch.mask)
        + pair_distance_reconstruction_loss_mlx(col_output.reconstructed_coords, col_batch.coords, col_batch.mask)
    )
    return float(metric_weight) * metric + float(atom_weight) * atom + float(distance_weight) * distance


def encode_positions(
    model: CheeseGraphTransformer,
    dataset: CheeseProjectionDataset,
    positions: np.ndarray,
    *,
    batch_size: int,
    pad_to: int | None,
) -> np.ndarray:
    model.eval()
    chunks = []
    for batch_positions in iter_batches(positions, batch_size=batch_size, shuffle=False, seed=0):
        batch = dataset.batch(batch_positions, pad_to=pad_to)
        embedding = model.encode_batch(batch)
        mx.eval(embedding)
        chunks.append(np.asarray(embedding, dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, model.config.embedding_dim), dtype=np.float32)


def evaluate_projection(
    model: CheeseGraphTransformer,
    dataset: CheeseProjectionDataset,
    positions: np.ndarray,
    *,
    batch_size: int,
    pad_to: int | None,
    loss_mode: str,
) -> dict[str, float]:
    if len(positions) == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "corr": 0.0, "n_pairs": 0.0}
    embeddings = encode_positions(model, dataset, positions, batch_size=batch_size, pad_to=pad_to)
    sim = dataset.teacher[np.ix_(positions, positions)]
    if loss_mode == "cosine_similarity":
        pred = embeddings @ embeddings.T
        target = 2.0 * sim - 1.0
    elif loss_mode == "euclidean_dissimilarity":
        diff = embeddings[:, None, :] - embeddings[None, :, :]
        pred = np.sqrt(np.sum(diff * diff, axis=-1) + 1.0e-8)
        target = 1.0 - sim
    else:
        raise ValueError("loss_mode must be 'cosine_similarity' or 'euclidean_dissimilarity'")
    mask = ~np.eye(len(positions), dtype=bool)
    err = pred[mask] - target[mask]
    pred_flat = pred[mask]
    target_flat = target[mask]
    corr = 0.0
    if pred_flat.size > 1 and float(np.std(pred_flat)) > 1.0e-12 and float(np.std(target_flat)) > 1.0e-12:
        corr = float(np.corrcoef(pred_flat, target_flat)[0, 1])
    return {
        "mse": float(np.mean(err * err)) if err.size else 0.0,
        "mae": float(np.mean(np.abs(err))) if err.size else 0.0,
        "rmse": float(np.sqrt(np.mean(err * err))) if err.size else 0.0,
        "corr": corr,
        "n_pairs": float(err.size),
    }


def save_embeddings(
    model: CheeseGraphTransformer,
    dataset: CheeseProjectionDataset,
    path: Path,
    *,
    batch_size: int,
    pad_to: int | None,
) -> None:
    positions = np.arange(dataset.n_molecules, dtype=np.int64)
    embeddings = encode_positions(model, dataset, positions, batch_size=batch_size, pad_to=pad_to)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=dataset.ids.astype(str),
        source_indices=dataset.source_indices.astype(np.int64),
        embeddings=embeddings.astype(np.float32),
        embedding_norm=np.linalg.norm(embeddings, axis=1).astype(np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Optional existing CheeseGraphTransformer weights to fine-tune from.",
    )
    parser.add_argument("--target", default=None, choices=[None, "q_reference", "q_esp", "q_resp"])
    parser.add_argument("--teacher-channel", choices=["combined", "shape", "electrostatic"], default="combined")
    parser.add_argument(
        "--loss-mode",
        choices=["euclidean_dissimilarity", "cosine_similarity"],
        default="euclidean_dissimilarity",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--bond-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--rbf", type=int, default=32)
    parser.add_argument("--metric-weight", type=float, default=1.0)
    parser.add_argument("--atom-weight", type=float, default=0.02)
    parser.add_argument("--distance-weight", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means all row/column batch blocks")
    parser.add_argument("--dynamic-pad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")

    dataset = CheeseProjectionDataset(
        args.data,
        args.teacher,
        target=args.target,
        teacher_channel=args.teacher_channel,
    )
    train_positions, valid_positions = split_positions(
        dataset.n_molecules,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    pad_to = None if args.dynamic_pad else dataset.max_atoms
    config = CheeseEmbeddingConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        bond_dim=args.bond_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_rbf=args.rbf,
        use_charges=True,
    )
    model = CheeseGraphTransformer(config)
    if args.init_weights is not None:
        if not args.init_weights.exists():
            raise FileNotFoundError(args.init_weights)
        model.load_weights(str(args.init_weights))
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(m, row_batch, col_batch, target_similarity):
        return train_loss(
            m,
            row_batch,
            col_batch,
            target_similarity,
            metric_weight=args.metric_weight,
            atom_weight=args.atom_weight,
            distance_weight=args.distance_weight,
            loss_mode=args.loss_mode,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out_dir / "best.safetensors"
    last_path = args.out_dir / "last.safetensors"
    metrics_path = args.out_dir / "metrics.csv"
    config_path = args.out_dir / "config.json"
    embeddings_path = args.out_dir / "embeddings.npz"

    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    print(
        f"Training openCHEESE projection n={dataset.n_molecules} train={len(train_positions)} "
        f"valid={len(valid_positions)} target={dataset.target} channel={dataset.teacher_channel} "
        f"loss={args.loss_mode} batch={args.batch_size}"
        + (f" init={args.init_weights}" if args.init_weights is not None else ""),
        flush=True,
    )

    rows: list[dict[str, float | int]] = []
    best_valid = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        step_losses = []
        rng = np.random.default_rng(args.seed + epoch)

        if args.steps_per_epoch > 0:
            block_pairs = [
                (
                    rng.choice(train_positions, size=min(args.batch_size, len(train_positions)), replace=False),
                    rng.choice(train_positions, size=min(args.batch_size, len(train_positions)), replace=False),
                )
                for _ in range(args.steps_per_epoch)
            ]
        else:
            row_blocks = list(iter_batches(train_positions, batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch))
            col_blocks = list(iter_batches(train_positions, batch_size=args.batch_size, shuffle=True, seed=args.seed + 17 * epoch))
            block_pairs = [(row_block, col_block) for row_block in row_blocks for col_block in col_blocks]

        for row_positions, col_positions in block_pairs:
            row_batch = dataset.batch(row_positions, pad_to=pad_to)
            col_batch = dataset.batch(col_positions, pad_to=pad_to)
            target_block = mx.array(dataset.teacher[np.ix_(row_positions, col_positions)], dtype=mx.float32)
            loss, grads = loss_and_grad(model, row_batch, col_batch, target_block)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            step_losses.append(float(loss))

        train_eval = evaluate_projection(
            model,
            dataset,
            train_positions,
            batch_size=args.batch_size,
            pad_to=pad_to,
            loss_mode=args.loss_mode,
        )
        valid_eval = evaluate_projection(
            model,
            dataset,
            valid_positions,
            batch_size=args.batch_size,
            pad_to=pad_to,
            loss_mode=args.loss_mode,
        )
        row = {
            "epoch": epoch,
            "step_loss": float(np.mean(step_losses)) if step_losses else 0.0,
            "train_mse": train_eval["mse"],
            "train_mae": train_eval["mae"],
            "train_rmse": train_eval["rmse"],
            "train_corr": train_eval["corr"],
            "valid_mse": valid_eval["mse"],
            "valid_mae": valid_eval["mae"],
            "valid_rmse": valid_eval["rmse"],
            "valid_corr": valid_eval["corr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        rows.append(row)
        if valid_eval["mse"] < best_valid:
            best_valid = valid_eval["mse"]
            best_epoch = epoch
            model.save_weights(str(best_path))
        if epoch == 1 or epoch == args.epochs or epoch % max(args.eval_every, 1) == 0:
            print(
                f"epoch {epoch:04d}/{args.epochs} "
                f"train_mse={train_eval['mse']:.5f} valid_mse={valid_eval['mse']:.5f} "
                f"valid_corr={valid_eval['corr']:.3f} best={best_valid:.5f}@{best_epoch}",
                flush=True,
            )

    model.save_weights(str(last_path))
    _write_metrics(metrics_path, rows)

    best_model = CheeseGraphTransformer(config)
    best_model.load_weights(str(best_path))
    save_embeddings(best_model, dataset, embeddings_path, batch_size=args.batch_size, pad_to=pad_to)

    summary = {
        "dataset": str(args.data),
        "teacher": str(args.teacher),
        "init_weights": None if args.init_weights is None else str(args.init_weights),
        "target": dataset.target,
        "teacher_channel": dataset.teacher_channel,
        "loss_mode": args.loss_mode,
        "best_epoch": best_epoch,
        "best_valid_mse": best_valid,
        "best_weights": str(best_path),
        "last_weights": str(last_path),
        "embeddings": str(embeddings_path),
        "metrics": str(metrics_path),
        "config": str(config_path),
        "seconds": time.perf_counter() - start_time,
        "final": rows[-1] if rows else {},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out_dir / 'summary.json'}", flush=True)


def _write_metrics(path: Path, rows: Sequence[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
