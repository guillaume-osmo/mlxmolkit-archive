#!/usr/bin/env python
"""GraphMVP-style 2D/3D pretraining for openCHEESE embeddings."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from opencheese import CheeseEmbeddingConfig, GraphMVPPretrainer, cheese_embedding_batch


DEFAULT_ENSEMBLES = Path("outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz")
DEFAULT_OUT = Path("outputs/cheese_projection/graphmvp_pretrain_k10_h128_l4")


class GraphMVPEnsembleDataset:
    """NPZ ensemble cache loader for GraphMVP-style 2D/3D pretraining."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        data = np.load(self.path, allow_pickle=False)
        self.metadata = json.loads(str(data["metadata_json"][0])) if "metadata_json" in data.files else {}
        self.source_indices = data["source_indices"].astype(np.int64)
        self.ids = data["ids"].astype(str)
        self.smiles = data["smiles"].astype(str) if "smiles" in data.files else np.asarray([""] * len(self.ids))
        self.ok = data["ok"].astype(bool)
        self.n_conformers = data["n_conformers"].astype(np.int32)
        self.atom_offsets = data["atom_offsets"].astype(np.int64)
        self.conformer_offsets = data["conformer_offsets"].astype(np.int64)
        self.coord_offsets = data["coord_offsets"].astype(np.int64)
        self.bond_offsets = data["bond_offsets"].astype(np.int64)
        self.atomic_numbers = data["atomic_numbers"].astype(np.int32)
        self.coords = data["coords"].astype(np.float32)
        self.charges = data["charges"].astype(np.float32) if "charges" in data.files else None
        self.bond_i = data["bond_i"].astype(np.int32)
        self.bond_j = data["bond_j"].astype(np.int32)
        self.bond_state = data["bond_state"].astype(np.int32)
        self.eligible = np.flatnonzero(self.ok & (self.n_conformers > 0)).astype(np.int64)
        if len(self.eligible) == 0:
            raise ValueError("no successful conformer rows in ensemble cache")

    @property
    def n_molecules(self) -> int:
        return int(len(self.eligible))

    @property
    def max_atoms(self) -> int:
        sizes = self.atom_offsets[1:] - self.atom_offsets[:-1]
        return int(np.max(sizes)) if len(sizes) else 0

    def molecule_arrays(self, cache_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        atom_start, atom_end = int(self.atom_offsets[cache_index]), int(self.atom_offsets[cache_index + 1])
        coord_start, coord_end = int(self.coord_offsets[cache_index]), int(self.coord_offsets[cache_index + 1])
        bond_start, bond_end = int(self.bond_offsets[cache_index]), int(self.bond_offsets[cache_index + 1])
        atoms = self.atomic_numbers[atom_start:atom_end]
        n_atoms = len(atoms)
        n_conf = int(self.n_conformers[cache_index])
        coords = self.coords[coord_start:coord_end].reshape(n_conf, n_atoms, 3)
        bonds = np.zeros((n_atoms, n_atoms), dtype=np.int32)
        local_i = self.bond_i[bond_start:bond_end]
        local_j = self.bond_j[bond_start:bond_end]
        states = self.bond_state[bond_start:bond_end]
        bonds[local_i, local_j] = states
        charges = (
            self.charges[atom_start:atom_end].astype(np.float32)
            if self.charges is not None
            else np.zeros((n_atoms,), dtype=np.float32)
        )
        return atoms, coords, bonds, charges

    def batch(
        self,
        positions: Sequence[int],
        rng: np.random.Generator,
        *,
        pad_to: int | None,
        use_charges_3d: bool,
    ):
        atoms_list = []
        coords_2d = []
        coords_3d = []
        bonds_list = []
        charges_3d = []
        ids = []
        for position in positions:
            cache_index = int(self.eligible[int(position)])
            atoms, conformers, bonds, charges = self.molecule_arrays(cache_index)
            conf_index = int(rng.integers(0, len(conformers)))
            atoms_list.append(atoms)
            coords_2d.append(np.zeros_like(conformers[conf_index], dtype=np.float32))
            coords_3d.append(conformers[conf_index])
            bonds_list.append(bonds)
            charges_3d.append(charges)
            ids.append(str(self.ids[cache_index]))
        batch_2d = cheese_embedding_batch(
            atoms_list,
            coords_2d,
            bonds_list,
            charges=None,
            ids=ids,
            pad_to=pad_to,
            compute_chiral_features=False,
        )
        batch_3d = cheese_embedding_batch(
            atoms_list,
            coords_3d,
            bonds_list,
            charges=charges_3d if use_charges_3d else None,
            ids=ids,
            pad_to=pad_to,
            compute_chiral_features=True,
        )
        return batch_2d, batch_3d


def split_positions(n: int, *, valid_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(positions)
    if valid_fraction <= 0 or n < 2:
        return positions, np.empty((0,), dtype=np.int64)
    n_valid = min(max(1, int(round(valid_fraction * n))), n - 1)
    return positions[n_valid:], positions[:n_valid]


def sample_batch(indices: np.ndarray, rng: np.random.Generator, batch_size: int) -> np.ndarray:
    size = min(int(batch_size), len(indices))
    return rng.choice(indices, size=size, replace=False)


def evaluate(
    model: GraphMVPPretrainer,
    dataset: GraphMVPEnsembleDataset,
    positions: np.ndarray,
    *,
    batch_size: int,
    pad_to: int | None,
    use_charges_3d: bool,
    temperature: float,
    uniformity_weight: float,
    variance_weight: float,
    uniformity_temperature: float,
    variance_target: float,
    reconstruction_loss: str,
    detach_target: bool,
    seed: int,
) -> dict[str, float]:
    if len(positions) == 0:
        return {
            "loss": 0.0,
            "contrastive": 0.0,
            "reconstruction": 0.0,
            "uniformity": 0.0,
            "variance": 0.0,
            "acc_2d_to_3d": 0.0,
            "acc_3d_to_2d": 0.0,
        }
    model.eval()
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        batch_2d, batch_3d = dataset.batch(
            batch_positions,
            rng,
            pad_to=pad_to,
            use_charges_3d=use_charges_3d,
        )
        loss = model(
            batch_2d,
            batch_3d,
            temperature=temperature,
            uniformity_weight=uniformity_weight,
            variance_weight=variance_weight,
            uniformity_temperature=uniformity_temperature,
            variance_target=variance_target,
            reconstruction_loss=reconstruction_loss,
            detach_target=detach_target,
        )
        mx.eval(
            loss.total,
            loss.contrastive,
            loss.reconstruction,
            loss.uniformity,
            loss.variance,
            loss.accuracy_2d_to_3d,
            loss.accuracy_3d_to_2d,
        )
        rows.append(
            {
                "loss": float(loss.total),
                "contrastive": float(loss.contrastive),
                "reconstruction": float(loss.reconstruction),
                "uniformity": float(loss.uniformity),
                "variance": float(loss.variance),
                "acc_2d_to_3d": float(loss.accuracy_2d_to_3d),
                "acc_3d_to_2d": float(loss.accuracy_3d_to_2d),
            }
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembles", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
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
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument(
        "--uniformity-weight",
        type=float,
        default=0.0,
        help="Weight for the unit-sphere uniformity anti-collapse loss.",
    )
    parser.add_argument(
        "--variance-weight",
        type=float,
        default=0.0,
        help="Weight for the batch-variance anti-collapse loss.",
    )
    parser.add_argument("--uniformity-temperature", type=float, default=2.0)
    parser.add_argument("--variance-target", type=float, default=0.05)
    parser.add_argument("--reconstruction-loss", choices=["l2", "l1", "cosine"], default="l2")
    parser.add_argument("--detach-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-charges-3d", action="store_true")
    parser.add_argument("--dynamic-pad", action="store_true")
    parser.add_argument("--eval-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("epochs, batch-size, and steps-per-epoch must be positive")
    dataset = GraphMVPEnsembleDataset(args.ensembles)
    train_positions, valid_positions = split_positions(
        dataset.n_molecules,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    pad_to = None if args.dynamic_pad else dataset.max_atoms
    config_2d = CheeseEmbeddingConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        bond_dim=args.bond_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_rbf=args.rbf,
        use_charges=False,
        use_chiral_features=False,
    )
    config_3d = CheeseEmbeddingConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        bond_dim=args.bond_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_rbf=args.rbf,
        use_charges=bool(args.use_charges_3d),
        use_chiral_features=True,
    )
    model = GraphMVPPretrainer(config_2d, config_3d)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def loss_fn(m, batch_2d, batch_3d):
        return m(
            batch_2d,
            batch_3d,
            temperature=args.temperature,
            contrastive_weight=args.contrastive_weight,
            reconstruction_weight=args.reconstruction_weight,
            uniformity_weight=args.uniformity_weight,
            variance_weight=args.variance_weight,
            uniformity_temperature=args.uniformity_temperature,
            variance_target=args.variance_target,
            reconstruction_loss=args.reconstruction_loss,
            detach_target=args.detach_target,
        ).total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.csv"
    summary_path = args.out_dir / "summary.json"
    best_path = args.out_dir / "best.safetensors"
    last_path = args.out_dir / "last.safetensors"
    config_path = args.out_dir / "config.json"
    config_path.write_text(
        json.dumps({"config_2d": asdict(config_2d), "config_3d": asdict(config_3d)}, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"GraphMVP pretraining n={dataset.n_molecules} train={len(train_positions)} "
        f"valid={len(valid_positions)} batch={args.batch_size} steps={args.steps_per_epoch}",
        flush=True,
    )
    best_valid = float("inf")
    best_epoch = 0
    rows = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        rng = np.random.default_rng(args.seed + epoch)
        step_losses = []
        for _ in range(args.steps_per_epoch):
            positions = sample_batch(train_positions, rng, args.batch_size)
            batch_2d, batch_3d = dataset.batch(
                positions,
                rng,
                pad_to=pad_to,
                use_charges_3d=bool(args.use_charges_3d),
            )
            loss, grads = loss_and_grad(model, batch_2d, batch_3d)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            step_losses.append(float(loss))

        train_eval = evaluate(
            model,
            dataset,
            train_positions[: min(len(train_positions), max(args.batch_size * 4, args.batch_size))],
            batch_size=args.batch_size,
            pad_to=pad_to,
            use_charges_3d=bool(args.use_charges_3d),
            temperature=args.temperature,
            uniformity_weight=args.uniformity_weight,
            variance_weight=args.variance_weight,
            uniformity_temperature=args.uniformity_temperature,
            variance_target=args.variance_target,
            reconstruction_loss=args.reconstruction_loss,
            detach_target=args.detach_target,
            seed=args.seed + 1000 + epoch,
        )
        valid_eval = evaluate(
            model,
            dataset,
            valid_positions,
            batch_size=args.batch_size,
            pad_to=pad_to,
            use_charges_3d=bool(args.use_charges_3d),
            temperature=args.temperature,
            uniformity_weight=args.uniformity_weight,
            variance_weight=args.variance_weight,
            uniformity_temperature=args.uniformity_temperature,
            variance_target=args.variance_target,
            reconstruction_loss=args.reconstruction_loss,
            detach_target=args.detach_target,
            seed=args.seed + 2000 + epoch,
        )
        row = {
            "epoch": epoch,
            "step_loss": float(np.mean(step_losses)) if step_losses else 0.0,
            "train_loss": train_eval["loss"],
            "train_contrastive": train_eval["contrastive"],
            "train_reconstruction": train_eval["reconstruction"],
            "train_uniformity": train_eval["uniformity"],
            "train_variance": train_eval["variance"],
            "train_acc_2d_to_3d": train_eval["acc_2d_to_3d"],
            "train_acc_3d_to_2d": train_eval["acc_3d_to_2d"],
            "valid_loss": valid_eval["loss"],
            "valid_contrastive": valid_eval["contrastive"],
            "valid_reconstruction": valid_eval["reconstruction"],
            "valid_uniformity": valid_eval["uniformity"],
            "valid_variance": valid_eval["variance"],
            "valid_acc_2d_to_3d": valid_eval["acc_2d_to_3d"],
            "valid_acc_3d_to_2d": valid_eval["acc_3d_to_2d"],
            "seconds": time.perf_counter() - epoch_start,
        }
        rows.append(row)
        if valid_eval["loss"] < best_valid:
            best_valid = valid_eval["loss"]
            best_epoch = epoch
            model.save_weights(str(best_path))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.eval_every) == 0:
            print(
                f"epoch {epoch:04d}/{args.epochs} step={row['step_loss']:.4f} "
                f"valid={valid_eval['loss']:.4f} cl={valid_eval['contrastive']:.4f} "
                f"rr={valid_eval['reconstruction']:.4f} uni={valid_eval['uniformity']:.4f} "
                f"var={valid_eval['variance']:.4f} acc={valid_eval['acc_2d_to_3d']:.3f}/"
                f"{valid_eval['acc_3d_to_2d']:.3f} best={best_valid:.4f}@{best_epoch}",
                flush=True,
            )

    model.save_weights(str(last_path))
    _write_metrics(metrics_path, rows)
    summary = {
        "ensembles": str(args.ensembles),
        "n_molecules": dataset.n_molecules,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid,
        "best_weights": str(best_path),
        "last_weights": str(last_path),
        "metrics": str(metrics_path),
        "config": str(config_path),
        "contrastive_weight": args.contrastive_weight,
        "reconstruction_weight": args.reconstruction_weight,
        "uniformity_weight": args.uniformity_weight,
        "variance_weight": args.variance_weight,
        "uniformity_temperature": args.uniformity_temperature,
        "variance_target": args.variance_target,
        "seconds": time.perf_counter() - start_time,
        "final": rows[-1] if rows else {},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {summary_path}", flush=True)


def _write_metrics(path: Path, rows: Sequence[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
