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
from opencheese.optimizers import MuonV2W
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
        teacher_transform: str = "raw",
        zscore_temperature: float = 1.0,
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
        self.raw_teacher = teacher[teacher_channel].astype(np.float32)
        self.shape_teacher = teacher["shape"].astype(np.float32)
        self.electrostatic_teacher = teacher["electrostatic"].astype(np.float32)
        if self.raw_teacher.shape != (len(self.source_indices), len(self.source_indices)):
            raise ValueError("combined teacher matrix shape does not match source_indices")

        metadata = json.loads(str(teacher["metadata_json"][0])) if "metadata_json" in teacher.files else {}
        self.target = str(target or metadata.get("target", "q_resp"))
        if self.target not in self.dataset.labels:
            raise ValueError(f"target {self.target!r} is not available in dataset labels")
        self.atom_counts = self._atom_counts()
        self.teacher_transform = str(teacher_transform)
        self.zscore_temperature = float(zscore_temperature)
        self.teacher, self.teacher_transform_metadata = transform_teacher_matrix(
            self.raw_teacher,
            transform=self.teacher_transform,
            atom_counts=self.atom_counts,
            zscore_temperature=self.zscore_temperature,
        )

    @property
    def n_molecules(self) -> int:
        return int(len(self.source_indices))

    @property
    def max_atoms(self) -> int:
        return self.dataset.max_atoms

    def _atom_counts(self) -> np.ndarray:
        counts = np.zeros((len(self.source_indices),), dtype=np.float32)
        for out_index, dataset_index in enumerate(self.source_indices):
            z, _, _, _, _ = self.dataset.molecule_arrays(int(dataset_index), self.target)
            counts[out_index] = float(len(z))
        return counts

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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def anchor_zscore_to_unit(score: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Row-normalize a teacher matrix and map row z-scores back to [0, 1]."""

    score = np.asarray(score, dtype=np.float32)
    n = int(score.shape[0])
    out = np.empty_like(score, dtype=np.float32)
    temperature = max(float(temperature), 1.0e-6)
    for i in range(n):
        mask = np.ones((score.shape[1],), dtype=bool)
        if score.shape[0] == score.shape[1]:
            mask[i] = False
        values = score[i, mask]
        mean = float(np.mean(values)) if values.size else float(score[i].mean())
        std = float(np.std(values)) if values.size else float(score[i].std())
        std = max(std, 1.0e-6)
        out[i] = sigmoid_np((score[i] - mean) / (std * temperature))
    if score.shape[0] == score.shape[1]:
        np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def size_residual_teacher_to_unit(
    score: np.ndarray,
    atom_counts: np.ndarray,
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Remove a simple molecular-size baseline before row normalization."""

    score = np.asarray(score, dtype=np.float32)
    sizes = np.maximum(np.asarray(atom_counts, dtype=np.float32), 1.0)
    left = sizes[:, None]
    right = sizes[None, :]
    ratio = np.minimum(left, right) / np.maximum(left, right)
    log_left = np.log(left)
    log_right = np.log(right)
    features = np.stack(
        [
            np.ones_like(score),
            ratio.astype(np.float32),
            np.abs(log_left - log_right).astype(np.float32),
            (0.5 * (log_left + log_right)).astype(np.float32),
        ],
        axis=-1,
    )
    mask = np.ones(score.shape, dtype=bool)
    if score.shape[0] == score.shape[1]:
        np.fill_diagonal(mask, False)
    x = features[mask]
    y = score[mask]
    coef, *_ = np.linalg.lstsq(x.astype(np.float64), y.astype(np.float64), rcond=None)
    baseline = np.tensordot(features, coef.astype(np.float32), axes=([-1], [0])).astype(np.float32)
    residual = score - baseline
    transformed = anchor_zscore_to_unit(residual, temperature=temperature)
    metadata = {
        "size_baseline_features": ["intercept", "min_size_over_max_size", "abs_log_size_delta", "mean_log_size"],
        "size_baseline_coefficients": [float(v) for v in coef],
        "size_baseline_r2": _r2_score_np(y, x @ coef),
    }
    return transformed, metadata


def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1.0e-12:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def transform_teacher_matrix(
    score: np.ndarray,
    *,
    transform: str,
    atom_counts: np.ndarray,
    zscore_temperature: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the training teacher matrix and metadata for its transform."""

    transform = str(transform)
    if transform == "raw":
        return np.asarray(score, dtype=np.float32), {"teacher_transform": "raw"}
    if transform == "anchor_zscore":
        return (
            anchor_zscore_to_unit(score, temperature=zscore_temperature),
            {"teacher_transform": "anchor_zscore", "zscore_temperature": float(zscore_temperature)},
        )
    if transform == "size_residual":
        transformed, metadata = size_residual_teacher_to_unit(
            score,
            atom_counts,
            temperature=zscore_temperature,
        )
        metadata.update({"teacher_transform": "size_residual", "zscore_temperature": float(zscore_temperature)})
        return transformed, metadata
    raise ValueError("teacher_transform must be 'raw', 'anchor_zscore', or 'size_residual'")


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


def embedding_distance_matrix_mlx(row_embeddings: mx.array, col_embeddings: mx.array) -> mx.array:
    diff = row_embeddings[:, None, :] - col_embeddings[None, :, :]
    return mx.sqrt(mx.sum(diff * diff, axis=-1) + mx.array(1.0e-8, dtype=row_embeddings.dtype))


def target_distance_mlx(
    target_similarity: mx.array,
    *,
    distance_transform: str,
    distance_scale: float,
    distance_power: float,
) -> mx.array:
    similarity = target_similarity.astype(mx.float32)
    if distance_transform == "one_minus_similarity":
        distance = mx.maximum(1.0 - similarity, mx.array(0.0, dtype=mx.float32))
    elif distance_transform == "neglog_similarity":
        distance = -mx.log(mx.clip(similarity, mx.array(1.0e-6, dtype=mx.float32), mx.array(1.0, dtype=mx.float32)))
    else:
        raise ValueError(f"unknown distance_transform {distance_transform!r}")
    if abs(float(distance_power) - 1.0) > 1.0e-8:
        distance = distance ** float(distance_power)
    return float(distance_scale) * distance


def weighted_distance_mse_mlx(
    pred_distance: mx.array,
    target_distance: mx.array,
    target_similarity: mx.array,
    *,
    top_sim_weight: float,
    top_sim_center: float,
    top_sim_temperature: float,
    valid_pair_mask: mx.array | None = None,
) -> mx.array:
    err2 = (pred_distance - target_distance) ** 2
    if valid_pair_mask is not None:
        valid = valid_pair_mask.astype(mx.float32)
    else:
        valid = mx.ones_like(err2)
    if top_sim_weight <= 0:
        return mx.sum(err2 * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))
    temperature = max(float(top_sim_temperature), 1.0e-6)
    top_gate = mx.sigmoid((target_similarity.astype(mx.float32) - float(top_sim_center)) / temperature)
    weights = (1.0 + float(top_sim_weight) * top_gate) * valid
    return mx.sum(weights * err2) / mx.maximum(mx.sum(weights), mx.array(1.0, dtype=mx.float32))


def pairwise_ranking_loss_mlx(
    pred_distance: mx.array,
    target_similarity: mx.array,
    *,
    min_delta: float,
    margin: float,
    temperature: float,
    valid_pair_mask: mx.array | None = None,
) -> mx.array:
    """Anchor-wise soft pairwise ranking loss.

    For each anchor row, if column j is more similar than column k by at least
    ``min_delta``, the predicted distance to j should be smaller than the
    predicted distance to k.
    """

    sim = target_similarity.astype(mx.float32)
    sim_delta = sim[:, :, None] - sim[:, None, :]
    mask = sim_delta > float(min_delta)
    if valid_pair_mask is not None:
        valid = valid_pair_mask.astype(mx.bool_)
        mask = mask & valid[:, :, None] & valid[:, None, :]
    pred_delta = pred_distance[:, :, None] - pred_distance[:, None, :]
    logits = (pred_delta + float(margin)) / max(float(temperature), 1.0e-6)
    loss = mx.logaddexp(mx.array(0.0, dtype=pred_distance.dtype), logits)
    weights = mx.where(mask, mx.maximum(sim_delta, 0.0), mx.array(0.0, dtype=sim.dtype))
    denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=mx.float32))
    return mx.sum(loss * weights) / denom


def soft_neighborhood_loss_mlx(
    pred_distance: mx.array,
    target_similarity: mx.array,
    *,
    target_temperature: float,
    pred_temperature: float,
    valid_pair_mask: mx.array | None = None,
) -> mx.array:
    """KL-style soft-neighborhood loss for retrieval ordering."""

    target_logits = target_similarity.astype(mx.float32) / max(float(target_temperature), 1.0e-6)
    pred_logits = -pred_distance / max(float(pred_temperature), 1.0e-6)
    if valid_pair_mask is not None:
        valid = valid_pair_mask.astype(mx.float32)
        target_logits = mx.where(valid > 0, target_logits, mx.array(-1.0e9, dtype=target_logits.dtype))
        pred_logits = mx.where(valid > 0, pred_logits, mx.array(-1.0e9, dtype=pred_logits.dtype))
        active = mx.sum(valid, axis=-1) > 0
        active_weight = active.astype(mx.float32)
    else:
        active_weight = mx.ones((target_similarity.shape[0],), dtype=mx.float32)
    target_prob = mx.stop_gradient(mx.softmax(target_logits, axis=-1))
    pred_log_prob = pred_logits - mx.logsumexp(pred_logits, axis=-1, keepdims=True)
    row_loss = -mx.sum(target_prob * pred_log_prob, axis=-1)
    return mx.sum(row_loss * active_weight) / mx.maximum(mx.sum(active_weight), mx.array(1.0, dtype=mx.float32))


def supervised_contrastive_loss_mlx(
    row_embeddings: mx.array,
    col_embeddings: mx.array,
    target_similarity: mx.array,
    *,
    temperature: float,
    positive_threshold: float,
    positive_top_k: int,
    valid_pair_mask: mx.array | None = None,
) -> mx.array:
    """InfoNCE-style loss using high-teacher-similarity pairs as positives."""

    logits = embedding_cosine_similarity_matrix_mlx(row_embeddings, col_embeddings)
    logits = logits / max(float(temperature), 1.0e-6)
    if valid_pair_mask is not None:
        valid = valid_pair_mask.astype(mx.float32)
        logits = mx.where(valid > 0, logits, mx.array(-1.0e9, dtype=logits.dtype))
    logits = logits - mx.max(logits, axis=-1, keepdims=True)
    log_prob = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    sim = target_similarity.astype(mx.float32)
    if valid_pair_mask is not None:
        valid = valid_pair_mask.astype(mx.float32)
        sim_for_topk = mx.where(valid > 0, sim, mx.array(-1.0e9, dtype=sim.dtype))
    else:
        valid = mx.ones_like(sim)
        sim_for_topk = sim
    positive = sim >= float(positive_threshold)
    if positive_top_k > 0 and sim.shape[1] > 0:
        k = min(int(positive_top_k), int(sim.shape[1]))
        cutoff = mx.sort(sim_for_topk, axis=-1)[:, -k]
        positive = positive | (sim_for_topk >= cutoff[:, None])
    positive = positive & (valid > 0)
    weights = mx.where(positive, mx.maximum(sim, 0.0), mx.array(0.0, dtype=mx.float32))
    pos_weight = mx.sum(weights, axis=-1)
    row_loss = -mx.sum(weights * log_prob, axis=-1) / mx.maximum(pos_weight, mx.array(1.0, dtype=mx.float32))
    active = pos_weight > 0
    active_weight = active.astype(mx.float32)
    return mx.sum(row_loss * active_weight) / mx.maximum(mx.sum(active_weight), mx.array(1.0, dtype=mx.float32))


def projection_metric_loss(
    row_embeddings: mx.array,
    col_embeddings: mx.array,
    target_similarity: mx.array,
    *,
    loss_mode: str,
    distance_transform: str = "one_minus_similarity",
    distance_scale: float = 1.0,
    distance_power: float = 1.0,
    top_sim_weight: float = 0.0,
    top_sim_center: float = 0.70,
    top_sim_temperature: float = 0.05,
    rank_weight: float = 0.0,
    rank_min_delta: float = 0.03,
    rank_margin: float = 0.02,
    rank_temperature: float = 0.05,
    soft_neighborhood_weight: float = 0.0,
    soft_target_temperature: float = 0.05,
    soft_pred_temperature: float = 0.05,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.07,
    contrastive_positive_threshold: float = 0.75,
    contrastive_positive_top_k: int = 0,
    valid_pair_mask: mx.array | None = None,
) -> mx.array:
    if loss_mode == "cosine_similarity":
        pred = embedding_cosine_similarity_matrix_mlx(row_embeddings, col_embeddings)
        target = 2.0 * target_similarity.astype(mx.float32) - 1.0
        err2 = (pred - target) ** 2
        if valid_pair_mask is not None:
            valid = valid_pair_mask.astype(mx.float32)
            loss = mx.sum(err2 * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))
        else:
            loss = mx.mean(err2)
        if contrastive_weight > 0:
            loss = loss + float(contrastive_weight) * supervised_contrastive_loss_mlx(
                row_embeddings,
                col_embeddings,
                target_similarity,
                temperature=contrastive_temperature,
                positive_threshold=contrastive_positive_threshold,
                positive_top_k=contrastive_positive_top_k,
                valid_pair_mask=valid_pair_mask,
            )
        return loss
    if loss_mode in {
        "euclidean_dissimilarity",
        "ranked_dissimilarity",
        "soft_neighborhood",
        "hybrid_shape",
        "contrastive_shape",
    }:
        pred = embedding_distance_matrix_mlx(row_embeddings, col_embeddings)
        target = target_distance_mlx(
            target_similarity,
            distance_transform=distance_transform,
            distance_scale=distance_scale,
            distance_power=distance_power,
        )
        loss = weighted_distance_mse_mlx(
            pred,
            target,
            target_similarity,
            top_sim_weight=top_sim_weight,
            top_sim_center=top_sim_center,
            top_sim_temperature=top_sim_temperature,
            valid_pair_mask=valid_pair_mask,
        )
        if loss_mode in {"ranked_dissimilarity", "hybrid_shape"} or rank_weight > 0:
            loss = loss + float(rank_weight) * pairwise_ranking_loss_mlx(
                pred,
                target_similarity,
                min_delta=rank_min_delta,
                margin=rank_margin,
                temperature=rank_temperature,
                valid_pair_mask=valid_pair_mask,
            )
        if loss_mode in {"soft_neighborhood", "hybrid_shape"} or soft_neighborhood_weight > 0:
            loss = loss + float(soft_neighborhood_weight) * soft_neighborhood_loss_mlx(
                pred,
                target_similarity,
                target_temperature=soft_target_temperature,
                pred_temperature=soft_pred_temperature,
                valid_pair_mask=valid_pair_mask,
            )
        if loss_mode in {"contrastive_shape", "hybrid_shape"} or contrastive_weight > 0:
            loss = loss + float(contrastive_weight) * supervised_contrastive_loss_mlx(
                row_embeddings,
                col_embeddings,
                target_similarity,
                temperature=contrastive_temperature,
                positive_threshold=contrastive_positive_threshold,
                positive_top_k=contrastive_positive_top_k,
                valid_pair_mask=valid_pair_mask,
            )
        return loss
    raise ValueError(f"unknown loss_mode {loss_mode!r}")


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
    distance_transform: str,
    distance_scale: float,
    distance_power: float,
    top_sim_weight: float,
    top_sim_center: float,
    top_sim_temperature: float,
    rank_weight: float,
    rank_min_delta: float,
    rank_margin: float,
    rank_temperature: float,
    soft_neighborhood_weight: float,
    soft_target_temperature: float,
    soft_pred_temperature: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    contrastive_positive_threshold: float,
    contrastive_positive_top_k: int,
    valid_pair_mask: mx.array | None = None,
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
        distance_transform=distance_transform,
        distance_scale=distance_scale,
        distance_power=distance_power,
        top_sim_weight=top_sim_weight,
        top_sim_center=top_sim_center,
        top_sim_temperature=top_sim_temperature,
        rank_weight=rank_weight,
        rank_min_delta=rank_min_delta,
        rank_margin=rank_margin,
        rank_temperature=rank_temperature,
        soft_neighborhood_weight=soft_neighborhood_weight,
        soft_target_temperature=soft_target_temperature,
        soft_pred_temperature=soft_pred_temperature,
        contrastive_weight=contrastive_weight,
        contrastive_temperature=contrastive_temperature,
        contrastive_positive_threshold=contrastive_positive_threshold,
        contrastive_positive_top_k=contrastive_positive_top_k,
        valid_pair_mask=valid_pair_mask,
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


def _rankdata_np(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        return np.empty((0,), dtype=np.float64)
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 0 or not np.isfinite(denom):
        return np.full_like(exp, 1.0 / max(1, exp.size), dtype=np.float64)
    return exp / denom


def _retrieval_metrics_np(
    pred_distance: np.ndarray,
    sim: np.ndarray,
    *,
    teacher_temperature: float = 0.05,
    pred_temperature: float = 0.05,
) -> dict[str, float]:
    n = int(sim.shape[0])
    if n < 3:
        return {
            "spearman": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "ndcg_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "teacher_perplexity": 0.0,
            "model_perplexity": 0.0,
            "soft_cross_entropy": 0.0,
            "soft_entropy": 0.0,
            "soft_kl": 0.0,
            "adaptive_recall": 0.0,
            "soft_mass_at_5": 0.0,
        }
    spearman_values = []
    recall5 = []
    recall10 = []
    ndcg5 = []
    ndcg10 = []
    teacher_perplexity = []
    model_perplexity = []
    soft_cross_entropy = []
    soft_entropy = []
    soft_kl = []
    adaptive_recall = []
    soft_mass5 = []
    teacher_tau = max(float(teacher_temperature), 1.0e-6)
    pred_tau = max(float(pred_temperature), 1.0e-6)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        truth_scores = sim[i, mask]
        pred_scores = -pred_distance[i, mask]
        if truth_scores.size > 1 and np.std(truth_scores) > 1.0e-12 and np.std(pred_scores) > 1.0e-12:
            spearman_values.append(float(np.corrcoef(_rankdata_np(truth_scores), _rankdata_np(pred_scores))[0, 1]))
        truth_order = np.argsort(truth_scores)[::-1]
        pred_order = np.argsort(pred_scores)[::-1]
        for k, recalls, ndcgs in ((5, recall5, ndcg5), (10, recall10, ndcg10)):
            kk = min(k, truth_scores.size)
            if kk <= 0:
                continue
            truth_top = set(int(x) for x in truth_order[:kk])
            pred_top = [int(x) for x in pred_order[:kk]]
            recalls.append(len(truth_top.intersection(pred_top)) / float(kk))
            gains = np.maximum(truth_scores[pred_top], 0.0)
            discounts = 1.0 / np.log2(np.arange(2, kk + 2, dtype=np.float64))
            dcg = float(np.sum(gains * discounts))
            ideal = float(np.sum(np.maximum(truth_scores[truth_order[:kk]], 0.0) * discounts))
            ndcgs.append(dcg / ideal if ideal > 1.0e-12 else 0.0)
        p_teacher = _softmax_np(truth_scores / teacher_tau)
        p_model = _softmax_np(pred_scores / pred_tau)
        entropy = -float(np.sum(p_teacher * np.log(np.clip(p_teacher, 1.0e-12, 1.0))))
        model_entropy = -float(np.sum(p_model * np.log(np.clip(p_model, 1.0e-12, 1.0))))
        cross_entropy = -float(np.sum(p_teacher * np.log(np.clip(p_model, 1.0e-12, 1.0))))
        teacher_perplexity.append(float(np.exp(entropy)))
        model_perplexity.append(float(np.exp(model_entropy)))
        soft_entropy.append(entropy)
        soft_cross_entropy.append(cross_entropy)
        soft_kl.append(max(0.0, cross_entropy - entropy))
        adaptive_k = min(truth_scores.size, max(1, int(np.ceil(np.exp(entropy)))))
        adaptive_truth_top = set(int(x) for x in truth_order[:adaptive_k])
        adaptive_pred_top = [int(x) for x in pred_order[:adaptive_k]]
        adaptive_recall.append(len(adaptive_truth_top.intersection(adaptive_pred_top)) / float(adaptive_k))
        soft_mass5.append(float(np.sum(p_teacher[pred_order[: min(5, truth_scores.size)]])))
    return {
        "spearman": float(np.mean(spearman_values)) if spearman_values else 0.0,
        "recall_at_5": float(np.mean(recall5)) if recall5 else 0.0,
        "recall_at_10": float(np.mean(recall10)) if recall10 else 0.0,
        "ndcg_at_5": float(np.mean(ndcg5)) if ndcg5 else 0.0,
        "ndcg_at_10": float(np.mean(ndcg10)) if ndcg10 else 0.0,
        "teacher_perplexity": float(np.mean(teacher_perplexity)) if teacher_perplexity else 0.0,
        "model_perplexity": float(np.mean(model_perplexity)) if model_perplexity else 0.0,
        "soft_cross_entropy": float(np.mean(soft_cross_entropy)) if soft_cross_entropy else 0.0,
        "soft_entropy": float(np.mean(soft_entropy)) if soft_entropy else 0.0,
        "soft_kl": float(np.mean(soft_kl)) if soft_kl else 0.0,
        "adaptive_recall": float(np.mean(adaptive_recall)) if adaptive_recall else 0.0,
        "soft_mass_at_5": float(np.mean(soft_mass5)) if soft_mass5 else 0.0,
    }


def evaluate_projection(
    model: CheeseGraphTransformer,
    dataset: CheeseProjectionDataset,
    positions: np.ndarray,
    *,
    batch_size: int,
    pad_to: int | None,
    loss_mode: str,
    distance_transform: str,
    distance_scale: float,
    distance_power: float,
    teacher_temperature: float = 0.05,
    pred_temperature: float = 0.05,
) -> dict[str, float]:
    if len(positions) == 0:
        return {
            "mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "corr": 0.0,
            "spearman": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "ndcg_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "teacher_perplexity": 0.0,
            "model_perplexity": 0.0,
            "soft_cross_entropy": 0.0,
            "soft_entropy": 0.0,
            "soft_kl": 0.0,
            "adaptive_recall": 0.0,
            "soft_mass_at_5": 0.0,
            "n_pairs": 0.0,
        }
    embeddings = encode_positions(model, dataset, positions, batch_size=batch_size, pad_to=pad_to)
    sim = dataset.teacher[np.ix_(positions, positions)]
    if loss_mode == "cosine_similarity":
        pred = embeddings @ embeddings.T
        target = 2.0 * sim - 1.0
        retrieval_distance = -pred
    elif loss_mode in {"euclidean_dissimilarity", "ranked_dissimilarity", "soft_neighborhood", "hybrid_shape", "contrastive_shape"}:
        diff = embeddings[:, None, :] - embeddings[None, :, :]
        pred = np.sqrt(np.sum(diff * diff, axis=-1) + 1.0e-8)
        if distance_transform == "one_minus_similarity":
            target_distance = np.maximum(1.0 - sim, 0.0)
        elif distance_transform == "neglog_similarity":
            target_distance = -np.log(np.clip(sim, 1.0e-6, 1.0))
        else:
            raise ValueError(f"unknown distance_transform {distance_transform!r}")
        target = float(distance_scale) * target_distance ** float(distance_power)
        retrieval_distance = pred
    else:
        raise ValueError(f"unknown loss_mode {loss_mode!r}")
    mask = ~np.eye(len(positions), dtype=bool)
    err = pred[mask] - target[mask]
    pred_flat = pred[mask]
    target_flat = target[mask]
    corr = 0.0
    if pred_flat.size > 1 and float(np.std(pred_flat)) > 1.0e-12 and float(np.std(target_flat)) > 1.0e-12:
        corr = float(np.corrcoef(pred_flat, target_flat)[0, 1])
    retrieval = _retrieval_metrics_np(
        retrieval_distance,
        sim,
        teacher_temperature=teacher_temperature,
        pred_temperature=pred_temperature,
    )
    out = {
        "mse": float(np.mean(err * err)) if err.size else 0.0,
        "mae": float(np.mean(np.abs(err))) if err.size else 0.0,
        "rmse": float(np.sqrt(np.mean(err * err))) if err.size else 0.0,
        "corr": corr,
        "n_pairs": float(err.size),
    }
    out.update(retrieval)
    return out


def sample_anchor_topk_block(
    dataset: CheeseProjectionDataset,
    train_positions: np.ndarray,
    rng: np.random.Generator,
    *,
    batch_size: int,
    anchor_rows: int,
    positive_k: int,
    hard_negative_k: int,
    random_negative_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = rng.choice(train_positions, size=min(anchor_rows, len(train_positions)), replace=False)
    train_positions = np.asarray(train_positions, dtype=np.int64)
    train_set = set(int(x) for x in train_positions)
    columns: list[int] = []
    n_random = max(1, int(round(max(batch_size, 1) * float(random_negative_fraction))))
    for anchor in anchors:
        scores = dataset.teacher[int(anchor), train_positions]
        order = train_positions[np.argsort(scores)[::-1]]
        order = np.asarray([int(x) for x in order if int(x) != int(anchor)], dtype=np.int64)
        if order.size:
            top = order[: max(1, positive_k)]
            columns.append(int(rng.choice(top)))
            hard_start = min(max(1, positive_k), order.size)
            hard_end = min(order.size, hard_start + max(1, hard_negative_k))
            hard = order[hard_start:hard_end] if hard_end > hard_start else order[-min(order.size, max(1, hard_negative_k)) :]
            columns.append(int(rng.choice(hard)))
    if len(train_positions):
        columns.extend(int(x) for x in rng.choice(train_positions, size=min(n_random, len(train_positions)), replace=False))
    columns.extend(int(x) for x in anchors)
    unique_columns = []
    seen = set()
    for value in columns:
        if value in train_set and value not in seen:
            unique_columns.append(value)
            seen.add(value)
    if len(unique_columns) > batch_size:
        unique_columns = list(rng.choice(np.asarray(unique_columns, dtype=np.int64), size=batch_size, replace=False))
    return np.asarray(anchors, dtype=np.int64), np.asarray(unique_columns, dtype=np.int64)


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
        "--teacher-transform",
        choices=["raw", "anchor_zscore", "size_residual"],
        default="raw",
        help="Preprocess the teacher scores before training.",
    )
    parser.add_argument(
        "--zscore-temperature",
        type=float,
        default=1.0,
        help="Temperature used by anchor_zscore and size_residual teacher transforms.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=[
            "euclidean_dissimilarity",
            "cosine_similarity",
            "ranked_dissimilarity",
            "soft_neighborhood",
            "hybrid_shape",
            "contrastive_shape",
        ],
        default="euclidean_dissimilarity",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for the train/validation split. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Seed for stochastic batch sampling. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--optimizer", choices=["adamw", "muonv2w"], default="adamw")
    parser.add_argument("--muon-lr", type=float, default=0.0, help="0 means 10x --lr")
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-polar", choices=["jordan", "polar_express", "gram_ns"], default="polar_express")
    parser.add_argument("--muon-filter", choices=["opencheese_hidden", "all_2d"], default="opencheese_hidden")
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--use-normuon", action="store_true")
    parser.add_argument("--muonplus-mode", choices=["none", "col", "row", "col_row", "row_col"], default="none")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--bond-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--rbf", type=int, default=32)
    parser.add_argument("--metric-weight", type=float, default=1.0)
    parser.add_argument("--atom-weight", type=float, default=0.02)
    parser.add_argument("--distance-weight", type=float, default=0.02)
    parser.add_argument(
        "--distance-transform",
        choices=["one_minus_similarity", "neglog_similarity"],
        default="one_minus_similarity",
    )
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--distance-power", type=float, default=1.0)
    parser.add_argument("--top-sim-weight", type=float, default=0.0)
    parser.add_argument("--top-sim-center", type=float, default=0.70)
    parser.add_argument("--top-sim-temperature", type=float, default=0.05)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--rank-min-delta", type=float, default=0.03)
    parser.add_argument("--rank-margin", type=float, default=0.02)
    parser.add_argument("--rank-temperature", type=float, default=0.05)
    parser.add_argument("--soft-neighborhood-weight", type=float, default=0.0)
    parser.add_argument("--soft-target-temperature", type=float, default=0.05)
    parser.add_argument("--soft-pred-temperature", type=float, default=0.05)
    parser.add_argument(
        "--perplexity-temperature",
        type=float,
        default=None,
        help="Teacher temperature for perplexity/soft-KL metrics. Defaults to --soft-target-temperature.",
    )
    parser.add_argument(
        "--perplexity-pred-temperature",
        type=float,
        default=None,
        help="Model temperature for perplexity/soft-KL metrics. Defaults to --soft-pred-temperature.",
    )
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-positive-threshold", type=float, default=0.75)
    parser.add_argument("--contrastive-positive-top-k", type=int, default=0)
    parser.add_argument("--sampler", choices=["random_blocks", "anchor_topk"], default="random_blocks")
    parser.add_argument("--anchor-rows", type=int, default=0, help="0 means batch-size//2 for anchor_topk")
    parser.add_argument("--positive-k", type=int, default=16)
    parser.add_argument("--hard-negative-k", type=int, default=128)
    parser.add_argument("--random-negative-fraction", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means all row/column batch blocks")
    parser.add_argument("--dynamic-pad", action="store_true")
    parser.add_argument(
        "--include-self-pairs",
        action="store_true",
        help="Include i==j pairs in training losses. By default they are excluded for degenerate shape learning.",
    )
    parser.add_argument("--no-charges", action="store_true", help="Train a pure shape model without charge features.")
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
        teacher_transform=args.teacher_transform,
        zscore_temperature=args.zscore_temperature,
    )
    split_seed = int(args.seed if args.split_seed is None else args.split_seed)
    sampler_seed = int(args.seed if args.sampler_seed is None else args.sampler_seed)
    perplexity_temperature = float(
        args.soft_target_temperature if args.perplexity_temperature is None else args.perplexity_temperature
    )
    perplexity_pred_temperature = float(
        args.soft_pred_temperature if args.perplexity_pred_temperature is None else args.perplexity_pred_temperature
    )
    train_positions, valid_positions = split_positions(
        dataset.n_molecules,
        valid_fraction=args.valid_fraction,
        seed=split_seed,
    )
    pad_to = None if args.dynamic_pad else dataset.max_atoms
    config = CheeseEmbeddingConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        bond_dim=args.bond_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_rbf=args.rbf,
        use_charges=not bool(args.no_charges),
    )
    model = CheeseGraphTransformer(config)
    if args.init_weights is not None:
        if not args.init_weights.exists():
            raise FileNotFoundError(args.init_weights)
        model.load_weights(str(args.init_weights))
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "muonv2w":
        optimizer = MuonV2W(
            muon_lr=float(args.muon_lr) if args.muon_lr > 0 else float(args.lr) * 10.0,
            adamw_lr=float(args.lr),
            muon_momentum=float(args.muon_momentum),
            muon_weight_decay=float(args.muon_weight_decay),
            adamw_weight_decay=float(args.weight_decay),
            muon_ns_steps=int(args.muon_ns_steps),
            polar_method=args.muon_polar,
            use_normuon=bool(args.use_normuon),
            muonplus_mode=args.muonplus_mode,
            filter_mode=args.muon_filter,
        )
    else:
        raise ValueError(f"unknown optimizer {args.optimizer!r}")

    def loss_fn(m, row_batch, col_batch, target_similarity, valid_pair_mask):
        return train_loss(
            m,
            row_batch,
            col_batch,
            target_similarity,
            metric_weight=args.metric_weight,
            atom_weight=args.atom_weight,
            distance_weight=args.distance_weight,
            loss_mode=args.loss_mode,
            distance_transform=args.distance_transform,
            distance_scale=args.distance_scale,
            distance_power=args.distance_power,
            top_sim_weight=args.top_sim_weight,
            top_sim_center=args.top_sim_center,
            top_sim_temperature=args.top_sim_temperature,
            rank_weight=args.rank_weight,
            rank_min_delta=args.rank_min_delta,
            rank_margin=args.rank_margin,
            rank_temperature=args.rank_temperature,
            soft_neighborhood_weight=args.soft_neighborhood_weight,
            soft_target_temperature=args.soft_target_temperature,
            soft_pred_temperature=args.soft_pred_temperature,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_positive_threshold=args.contrastive_positive_threshold,
            contrastive_positive_top_k=args.contrastive_positive_top_k,
            valid_pair_mask=valid_pair_mask,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out_dir / "best.safetensors"
    best_recall5_path = args.out_dir / "best_recall_at_5.safetensors"
    best_spearman_path = args.out_dir / "best_spearman.safetensors"
    last_path = args.out_dir / "last.safetensors"
    metrics_path = args.out_dir / "metrics.csv"
    config_path = args.out_dir / "config.json"
    embeddings_path = args.out_dir / "embeddings.npz"
    embeddings_recall5_path = args.out_dir / "embeddings_recall_at_5.npz"
    embeddings_spearman_path = args.out_dir / "embeddings_spearman.npz"

    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    print(
        f"Training openCHEESE projection n={dataset.n_molecules} train={len(train_positions)} "
        f"valid={len(valid_positions)} target={dataset.target} channel={dataset.teacher_channel} "
        f"transform={args.teacher_transform} loss={args.loss_mode} optimizer={args.optimizer} "
        f"sampler={args.sampler} charges={not args.no_charges} batch={args.batch_size} "
        f"split_seed={split_seed} sampler_seed={sampler_seed} self_pairs={bool(args.include_self_pairs)}"
        + (f" init={args.init_weights}" if args.init_weights is not None else ""),
        flush=True,
    )

    rows: list[dict[str, float | int]] = []
    best_valid = float("inf")
    best_epoch = 0
    best_recall5 = -float("inf")
    best_recall5_epoch = 0
    best_spearman = -float("inf")
    best_spearman_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        step_losses = []
        rng = np.random.default_rng(sampler_seed + epoch)

        if args.sampler == "anchor_topk":
            steps = args.steps_per_epoch if args.steps_per_epoch > 0 else max(1, int(np.ceil(len(train_positions) / args.batch_size)))
            anchor_rows = int(args.anchor_rows) if args.anchor_rows > 0 else max(1, args.batch_size // 2)
            block_pairs = [
                sample_anchor_topk_block(
                    dataset,
                    train_positions,
                    rng,
                    batch_size=args.batch_size,
                    anchor_rows=anchor_rows,
                    positive_k=args.positive_k,
                    hard_negative_k=args.hard_negative_k,
                    random_negative_fraction=args.random_negative_fraction,
                )
                for _ in range(steps)
            ]
        elif args.steps_per_epoch > 0:
            block_pairs = [
                (
                    rng.choice(train_positions, size=min(args.batch_size, len(train_positions)), replace=False),
                    rng.choice(train_positions, size=min(args.batch_size, len(train_positions)), replace=False),
                )
                for _ in range(args.steps_per_epoch)
            ]
        else:
            row_blocks = list(iter_batches(train_positions, batch_size=args.batch_size, shuffle=True, seed=sampler_seed + epoch))
            col_blocks = list(iter_batches(train_positions, batch_size=args.batch_size, shuffle=True, seed=sampler_seed + 17 * epoch))
            block_pairs = [(row_block, col_block) for row_block in row_blocks for col_block in col_blocks]

        for row_positions, col_positions in block_pairs:
            row_batch = dataset.batch(row_positions, pad_to=pad_to)
            col_batch = dataset.batch(col_positions, pad_to=pad_to)
            target_block = mx.array(dataset.teacher[np.ix_(row_positions, col_positions)], dtype=mx.float32)
            if args.include_self_pairs:
                valid_pair_mask = mx.ones(target_block.shape, dtype=mx.float32)
            else:
                valid_pair_mask = mx.array(
                    (np.asarray(row_positions)[:, None] != np.asarray(col_positions)[None, :]).astype(np.float32)
                )
            loss, grads = loss_and_grad(model, row_batch, col_batch, target_block, valid_pair_mask)
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
            distance_transform=args.distance_transform,
            distance_scale=args.distance_scale,
            distance_power=args.distance_power,
            teacher_temperature=perplexity_temperature,
            pred_temperature=perplexity_pred_temperature,
        )
        valid_eval = evaluate_projection(
            model,
            dataset,
            valid_positions,
            batch_size=args.batch_size,
            pad_to=pad_to,
            loss_mode=args.loss_mode,
            distance_transform=args.distance_transform,
            distance_scale=args.distance_scale,
            distance_power=args.distance_power,
            teacher_temperature=perplexity_temperature,
            pred_temperature=perplexity_pred_temperature,
        )
        row = {
            "epoch": epoch,
            "step_loss": float(np.mean(step_losses)) if step_losses else 0.0,
            "train_mse": train_eval["mse"],
            "train_mae": train_eval["mae"],
            "train_rmse": train_eval["rmse"],
            "train_corr": train_eval["corr"],
            "train_spearman": train_eval["spearman"],
            "train_recall_at_5": train_eval["recall_at_5"],
            "train_recall_at_10": train_eval["recall_at_10"],
            "train_ndcg_at_5": train_eval["ndcg_at_5"],
            "train_ndcg_at_10": train_eval["ndcg_at_10"],
            "train_teacher_perplexity": train_eval["teacher_perplexity"],
            "train_model_perplexity": train_eval["model_perplexity"],
            "train_soft_kl": train_eval["soft_kl"],
            "train_soft_cross_entropy": train_eval["soft_cross_entropy"],
            "train_adaptive_recall": train_eval["adaptive_recall"],
            "train_soft_mass_at_5": train_eval["soft_mass_at_5"],
            "valid_mse": valid_eval["mse"],
            "valid_mae": valid_eval["mae"],
            "valid_rmse": valid_eval["rmse"],
            "valid_corr": valid_eval["corr"],
            "valid_spearman": valid_eval["spearman"],
            "valid_recall_at_5": valid_eval["recall_at_5"],
            "valid_recall_at_10": valid_eval["recall_at_10"],
            "valid_ndcg_at_5": valid_eval["ndcg_at_5"],
            "valid_ndcg_at_10": valid_eval["ndcg_at_10"],
            "valid_teacher_perplexity": valid_eval["teacher_perplexity"],
            "valid_model_perplexity": valid_eval["model_perplexity"],
            "valid_soft_kl": valid_eval["soft_kl"],
            "valid_soft_cross_entropy": valid_eval["soft_cross_entropy"],
            "valid_adaptive_recall": valid_eval["adaptive_recall"],
            "valid_soft_mass_at_5": valid_eval["soft_mass_at_5"],
            "seconds": time.perf_counter() - epoch_start,
        }
        rows.append(row)
        if valid_eval["mse"] < best_valid:
            best_valid = valid_eval["mse"]
            best_epoch = epoch
            model.save_weights(str(best_path))
        if valid_eval["recall_at_5"] > best_recall5:
            best_recall5 = valid_eval["recall_at_5"]
            best_recall5_epoch = epoch
            model.save_weights(str(best_recall5_path))
        if valid_eval["spearman"] > best_spearman:
            best_spearman = valid_eval["spearman"]
            best_spearman_epoch = epoch
            model.save_weights(str(best_spearman_path))
        if epoch == 1 or epoch == args.epochs or epoch % max(args.eval_every, 1) == 0:
            print(
                f"epoch {epoch:04d}/{args.epochs} "
                f"train_mse={train_eval['mse']:.5f} valid_mse={valid_eval['mse']:.5f} "
                f"valid_corr={valid_eval['corr']:.3f} valid_spearman={valid_eval['spearman']:.3f} "
                f"recall5={valid_eval['recall_at_5']:.3f} soft_kl={valid_eval['soft_kl']:.3f} "
                f"ppl={valid_eval['teacher_perplexity']:.1f} best_mse={best_valid:.5f}@{best_epoch} "
                f"best_r5={best_recall5:.3f}@{best_recall5_epoch}",
                flush=True,
            )

    model.save_weights(str(last_path))
    _write_metrics(metrics_path, rows)

    best_model = CheeseGraphTransformer(config)
    best_model.load_weights(str(best_path))
    save_embeddings(best_model, dataset, embeddings_path, batch_size=args.batch_size, pad_to=pad_to)
    best_recall5_model = CheeseGraphTransformer(config)
    best_recall5_model.load_weights(str(best_recall5_path))
    save_embeddings(best_recall5_model, dataset, embeddings_recall5_path, batch_size=args.batch_size, pad_to=pad_to)
    best_spearman_model = CheeseGraphTransformer(config)
    best_spearman_model.load_weights(str(best_spearman_path))
    save_embeddings(best_spearman_model, dataset, embeddings_spearman_path, batch_size=args.batch_size, pad_to=pad_to)

    summary = {
        "dataset": str(args.data),
        "teacher": str(args.teacher),
        "init_weights": None if args.init_weights is None else str(args.init_weights),
        "target": dataset.target,
        "teacher_channel": dataset.teacher_channel,
        "teacher_transform": args.teacher_transform,
        "teacher_transform_metadata": dataset.teacher_transform_metadata,
        "loss_mode": args.loss_mode,
        "optimizer": args.optimizer,
        "sampler": args.sampler,
        "seed": int(args.seed),
        "split_seed": split_seed,
        "sampler_seed": sampler_seed,
        "include_self_pairs": bool(args.include_self_pairs),
        "use_charges": not bool(args.no_charges),
        "distance_transform": args.distance_transform,
        "distance_scale": args.distance_scale,
        "distance_power": args.distance_power,
        "top_sim_weight": args.top_sim_weight,
        "rank_weight": args.rank_weight,
        "soft_neighborhood_weight": args.soft_neighborhood_weight,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_temperature": args.contrastive_temperature,
        "contrastive_positive_threshold": args.contrastive_positive_threshold,
        "contrastive_positive_top_k": args.contrastive_positive_top_k,
        "perplexity_temperature": perplexity_temperature,
        "perplexity_pred_temperature": perplexity_pred_temperature,
        "best_epoch": best_epoch,
        "best_valid_mse": best_valid,
        "best_weights": str(best_path),
        "best_recall_at_5_epoch": best_recall5_epoch,
        "best_valid_recall_at_5": best_recall5,
        "best_recall_at_5_weights": str(best_recall5_path),
        "best_spearman_epoch": best_spearman_epoch,
        "best_valid_spearman": best_spearman,
        "best_spearman_weights": str(best_spearman_path),
        "last_weights": str(last_path),
        "embeddings": str(embeddings_path),
        "embeddings_recall_at_5": str(embeddings_recall5_path),
        "embeddings_spearman": str(embeddings_spearman_path),
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
