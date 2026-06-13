"""GraphMVP-style 2D/3D pretraining for openCHEESE.

This is an MLX-native adaptation of the useful GraphMVP objective shape:
contrast 2D graph and 3D conformer views for the same molecule, then reconstruct
each view's representation from the other view through small projector heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from opencheese.embedding import (
    CheeseEmbeddingBatch,
    CheeseEmbeddingConfig,
    CheeseGraphTransformer,
    embedding_cosine_similarity_matrix_mlx,
    l2_normalize_mlx,
)


@dataclass(frozen=True)
class GraphMVPLoss:
    total: mx.array
    contrastive: mx.array
    reconstruction: mx.array
    uniformity: mx.array
    variance: mx.array
    accuracy_2d_to_3d: mx.array
    accuracy_3d_to_2d: mx.array


class RepresentationProjector(nn.Module):
    """Small MLP used for representation reconstruction between views."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(hidden_dim or embedding_dim)
        self.layers = [
            nn.Linear(int(embedding_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(embedding_dim)),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class GraphMVPPretrainer(nn.Module):
    """Two-view openCHEESE pretraining wrapper.

    The 2D encoder consumes atom/bond topology with zeroed coordinates. The 3D
    encoder consumes conformer coordinates. Both are ordinary
    :class:`CheeseGraphTransformer` modules so their weights can seed later
    openCHEESE projection fine-tuning.
    """

    def __init__(
        self,
        config_2d: CheeseEmbeddingConfig,
        config_3d: CheeseEmbeddingConfig,
        *,
        projector_hidden_dim: int | None = None,
    ):
        super().__init__()
        if int(config_2d.embedding_dim) != int(config_3d.embedding_dim):
            raise ValueError("2D and 3D embedding dimensions must match")
        self.model_2d = CheeseGraphTransformer(config_2d)
        self.model_3d = CheeseGraphTransformer(config_3d)
        self.project_2d_to_3d = RepresentationProjector(config_2d.embedding_dim, projector_hidden_dim)
        self.project_3d_to_2d = RepresentationProjector(config_2d.embedding_dim, projector_hidden_dim)

    def encode_2d(self, batch: CheeseEmbeddingBatch) -> mx.array:
        return self.model_2d.encode_batch(batch)

    def encode_3d(self, batch: CheeseEmbeddingBatch) -> mx.array:
        return self.model_3d.encode_batch(batch)

    def __call__(
        self,
        batch_2d: CheeseEmbeddingBatch,
        batch_3d: CheeseEmbeddingBatch,
        *,
        temperature: float = 0.1,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        uniformity_weight: float = 0.0,
        variance_weight: float = 0.0,
        uniformity_temperature: float = 2.0,
        variance_target: float = 0.05,
        reconstruction_loss: Literal["l2", "l1", "cosine"] = "l2",
        detach_target: bool = True,
    ) -> GraphMVPLoss:
        embedding_2d = self.encode_2d(batch_2d)
        embedding_3d = self.encode_3d(batch_3d)
        contrastive, acc_2d, acc_3d = dual_info_nce_loss_mlx(
            embedding_2d,
            embedding_3d,
            temperature=temperature,
        )
        reconstruction = bidirectional_representation_reconstruction_loss_mlx(
            embedding_2d,
            embedding_3d,
            self.project_2d_to_3d,
            self.project_3d_to_2d,
            loss=reconstruction_loss,
            detach_target=detach_target,
        )
        uniformity = symmetric_sphere_uniformity_loss_mlx(
            embedding_2d,
            embedding_3d,
            temperature=uniformity_temperature,
        )
        variance = symmetric_batch_variance_loss_mlx(
            embedding_2d,
            embedding_3d,
            target_std=variance_target,
        )
        total = (
            float(contrastive_weight) * contrastive
            + float(reconstruction_weight) * reconstruction
            + float(uniformity_weight) * uniformity
            + float(variance_weight) * variance
        )
        return GraphMVPLoss(
            total=total,
            contrastive=contrastive,
            reconstruction=reconstruction,
            uniformity=uniformity,
            variance=variance,
            accuracy_2d_to_3d=acc_2d,
            accuracy_3d_to_2d=acc_3d,
        )


def info_nce_loss_mlx(
    query: mx.array,
    key: mx.array,
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> tuple[mx.array, mx.array]:
    """Return row-wise InfoNCE loss and top-1 matching accuracy."""

    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batches must have the same size")
    if normalize:
        query = l2_normalize_mlx(query)
        key = l2_normalize_mlx(key)
    logits = query @ mx.transpose(key, (1, 0))
    logits = logits / max(float(temperature), 1.0e-6)
    labels = mx.arange(query.shape[0], dtype=mx.int32)
    loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
    pred = mx.argmax(logits, axis=1).astype(mx.int32)
    accuracy = mx.mean((pred == labels).astype(mx.float32))
    return loss, accuracy


def dual_info_nce_loss_mlx(
    view_a: mx.array,
    view_b: mx.array,
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Symmetric InfoNCE: 2D->3D and 3D->2D."""

    loss_ab, acc_ab = info_nce_loss_mlx(view_a, view_b, temperature=temperature, normalize=normalize)
    loss_ba, acc_ba = info_nce_loss_mlx(view_b, view_a, temperature=temperature, normalize=normalize)
    return 0.5 * (loss_ab + loss_ba), acc_ab, acc_ba


def representation_reconstruction_loss_mlx(
    source: mx.array,
    target: mx.array,
    projector: RepresentationProjector,
    *,
    loss: Literal["l2", "l1", "cosine"] = "l2",
    detach_target: bool = True,
) -> mx.array:
    """Reconstruct a target representation from a source representation."""

    pred = projector(source)
    target = mx.stop_gradient(target) if detach_target else target
    if loss == "l2":
        return mx.mean((pred - target) ** 2)
    if loss == "l1":
        return mx.mean(mx.abs(pred - target))
    if loss == "cosine":
        pred = l2_normalize_mlx(pred)
        target = l2_normalize_mlx(target)
        return -mx.mean(mx.sum(pred * target, axis=-1))
    raise ValueError("loss must be 'l2', 'l1', or 'cosine'")


def bidirectional_representation_reconstruction_loss_mlx(
    view_a: mx.array,
    view_b: mx.array,
    projector_a_to_b: RepresentationProjector,
    projector_b_to_a: RepresentationProjector,
    *,
    loss: Literal["l2", "l1", "cosine"] = "l2",
    detach_target: bool = True,
) -> mx.array:
    loss_ab = representation_reconstruction_loss_mlx(
        view_a,
        view_b,
        projector_a_to_b,
        loss=loss,
        detach_target=detach_target,
    )
    loss_ba = representation_reconstruction_loss_mlx(
        view_b,
        view_a,
        projector_b_to_a,
        loss=loss,
        detach_target=detach_target,
    )
    return 0.5 * (loss_ab + loss_ba)


def sphere_uniformity_loss_mlx(
    embeddings: mx.array,
    *,
    temperature: float = 2.0,
    eps: float = 1.0e-8,
) -> mx.array:
    """Uniformity loss on the unit sphere.

    Collapsed embeddings have loss near zero. Well-spread embeddings produce a
    more negative value, so minimizing this term discourages the common
    contrastive-pretraining collapse where every molecule maps to one point.
    """

    embeddings = l2_normalize_mlx(embeddings)
    n = int(embeddings.shape[0])
    if n < 2:
        return mx.array(0.0, dtype=embeddings.dtype)
    cosine = embeddings @ mx.transpose(embeddings, (1, 0))
    sqdist = mx.maximum(2.0 - 2.0 * cosine, mx.array(0.0, dtype=embeddings.dtype))
    mask = 1.0 - mx.eye(n, dtype=embeddings.dtype)
    values = mx.exp(-float(temperature) * sqdist) * mask
    denom = mx.maximum(mx.sum(mask), mx.array(float(eps), dtype=embeddings.dtype))
    return mx.log(mx.sum(values) / denom + mx.array(float(eps), dtype=embeddings.dtype))


def symmetric_sphere_uniformity_loss_mlx(
    view_a: mx.array,
    view_b: mx.array,
    *,
    temperature: float = 2.0,
) -> mx.array:
    return 0.5 * (
        sphere_uniformity_loss_mlx(view_a, temperature=temperature)
        + sphere_uniformity_loss_mlx(view_b, temperature=temperature)
    )


def batch_variance_loss_mlx(
    embeddings: mx.array,
    *,
    target_std: float = 0.05,
    eps: float = 1.0e-4,
) -> mx.array:
    """Batch variance floor used as a cheap anti-collapse regularizer."""

    n = int(embeddings.shape[0])
    if n < 2:
        return mx.array(0.0, dtype=embeddings.dtype)
    centered = embeddings - mx.mean(embeddings, axis=0, keepdims=True)
    variance = mx.mean(centered * centered, axis=0)
    std = mx.sqrt(variance + mx.array(float(eps), dtype=embeddings.dtype))
    return mx.mean(mx.maximum(float(target_std) - std, mx.array(0.0, dtype=embeddings.dtype)))


def symmetric_batch_variance_loss_mlx(
    view_a: mx.array,
    view_b: mx.array,
    *,
    target_std: float = 0.05,
) -> mx.array:
    return 0.5 * (
        batch_variance_loss_mlx(view_a, target_std=target_std)
        + batch_variance_loss_mlx(view_b, target_std=target_std)
    )


__all__ = [
    "GraphMVPLoss",
    "GraphMVPPretrainer",
    "RepresentationProjector",
    "batch_variance_loss_mlx",
    "bidirectional_representation_reconstruction_loss_mlx",
    "dual_info_nce_loss_mlx",
    "info_nce_loss_mlx",
    "representation_reconstruction_loss_mlx",
    "sphere_uniformity_loss_mlx",
    "symmetric_batch_variance_loss_mlx",
    "symmetric_sphere_uniformity_loss_mlx",
]
