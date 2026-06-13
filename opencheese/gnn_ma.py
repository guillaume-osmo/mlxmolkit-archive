"""GNN-MA-inspired soft molecular alignment layers for openCHEESE.

The published GNN-MA model uses cross-graph attention to learn atom-level soft
correspondences for ligand-based virtual screening. This module keeps the core
idea in an MLX-native, 3D-compatible form: it scores pairs from padded atom
embeddings and masks, so it can sit on top of ``CheeseGraphTransformer`` atom
states or any other openCHEESE encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from opencheese.embedding import masked_mean_mlx


@dataclass(frozen=True)
class SoftAlignmentResult:
    """Pair score plus atom-level soft-alignment matrices."""

    logits: mx.array
    attention_ab: mx.array
    attention_ba: mx.array
    pooled_a: mx.array
    pooled_b: mx.array


def masked_softmax_mlx(scores: mx.array, mask: mx.array, *, axis: int = -1) -> mx.array:
    """Softmax with a broadcastable binary validity mask."""

    mask = mask.astype(mx.float32)
    masked = mx.where(mask > 0, scores, mx.array(-1.0e9, dtype=scores.dtype))
    probs = mx.softmax(masked, axis=axis) * mask
    denom = mx.sum(probs, axis=axis, keepdims=True)
    return probs / mx.maximum(denom, mx.array(1.0e-8, dtype=probs.dtype))


def cross_graph_attention_weights_mlx(
    query: mx.array,
    key: mx.array,
    key_mask: mx.array | None = None,
    *,
    temperature: float | None = None,
) -> mx.array:
    """Return ``query -> key`` atom correspondence weights.

    ``query`` is ``(B, NA, H)`` and ``key`` is ``(B, NB, H)``. The result is
    ``(B, NA, NB)``. This is the atom-level soft alignment matrix.
    """

    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape (batch, atoms, hidden)")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key batch/hidden dimensions must match")
    scale = float(temperature) if temperature is not None else query.shape[-1] ** 0.5
    logits = (query @ mx.transpose(key, (0, 2, 1))) / max(scale, 1.0e-6)
    if key_mask is None:
        return mx.softmax(logits, axis=-1)
    mask = key_mask.astype(mx.float32)[:, None, :]
    return masked_softmax_mlx(logits, mask, axis=-1)


class CrossGraphSoftAlignmentScorer(nn.Module):
    """Lightweight GNN-MA-style pair scorer over atom embeddings.

    The full GNN-MA architecture also performs edge-to-edge attention over
    flattened bond pairs. For openCHEESE we keep this first layer atom-level so
    it can be used cheaply in shortlist re-ranking or as an auxiliary loss.
    """

    def __init__(self, hidden_dim: int, *, pair_hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim)
        pair_hidden = int(pair_hidden_dim or hidden)
        self.hidden_dim = hidden
        self.norm = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden * 4, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, 1),
        )

    def __call__(
        self,
        atom_a: mx.array,
        mask_a: mx.array,
        atom_b: mx.array,
        mask_b: mx.array,
    ) -> SoftAlignmentResult:
        if atom_a.ndim != 3 or atom_b.ndim != 3:
            raise ValueError("atom embeddings must have shape (batch, atoms, hidden)")
        if atom_a.shape[0] != atom_b.shape[0]:
            raise ValueError("paired atom embedding batches must have the same size")
        if atom_a.shape[-1] != self.hidden_dim or atom_b.shape[-1] != self.hidden_dim:
            raise ValueError(f"atom embeddings must have hidden_dim={self.hidden_dim}")

        mask_a = mask_a.astype(mx.float32)
        mask_b = mask_b.astype(mx.float32)
        a = self.norm(atom_a) * mask_a[:, :, None]
        b = self.norm(atom_b) * mask_b[:, :, None]
        qa = self.q_proj(a)
        qb = self.q_proj(b)
        ka = self.k_proj(a)
        kb = self.k_proj(b)
        va = self.v_proj(a)
        vb = self.v_proj(b)

        attention_ab = cross_graph_attention_weights_mlx(qa, kb, mask_b)
        attention_ba = cross_graph_attention_weights_mlx(qb, ka, mask_a)
        context_ab = attention_ab @ vb
        context_ba = attention_ba @ va

        fused_a = self._fuse(a, context_ab) * mask_a[:, :, None]
        fused_b = self._fuse(b, context_ba) * mask_b[:, :, None]
        pooled_a = masked_mean_mlx(fused_a, mask_a, axis=1)
        pooled_b = masked_mean_mlx(fused_b, mask_b, axis=1)
        pair = mx.concatenate(
            [pooled_a, pooled_b, mx.abs(pooled_a - pooled_b), pooled_a * pooled_b],
            axis=-1,
        )
        logits = mx.squeeze(self.readout(pair), axis=-1)
        return SoftAlignmentResult(
            logits=logits,
            attention_ab=attention_ab,
            attention_ba=attention_ba,
            pooled_a=pooled_a,
            pooled_b=pooled_b,
        )

    def _fuse(self, atom: mx.array, context: mx.array) -> mx.array:
        return self.fusion(mx.concatenate([atom, context, mx.abs(atom - context), atom * context], axis=-1))


__all__ = [
    "CrossGraphSoftAlignmentScorer",
    "SoftAlignmentResult",
    "cross_graph_attention_weights_mlx",
    "masked_softmax_mlx",
]
