"""
Dense-vector cosine similarity (Metal-backed via MLX matmul).

For the binary-fingerprint Tanimoto pipeline see :mod:`mlxmolkit.tanimoto_metal_u32`.
This module is the corresponding primitive for **dense float fingerprints**
(e.g. ERG, autocorrelation, learned embeddings). MLX's matmul on Apple silicon
is already GPU-resident, so we don't write a custom Metal kernel — the only
work here is row-wise L2 normalization plus a `(a @ b.T)` matmul.

Public API:

- :func:`l2_normalize_rows`     — row-wise L2 normalization (eps-stable)
- :func:`cosine_matrix_dense`   — pairwise cosine `(Na, Nb)`
- :func:`max_cosine_to_set`     — `(Na,)` max cosine of each query against a set
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx


def l2_normalize_rows(x: mx.array, eps: float = 1e-9) -> mx.array:
    """Row-wise L2 normalization. `x` shape (N, D) → unit-norm rows.

    Uses ``mx.sqrt(mx.sum(x*x))`` (matmul-only path) so the whole thing stays
    on the active stream without falling back to ``mx.linalg`` CPU paths.
    """
    norms = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + eps
    return x / norms


def cosine_matrix_dense(
    a: mx.array,
    b: Optional[mx.array] = None,
    *,
    eps: float = 1e-9,
) -> mx.array:
    """Pairwise cosine similarity between row sets ``a`` and ``b``.

    Parameters
    ----------
    a : mx.array (Na, D), float
    b : mx.array (Nb, D), float, optional
        Defaults to ``a`` (square self-similarity).

    Returns
    -------
    sims : mx.array (Na, Nb), float32 in ``[-1, 1]``
    """
    if b is None:
        b = a
    a_n = l2_normalize_rows(a, eps=eps)
    b_n = l2_normalize_rows(b, eps=eps)
    return a_n @ b_n.T


def max_cosine_to_set(
    query: mx.array,
    reference: mx.array,
    *,
    eps: float = 1e-9,
) -> mx.array:
    """Per-row max cosine similarity of each query against the reference set.

    Returns a length-``Nq`` vector. If ``reference`` is empty, returns zeros
    (no reference → no similarity, which is the natural "no novelty signal yet"
    case in streaming sampling).
    """
    if reference.shape[0] == 0:
        return mx.zeros((query.shape[0],), dtype=mx.float32)
    return cosine_matrix_dense(query, reference, eps=eps).max(axis=-1)
