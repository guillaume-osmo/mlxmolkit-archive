"""Lower-triangle packing for NDDO two-centre integrals.

The two-electron integral (mu nu | lam sig) is symmetric under mu<->nu and
under lam<->sig, so storing it as a dense 4-index block wastes most of the
room: an sp atom has 16 ordered orbital pairs but only 10 distinct ones, and a
d atom has 81 against 45.

Packing collapses each centre onto its distinct pairs. The index is the
standard lower-triangle one,

    pack(i, j) = max(i,j) * (max(i,j) + 1) / 2 + min(i,j)

which is the convention the vendored PYSEQM TETCI port already emits, so the
d-orbital integrals arrive packed and are currently expanded on the way in.

The useful property is that the packings **nest**: for orbitals below 4 the
9-basis index equals the 4-basis index, so an sp pair block is literally the
leading 10x10 corner of a d pair block. One convention covers H (1 pair), sp
(10) and spd (45) — there is no need for separate sp and d storage, and no
branch on shell type when indexing.

    n orbitals      1     4     9
    packed pairs    1    10    45
    dense pairs     1    16    81
"""
from __future__ import annotations

import numpy as np

# Cache the (n, n) index matrices — they are tiny and rebuilt constantly.
_INDEX_CACHE: dict[int, np.ndarray] = {}


def packed_size(n_orbitals: int) -> int:
    """Number of distinct orbital pairs on a centre with `n_orbitals`."""
    return n_orbitals * (n_orbitals + 1) // 2


def pack_index(i: int, j: int) -> int:
    """Lower-triangle packed index of the orbital pair (i, j)."""
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j


def index_matrix(n_orbitals: int) -> np.ndarray:
    """(n, n) array whose [i, j] entry is `pack_index(i, j)`.

    Lets a dense block be gathered from a packed one by fancy indexing rather
    than a Python loop.
    """
    cached = _INDEX_CACHE.get(n_orbitals)
    if cached is not None:
        return cached
    i = np.arange(n_orbitals)[:, None]
    j = np.arange(n_orbitals)[None, :]
    hi = np.maximum(i, j)
    lo = np.minimum(i, j)
    matrix = (hi * (hi + 1) // 2 + lo).astype(np.int32)
    _INDEX_CACHE[n_orbitals] = matrix
    return matrix


def pack(dense: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Pack a dense (nA, nA, nB, nB) block into (packed(nA), packed(nB)).

    The dense block must already be symmetric in its first and last index
    pairs, which every NDDO two-electron block is; :func:`unpack` inverts this
    exactly.
    """
    out = np.zeros((packed_size(n_a), packed_size(n_b)), dtype=np.float64)
    rows = index_matrix(n_a).ravel()
    cols = index_matrix(n_b).ravel()
    # One scatter rather than a loop over the first index pair. Several
    # (mu, nu) map to the same packed row — that is the point of packing — and
    # they carry equal values by the symmetry this function assumes, so the
    # repeated writes agree and last-wins is harmless.
    out[rows[:, None], cols[None, :]] = dense.reshape(rows.size, cols.size)
    return out


def unpack(packed: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Expand a packed block back to dense (nA, nA, nB, nB).

    Used by the CPU reference path, which keeps the existing einsum
    contractions rather than rewriting them against packed indices.
    """
    ia = index_matrix(n_a)
    ib = index_matrix(n_b)
    return packed[np.ix_(ia.ravel(), ib.ravel())].reshape(n_a, n_a, n_b, n_b)


def pair_block_size(n_a: int, n_b: int) -> int:
    """Flat storage a single (A, B) pair block occupies."""
    return packed_size(n_a) * packed_size(n_b)


def pack_batch(dense: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Pack (P, nA, nA, nB, nB) -> (P, packed(nA), packed(nB)) in one scatter.

    :func:`pack` called per pair cost 132 ms across 64640 calls on an
    800-molecule batch. The index map does not depend on the pair, so the whole
    stack is scattered at once.
    """
    P = dense.shape[0]
    out = np.zeros((P, packed_size(n_a), packed_size(n_b)))
    rows = index_matrix(n_a).ravel()
    cols = index_matrix(n_b).ravel()
    out[:, rows[:, None], cols[None, :]] = dense.reshape(P, rows.size, cols.size)
    return out
