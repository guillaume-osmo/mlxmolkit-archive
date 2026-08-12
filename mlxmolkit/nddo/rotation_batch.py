"""Rotate two-centre integrals for many atom pairs at once.

:func:`~mlxmolkit.nddo.rotation.rotate_integrals_to_molecular_frame` walks a
four-deep Python loop over the 100 distinct orbital-pair combinations of an
sp-sp pair, doing scalar arithmetic in each. That costs ~75 us per pair and
dominates everything built on it: a benzaldehyde gradient spent 78% of its time
there, across 6544 calls.

Nothing about the arithmetic requires that. Every element of the rotated tensor
is a *linear* combination of the local-frame integrals ``ri``, with coefficients
that are polynomials in the three rotation vectors:

    w[kk, ll, mm, nn] = sum_t C[kk, ll, mm, nn, t](r0, r1, r2) * ri[t]

so the loop can be replaced by broadcasting over a leading pair axis. The 100
combinations fall into nine categories by which of kk, ll, mm, nn are zero, and
each category is one vectorised expression over all pairs at once.

This module keeps the same convention and the same ``ri`` layout as the scalar
version, which remains the reference it is tested against.
"""
from __future__ import annotations

import os

import numpy as np

# The 100 (kk, ll, mm, nn) combinations with kk >= ll and mm >= nn, grouped by
# category. Built once at import.
_CATEGORIES: dict[str, np.ndarray] = {}


def _build_categories() -> None:
    buckets: dict[str, list[tuple[int, int, int, int]]] = {
        "ss_ss": [], "ss_ps": [], "ss_pp": [],
        "ps_ss": [], "ps_ps": [], "ps_pp": [],
        "pp_ss": [], "pp_ps": [], "pp_pp": [],
    }
    for kk in range(4):
        for ll in range(kk + 1):
            for mm in range(4):
                for nn in range(mm + 1):
                    left = "ss" if kk == 0 else ("ps" if ll == 0 else "pp")
                    right = "ss" if mm == 0 else ("ps" if nn == 0 else "pp")
                    buckets[f"{left}_{right}"].append((kk, ll, mm, nn))
    for name, combos in buckets.items():
        _CATEGORIES[name] = np.array(combos, dtype=np.int64).reshape(-1, 4)


_build_categories()


def rotate_xx_batch(ri: np.ndarray, r0: np.ndarray, r1: np.ndarray,
                    r2: np.ndarray) -> np.ndarray:
    """Rotated (P, 4, 4, 4, 4) tensors for P heavy-heavy pairs.

    Args:
        ri: (P, 22) local-frame integrals, the layout the scalar routine uses.
        r0, r1, r2: (P, 3) rotation vectors — sigma, pi_x, pi_y.

    Returns:
        (P, 4, 4, 4, 4), symmetrised in the first index pair and the last, the
        same as the scalar routine produces.
    """
    P = ri.shape[0]
    w = np.zeros((P, 4, 4, 4, 4))

    def put(combos: np.ndarray, values: np.ndarray) -> None:
        """Scatter `values` (P, n_combos) into all four symmetric positions."""
        kk, ll, mm, nn = combos[:, 0], combos[:, 1], combos[:, 2], combos[:, 3]
        w[:, kk, ll, mm, nn] = values
        w[:, ll, kk, mm, nn] = values
        w[:, kk, ll, nn, mm] = values
        w[:, ll, kk, nn, mm] = values

    def orbital_indices(combos):
        """(kk, ll, mm, nn) -> the p-orbital indices k, l, m, n = idx - 1."""
        return (combos[:, 0] - 1, combos[:, 1] - 1,
                combos[:, 2] - 1, combos[:, 3] - 1)

    # Each block below is the scalar branch of the same name, broadcast over
    # the pair axis. `g` indexes combinations within the category.
    c = _CATEGORIES["ss_ss"]
    put(c, np.broadcast_to(ri[:, 0:1], (P, c.shape[0])))

    c = _CATEGORIES["ss_ps"]
    _, _, m, _ = orbital_indices(c)
    put(c, ri[:, 4:5] * r0[:, m])

    c = _CATEGORIES["ss_pp"]
    _, _, m, n = orbital_indices(c)
    put(c, ri[:, 10:11] * r0[:, m] * r0[:, n]
         + ri[:, 11:12] * (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]))

    c = _CATEGORIES["ps_ss"]
    k, _, _, _ = orbital_indices(c)
    put(c, ri[:, 1:2] * r0[:, k])

    c = _CATEGORIES["ps_ps"]
    k, _, m, _ = orbital_indices(c)
    put(c, ri[:, 5:6] * r0[:, k] * r0[:, m]
         + ri[:, 6:7] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m]))

    c = _CATEGORIES["ps_pp"]
    k, _, m, n = orbital_indices(c)
    t0 = r0[:, k] * r0[:, m] * r0[:, n]
    t1 = (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]) * r0[:, k]
    mix = (r1[:, k] * (r1[:, n] * r0[:, m] + r1[:, m] * r0[:, n])
           + r2[:, k] * (r2[:, m] * r0[:, n] + r2[:, n] * r0[:, m]))
    put(c, ri[:, 12:13] * t0 + ri[:, 13:14] * t1 + ri[:, 14:15] * mix)

    c = _CATEGORIES["pp_ss"]
    k, l, _, _ = orbital_indices(c)
    put(c, ri[:, 2:3] * r0[:, k] * r0[:, l]
         + ri[:, 3:4] * (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]))

    c = _CATEGORIES["pp_ps"]
    k, l, m, _ = orbital_indices(c)
    t0 = r0[:, k] * r0[:, l] * r0[:, m]
    t1 = (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]) * r0[:, m]
    t2a = r1[:, l] * r1[:, m] + r2[:, l] * r2[:, m]
    t2b = r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m]
    put(c, ri[:, 7:8] * t0 + ri[:, 8:9] * t1
         + ri[:, 9:10] * (r0[:, k] * t2a + r0[:, l] * t2b))

    c = _CATEGORIES["pp_pp"]
    k, l, m, n = orbital_indices(c)
    t0 = r0[:, k] * r0[:, l] * r0[:, m] * r0[:, n]
    t1 = (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]) * r0[:, m] * r0[:, n]
    t2 = (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]) * r0[:, k] * r0[:, l]
    quad = (r1[:, k] * r1[:, l] * r1[:, m] * r1[:, n]
            + r2[:, k] * r2[:, l] * r2[:, m] * r2[:, n])
    mix1 = r0[:, m] * (r1[:, l] * r1[:, n] + r2[:, l] * r2[:, n])
    mix2 = r0[:, n] * (r1[:, l] * r1[:, m] + r2[:, l] * r2[:, m])
    val5 = (r0[:, k] * (mix1 + mix2)
            + r0[:, l] * (r0[:, m] * (r1[:, k] * r1[:, n] + r2[:, k] * r2[:, n])
                          + r0[:, n] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m])))
    mix3 = (r1[:, k] * r1[:, l] * r2[:, m] * r2[:, n]
            + r2[:, k] * r2[:, l] * r1[:, m] * r1[:, n])
    cross = ((r1[:, k] * r2[:, l] + r2[:, k] * r1[:, l])
             * (r1[:, m] * r2[:, n] + r2[:, m] * r1[:, n]))
    put(c, ri[:, 15:16] * t0 + ri[:, 16:17] * t1 + ri[:, 17:18] * t2
         + ri[:, 18:19] * quad + ri[:, 19:20] * val5
         + ri[:, 20:21] * mix3 + ri[:, 21:22] * cross)

    return w


def rotate_xh_batch(ri: np.ndarray, r0: np.ndarray, r1: np.ndarray,
                    r2: np.ndarray) -> np.ndarray:
    """Rotated tensors for P heavy-hydrogen pairs, (P, 4, 4, 4, 4).

    Only the (kl|00) column is non-zero: B carries a single s orbital.
    """
    P = ri.shape[0]
    w = np.zeros((P, 4, 4, 4, 4))
    w[:, 0, 0, 0, 0] = ri[:, 0]

    k = np.arange(3)
    ps = ri[:, 1:2] * r0[:, k]
    w[:, k + 1, 0, 0, 0] = ps
    w[:, 0, k + 1, 0, 0] = ps

    kk = k[:, None]
    ll = k[None, :]
    pp = (ri[:, 2:3, None] * r0[:, kk] * r0[:, ll]
          + ri[:, 3:4, None] * (r1[:, kk] * r1[:, ll] + r2[:, kk] * r2[:, ll]))
    w[:, 1:4, 1:4, 0, 0] = pp
    return w


def rotate_hh_batch(ri: np.ndarray) -> np.ndarray:
    """Rotated tensors for P hydrogen-hydrogen pairs — one element each."""
    w = np.zeros((ri.shape[0], 4, 4, 4, 4))
    w[:, 0, 0, 0, 0] = ri[:, 0]
    return w



# Off by default, and the reason is measured rather than assumed.
#
# rotate_xx_batch_mlx beats the NumPy rotation on the arithmetic alone — 11.9x
# at 24000 pairs. End to end through the batched SCF it buys nothing:
#
#     800 molecules, 5 repeats     median      min     max
#       NumPy rotation             1.94 s     1.89    1.99
#       MLX rotation               1.96 s     1.95    1.98
#     median ratio 0.99x, distributions overlap
#
# The rotation is ~26% of prepare_batch, so Amdahl caps the gain near 1.2x
# before anything else; the float64 -> float32 -> device -> host round trip at
# the boundary then spends the rest. It also costs 7.3e-04 eV of agreement with
# the NumPy path, which is float32.
#
# This becomes worth switching on when the whole of prepare_batch is on device
# and there is no boundary to cross. Until then the flag exists so the
# experiment can be repeated, not because it should be enabled:
#     MLXMOLKIT_ROTATE_MLX_MIN=0 forces the GPU path.
MLX_MIN_PAIRS = int(os.environ.get("MLXMOLKIT_ROTATE_MLX_MIN", "100000000"))


def _use_mlx(n_pairs: int, forced: bool | None = None) -> bool:
    """Whether to rotate on the GPU.

    `forced` is the caller's decision and wins: the GPU path is float32, so it
    is correct only where the caller has already accepted float32 downstream.
    Passing None falls back to the environment threshold, which exists so the
    experiment can be repeated by hand.
    """
    if forced is False:
        return False
    if forced is not True and n_pairs < MLX_MIN_PAIRS:
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def rotate_pairs(pair_params, pair_coords, use_mlx: bool | None = None):
    """Rotated tensors for a heterogeneous list of sp pairs.

    Args:
        pair_params: sequence of (pA, pB) ElementParams, sp only (n_basis <= 4).
        pair_coords: sequence of (coordA, coordB).
        use_mlx: force the GPU rotation on or off. MLX cannot do float64 on
            Metal — `float64 is not supported on the GPU` — so the GPU path
            returns float32, which shifts a rotated integral by ~1e-6 eV in
            absolute terms. That is chemically nothing and eleven orders below
            anything an SCF resolves, but it is 130x the agreement the batch
            path otherwise has with the float64 sequential solver, so it is the
            caller's decision, not a heuristic's. None keeps the environment
            threshold.

    Returns:
        (n_pairs, 4, 4, 4, 4). ``e1b`` and ``e2a`` are not returned because
        they are just slices of it: ``e1b = -Z_B * w[:, :, 0, 0]`` and
        ``e2a = -Z_A * w[0, 0, :, :]``, which holds for every pair type
        (verified to round-off against the scalar routine).

    Pairs are grouped by type — HH, XH, XX — because each has its own set of
    local-frame integrals, and each group is rotated in one vectorised call.
    """
    from .overlap_batch import _rotations
    from .two_center_integrals import two_center_integrals_batch

    n = len(pair_params)
    out = np.zeros((n, 4, 4, 4, 4))
    if n == 0:
        return out

    # No per-pair Python work: coordinates, distances and rotation vectors are
    # built as whole arrays. The loop this replaced cost 124 ms on an 800
    # molecule batch — five times the rotation arithmetic it was feeding.
    ca = np.array([c[0] for c in pair_coords], dtype=np.float64)
    cb = np.array([c[1] for c in pair_coords], dtype=np.float64)
    delta = cb - ca
    dist = np.linalg.norm(delta, axis=1)
    live = dist >= 1e-10

    ri_all, kinds = two_center_integrals_batch(pair_params, dist)
    kinds = np.asarray(kinds)

    # HX is XH with the pair reversed; the batch already solved it in that
    # order, so only the geometry is flipped here and the block transposed at
    # the end.
    swapped = kinds == "HX"
    signed = np.where(swapped[:, None], -delta, delta)

    safe = np.where(live, dist, 1.0)
    rot = _rotations(-signed / safe[:, None])
    r0, r1, r2 = rot[:, 0, :], rot[:, 1, :], rot[:, 2, :]

    for pair_type in ("HH", "XH", "XX"):
        sel = np.flatnonzero(live & ((kinds == pair_type)
                                     | (swapped if pair_type == "XH" else False)))
        if sel.size == 0:
            continue
        if pair_type == "XX":
            if _use_mlx(sel.size, use_mlx):
                import mlx.core as mx
                arrs = [mx.array(a[sel].astype(np.float32))
                        for a in (ri_all, r0, r1, r2)]
                out[sel] = np.asarray(rotate_xx_batch_mlx(*arrs), dtype=np.float64)
            else:
                out[sel] = rotate_xx_batch(ri_all[sel], r0[sel], r1[sel], r2[sel])
        elif pair_type == "XH":
            out[sel] = rotate_xh_batch(ri_all[sel], r0[sel], r1[sel], r2[sel])
        else:
            out[sel] = rotate_hh_batch(ri_all[sel])

    flip = np.flatnonzero(swapped & live)
    if flip.size:
        out[flip] = np.transpose(out[flip], (0, 3, 4, 1, 2))
    return out


# ---------------------------------------------------------------------------

def _fused_scatter_plan():
    """Flat destination indices for every (category, symmetry) slot, once.

    The rotation writes each computed value into four positions — the tensor is
    symmetric in its first index pair and its last — across nine categories, so
    a naive implementation issues 36 scatters. Measured on (24000, 256): 36
    separate scatters cost 5.69 ms against 1.10 ms for a single fused one, a
    5.2x difference that is dispatch overhead, not scatter work. (For
    reference, a gather of the same shape is 0.22-0.30 ms: scatter is
    intrinsically 1.6-3.3x dearer, so avoid it where a gather will do.)

    So the plan is built once and every value is written in one call.
    """
    order = ["ss_ss", "ss_ps", "ss_pp", "ps_ss", "ps_ps", "ps_pp",
             "pp_ss", "pp_ps", "pp_pp"]
    dest = []
    for name in order:
        c = _CATEGORIES[name]
        kk, ll, mm, nn = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        for a, b, x, y in ((kk, ll, mm, nn), (ll, kk, mm, nn),
                           (kk, ll, nn, mm), (ll, kk, nn, mm)):
            dest.append(a * 64 + b * 16 + x * 4 + y)
    return order, np.concatenate(dest).astype(np.int32)


_MLX_ORDER, _MLX_DEST = _fused_scatter_plan()


def rotate_xx_batch_mlx(ri, r0, r1, r2):
    """`rotate_xx_batch` on the GPU. Takes and returns ``mx.array``.

    Identical arithmetic, expressed in MLX, with every write fused into one
    scatter — see :func:`_fused_scatter_plan` for why that matters.

    Worth it only in bulk. Below ~1000 pairs the dispatch overhead exceeds the
    gain and the NumPy path is faster; a 300-molecule batch produces ~24000.
    """
    import mlx.core as mx

    P = ri.shape[0]

    def orb(name):
        c = _CATEGORIES[name]
        return (mx.array(c[:, 0] - 1), mx.array(c[:, 1] - 1),
                mx.array(c[:, 2] - 1), mx.array(c[:, 3] - 1))

    values = {}

    c = _CATEGORIES["ss_ss"]
    values["ss_ss"] = mx.broadcast_to(ri[:, 0:1], (P, c.shape[0]))

    _, _, m, n = orb("ss_ps")
    values["ss_ps"] = ri[:, 4:5] * r0[:, m]

    _, _, m, n = orb("ss_pp")
    values["ss_pp"] = (ri[:, 10:11] * r0[:, m] * r0[:, n]
                       + ri[:, 11:12] * (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]))

    k, _, _, _ = orb("ps_ss")
    values["ps_ss"] = ri[:, 1:2] * r0[:, k]

    k, _, m, _ = orb("ps_ps")
    values["ps_ps"] = (ri[:, 5:6] * r0[:, k] * r0[:, m]
                       + ri[:, 6:7] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m]))

    k, _, m, n = orb("ps_pp")
    values["ps_pp"] = (
        ri[:, 12:13] * (r0[:, k] * r0[:, m] * r0[:, n])
        + ri[:, 13:14] * ((r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]) * r0[:, k])
        + ri[:, 14:15] * (r1[:, k] * (r1[:, n] * r0[:, m] + r1[:, m] * r0[:, n])
                          + r2[:, k] * (r2[:, m] * r0[:, n] + r2[:, n] * r0[:, m])))

    k, l, _, _ = orb("pp_ss")
    values["pp_ss"] = (ri[:, 2:3] * r0[:, k] * r0[:, l]
                       + ri[:, 3:4] * (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]))

    k, l, m, _ = orb("pp_ps")
    values["pp_ps"] = (
        ri[:, 7:8] * (r0[:, k] * r0[:, l] * r0[:, m])
        + ri[:, 8:9] * ((r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]) * r0[:, m])
        + ri[:, 9:10] * (r0[:, k] * (r1[:, l] * r1[:, m] + r2[:, l] * r2[:, m])
                         + r0[:, l] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m])))

    k, l, m, n = orb("pp_pp")
    t0 = r0[:, k] * r0[:, l] * r0[:, m] * r0[:, n]
    t1 = (r1[:, k] * r1[:, l] + r2[:, k] * r2[:, l]) * r0[:, m] * r0[:, n]
    t2 = (r1[:, m] * r1[:, n] + r2[:, m] * r2[:, n]) * r0[:, k] * r0[:, l]
    quad = (r1[:, k] * r1[:, l] * r1[:, m] * r1[:, n]
            + r2[:, k] * r2[:, l] * r2[:, m] * r2[:, n])
    val5 = (r0[:, k] * (r0[:, m] * (r1[:, l] * r1[:, n] + r2[:, l] * r2[:, n])
                        + r0[:, n] * (r1[:, l] * r1[:, m] + r2[:, l] * r2[:, m]))
            + r0[:, l] * (r0[:, m] * (r1[:, k] * r1[:, n] + r2[:, k] * r2[:, n])
                          + r0[:, n] * (r1[:, k] * r1[:, m] + r2[:, k] * r2[:, m])))
    mix3 = (r1[:, k] * r1[:, l] * r2[:, m] * r2[:, n]
            + r2[:, k] * r2[:, l] * r1[:, m] * r1[:, n])
    cross = ((r1[:, k] * r2[:, l] + r2[:, k] * r1[:, l])
             * (r1[:, m] * r2[:, n] + r2[:, m] * r1[:, n]))
    values["pp_pp"] = (ri[:, 15:16] * t0 + ri[:, 16:17] * t1 + ri[:, 17:18] * t2
                       + ri[:, 18:19] * quad + ri[:, 19:20] * val5
                       + ri[:, 20:21] * mix3 + ri[:, 21:22] * cross)

    # Each category's values repeated for its four symmetric destinations,
    # concatenated in the same order the plan was built, then written once.
    payload = mx.concatenate(
        [values[name] for name in _MLX_ORDER for _ in range(4)], axis=1)
    flat = mx.zeros((P, 256))
    flat[:, mx.array(_MLX_DEST)] = payload
    return mx.reshape(flat, (P, 4, 4, 4, 4))
