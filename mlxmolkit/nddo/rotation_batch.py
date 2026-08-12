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


def rotate_pairs(pair_params, pair_coords):
    """Rotated tensors for a heterogeneous list of sp pairs.

    Args:
        pair_params: sequence of (pA, pB) ElementParams, sp only (n_basis <= 4).
        pair_coords: sequence of (coordA, coordB).

    Returns:
        (n_pairs, 4, 4, 4, 4). ``e1b`` and ``e2a`` are not returned because
        they are just slices of it: ``e1b = -Z_B * w[:, :, 0, 0]`` and
        ``e2a = -Z_A * w[0, 0, :, :]``, which holds for every pair type
        (verified to round-off against the scalar routine).

    Pairs are grouped by type — HH, XH, XX — because each has its own set of
    local-frame integrals, and each group is rotated in one vectorised call.
    """
    from .rotation import _rotation_matrix
    from .two_center_integrals import two_center_integrals

    n = len(pair_params)
    out = np.zeros((n, 4, 4, 4, 4))
    groups: dict[str, list[int]] = {"HH": [], "XH": [], "XX": []}
    swapped: list[int] = []
    ri_all = np.zeros((n, 22))
    r0 = np.zeros((n, 3))
    r1 = np.zeros((n, 3))
    r2 = np.zeros((n, 3))

    for idx, ((pA, pB), (rA, rB)) in enumerate(zip(pair_params, pair_coords)):
        delta = rB - rA
        R = float(np.linalg.norm(delta))
        if R < 1e-10:
            continue
        ri, _core, pair_type = two_center_integrals(pA, pB, R)
        if pair_type == "HX":
            # A is the hydrogen. The scalar routine solves the pair the other
            # way round and transposes, so do the same: recompute in the
            # swapped orientation and flip the result at the end.
            pA, pB = pB, pA
            rA, rB = rB, rA
            delta = rB - rA
            ri, _core, pair_type = two_center_integrals(pA, pB, R)
            swapped.append(idx)
        rot = _rotation_matrix(-delta / R)
        r0[idx], r1[idx], r2[idx] = rot[0], rot[1], rot[2]
        ri_all[idx, :len(ri)] = ri
        groups[pair_type].append(idx)

    for pair_type, idxs in groups.items():
        if not idxs:
            continue
        sel = np.array(idxs)
        if pair_type == "XX":
            out[sel] = rotate_xx_batch(ri_all[sel], r0[sel], r1[sel], r2[sel])
        elif pair_type == "XH":
            out[sel] = rotate_xh_batch(ri_all[sel], r0[sel], r1[sel], r2[sel])
        else:
            out[sel] = rotate_hh_batch(ri_all[sel])

    if swapped:
        sel = np.array(swapped)
        out[sel] = np.transpose(out[sel], (0, 3, 4, 1, 2))
    return out
