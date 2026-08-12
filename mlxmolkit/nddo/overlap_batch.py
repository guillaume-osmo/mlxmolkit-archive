"""Slater overlaps for many atom pairs at once.

:func:`~mlxmolkit.nddo.overlap.overlap_molecular_frame` is called once per pair
— 24240 times for a 300-molecule PM6 batch, ~29% of ``prepare_batch``, which is
itself 86% of a batched SCF. It is the largest scalar cost left in the pipeline.

The six ``jcall`` branches in that function look like six different
calculations, but every Slater term in all of them has one form::

    S = zA**eA * zB**eB * R**p / d  *  sum_k  c_k * A[i_k] * B[j_k]

differing only in the exponents, the divisor, and a list of (i, j, c) triples.
So they are a table, not six code paths: :data:`TABLE` holds the coefficients
and :func:`_s_terms` is the single evaluator. Products of sums in the original
— ``(A[5]-A[3]) * (B[0]-B[2])`` — are expanded into triples here, because the
expanded form is what the evaluator consumes.

Correctness rests on the exhaustive check in tests/test_overlap_batch.py, not
on care taken transcribing: a transposed index is invisible by inspection and
unmissable against the scalar.
"""
from __future__ import annotations

import numpy as np

from .params import ANG_TO_BOHR, principal_qn

_R3, _R10, _R30 = np.sqrt(3.0), np.sqrt(10.0), np.sqrt(30.0)

# (qnA, qnB) -> jcall. Anything absent falls back to the scalar routine.
_JCALL = {(1, 1): 2, (2, 1): 3, (2, 2): 4, (3, 1): 431, (3, 2): 5, (3, 3): 6}

# Which (A, B) integral pair each Slater term is built from.
_SET = {"S111": "ss", "S211": "ps", "S121": "sp", "S221": "pp", "S222": "pp"}

# jcall -> term -> (zetaA orbital, exp, zetaB orbital, exp, R power, divisor,
#                   [(i, j, coeff), ...])
TABLE: dict[int, dict[str, tuple]] = {
    2: {
        "S111": ("s", 1.5, "s", 1.5, 3, 4.0,
                 [(2, 0, 1), (0, 2, -1)]),
    },
    3: {
        "S111": ("s", 2.5, "s", 1.5, 4, 8.0 * _R3,
                 [(3, 0, 1), (0, 3, -1), (2, 1, 1), (1, 2, -1)]),
        "S211": ("p", 2.5, "s", 1.5, 4, 8.0,
                 [(2, 0, 1), (0, 2, -1), (3, 1, 1), (1, 3, -1)]),
    },
    4: {
        "S111": ("s", 2.5, "s", 2.5, 5, 48.0,
                 [(4, 0, 1), (0, 4, 1), (2, 2, -2)]),
        "S211": ("p", 2.5, "s", 2.5, 5, 16.0 * _R3,
                 [(3, 0, 1), (3, 2, -1), (1, 2, -1), (1, 4, 1),
                  (0, 3, 1), (2, 3, -1), (2, 1, -1), (4, 1, 1)]),
        "S121": ("s", 2.5, "p", 2.5, 5, 16.0 * _R3,
                 [(3, 0, 1), (3, 2, -1), (1, 2, -1), (1, 4, 1),
                  (0, 3, -1), (2, 3, 1), (2, 1, 1), (4, 1, -1)]),
        "S221": ("p", 2.5, "p", 2.5, 5, 16.0,
                 [(4, 2, -1), (0, 2, -1), (2, 4, 1), (2, 0, 1)]),
        "S222": ("p", 2.5, "p", 2.5, 5, 32.0,
                 [(4, 0, 1), (4, 2, -1), (0, 4, -1), (2, 4, 1),
                  (2, 0, -1), (0, 2, 1)]),
    },
    431: {
        "S111": ("s", 3.5, "s", 1.5, 5, 24.0 * _R10,
                 [(4, 0, 1), (3, 1, 2), (1, 3, -2), (0, 4, -1)]),
        "S211": ("p", 3.5, "s", 1.5, 5, 8.0 * _R30,
                 [(3, 0, 1), (3, 2, 1), (1, 4, -1), (1, 2, -1),
                  (2, 1, 1), (4, 1, 1), (2, 3, -1), (0, 3, -1)]),
    },
    5: {
        "S111": ("s", 3.5, "s", 2.5, 6, 48.0 * _R30,
                 [(5, 0, 1), (4, 1, 1), (3, 2, -2), (2, 3, -2),
                  (1, 4, 1), (0, 5, 1)]),
        "S211": ("p", 3.5, "s", 2.5, 6, 48.0 * _R10,
                 [(4, 0, 1), (5, 1, 1), (3, 3, -2), (2, 2, -2),
                  (1, 5, 1), (0, 4, 1)]),
        "S121": ("s", 3.5, "p", 2.5, 6, 48.0 * _R10,
                 [(4, 0, 1), (5, 1, -1), (3, 1, 2), (4, 2, -2),
                  (1, 3, -2), (2, 4, 2), (0, 4, -1), (1, 5, 1)]),
        "S221": ("p", 3.5, "p", 2.5, 6, 16.0 * _R30,
                 [(3, 0, 1), (5, 2, -1), (2, 1, 1), (4, 3, -1),
                  (1, 2, -1), (3, 4, 1), (0, 3, -1), (2, 5, 1)]),
        "S222": ("p", 3.5, "p", 2.5, 6, 32.0 * _R30,
                 [(5, 0, 1), (5, 2, -1), (3, 0, -1), (3, 2, 1),
                  (4, 1, 1), (4, 3, -1), (2, 1, -1), (2, 3, 1),
                  (3, 2, -1), (3, 4, 1), (1, 2, 1), (1, 4, -1),
                  (2, 3, -1), (2, 5, 1), (0, 3, 1), (0, 5, -1)]),
    },
    6: {
        "S111": ("s", 3.5, "s", 3.5, 7, 1440.0,
                 [(6, 0, 1), (4, 2, -3), (2, 4, 3), (0, 6, -1)]),
        "S211": ("p", 3.5, "s", 3.5, 7, 480.0 * _R3,
                 [(5, 0, 1), (6, 1, 1), (4, 1, -1), (5, 2, -1),
                  (3, 2, -2), (4, 3, -2), (2, 3, 2), (3, 4, 2),
                  (1, 4, 1), (2, 5, 1), (0, 5, -1), (1, 6, -1)]),
        "S121": ("s", 3.5, "p", 3.5, 7, 480.0 * _R3,
                 [(5, 0, 1), (6, 1, -1), (4, 1, 1), (5, 2, -1),
                  (3, 2, -2), (4, 3, 2), (2, 3, -2), (3, 4, 2),
                  (1, 4, 1), (2, 5, -1), (0, 5, 1), (1, 6, -1)]),
        "S221": ("p", 3.5, "p", 3.5, 7, 480.0,
                 [(4, 0, 1), (6, 2, -1), (2, 2, -2), (4, 4, 2),
                  (0, 4, 1), (2, 6, -1)]),
        "S222": ("p", 3.5, "p", 3.5, 7, 960.0,
                 [(6, 0, 1), (6, 2, -1), (4, 0, -1), (4, 2, 1),
                  (4, 2, -2), (4, 4, 2), (2, 2, 2), (2, 4, -2),
                  (2, 4, 1), (2, 6, -1), (0, 4, -1), (0, 6, 1)]),
    },
}


def _aintgs(alpha: np.ndarray, n_max: int = 7) -> np.ndarray:
    """(P,) -> (P, n_max). Vectorised :func:`~mlxmolkit.nddo.overlap._aintgs`."""
    out = np.zeros((alpha.size, n_max))
    live = np.abs(alpha) >= 1e-10
    safe = np.where(live, alpha, 1.0)          # keep the division finite
    out[:, 0] = np.where(live, np.exp(-safe) / safe, 0.0)
    for k in range(1, n_max):
        out[:, k] = np.where(live, out[:, 0] + k * out[:, k - 1] / safe, 0.0)
    return out


# Taylor coefficients for 1e-6 < |beta| <= 0.5, from the scalar routine.
_EVEN = np.array([(2.0, 1/3, 1/60, 1/2520), (2/3, 1/5, 1/84, 1/3240),
                  (2/5, 1/7, 1/108, 1/3960), (2/7, 1/9, 1/132, 1/4680),
                  (2/9, 1/11, 1/156, 1/5400), (2/11, 1/13, 1/180, 1/6120),
                  (2/13, 1/15, 1/204, 1/6840)])
_ODD = np.array([(-2/3, -1/15, -1/420), (-2/5, -1/21, -1/540),
                 (-2/7, -1/27, -1/660), (-2/9, -1/33, -1/780),
                 (-2/11, -1/39, -1/900), (-2/13, -1/45, -1/1020)])


def _bintgs(beta: np.ndarray, n_max: int = 7) -> np.ndarray:
    """(P,) -> (P, n_max). Vectorised :func:`~mlxmolkit.nddo.overlap._bintgs`.

    Three regimes by |beta|, evaluated everywhere and selected by mask. The
    recurrence branch divides by beta, so it runs on a substituted-safe copy;
    a NaN in an unused lane would propagate through the select.
    """
    x = np.asarray(beta, dtype=np.float64)
    out = np.zeros((x.size, n_max))
    tiny = np.abs(x) <= 1e-6
    taylor = (~tiny) & (np.abs(x) <= 0.5)
    rec = np.abs(x) > 0.5

    safe = np.where(rec, x, 1.0)
    xc = np.clip(safe, -500.0, 500.0)
    tx, tmx = np.exp(xc) / safe, -np.exp(-xc) / safe
    b_rec = np.zeros((x.size, n_max))
    b_rec[:, 0] = tx + tmx
    sign = 1.0
    for k in range(1, n_max):
        sign = -sign
        b_rec[:, k] = sign * tx + tmx + k * b_rec[:, k - 1] / safe

    x2 = x * x
    for k in range(n_max):
        if k % 2 == 0:
            c = _EVEN[k // 2]
            b_tay = c[0] + c[1] * x2 + c[2] * x2 * x2 + c[3] * x2 * x2 * x2
            b_tiny = 2.0 / (k + 1)
        else:
            c = _ODD[k // 2]
            b_tay = c[0] * x + c[1] * x * x2 + c[2] * x * x2 * x2
            b_tiny = 0.0
        out[:, k] = np.where(tiny, b_tiny, np.where(taylor, b_tay, b_rec[:, k]))
    return out


def _s_terms(jcall, zeta, AB, R_bohr, has_pA, has_pB):
    """The five Slater terms for one jcall, over a pair axis."""
    zero = np.zeros_like(R_bohr)
    out = {k: zero.copy() for k in _SET}
    for term, (oA, eA, oB, eB, p, div, triples) in TABLE[jcall].items():
        A, B = AB[_SET[term]]
        acc = zero.copy()
        for i, j, c in triples:
            acc += c * A[:, i] * B[:, j]
        val = (zeta["A" + oA] ** eA * zeta["B" + oB] ** eB
               * R_bohr ** p * acc / div)
        if term == "S111":
            keep = np.ones_like(has_pA)
        elif term == "S211" and jcall in (3, 431):
            keep = has_pA                      # heavy-H: B has no p orbitals
        else:
            keep = has_pA & has_pB
        out[term] = np.where(keep, val, 0.0)
    return out


def _rotations(unit: np.ndarray) -> np.ndarray:
    """(P, 3) unit vectors -> (P, 3, 3), matching the scalar quaternion form."""
    vx, vy, vz = unit[:, 0], unit[:, 1], unit[:, 2]
    w = 1.0 + vx
    anti = np.abs(w) < 1e-7
    safe_w = np.where(anti, 1.0, w)
    norm = np.sqrt(vz * vz + vy * vy + safe_w * safe_w)
    qy, qz, qw = vz / norm, -vy / norm, safe_w / norm

    R = np.empty((unit.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = -2 * qz * qw
    R[:, 0, 2] = 2 * qy * qw
    R[:, 1, 0] = 2 * qz * qw
    R[:, 1, 1] = 1 - 2 * qz * qz
    R[:, 1, 2] = 2 * qy * qz
    R[:, 2, 0] = -2 * qy * qw
    R[:, 2, 1] = 2 * qy * qz
    R[:, 2, 2] = 1 - 2 * qy * qy
    R[anti] = np.diag([-1.0, -1.0, 1.0])       # v ~ (-1,0,0): 180 deg about z
    return R


def _assemble(jcall, S, rot, n_a, n_b):
    """Slater terms + rotation -> (P, 4, 4) molecular-frame overlap blocks."""
    P = rot.shape[0]
    di = np.zeros((P, 4, 4))
    di[:, 0, 0] = S["S111"]
    r0, r1, r2 = rot[:, 0, :], rot[:, 1, :], rot[:, 2, :]

    if jcall in (3, 431):
        if n_a > 1:
            di[:, 1:4, 0] = S["S211"][:, None] * r0
    elif jcall in (4, 5, 6) and n_a > 1 and n_b > 1:
        di[:, 1:4, 0] = S["S211"][:, None] * r0
        di[:, 0, 1:4] = -S["S121"][:, None] * r0
        di[:, 1:4, 1:4] = (
            -S["S221"][:, None, None] * r0[:, :, None] * r0[:, None, :]
            + S["S222"][:, None, None] * (r1[:, :, None] * r1[:, None, :]
                                          + r2[:, :, None] * r2[:, None, :]))
    return di


def overlap_pairs(pair_specs):
    """Molecular-frame overlaps for a heterogeneous list of sp pairs.

    Args:
        pair_specs: sequence of (pA, pB, coordA, coordB).

    Returns:
        list of (nA, nB) arrays, one per input, matching
        :func:`~mlxmolkit.nddo.overlap.overlap_molecular_frame` element for
        element. Pairs the table does not cover — d orbitals, or principal
        quantum numbers above 3 such as Br and I — fall through to the scalar
        routine rather than being guessed at.
    """
    from .overlap import overlap_molecular_frame

    out: list[np.ndarray | None] = [None] * len(pair_specs)
    groups: dict[tuple, list[int]] = {}

    for k, (pA, pB, rA, rB) in enumerate(pair_specs):
        if float(np.linalg.norm(np.asarray(rB) - np.asarray(rA))) < 1e-10:
            # Coincident centres: the scalar short-circuits to an identity
            # block. Without this the rotation divides by zero and returns a
            # silent NaN, which is worse than either answer.
            block = np.zeros((pA.n_basis, pB.n_basis))
            np.fill_diagonal(block, 1.0)
            out[k] = block
            continue
        swap = principal_qn(pA.Z) < principal_qn(pB.Z)
        a, b = (pB, pA) if swap else (pA, pB)
        jcall = _JCALL.get((principal_qn(a.Z), principal_qn(b.Z)))
        if jcall is None or a.n_basis > 4 or b.n_basis > 4:
            out[k] = overlap_molecular_frame(pA, pB, rA, rB)
            continue
        groups.setdefault((jcall, a.n_basis, b.n_basis, swap), []).append(k)

    for (jcall, n_a, n_b, swap), idxs in groups.items():
        sel = np.array(idxs)
        pa = [pair_specs[i][1 if swap else 0] for i in sel]
        pb = [pair_specs[i][0 if swap else 1] for i in sel]
        ca = np.array([pair_specs[i][3 if swap else 2] for i in sel])
        cb = np.array([pair_specs[i][2 if swap else 3] for i in sel])

        delta = cb - ca
        R = np.linalg.norm(delta, axis=1)
        R_bohr = R * ANG_TO_BOHR
        zeta = {"As": np.array([p.zeta_s for p in pa]),
                "Ap": np.array([p.zeta_p for p in pa]),
                "Bs": np.array([p.zeta_s for p in pb]),
                "Bp": np.array([p.zeta_p for p in pb])}
        AB = {}
        for key, (za, zb) in (("ss", ("As", "Bs")), ("ps", ("Ap", "Bs")),
                              ("sp", ("As", "Bp")), ("pp", ("Ap", "Bp"))):
            AB[key] = (_aintgs(0.5 * R_bohr * (zeta[za] + zeta[zb])),
                       _bintgs(0.5 * R_bohr * (zeta[za] - zeta[zb])))

        has_pA = np.full(sel.size, n_a > 1)
        has_pB = np.full(sel.size, n_b > 1)
        S = _s_terms(jcall, zeta, AB, R_bohr, has_pA, has_pB)
        blocks = _assemble(jcall, S, _rotations(delta / R[:, None]), n_a, n_b)

        for pos, i in enumerate(sel):
            block = blocks[pos][:n_a, :n_b]
            out[i] = block.T.copy() if swap else block.copy()

    return out
