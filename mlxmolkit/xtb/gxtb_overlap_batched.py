# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT
"""GPU-batched cartesian-Gaussian overlap (MLX), bit-exact vs basis.overlap_matrix.

The per-molecule numpy path (`basis._primitive_overlap` in a Python quad-loop over
AO-pairs x primitive-pairs) is the dominant g-xTB cost (~50% of SCF wall time;
O(nao^2 * nprim^2) Python calls). This computes the FULL overlap for a BATCH of
molecules in one vectorized GPU pass, with no Python inner loops.

Math (identical to the OS recurrence in basis.py): for l<=2 the per-axis factor
f(la,lb) is a closed-form polynomial in PA, PB, 1/(2p); S = sum_ij c_i c_j *
(pi/p)^1.5 exp(-mu R^2) * f_x * f_y * f_z. Padding (prims and AOs) carries coeff=0
so padded contributions vanish; padded AO rows/cols come out 0 and are sliced off.
"""
from __future__ import annotations
import numpy as np
import mlx.core as mx

__all__ = ["prep_basis", "batch_overlap", "overlap_matrix_mlx"]


def prep_basis(basis: list) -> dict:
    """Pack a list[BasisFunction] into padded numpy arrays (one molecule)."""
    nao = len(basis)
    pmax = max(len(b.alphas) for b in basis)
    alpha = np.ones((nao, pmax), np.float64)      # pad alpha=1 (p>0, harmless)
    coeff = np.zeros((nao, pmax), np.float64)      # pad coeff=0 -> no contribution
    center = np.zeros((nao, 3), np.float64)
    lxyz = np.zeros((nao, 3), np.int32)
    for i, b in enumerate(basis):
        k = len(b.alphas)
        alpha[i, :k] = b.alphas
        coeff[i, :k] = b.coeffs
        center[i] = b.center
        lxyz[i] = b.l_xyz
    return {"nao": nao, "pmax": pmax, "alpha": alpha, "coeff": coeff,
            "center": center, "lxyz": lxyz}


def _stack(preps: list[dict]) -> dict:
    """Pad a list of single-molecule preps to a common (nao_max, pmax_max) batch."""
    B = len(preps)
    nao_max = max(p["nao"] for p in preps)
    pmax = max(p["pmax"] for p in preps)
    alpha = np.ones((B, nao_max, pmax), np.float64)
    coeff = np.zeros((B, nao_max, pmax), np.float64)
    center = np.zeros((B, nao_max, 3), np.float64)
    lxyz = np.zeros((B, nao_max, 3), np.int32)
    naos = np.array([p["nao"] for p in preps])
    for b, p in enumerate(preps):
        n, k = p["nao"], p["pmax"]
        alpha[b, :n, :k] = p["alpha"]
        coeff[b, :n, :k] = p["coeff"]
        center[b, :n] = p["center"]
        lxyz[b, :n] = p["lxyz"]
    return {"alpha": alpha, "coeff": coeff, "center": center, "lxyz": lxyz, "naos": naos}


def _os_factor(PA, PB, i2p, la, lb):
    """Per-axis Obara-Saika factor f(la,lb) for l in {0,1,2}, fully broadcast."""
    PA2, PB2 = PA * PA, PB * PB
    T = [[None] * 3 for _ in range(3)]
    T[0][0] = mx.ones_like(PA)
    T[1][0] = PA;            T[2][0] = PA2 + i2p
    T[0][1] = PB;            T[0][2] = PB2 + i2p
    T[1][1] = PA * PB + i2p
    T[2][1] = PA2 * PB + i2p * PB + 2.0 * i2p * PA
    T[1][2] = PB2 * PA + i2p * PA + 2.0 * i2p * PB
    T[2][2] = PA2 * PB2 + i2p * (PA2 + PB2) + 4.0 * i2p * PA * PB + 3.0 * i2p * i2p
    out = mx.zeros_like(PA)
    for a in range(3):
        ma = (la == a)
        for b in range(3):
            out = out + ma * (lb == b) * T[a][b]
    return out


def batch_overlap(preps: list[dict]) -> list:
    """Overlap matrices for a batch of molecules. Returns list of (nao_i,nao_i) arrays."""
    st = _stack(preps)
    al = mx.array(st["alpha"]); co = mx.array(st["coeff"])
    ct = mx.array(st["center"]); lx = mx.array(st["lxyz"])
    # dims: (B, mu, nu, i, j)
    a_a = al[:, :, None, :, None]      # alpha of bra prim i
    a_b = al[:, None, :, None, :]      # alpha of ket prim j
    c_a = co[:, :, None, :, None]
    c_b = co[:, None, :, None, :]
    p = a_a + a_b
    i2p = 0.5 / p
    base = (np.pi / p) ** 1.5
    mu = a_a * a_b / p
    fac = c_a * c_b
    for ax in range(3):
        Aa = ct[:, :, ax][:, :, None, None, None]
        Ab = ct[:, :, ax][:, None, :, None, None]
        P = (a_a * Aa + a_b * Ab) / p
        fac = fac * _os_factor(P - Aa, P - Ab, i2p,
                               lx[:, :, ax][:, :, None, None, None],
                               lx[:, :, ax][:, None, :, None, None])
    # R^2 between centers (B, mu, nu, 1, 1)
    d = ct[:, :, None, :] - ct[:, None, :, :]
    R2 = mx.sum(d * d, axis=-1)[:, :, :, None, None]
    S = mx.sum(fac * base * mx.exp(-mu * R2), axis=(3, 4))   # (B, nao, nao)
    mx.eval(S)
    Snp = np.array(S)
    return [Snp[b, :n, :n] for b, n in enumerate(st["naos"])]


def overlap_matrix_mlx(basis: list) -> np.ndarray:
    """Drop-in single-molecule replacement for basis.overlap_matrix."""
    return batch_overlap([prep_basis(basis)])[0]


# --------------------------------------------------------------------------- #
#  Multipole integrals (point 1): S, dpint(3), qpint(6) — batched on GPU      #
# --------------------------------------------------------------------------- #
def _aug_table(PA, PB, i2p, amax=4, bmax=2):
    """Augmented 1D OS table T[a][b], a in [0,amax], b in [0,bmax] (broadcast)."""
    T = [[None] * (bmax + 1) for _ in range(amax + 1)]
    T[0][0] = mx.ones_like(PA)
    for a in range(1, amax + 1):
        T[a][0] = PA * T[a - 1][0] + (i2p * (a - 1) * T[a - 2][0] if a >= 2 else 0.0)
    for b in range(1, bmax + 1):
        for a in range(amax + 1):
            t = PB * T[a][b - 1]
            if b >= 2: t = t + i2p * (b - 1) * T[a][b - 2]
            if a >= 1: t = t + i2p * a * T[a - 1][b - 1]
            T[a][b] = t
    return T


def _pick(T, ta, lb):
    """Select T[ta][lb] by integer-equality masks (ta, lb broadcast arrays)."""
    out = mx.zeros_like(T[0][0])
    for a in range(len(T)):
        ma = (ta == a)
        for b in range(len(T[0])):
            out = out + ma * (lb == b) * T[a][b]
    return out


def batch_multipole(preps: list[dict]):
    """Batched S, dpint(3), qpint(6) in CAO basis (origin = frame 0).

    Returns (S_list, dp_list, qp_list); dp[b]=(3,n,n), qp[b]=(6,n,n in xtb order
    xx,yy,zz,xy,xz,yz). Bit-identical math to multipole_integrals.multipole_matrices.
    """
    st = _stack(preps)
    al = mx.array(st["alpha"]); co = mx.array(st["coeff"])
    ct = mx.array(st["center"]); lx = mx.array(st["lxyz"])
    a_a = al[:, :, None, :, None]; a_b = al[:, None, :, None, :]
    p = a_a + a_b; i2p = 0.5 / p
    w = co[:, :, None, :, None] * co[:, None, :, None, :] * (np.pi / p) ** 1.5
    d = ct[:, :, None, :] - ct[:, None, :, :]
    w = w * mx.exp(-(a_a * a_b / p) * mx.sum(d * d, axis=-1)[:, :, :, None, None])
    s0 = [None] * 3; s1 = [None] * 3; s2 = [None] * 3; Aa = [None] * 3
    for ax in range(3):
        A = ct[:, :, ax][:, :, None, None, None]
        B = ct[:, None, :, ax][:, :, :, None, None]
        P = (a_a * A + a_b * B) / p
        T = _aug_table(P - A, P - B, i2p)
        la = lx[:, :, ax][:, :, None, None, None]; lb = lx[:, None, :, ax][:, :, :, None, None]
        s0[ax] = _pick(T, la, lb); s1[ax] = _pick(T, la + 1, lb); s2[ax] = _pick(T, la + 2, lb)
        Aa[ax] = A
    def red(x): return mx.sum(w * x, axis=(3, 4))
    S = red(s0[0] * s0[1] * s0[2])
    m = [s1[k] + Aa[k] * s0[k] for k in range(3)]            # 1st-moment per axis
    dp = mx.stack([red(m[0] * s0[1] * s0[2]), red(s0[0] * m[1] * s0[2]),
                   red(s0[0] * s0[1] * m[2])], axis=1)
    qd = [s2[k] + 2.0 * Aa[k] * s1[k] + Aa[k] * Aa[k] * s0[k] for k in range(3)]
    qp = mx.stack([
        red(qd[0] * s0[1] * s0[2]), red(s0[0] * qd[1] * s0[2]), red(s0[0] * s0[1] * qd[2]),
        red(m[0] * m[1] * s0[2]),   red(m[0] * s0[1] * m[2]),   red(s0[0] * m[1] * m[2]),
    ], axis=1)
    mx.eval(S, dp, qp)
    S, dp, qp = np.array(S), np.array(dp), np.array(qp)
    n = st["naos"]
    return ([S[b, :k, :k] for b, k in enumerate(n)],
            [dp[b, :, :k, :k] for b, k in enumerate(n)],
            [qp[b, :, :k, :k] for b, k in enumerate(n)])


# --------------------------------------------------------------------------- #
#  Batched generalized eigensolver (point 2): F C = S C e, on GPU             #
# --------------------------------------------------------------------------- #
def batched_eigh_general(F, S):
    """Solve F_b C_b = S_b C_b diag(e_b) for a BUCKET of same-size (B,n,n) F,S.

    Delegates to the project's GPU batched generalized solver
    ``mlx_addons.linalg.gen_eigh`` (Cholesky reduction; Metal triangular-solve
    kernels for n<=128). Bucket molecules by nao, then one call per bucket.
    Returns (eigvals (B,n) ascending, C (B,n,n) with C^T S C = I).
    """
    from mlx_addons.linalg import gen_eigh
    e, C = gen_eigh(mx.array(F), mx.array(S))
    mx.eval(e, C)
    return np.array(e), np.array(C)
