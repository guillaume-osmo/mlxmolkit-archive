# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT
"""Vectorized AES assembly: mmompop / setvsdq drop-ins (no Python pair loops).

Numerically identical to aes.mmompop / aes.setvsdq. These are the last numpy hot
spots in the g-xTB SCF (~16% of per-mol wall time); both were O(nao^2)/O(nat^2)
Python loops. mmompop reduces (by μ↔ν integral symmetry) to per-AO row-sums + a
per-atom segment scatter; setvsdq to (nat,nat,3) broadcasts + a per-atom CT tail.
"""
from __future__ import annotations
import numpy as np
from .aes import GFN2_PARAMS

# qp layouts:  qpint = (xx,yy,zz,xy,xz,yz);  mmompop = (xx,xy,yy,xz,yz,zz)
_DIAG_MM = (0, 2, 5)                      # mmompop slot for axis x,y,z
_OFF = ((1, 0, 1, 3), (2, 0, 3, 4), (2, 1, 4, 5))  # (k,l, mm_slot, qpint_idx)
_IDX = np.array([[0, 1, 3], [1, 2, 4], [3, 4, 5]])  # mmompop 3x3 symmetric map


def mmompop_fast(P, S, dpint, qpint, aoat, coords_bohr):
    """Faithful masked transcription of aes.mmompop (lower tri + diagonal AO).

    Per AO-pair (i,j) the original adds tii (with r=coords[atom_i]) to atom_i and
    tjj (r=coords[atom_j]) to atom_j over j<i, plus an i==i diagonal term. We build
    the full per-pair tii/tjj, mask the strict-lower triangle, and segment-scatter.
    """
    nat = coords_bohr.shape[0]
    nao = P.shape[0]
    ps = P * S                                        # symmetric (nao,nao)
    pdm = P[None] * dpint                             # (3,nao,nao)
    pqm = P[None] * qpint                             # (6,nao,nao) qpint layout
    ra = coords_bohr[aoat]                            # (nao,3)
    low = np.tril(np.ones((nao, nao)), -1)            # i>j
    diag = np.eye(nao)

    def scatter(tii, tjj):
        vals = (low * tii).sum(1) + (low * tjj).sum(0) + (diag * tii).sum(1)
        return np.bincount(aoat, vals, nat)

    dipm = np.zeros((3, nat)); qp = np.zeros((6, nat))
    ri = ra[:, None, :]; rj = ra[None, :, :]          # broadcast over (i,j)
    for k in range(3):
        dipm[k] = scatter(ri[:, :, k] * ps - pdm[k], rj[:, :, k] * ps - pdm[k])
        d = _DIAG_MM[k]
        qp[d] = scatter(2.0 * pdm[k] * ri[:, :, k] - ri[:, :, k] ** 2 * ps - pqm[k],
                        2.0 * pdm[k] * rj[:, :, k] - rj[:, :, k] ** 2 * ps - pqm[k])
    for k, l, mm, qi in _OFF:
        qp[mm] = scatter(
            pdm[k] * ri[:, :, l] + pdm[l] * ri[:, :, k] - ri[:, :, l] * ri[:, :, k] * ps - pqm[qi],
            pdm[k] * rj[:, :, l] + pdm[l] * rj[:, :, k] - rj[:, :, l] * rj[:, :, k] * ps - pqm[qi])
    # traceless transform (aes.mmompop:250-260): qp *= 1.5; diag -= 0.5*(xx+yy+zz)
    tr = 0.5 * (qp[0] + qp[2] + qp[5])
    qp *= 1.5
    qp[0] -= tr; qp[2] -= tr; qp[5] -= tr
    return dipm, qp


def setvsdq_fast(atoms, coords_bohr, q, dipm, qp, gab3, gab5):
    nat = coords_bohr.shape[0]
    ra = coords_bohr                                  # (nat,3)  receiver i
    dra = ra[:, None, :] - ra[None, :, :]             # (i,j,3)
    g3, g5, qj = gab3, gab5, q[None, :]               # (i,j),(i,j),(1,j)
    r2a = np.einsum("il,il->i", ra, ra)[:, None]      # |r_i|^2
    r2ab = np.einsum("ijl,ijl->ij", dra, dra)
    t1a = np.einsum("il,ijl->ij", ra, dra)
    t2a = np.einsum("lj,ijl->ij", dipm, dra)          # dipm of source j
    t3a = np.einsum("il,lj->ij", ra, dipm)
    Qmat = qp[_IDX]                                   # (3,3,nat) symmetric, source j
    dQd = np.einsum("ijl,lmj,ijm->ij", dra, Qmat, dra)
    dum5 = -dQd - 1.5 * qj * t1a ** 2 + t3a * r2ab - 3.0 * t1a * t2a + 0.5 * qj * r2a * r2ab
    dum3 = -t1a * qj - t2a
    vs = np.sum(dum5 * g5 + dum3 * g3, axis=1)
    vd = np.zeros((3, nat)); vq = np.zeros((6, nat))
    for k in range(3):
        vd[k] = np.sum(g3 * dra[:, :, k] * qj
                       + g5 * (3.0 * dra[:, :, k] * t2a - r2ab * dipm[k][None, :]
                               - qj * r2ab * ra[:, k][:, None] + 3.0 * qj * dra[:, :, k] * t1a), axis=1)
        vq[k] = np.sum(g5 * qj * (-1.5 * dra[:, :, k] ** 2 + 0.5 * r2ab), axis=1)
    for k, l, _mm, qi in _OFF:
        vq[qi] = np.sum(-3.0 * qj * g5 * dra[:, :, k] * dra[:, :, l], axis=1)
    # --- per-atom CT correction (qp here in mmompop layout) ---
    qs1 = np.array([GFN2_PARAMS[int(Z)].dip_kernel for Z in atoms]) * 2.0
    qs2 = np.array([GFN2_PARAMS[int(Z)].quad_kernel for Z in atoms]) * 6.0
    qk = np.array([GFN2_PARAMS[int(Z)].quad_kernel for Z in atoms])
    t3 = np.zeros(nat); t2 = np.zeros(nat)
    for k in range(3):
        vd[k] -= qs1 * dipm[k]
        t3 += ra[:, k] * dipm[k] * qs1
    for k, l, mm, kvq in _OFF:                        # mm = mmompop slot of qp; kvq = qpint slot of vq
        vq[kvq] -= qp[mm] * qs2
        t3 -= ra[:, k] * ra[:, l] * qp[mm] * qs2
        vd[k] += ra[:, l] * qp[mm] * qs2
        vd[l] += ra[:, k] * qp[mm] * qs2
    for k in range(3):
        d = _DIAG_MM[k]
        vq[k] -= qp[d] * qs2 * 0.5
        t3 -= ra[:, k] ** 2 * qp[d] * qs2 * 0.5
        vd[k] += ra[:, k] * qp[d] * qs2
        t2 += qp[d]
    vs += t3
    t2 = t2 * qk
    for k in range(3):
        vq[k] += t2
        vd[k] -= 2.0 * ra[:, k] * t2
        vs += t2 * ra[:, k] ** 2
    return vs, vd, vq
