# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Vectorized experimental AES helpers.

These are CPU/NumPy accelerators that preserve the reference path's
float64 numerics. They live beside :mod:`aes` while the fast GFN2 path
is still being hardened.
"""

from __future__ import annotations

import numpy as np

from .params_gfn2 import GFN2_PARAMS


def mulliken_shell_charges_vectorized(
    P: np.ndarray,
    S: np.ndarray,
    bf_to_shell: np.ndarray,
    n_shell: int,
    z_ref: np.ndarray,
) -> np.ndarray:
    """Vectorized version of :func:`scf_gfn2._mulliken_shell_charges`."""
    shell_idx = np.asarray(bf_to_shell, dtype=np.int64)
    ps_diag = np.einsum("ik,ki->i", P, S, optimize=True)
    pop = np.bincount(
        shell_idx,
        weights=ps_diag,
        minlength=int(n_shell),
    ).astype(np.float64, copy=False)
    return np.asarray(z_ref, dtype=np.float64) - pop


def _atom_reduce_pair_values(
    values_i: np.ndarray,
    values_j: np.ndarray,
    values_diag: np.ndarray,
    aoat: np.ndarray,
    nat: int,
) -> np.ndarray:
    row_atoms = np.broadcast_to(aoat[:, None], values_i.shape[1:]).ravel()
    col_atoms = np.broadcast_to(aoat[None, :], values_i.shape[1:]).ravel()
    out = np.empty((values_i.shape[0], nat), dtype=np.float64)
    for k in range(values_i.shape[0]):
        out[k] = np.bincount(
            row_atoms,
            weights=(values_i[k] + values_diag[k]).ravel(),
            minlength=nat,
        )
        out[k] += np.bincount(
            col_atoms,
            weights=values_j[k].ravel(),
            minlength=nat,
        )
    return out


def mmompop_vectorized(
    P: np.ndarray,
    S: np.ndarray,
    dpint: np.ndarray,
    qpint: np.ndarray,
    aoat: np.ndarray,
    coords_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized NumPy version of :func:`mlxmolkit.xtb.aes.mmompop`."""
    ao = np.asarray(aoat, dtype=np.int64)
    nat = coords_bohr.shape[0]
    nao = P.shape[0]

    idx = np.arange(nao)
    lower = (idx[:, None] > idx[None, :]).astype(np.float64)
    diag = np.eye(nao, dtype=np.float64)

    pji = P.T
    sji = S.T
    ps = pji * sji
    dji = np.transpose(dpint, axes=(0, 2, 1))
    qji = np.transpose(qpint, axes=(0, 2, 1))

    ao_coords = coords_bohr[ao]
    ri = ao_coords.T[:, :, None]
    rj = ao_coords.T[:, None, :]

    dip_i = ri * ps[None, :, :] - pji[None, :, :] * dji
    dip_j = rj * ps[None, :, :] - pji[None, :, :] * dji
    dipm = _atom_reduce_pair_values(
        dip_i * lower[None, :, :],
        dip_j * lower[None, :, :],
        dip_i * diag[None, :, :],
        ao,
        nat,
    )

    qp_i_parts = []
    qp_j_parts = []
    for axis, qint in [(0, 0), (1, 1), (2, 2)]:
        qi = 2.0 * pji * dji[axis] * ri[axis] - ri[axis] * ri[axis] * ps - pji * qji[qint]
        qj = 2.0 * pji * dji[axis] * rj[axis] - rj[axis] * rj[axis] * ps - pji * qji[qint]
        qp_i_parts.append(qi)
        qp_j_parts.append(qj)

    def offdiag(a: int, b: int, qint: int) -> tuple[np.ndarray, np.ndarray]:
        qi = (
            pji * dji[a] * ri[b]
            + pji * dji[b] * ri[a]
            - ri[a] * ri[b] * ps
            - pji * qji[qint]
        )
        qj = (
            pji * dji[a] * rj[b]
            + pji * dji[b] * rj[a]
            - rj[a] * rj[b] * ps
            - pji * qji[qint]
        )
        return qi, qj

    qxx_i, qyy_i, qzz_i = qp_i_parts
    qxx_j, qyy_j, qzz_j = qp_j_parts
    qxy_i, qxy_j = offdiag(1, 0, 3)
    qxz_i, qxz_j = offdiag(2, 0, 4)
    qyz_i, qyz_j = offdiag(2, 1, 5)

    qp_i = np.stack([qxx_i, qxy_i, qyy_i, qxz_i, qyz_i, qzz_i], axis=0)
    qp_j = np.stack([qxx_j, qxy_j, qyy_j, qxz_j, qyz_j, qzz_j], axis=0)
    qp = _atom_reduce_pair_values(
        qp_i * lower[None, :, :],
        qp_j * lower[None, :, :],
        qp_i * diag[None, :, :],
        ao,
        nat,
    )

    tr = 0.5 * (qp[0] + qp[2] + qp[5])
    qp *= 1.5
    qp[0] -= tr
    qp[2] -= tr
    qp[5] -= tr
    return dipm, qp


def setvsdq_vectorized(
    atoms: list[int],
    coords_bohr: np.ndarray,
    q: np.ndarray,
    dipm: np.ndarray,
    qp: np.ndarray,
    gab3: np.ndarray,
    gab5: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized NumPy version of :func:`mlxmolkit.xtb.aes.setvsdq`."""
    atoms_arr = np.asarray(atoms, dtype=np.int64)
    coords = np.asarray(coords_bohr, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    dipm = np.asarray(dipm, dtype=np.float64)
    qp = np.asarray(qp, dtype=np.float64)
    nat = coords.shape[0]

    ra = coords[None, :, :]          # i atom
    dra = coords[None, :, :] - coords[:, None, :]  # [j, i, xyz]
    qj = q[:, None]
    dip_j = dipm.T[:, None, :]

    r2a = np.sum(coords * coords, axis=1)[None, :]
    r2ab = np.sum(dra * dra, axis=2)
    t1 = np.sum(ra * dra, axis=2)
    t2 = np.sum(dip_j * dra, axis=2)
    t3 = np.sum(ra * dip_j, axis=2)

    idx = np.array([[0, 1, 3],
                    [1, 2, 4],
                    [3, 4, 5]], dtype=np.int64)
    qp_mat = qp.T[:, idx]  # [j, xyz, xyz], mmompop layout
    dum5 = -np.einsum("jab,jia,jib->ji", qp_mat, dra, dra)
    dum5 -= 1.5 * qj * t1 * t1
    dum5 += t3 * r2ab - 3.0 * t1 * t2 + 0.5 * qj * r2a * r2ab
    dum3 = -t1 * qj - t2
    vs = np.sum(dum5 * gab5 + dum3 * gab3, axis=0)

    dum3_vec = dra * qj[:, :, None]
    dum5_vec = (
        3.0 * dra * t2[:, :, None]
        - r2ab[:, :, None] * dip_j
        - qj[:, :, None] * r2ab[:, :, None] * ra
        + 3.0 * qj[:, :, None] * dra * t1[:, :, None]
    )
    vd = np.sum(dum3_vec * gab3[:, :, None] + dum5_vec * gab5[:, :, None], axis=0).T

    vq = np.zeros((6, nat), dtype=np.float64)
    qg5 = qj * gab5
    for axis in range(3):
        vq[axis] += np.sum(
            -1.5 * qg5 * dra[:, :, axis] * dra[:, :, axis]
            + 0.5 * r2ab * qg5,
            axis=0,
        )
    for l1, l2, slot in [(1, 0, 3), (2, 0, 4), (2, 1, 5)]:
        vq[slot] += np.sum(-3.0 * qg5 * dra[:, :, l2] * dra[:, :, l1], axis=0)

    dip_kernel = np.array([GFN2_PARAMS[int(z)].dip_kernel for z in atoms_arr])
    quad_kernel = np.array([GFN2_PARAMS[int(z)].quad_kernel for z in atoms_arr])
    qs1 = 2.0 * dip_kernel
    qs2 = 6.0 * quad_kernel

    # CT correction.
    vs += np.sum(coords.T * dipm, axis=0) * qs1
    vd -= qs1[None, :] * dipm

    for l1, l2, qp_slot, vq_slot in [(1, 0, 1, 3), (2, 0, 3, 4), (2, 1, 4, 5)]:
        vq[vq_slot] -= qp[qp_slot] * qs2
        vs -= coords[:, l1] * coords[:, l2] * qp[qp_slot] * qs2
        vd[l1] += coords[:, l2] * qp[qp_slot] * qs2
        vd[l2] += coords[:, l1] * qp[qp_slot] * qs2

    for axis, qp_slot in [(0, 0), (1, 2), (2, 5)]:
        vq[axis] -= qp[qp_slot] * qs2 * 0.5
        vs -= coords[:, axis] * coords[:, axis] * qp[qp_slot] * qs2 * 0.5
        vd[axis] += coords[:, axis] * qp[qp_slot] * qs2

    trace_q = qp[0] + qp[2] + qp[5]
    t2a = trace_q * quad_kernel
    vq[0] += t2a
    vq[1] += t2a
    vq[2] += t2a
    vd -= 2.0 * coords.T * t2a[None, :]
    vs += np.sum(coords * coords, axis=1) * t2a
    return vs, vd, vq
