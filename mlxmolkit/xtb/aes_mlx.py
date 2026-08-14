# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Experimental MLX kernels for GFN2 AES.

This module intentionally lives beside :mod:`mlxmolkit.xtb.aes` instead
of replacing it. The NumPy implementation remains the reference path;
functions here are opt-in building blocks for benchmarking and gradual
SCF kernelization.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .params_gfn2 import (
    GFN2_GLOBALS,
    GFN2_PARAMS,
    _GFN2_MULTI_RAD,
    _GFN2_VALENCE_CN,
)


def _table(values) -> mx.array:
    arr = np.zeros(119, dtype=np.float32)
    if hasattr(values, "items"):
        iterator = values.items()
    else:
        iterator = enumerate(values)
    for z, v in iterator:
        if int(z) < arr.shape[0]:
            arr[int(z)] = float(v)
    return mx.array(arr)


_MULTI_RAD = _table(_GFN2_MULTI_RAD)
_VALENCE_CN = _table(_GFN2_VALENCE_CN)
_DIP_KERNEL = _table({z: p.dip_kernel for z, p in GFN2_PARAMS.items()})
_QUAD_KERNEL = _table({z: p.quad_kernel for z, p in GFN2_PARAMS.items()})


def get_radcn_mlx(
    atoms: mx.array,
    cn: mx.array,
    *,
    shift: float | None = None,
    expo: float | None = None,
    rmax: float | None = None,
) -> mx.array:
    """MLX version of :func:`mlxmolkit.xtb.aes.get_radcn`.

    Args:
        atoms: ``(nat,)`` int atomic numbers.
        cn: ``(nat,)`` coordination numbers.

    Returns:
        ``(nat,)`` cutoff radii in Bohr.
    """
    g = GFN2_GLOBALS
    if shift is None:
        shift = g.aesshift
    if expo is None:
        expo = g.aesexp
    if rmax is None:
        rmax = g.aesrmax

    atoms = atoms.astype(mx.int32)
    dtype = cn.dtype
    rco = mx.take(_MULTI_RAD, atoms).astype(dtype)
    valcn = mx.take(_VALENCE_CN, atoms).astype(dtype)
    t1 = cn - valcn - mx.array(shift, dtype=dtype)
    sigm = 1.0 / (1.0 + mx.exp(-mx.array(expo, dtype=dtype) * t1))
    return rco + (mx.array(rmax, dtype=dtype) - rco) * sigm


def mmomgabzero_mlx(
    coords_bohr: mx.array,
    radcn: mx.array,
    *,
    kdmp3: float | None = None,
    kdmp5: float | None = None,
) -> tuple[mx.array, mx.array]:
    """MLX vectorized ``gab3``/``gab5`` damping matrices."""
    g = GFN2_GLOBALS
    if kdmp3 is None:
        kdmp3 = g.aesdmp3
    if kdmp5 is None:
        kdmp5 = g.aesdmp5

    dtype = coords_bohr.dtype
    diff = coords_bohr[:, None, :] - coords_bohr[None, :, :]
    r2 = mx.sum(diff * diff, axis=-1)
    n = coords_bohr.shape[0]
    eye = mx.eye(n, dtype=dtype)
    # Use R=1 on the diagonal while forming powers; the final mask
    # zeros diagonal entries. This avoids inf*0 -> NaN on Metal.
    r = mx.sqrt(mx.maximum(r2, mx.array(1e-30, dtype=dtype)) + eye)
    inv_r = 1.0 / r
    rco = 0.5 * (radcn[:, None] + radcn[None, :])

    damp3 = 1.0 / (1.0 + 6.0 * mx.power(rco * inv_r, kdmp3))
    damp5 = 1.0 / (1.0 + 6.0 * mx.power(rco * inv_r, kdmp5))
    mask = 1.0 - eye
    gab3 = damp3 * mx.power(inv_r, 3.0) * mask
    gab5 = damp5 * mx.power(inv_r, 5.0) * mask
    return gab3, gab5


def fockelectro_mlx(
    P: mx.array,
    S: mx.array,
    dpint: mx.array,
    qpint: mx.array,
    aoat: mx.array,
    vs: mx.array,
    vd: mx.array,
    vq: mx.array,
) -> tuple[mx.array, mx.array]:
    """MLX vectorized AES Fock builder.

    Mirrors :func:`mlxmolkit.xtb.aes.fockelectro` and returns
    ``(F_aes, e_aes)``. Inputs are expected to already be MLX arrays:

    - ``P, S``: ``(nao, nao)``
    - ``dpint``: ``(3, nao, nao)``
    - ``qpint``: ``(6, nao, nao)``
    - ``aoat``: ``(nao,)`` int atom index per AO
    - ``vs, vd, vq``: AES potentials from the reference or future MLX path
    """
    ao = aoat.astype(mx.int32)
    vs_ao = mx.take(vs, ao)
    fji = S * (vs_ao[None, :] + vs_ao[:, None])

    for k in range(3):
        vd_ao = mx.take(vd[k], ao)
        fji = fji + mx.transpose(dpint[k]) * (vd_ao[None, :] + vd_ao[:, None])

    for k in range(6):
        vq_ao = mx.take(vq[k], ao)
        fji = fji + mx.transpose(qpint[k]) * (vq_ao[None, :] + vq_ao[:, None])

    F = 0.5 * fji
    e_aes = 0.25 * mx.sum(P * fji)
    return F, e_aes


def _atom_reduce_pair_values(
    values_i: mx.array,
    values_j: mx.array,
    values_diag: mx.array,
    aoat: mx.array,
    nat: int,
) -> mx.array:
    """Reduce component × AO × AO pair contributions to atoms."""
    atom_ids = mx.arange(nat, dtype=aoat.dtype)
    row_mask = (aoat[None, :, None] == atom_ids[:, None, None]).astype(values_i.dtype)
    col_mask = (aoat[None, None, :] == atom_ids[:, None, None]).astype(values_i.dtype)
    out = mx.sum(values_i[:, None, :, :] * row_mask[None, :, :, :], axis=(2, 3))
    out = out + mx.sum(values_j[:, None, :, :] * col_mask[None, :, :, :], axis=(2, 3))
    out = out + mx.sum(values_diag[:, None, :, :] * row_mask[None, :, :, :], axis=(2, 3))
    return out


def mmompop_mlx(
    P: mx.array,
    S: mx.array,
    dpint: mx.array,
    qpint: mx.array,
    aoat: mx.array,
    coords_bohr: mx.array,
) -> tuple[mx.array, mx.array]:
    """MLX vectorized Mulliken cumulative atomic multipoles.

    Mirrors :func:`mlxmolkit.xtb.aes.mmompop`. The returned quadrupole
    uses mmompop layout ``(xx, xy, yy, xz, yz, zz)``.
    """
    ao = aoat.astype(mx.int32)
    nat = coords_bohr.shape[0]
    nao = P.shape[0]
    dtype = P.dtype

    idx = mx.arange(nao, dtype=mx.int32)
    lower = (idx[:, None] > idx[None, :]).astype(dtype)
    diag = mx.eye(nao, dtype=dtype)

    # Formula variables indexed as [i, j], matching the original loops
    # where i is the higher AO index and j is the lower AO index.
    pji = mx.transpose(P)
    sji = mx.transpose(S)
    ps = pji * sji
    dji = mx.transpose(dpint, axes=(0, 2, 1))
    qji = mx.transpose(qpint, axes=(0, 2, 1))

    ao_coords = mx.take(coords_bohr, ao, axis=0)
    ri = mx.transpose(ao_coords, axes=(1, 0))[:, :, None]
    rj = mx.transpose(ao_coords, axes=(1, 0))[:, None, :]

    dip_i = ri * ps[None, :, :] - pji[None, :, :] * dji
    dip_j = rj * ps[None, :, :] - pji[None, :, :] * dji
    dipm = _atom_reduce_pair_values(
        dip_i * lower[None, :, :],
        dip_j * lower[None, :, :],
        dip_i * diag[None, :, :],
        ao,
        nat,
    )

    # Build quadrupoles in mmompop layout: xx, xy, yy, xz, yz, zz.
    qp_i_parts = []
    qp_j_parts = []

    # xx, yy, zz diagonal slots.
    for axis, qint in [(0, 0), (1, 1), (2, 2)]:
        qi = 2.0 * pji * dji[axis] * ri[axis] - ri[axis] * ri[axis] * ps - pji * qji[qint]
        qj = 2.0 * pji * dji[axis] * rj[axis] - rj[axis] * rj[axis] * ps - pji * qji[qint]
        qp_i_parts.append(qi)
        qp_j_parts.append(qj)

    # Reorder/insert off-diagonal slots into mmompop layout.
    def offdiag(a: int, b: int, qint: int) -> tuple[mx.array, mx.array]:
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

    qp_i = mx.stack([qxx_i, qxy_i, qyy_i, qxz_i, qyz_i, qzz_i], axis=0)
    qp_j = mx.stack([qxx_j, qxy_j, qyy_j, qxz_j, qyz_j, qzz_j], axis=0)
    qp = _atom_reduce_pair_values(
        qp_i * lower[None, :, :],
        qp_j * lower[None, :, :],
        qp_i * diag[None, :, :],
        ao,
        nat,
    )

    tr = 0.5 * (qp[0] + qp[2] + qp[5])
    qp = qp * 1.5
    qp = qp.at[0].subtract(tr)
    qp = qp.at[2].subtract(tr)
    qp = qp.at[5].subtract(tr)
    return dipm, qp


def atom_ct_energy_mlx(
    atoms: mx.array,
    dipm: mx.array,
    qp: mx.array,
) -> mx.array:
    """MLX per-atom CT/polarization energy from AES multipoles.

    This is a small helper for future ``aniso_electro_mlx`` work. The
    ``qp`` layout is mmompop order ``(xx, xy, yy, xz, yz, zz)``.
    """
    atoms = atoms.astype(mx.int32)
    dip_kernel = mx.take(_DIP_KERNEL, atoms).astype(dipm.dtype)
    quad_kernel = mx.take(_QUAD_KERNEL, atoms).astype(dipm.dtype)
    dip2 = mx.sum(dipm * dipm, axis=0)
    qp2 = mx.sum(qp * qp, axis=0)
    return mx.sum(dip_kernel * dip2 + quad_kernel * qp2)
