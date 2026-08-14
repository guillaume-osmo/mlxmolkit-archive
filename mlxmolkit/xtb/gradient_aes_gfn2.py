# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Analytical/semi-analytical AES gradient pieces for GFN2-xTB."""

from __future__ import annotations

import numpy as np

from .aes import aniso_electro, get_radcn, mmomgabzero
from .aes_fast import mmompop_vectorized
from .basis import build_basis, sao_basis_metadata
from .gradient_gfn0 import cn_gradient
from .gradient_gfn2 import _ANG_TO_BOHR
from .multipole_integrals import multipole_matrices
from .multipole_grad import multipole_gradient
from .multipole_integrals_cpp import (
    CPP_AVAILABLE,
    mmompop_cpp,
    mmompop_chain_gradient_cpp,
    overlap_gradient_cpp,
    multipole_gradient_cpp,
    multipole_matrices_cpp,
)
from .params_gfn2 import GFN2_PARAMS
from .params_gfn2 import GFN2_GLOBALS, _GFN2_MULTI_RAD, _GFN2_VALENCE_CN
from .scf_gfn2 import gfn2_n_gauss


_QP_INT_TO_MM = np.array([0, 2, 5, 1, 3, 4], dtype=np.int64)
_QP_MM_DIAG = np.array([0, 2, 5], dtype=np.int64)


def _aniso_moment_derivatives(
    atoms: list[int],
    coords_bohr: np.ndarray,
    q: np.ndarray,
    dipm: np.ndarray,
    qp: np.ndarray,
    gab3: np.ndarray,
    gab5: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dE/ddipm, dE/dqp)`` for :func:`aniso_electro`.

    ``qp`` and ``dE/dqp`` use mmompop layout ``(xx, xy, yy, xz, yz, zz)``.
    """

    nat = coords_bohr.shape[0]
    d_dip = np.zeros_like(dipm)
    d_qp = np.zeros_like(qp)
    idx = np.array([[0, 1, 3],
                    [1, 2, 4],
                    [3, 4, 5]], dtype=np.int64)

    # Per-atom CT/polarization correction.
    for i, Z in enumerate(atoms):
        p = GFN2_PARAMS[int(Z)]
        d_dip[:, i] += 2.0 * p.dip_kernel * dipm[:, i]
        d_qp[_QP_MM_DIAG, i] += 2.0 * p.quad_kernel * qp[_QP_MM_DIAG, i]
        d_qp[[1, 3, 4], i] += 4.0 * p.quad_kernel * qp[[1, 3, 4], i]

    for i in range(nat):
        qi = q[i]
        for j in range(i):
            qj = q[j]
            rij = coords_bohr[j] - coords_bohr[i]
            r2 = float(np.dot(rij, rij))
            g3 = gab3[j, i]
            g5 = gab5[j, i]

            for k in range(3):
                d_dip[k, i] += qj * rij[k] * g3
                d_dip[k, j] -= qi * rij[k] * g3

                d_dip[k, i] += dipm[k, j] * r2 * g5
                d_dip[k, j] += dipm[k, i] * r2 * g5
                for l in range(3):
                    d_dip[l, i] -= 3.0 * dipm[k, j] * rij[l] * rij[k] * g5
                    d_dip[k, j] -= 3.0 * dipm[l, i] * rij[l] * rij[k] * g5

                    kl = idx[l, k]
                    rr = rij[l] * rij[k]
                    d_qp[kl, i] += qj * rr * g5
                    d_qp[kl, j] += qi * rr * g5

    return d_dip, d_qp


def _traceless_qp_derivative(raw: np.ndarray) -> np.ndarray:
    out = 1.5 * raw
    tr = 0.5 * (raw[0] + raw[2] + raw[5])
    out[0] -= tr
    out[2] -= tr
    out[5] -= tr
    return out


def _mmompop_chain_gradient_bohr(
    P: np.ndarray,
    S: np.ndarray,
    dpint: np.ndarray,
    qpint: np.ndarray,
    aoat: np.ndarray,
    coords_bohr: np.ndarray,
    dSA: np.ndarray,
    dSB: np.ndarray,
    dDA: np.ndarray,
    dDB: np.ndarray,
    dQA: np.ndarray,
    dQB: np.ndarray,
    dE_dip: np.ndarray,
    dE_qp: np.ndarray,
) -> np.ndarray:
    """Contract the Mulliken multipole chain into a nuclear gradient.

    All derivatives are with respect to Bohr coordinates. The returned
    gradient is therefore Hartree / Bohr.
    """

    nat = coords_bohr.shape[0]
    nao = S.shape[0]
    grad = np.zeros((nat, 3), dtype=np.float64)
    ao = np.asarray(aoat, dtype=np.int64)

    for atom in range(nat):
        for beta in range(3):
            ddip = np.zeros((3, nat), dtype=np.float64)
            dqp_raw = np.zeros((6, nat), dtype=np.float64)

            for i in range(nao):
                ii = int(ao[i])
                ra = coords_bohr[ii]
                for j in range(i):
                    jj = int(ao[j])
                    rb = coords_bohr[jj]
                    pij = float(P[j, i])
                    ps = pij * float(S[j, i])

                    dS = 0.0
                    if jj == atom:
                        dS += dSA[beta, j, i]
                    if ii == atom:
                        dS += dSB[beta, j, i]
                    dps = pij * dS

                    dD = np.zeros(3, dtype=np.float64)
                    dQ = np.zeros(6, dtype=np.float64)
                    for k in range(3):
                        if jj == atom:
                            dD[k] += dDA[beta, k, j, i]
                        if ii == atom:
                            dD[k] += dDB[beta, k, j, i]
                    for k in range(6):
                        if jj == atom:
                            dQ[k] += dQA[beta, k, j, i]
                        if ii == atom:
                            dQ[k] += dQB[beta, k, j, i]

                    for k in range(3):
                        pdmk = pij * float(dpint[k, j, i])
                        dpdmk = pij * dD[k]
                        ddip[k, ii] += (
                            (ps if (ii == atom and k == beta) else 0.0)
                            + ra[k] * dps
                            - dpdmk
                        )
                        ddip[k, jj] += (
                            (ps if (jj == atom and k == beta) else 0.0)
                            + rb[k] * dps
                            - dpdmk
                        )

                        for l in range(k):
                            if (k, l) == (1, 0):
                                qint_idx = 3
                                mm_idx = 1
                            elif (k, l) == (2, 0):
                                qint_idx = 4
                                mm_idx = 3
                            elif (k, l) == (2, 1):
                                qint_idx = 5
                                mm_idx = 4
                            else:  # pragma: no cover
                                raise RuntimeError("unreachable")

                            pdml = pij * float(dpint[l, j, i])
                            dpdml = pij * dD[l]
                            pqm = pij * float(qpint[qint_idx, j, i])
                            dpqm = pij * dQ[qint_idx]

                            for target, r in ((ii, ra), (jj, rb)):
                                explicit = 0.0
                                if target == atom:
                                    explicit += pdmk if l == beta else 0.0
                                    explicit += pdml if k == beta else 0.0
                                    explicit -= (
                                        (r[k] if l == beta else 0.0)
                                        + (r[l] if k == beta else 0.0)
                                    ) * ps
                                dqp_raw[mm_idx, target] += (
                                    dpdmk * r[l]
                                    + pdmk * (1.0 if target == atom and l == beta else 0.0)
                                    + dpdml * r[k]
                                    + pdml * (1.0 if target == atom and k == beta else 0.0)
                                    - r[l] * r[k] * dps
                                    - ((r[k] if target == atom and l == beta else 0.0)
                                       + (r[l] if target == atom and k == beta else 0.0)) * ps
                                    - dpqm
                                )

                        qint_idx = k
                        mm_idx = (0, 2, 5)[k]
                        pqm = pij * float(qpint[qint_idx, j, i])
                        dpqm = pij * dQ[qint_idx]
                        for target, r in ((ii, ra), (jj, rb)):
                            delta = 1.0 if target == atom and k == beta else 0.0
                            dqp_raw[mm_idx, target] += (
                                2.0 * dpdmk * r[k]
                                + 2.0 * pdmk * delta
                                - r[k] * r[k] * dps
                                - 2.0 * r[k] * delta * ps
                                - dpqm
                            )

            for i in range(nao):
                ii = int(ao[i])
                if ii != atom and atom not in (ii,):
                    pass
                pij = float(P[i, i])
                ps = pij * float(S[i, i])
                ra = coords_bohr[ii]

                dS = 0.0
                if ii == atom:
                    dS += dSA[beta, i, i] + dSB[beta, i, i]
                dps = pij * dS
                dD = np.zeros(3, dtype=np.float64)
                dQ = np.zeros(6, dtype=np.float64)
                if ii == atom:
                    dD = dDA[beta, :, i, i] + dDB[beta, :, i, i]
                    dQ = dQA[beta, :, i, i] + dQB[beta, :, i, i]

                for k in range(3):
                    pdmk = pij * float(dpint[k, i, i])
                    dpdmk = pij * dD[k]
                    delta_k = 1.0 if ii == atom and k == beta else 0.0
                    ddip[k, ii] += delta_k * ps + ra[k] * dps - dpdmk

                    for l in range(k):
                        if (k, l) == (1, 0):
                            qint_idx = 3
                            mm_idx = 1
                        elif (k, l) == (2, 0):
                            qint_idx = 4
                            mm_idx = 3
                        elif (k, l) == (2, 1):
                            qint_idx = 5
                            mm_idx = 4
                        else:  # pragma: no cover
                            raise RuntimeError("unreachable")

                        pdml = pij * float(dpint[l, i, i])
                        dpdml = pij * dD[l]
                        dpqm = pij * dQ[qint_idx]
                        delta_l = 1.0 if ii == atom and l == beta else 0.0
                        dqp_raw[mm_idx, ii] += (
                            dpdmk * ra[l]
                            + pdmk * delta_l
                            + dpdml * ra[k]
                            + pdml * delta_k
                            - ra[l] * ra[k] * dps
                            - (delta_l * ra[k] + ra[l] * delta_k) * ps
                            - dpqm
                        )

                    qint_idx = k
                    mm_idx = (0, 2, 5)[k]
                    dpqm = pij * dQ[qint_idx]
                    dqp_raw[mm_idx, ii] += (
                        2.0 * dpdmk * ra[k]
                        + 2.0 * pdmk * delta_k
                        - ra[k] * ra[k] * dps
                        - 2.0 * ra[k] * delta_k * ps
                        - dpqm
                    )

            dqp = _traceless_qp_derivative(dqp_raw)
            grad[atom, beta] = (
                float(np.sum(dE_dip * ddip))
                + float(np.sum(dE_qp * dqp))
            )

    return grad


def _fixed_moment_aes_energy(
    atoms: list[int],
    coords_ang: np.ndarray,
    q_at: np.ndarray,
    dipm: np.ndarray,
    qp: np.ndarray,
) -> float:
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    cn, _ = cn_gradient(atoms, coords_ang)
    radcn = get_radcn(atoms, cn)
    gab3, gab5 = mmomgabzero(coords_bohr, radcn)
    e_pair, e_polar = aniso_electro(atoms, coords_bohr, q_at, dipm, qp, gab3, gab5)
    return e_pair + e_polar


def _radcn_with_derivative(atoms: list[int], cn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = GFN2_GLOBALS
    rad = np.zeros(len(atoms), dtype=np.float64)
    drad_dcn = np.zeros(len(atoms), dtype=np.float64)
    for i, Z in enumerate(atoms):
        rco = float(_GFN2_MULTI_RAD[int(Z)])
        t1 = float(cn[i]) - float(_GFN2_VALENCE_CN[int(Z)]) - g.aesshift
        sigm = 1.0 / (1.0 + np.exp(-g.aesexp * t1))
        rad[i] = rco + (g.aesrmax - rco) * sigm
        drad_dcn[i] = (g.aesrmax - rco) * g.aesexp * sigm * (1.0 - sigm)
    return rad, drad_dcn


def _gab_value_derivatives(
    R: float,
    rco: float,
    power: int,
    kdmp: float,
) -> tuple[float, float, float]:
    rabinv = 1.0 / R
    ratio = rco * rabinv
    u = 6.0 * ratio ** kdmp
    damp = 1.0 / (1.0 + u)
    gab = damp * rabinv ** power
    dg_dR = rabinv ** (power + 1) * (kdmp * u * damp * damp - power * damp)
    if abs(rco) < 1e-14:
        dg_drco = 0.0
    else:
        dg_drco = -kdmp * u * damp * damp * rabinv ** power / rco
    return gab, dg_dR, dg_drco


def _fixed_moment_aes_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    q_at: np.ndarray,
    dipm: np.ndarray,
    qp: np.ndarray,
    cn: np.ndarray | None = None,
    dcn_dr: np.ndarray | None = None,
) -> np.ndarray:
    """Explicit coordinate/damping gradient at fixed AES multipoles."""

    coords_ang = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords_ang * _ANG_TO_BOHR
    nat = len(atoms)
    if cn is None or dcn_dr is None:
        cn, dcn_dr = cn_gradient(atoms, coords_ang)
    radcn, drad_dcn = _radcn_with_derivative(atoms, cn)
    g = GFN2_GLOBALS

    idx = np.array([[0, 1, 3],
                    [1, 2, 4],
                    [3, 4, 5]], dtype=np.int64)
    grad_bohr = np.zeros((nat, 3), dtype=np.float64)
    dE_drad = np.zeros(nat, dtype=np.float64)

    for i in range(nat):
        qi = q_at[i]
        for j in range(i):
            qj = q_at[j]
            rij = coords_bohr[j] - coords_bohr[i]
            R = float(np.linalg.norm(rij))
            if R < 1e-14:
                continue
            rco = 0.5 * (radcn[i] + radcn[j])
            gab3, dg3_dR, dg3_drco = _gab_value_derivatives(R, rco, 3, g.aesdmp3)
            gab5, dg5_dR, dg5_drco = _gab_value_derivatives(R, rco, 5, g.aesdmp5)
            r2 = float(np.dot(rij, rij))

            A = 0.0
            Bq = 0.0
            Bdd = 0.0
            dA = np.zeros(3, dtype=np.float64)
            dBq = np.zeros(3, dtype=np.float64)
            dBdd = np.zeros(3, dtype=np.float64)
            dip_dot = float(np.dot(dipm[:, j], dipm[:, i]))

            for k in range(3):
                A += qj * dipm[k, i] * rij[k]
                A -= dipm[k, j] * qi * rij[k]
                dA[k] += qj * dipm[k, i] - qi * dipm[k, j]

                dBdd[k] += 2.0 * dip_dot * rij[k]
                for l in range(3):
                    kl = idx[l, k]
                    rr = rij[l] * rij[k]
                    c_q = qj * qp[kl, i] + qi * qp[kl, j]
                    Bq += c_q * rr
                    dBq[k] += c_q * rij[l]
                    dBq[l] += c_q * rij[k]

                    Bdd -= 3.0 * dipm[k, j] * dipm[l, i] * rr
                    dBdd[l] -= 3.0 * dipm[k, j] * dipm[l, i] * rij[k]
                    dBdd[k] -= 3.0 * dipm[k, j] * dipm[l, i] * rij[l]

            Bdd += dip_dot * r2
            B = Bq + Bdd
            dB = dBq + dBdd
            dE_dr = gab3 * dA + gab5 * dB
            dE_dr += (A * dg3_dR + B * dg5_dR) * rij / R

            grad_bohr[i] -= dE_dr
            grad_bohr[j] += dE_dr
            dE_drco = A * dg3_drco + B * dg5_drco
            dE_drad[i] += 0.5 * dE_drco
            dE_drad[j] += 0.5 * dE_drco

    grad_ang = grad_bohr * _ANG_TO_BOHR
    for i in range(nat):
        if drad_dcn[i] != 0.0 and dE_drad[i] != 0.0:
            grad_ang += dE_drad[i] * drad_dcn[i] * dcn_dr[i]
    return grad_ang


def aes_gradient_frozen_density(
    atoms: list[int],
    coords_ang: np.ndarray,
    P_sao: np.ndarray,
    qsh: np.ndarray,
    shell_atom: np.ndarray,
    *,
    h_explicit: float = 1e-3,
) -> np.ndarray:
    """Gradient of ``E_aes`` with frozen density and shell charges.

    This replaces the expensive coordinate finite difference that
    rebuilds multipole integrals at every displaced geometry. The
    Mulliken multipole chain is analytical; only the cheap explicit
    pair/damping-coordinate piece is finite-differenced with dipoles and
    quadrupoles held fixed. ``h_explicit`` is accepted for API
    compatibility with the previous finite-difference implementation
    and is no longer used.
    """

    coords = np.asarray(coords_ang, dtype=np.float64)
    cao_basis = build_basis(
        atoms,
        coords,
        params_dict=GFN2_PARAMS,
        n_gauss_fn=gfn2_n_gauss,
    )
    if CPP_AVAILABLE:
        S_cao, dp_cao, qp_cao = multipole_matrices_cpp(cao_basis)
    else:
        S_cao, dp_cao, qp_cao = multipole_matrices(cao_basis)
    sao_basis, T = sao_basis_metadata(cao_basis)
    n_sao = T.shape[0]
    T_is_identity = (
        T.shape[0] == T.shape[1] and np.array_equal(T, np.eye(T.shape[0]))
    )

    if T_is_identity:
        S = S_cao
        dpint = dp_cao
        qpint = qp_cao
    else:
        S = T @ S_cao @ T.T
        dpint = np.empty((3, n_sao, n_sao), dtype=np.float64)
        qpint = np.empty((6, n_sao, n_sao), dtype=np.float64)
        for k in range(3):
            dpint[k] = T @ dp_cao[k] @ T.T
        for k in range(6):
            qpint[k] = T @ qp_cao[k] @ T.T

    aoat = np.array([b.atom_idx for b in sao_basis], dtype=np.int64)
    from .overlap_grad import overlap_gradient

    if CPP_AVAILABLE:
        dSA_cao, dSB_cao = overlap_gradient_cpp(cao_basis)
    else:
        dSA_cao, dSB_cao = overlap_gradient(cao_basis)
    if T_is_identity:
        dSA = dSA_cao
        dSB = dSB_cao
    else:
        dSA = np.empty((3, n_sao, n_sao), dtype=np.float64)
        dSB = np.empty((3, n_sao, n_sao), dtype=np.float64)
        for beta in range(3):
            dSA[beta] = T @ dSA_cao[beta] @ T.T
            dSB[beta] = T @ dSB_cao[beta] @ T.T

    return aes_gradient_frozen_density_state(
        atoms,
        coords,
        P_sao,
        qsh,
        shell_atom,
        S,
        dpint,
        qpint,
        aoat,
        cao_basis,
        T,
        dSA,
        dSB,
    )


def aes_gradient_frozen_density_state(
    atoms: list[int],
    coords_ang: np.ndarray,
    P_sao: np.ndarray,
    qsh: np.ndarray,
    shell_atom: np.ndarray,
    S: np.ndarray,
    dpint: np.ndarray,
    qpint: np.ndarray,
    aoat: np.ndarray,
    cao_basis,
    T: np.ndarray,
    dSA: np.ndarray,
    dSB: np.ndarray,
    cn: np.ndarray | None = None,
    dcn_dr: np.ndarray | None = None,
) -> np.ndarray:
    """AES gradient reusing already-built SCF/gradient state."""

    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords * _ANG_TO_BOHR
    nat = len(atoms)
    n_sao = T.shape[0]
    T_is_identity = (
        T.shape[0] == T.shape[1] and np.array_equal(T, np.eye(T.shape[0]))
    )
    q_at = np.zeros(nat, dtype=np.float64)
    for ish, atom in enumerate(shell_atom):
        q_at[int(atom)] += qsh[ish]

    if cn is None:
        cn, _ = cn_gradient(atoms, coords)
    radcn = get_radcn(atoms, cn)
    gab3, gab5 = mmomgabzero(coords_bohr, radcn)
    if CPP_AVAILABLE:
        dipm, qp = mmompop_cpp(P_sao, S, dpint, qpint, aoat, coords_bohr)
    else:
        dipm, qp = mmompop_vectorized(P_sao, S, dpint, qpint, aoat, coords_bohr)
    dE_dip, dE_dqp = _aniso_moment_derivatives(
        atoms, coords_bohr, q_at, dipm, qp, gab3, gab5,
    )

    if CPP_AVAILABLE:
        dDA_cao, dDB_cao, dQA_cao, dQB_cao = multipole_gradient_cpp(cao_basis)
    else:
        dDA_cao, dDB_cao, dQA_cao, dQB_cao = multipole_gradient(cao_basis)

    if T_is_identity:
        dDA = dDA_cao
        dDB = dDB_cao
        dQA = dQA_cao
        dQB = dQB_cao
    else:
        dDA = np.empty((3, 3, n_sao, n_sao), dtype=np.float64)
        dDB = np.empty((3, 3, n_sao, n_sao), dtype=np.float64)
        dQA = np.empty((3, 6, n_sao, n_sao), dtype=np.float64)
        dQB = np.empty((3, 6, n_sao, n_sao), dtype=np.float64)
        for beta in range(3):
            for k in range(3):
                dDA[beta, k] = T @ dDA_cao[beta, k] @ T.T
                dDB[beta, k] = T @ dDB_cao[beta, k] @ T.T
            for k in range(6):
                dQA[beta, k] = T @ dQA_cao[beta, k] @ T.T
                dQB[beta, k] = T @ dQB_cao[beta, k] @ T.T

    if CPP_AVAILABLE:
        g_chain_bohr = mmompop_chain_gradient_cpp(
            P_sao,
            S,
            dpint,
            qpint,
            aoat,
            coords_bohr,
            dSA,
            dSB,
            dDA,
            dDB,
            dQA,
            dQB,
            dE_dip,
            dE_dqp,
        )
    else:
        g_chain_bohr = _mmompop_chain_gradient_bohr(
            P_sao,
            S,
            dpint,
            qpint,
            aoat,
            coords_bohr,
            dSA,
            dSB,
            dDA,
            dDB,
            dQA,
            dQB,
            dE_dip,
            dE_dqp,
        )
    g_explicit_ang = _fixed_moment_aes_gradient(
        atoms, coords, q_at, dipm, qp, cn=cn, dcn_dr=dcn_dr
    )
    return g_chain_bohr * _ANG_TO_BOHR + g_explicit_ang
