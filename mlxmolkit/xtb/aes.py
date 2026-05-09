# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN2 anisotropic electrostatics (AES) — verbatim port of
``xtb/src/aespot.F90``.

Adds the multipole interactions to the SCF on top of the monopole
electrostatics (gam-Coulomb) already in :mod:`scf_gfn2`. The pieces:

- :func:`mmompop`: Mulliken atomic dipole and quadrupole moments from
  the density matrix and the (S, dpint, qpint) integrals (xtb's
  ``mmompop_cpu`` at aespot.F90:289-427).
- :func:`get_radcn`: CN-dependent atomic multipole cutoff radii
  (aespot.F90:1267-1281).
- :func:`mmomgabzero`: damped Coulomb operators ``gab3 = damp/R³``,
  ``gab5 = damp/R⁵`` for the multipole-multipole interactions
  (aespot.F90:1219-1255).
- :func:`aniso_electro`: AES energy = q-dip + q-qpole + dip-dip +
  the per-atom CT (charge-transfer) correction with dipKernel /
  quadKernel kernels (aespot.F90:569-664).
- :func:`setvsdq`: build the per-atom potentials ``vs[nat]``,
  ``vd[3, nat]``, ``vq[6, nat]`` (potentials proportional to S, D, Q
  integrals respectively — aespot.F90:727-848).
- :func:`fockelectro`: assemble the AES Fock contribution
  ``F_AES[μ, ν] = ¼ Σ ((S, dpint, qpint) at μν) · (v[atom_μ] +
  v[atom_ν])`` (aespot.F90:676-709).

Conventions:
- ``qp`` (multipole moments output by :func:`mmompop`) uses xtb's
  mmompop layout: ``(xx, xy, yy, xz, yz, zz)`` (= ``lin(l1, l2)``).
- ``vq`` (potential output by :func:`setvsdq`) uses xtb's **qpint**
  layout: ``(xx, yy, zz, xy, xz, yz)`` — same as the integral-routine.
  The reason is that vq multiplies qpint in :func:`fockelectro`, so it
  must match the qpint slot order. xtb itself does this layout switch
  (compare aespot.F90:782 vs aespot.F90:786 — qp is read with mmompop
  ``lin`` packing, but vq is written with qpint ``ki = l1+l2+1``).
- ``dipm`` (and ``vd``) is just (x, y, z).
"""

from __future__ import annotations

import numpy as np

from .params_gfn2 import (
    GFN2_GLOBALS,
    GFN2_PARAMS,
    _GFN2_MULTI_RAD,
    _GFN2_VALENCE_CN,
)


# Map (l1, l2) → packed index for xtb's qp/vq layout (mmompop convention)
# (xx, xy, yy, xz, yz, zz). Symmetric in (l1, l2). 0-indexed Python.
_LIN = ((0, 1, 3),
        (1, 2, 4),
        (3, 4, 5))


def _lin(l1: int, l2: int) -> int:
    """Same as xtb's ``lin(l1, l2)`` for a 3×3 → 6 packing."""
    return _LIN[l1][l2]


def get_radcn(
    atoms: list[int],
    cn: np.ndarray,
    *,
    shift: float | None = None,
    expo: float | None = None,
    rmax: float | None = None,
) -> np.ndarray:
    """CN-dependent multipole cutoff radii (aespot.F90:1267-1281).

    Returns ``(n_atoms,)`` array in atomic units. Mirrors xtb:

        radcn[i] = multiRad[Z_i] + (rmax - multiRad[Z_i])
                                · 1 / (1 + exp(-expo · t1))
        t1 = cn[i] − valenceCN[Z_i] − shift
    """
    g = GFN2_GLOBALS
    if shift is None: shift = g.aesshift
    if expo  is None: expo  = g.aesexp
    if rmax  is None: rmax  = g.aesrmax
    n = len(atoms)
    out = np.zeros(n, dtype=np.float64)
    for i, Z in enumerate(atoms):
        rco = float(_GFN2_MULTI_RAD[int(Z)])
        t1 = float(cn[i]) - float(_GFN2_VALENCE_CN[int(Z)]) - shift
        sigm = 1.0 / (1.0 + np.exp(-expo * t1))
        out[i] = rco + (rmax - rco) * sigm
    return out


def mmomgabzero(
    coords_bohr: np.ndarray,
    radcn: np.ndarray,
    *,
    kdmp3: float | None = None,
    kdmp5: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Damped multipole Coulomb operators (aespot.F90:1219-1255).

    ``gab3 = 1/(R³) · damp(kdmp3)`` for q-dip term (R^-2 derivative).
    ``gab5 = 1/(R⁵) · damp(kdmp5)`` for q-qpole and dip-dip (R^-3).

    The damping is xtb's ``dzero``::

        rco = 0.5·(rad_i + rad_j)
        damp = 1 / (1 + 6·(rco / R)^dex)

    Same atom (i = j) entry is left as zero — the per-atom CT term is
    handled separately in :func:`aniso_electro`.
    """
    g = GFN2_GLOBALS
    if kdmp3 is None: kdmp3 = g.aesdmp3
    if kdmp5 is None: kdmp5 = g.aesdmp5
    n = coords_bohr.shape[0]
    gab3 = np.zeros((n, n), dtype=np.float64)
    gab5 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i):
            R = float(np.linalg.norm(coords_bohr[i] - coords_bohr[j]))
            if R < 1e-12:
                continue
            rabinv = 1.0 / R
            rco = 0.5 * (radcn[i] + radcn[j])
            damp3 = 1.0 / (1.0 + 6.0 * (rco * rabinv) ** kdmp3)
            damp5 = 1.0 / (1.0 + 6.0 * (rco * rabinv) ** kdmp5)
            gab3[i, j] = gab3[j, i] = damp3 * rabinv ** 3
            gab5[i, j] = gab5[j, i] = damp5 * rabinv ** 5
    return gab3, gab5


def mmompop(
    P: np.ndarray,
    S: np.ndarray,
    dpint: np.ndarray,
    qpint: np.ndarray,
    aoat: np.ndarray,
    coords_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mulliken cumulative atomic multipole moments (CAMM).

    Computes per-atom cumulative dipole moments
    ``dipm[3, nat]`` and traceless quadrupole moments
    ``qp[6, nat]`` in xtb's mmompop layout
    ``(xx, xy, yy, xz, yz, zz)`` (aespot.F90:289-427).

    Args:
        P: ``(nao, nao)`` density matrix in SAO basis.
        S: ``(nao, nao)`` overlap matrix.
        dpint: ``(3, nao, nao)`` dipole AO integrals (xyz).
        qpint: ``(6, nao, nao)`` quadrupole AO integrals in
            **integral-routine layout** (xx, yy, zz, xy, xz, yz).
        aoat: ``(nao,)`` atom index for each AO.
        coords_bohr: ``(nat, 3)`` atom positions in Bohr.

    Returns:
        ``(dipm, qp)`` with ``dipm`` shape ``(3, nat)`` and ``qp``
        shape ``(6, nat)`` in xtb's mmompop layout.
    """
    nao = S.shape[0]
    nat = coords_bohr.shape[0]
    dipm = np.zeros((3, nat), dtype=np.float64)
    qp = np.zeros((6, nat), dtype=np.float64)

    # Index mapping from integral-routine qpint layout (xx, yy, zz, xy, xz, yz)
    # to xtb's mmompop "lin" packing (xx, xy, yy, xz, yz, zz).
    # The xtb code uses kj = k+l+1 to look up the integral-layout entry
    # for the (l, k) cross term, so kj=4 → xy, kj=5 → xz, kj=6 → yz.
    # We translate by a fixed map.
    int_to_lin_diag = {0: 0, 1: 2, 2: 5}    # xx→pos0, yy→pos2, zz→pos5
    int_to_lin_offdiag = {3: 1, 4: 3, 5: 4} # xy→pos1, xz→pos3, yz→pos4

    for i in range(nao):
        for j in range(nao):
            if j >= i:
                continue
            ii = int(aoat[i])
            jj = int(aoat[j])
            ra = coords_bohr[ii]
            rb = coords_bohr[jj]
            pij = float(P[j, i])
            ps = pij * float(S[j, i])
            for k in range(3):
                xk1 = ra[k]
                xk2 = rb[k]
                pdmk = pij * float(dpint[k, j, i])
                tii = xk1 * ps - pdmk
                tjj = xk2 * ps - pdmk
                dipm[k, jj] += tjj
                dipm[k, ii] += tii
                # off-diagonal qp components (l < k)
                for l in range(k):
                    kl_lin = int_to_lin_offdiag[3 + (k - 1) * (k) // 2 + l]   # not quite right...
                    # simpler: explicit map
                    if (k, l) == (1, 0): kl_lin = 1     # xy
                    elif (k, l) == (2, 0): kl_lin = 3   # xz
                    elif (k, l) == (2, 1): kl_lin = 4   # yz
                    else:
                        raise RuntimeError("unreachable")
                    # corresponding integral-layout slot for the (k,l) cross term:
                    if (k, l) == (1, 0): qint_idx = 3   # xy
                    elif (k, l) == (2, 0): qint_idx = 4 # xz
                    elif (k, l) == (2, 1): qint_idx = 5 # yz
                    xl1 = ra[l]
                    xl2 = rb[l]
                    pdml = pij * float(dpint[l, j, i])
                    pqm = pij * float(qpint[qint_idx, j, i])
                    tii = pdmk * xl1 + pdml * xk1 - xl1 * xk1 * ps - pqm
                    tjj = pdmk * xl2 + pdml * xk2 - xl2 * xk2 * ps - pqm
                    qp[kl_lin, jj] += tjj
                    qp[kl_lin, ii] += tii
                # diagonal qp[kk]: lin position
                kl_lin = int_to_lin_diag[k]   # 0, 2, 5 for x, y, z
                qint_idx = k                   # xx, yy, zz in integral layout
                pqm = pij * float(qpint[qint_idx, j, i])
                tii = 2.0 * pdmk * xk1 - xk1 * xk1 * ps - pqm
                tjj = 2.0 * pdmk * xk2 - xk2 * xk2 * ps - pqm
                qp[kl_lin, jj] += tjj
                qp[kl_lin, ii] += tii

    # Diagonal AO contribution (i == j case in xtb's second loop)
    for i in range(nao):
        ii = int(aoat[i])
        ra = coords_bohr[ii]
        pij = float(P[i, i])
        ps = pij * float(S[i, i])
        for k in range(3):
            xk1 = ra[k]
            pdmk = pij * float(dpint[k, i, i])
            tii = xk1 * ps - pdmk
            dipm[k, ii] += tii
            for l in range(k):
                if (k, l) == (1, 0): kl_lin = 1; qint_idx = 3
                elif (k, l) == (2, 0): kl_lin = 3; qint_idx = 4
                elif (k, l) == (2, 1): kl_lin = 4; qint_idx = 5
                else: raise RuntimeError("unreachable")
                xl1 = ra[l]
                pdml = pij * float(dpint[l, i, i])
                pqm = pij * float(qpint[qint_idx, i, i])
                tii = pdmk * xl1 + pdml * xk1 - xl1 * xk1 * ps - pqm
                qp[kl_lin, ii] += tii
            kl_lin = (0, 2, 5)[k]
            qint_idx = k
            pqm = pij * float(qpint[qint_idx, i, i])
            tii = 2.0 * pdmk * xk1 - xk1 * xk1 * ps - pqm
            qp[kl_lin, ii] += tii

    # Trace removal (aespot.F90:418-425):  q[1] += -tii/2; q[3] += -tii/2;
    # q[6] += -tii/2 with tii = q[1]+q[3]+q[6]; then scale qp by 1.5.
    # (xtb indexing q[1..6] maps to our [0, 1, 2, 3, 4, 5] in lin order
    # = (xx, xy, yy, xz, yz, zz); diagonals are at positions 0, 2, 5.)
    for ia in range(nat):
        tr = qp[0, ia] + qp[2, ia] + qp[5, ia]
        tr = 0.5 * tr
        qp[:, ia] *= 1.5
        qp[0, ia] -= tr
        qp[2, ia] -= tr
        qp[5, ia] -= tr
    return dipm, qp


def aniso_electro(
    atoms: list[int],
    coords_bohr: np.ndarray,
    q: np.ndarray,
    dipm: np.ndarray,
    qp: np.ndarray,
    gab3: np.ndarray,
    gab5: np.ndarray,
) -> tuple[float, float]:
    """AES energy (aespot.F90:569-662). Returns ``(E_aes, E_polar)``.

    ``E_aes`` is the sum of three terms (q-dip, q-qpole, dip-dip)
    weighted by gab3/gab5, summed over unique atom pairs.
    ``E_polar`` is the per-atom CT correction with dipKernel /
    quadKernel.

    The qp layout here is xtb's mmompop ``(xx, xy, yy, xz, yz, zz)``.
    """
    nat = coords_bohr.shape[0]
    # idx[k, l] = lin(k, l) - 1  (mmompop layout)
    idx = np.array([[0, 1, 3],
                    [1, 2, 4],
                    [3, 4, 5]], dtype=np.int64)

    # Polarization (per-atom CT correction)
    e_polar = 0.0
    for i, Z in enumerate(atoms):
        p = GFN2_PARAMS[int(Z)]
        tt = float(np.dot(dipm[:, i], dipm[:, i]))
        tt3 = 0.0
        for k in range(3):
            for l in range(3):
                kl = idx[l, k]
                tt3 += qp[kl, i] * qp[kl, i]
        e_polar += p.dip_kernel * tt + tt3 * p.quad_kernel

    # Pair sum
    e01 = 0.0
    e02 = 0.0
    e11 = 0.0
    for i in range(nat):
        for j in range(i):
            q1 = q[i]
            qj = q[j]
            rij = coords_bohr[j] - coords_bohr[i]
            r2 = float(np.dot(rij, rij))
            ed = 0.0
            eq = 0.0
            edd = 0.0
            for k in range(3):
                ed += qj * dipm[k, i] * rij[k]
                ed -= dipm[k, j] * q1 * rij[k]
                for l in range(3):
                    kl = idx[l, k]
                    tt = rij[l] * rij[k]
                    tt3 = 3.0 * tt
                    eq += qj * qp[kl, i] * tt
                    eq += qp[kl, j] * q1 * tt
                    edd -= dipm[k, j] * dipm[l, i] * tt3
                edd += dipm[k, j] * dipm[k, i] * r2
            e01 += ed * gab3[j, i]
            e02 += eq * gab5[j, i]
            e11 += edd * gab5[j, i]

    return e01 + e02 + e11, e_polar


def setvsdq(
    atoms: list[int],
    coords_bohr: np.ndarray,
    q: np.ndarray,
    dipm: np.ndarray,
    qp: np.ndarray,
    gab3: np.ndarray,
    gab5: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-atom potentials ``vs[nat]``, ``vd[3, nat]``, ``vq[6, nat]``
    in xtb's mmompop layout (aespot.F90:727-848).

    These multiply the AO integrals (S, dpint, qpint respectively) when
    assembling the AES Fock contribution in :func:`fockelectro`.
    """
    nat = coords_bohr.shape[0]
    vs = np.zeros(nat, dtype=np.float64)
    vd = np.zeros((3, nat), dtype=np.float64)
    vq = np.zeros((6, nat), dtype=np.float64)

    idx = np.array([[0, 1, 3],
                    [1, 2, 4],
                    [3, 4, 5]], dtype=np.int64)

    for i in range(nat):
        ra = coords_bohr[i]
        stmp = 0.0
        dtmp = np.zeros(3, dtype=np.float64)
        qtmp = np.zeros(6, dtype=np.float64)
        for j in range(nat):
            g3 = gab3[j, i]
            g5 = gab5[j, i]
            rb = coords_bohr[j]
            dra = ra - rb
            r2a = 0.0
            r2ab = 0.0
            t1a = 0.0
            t2a = 0.0
            t3a = 0.0
            dum5a = 0.0
            for l1 in range(3):
                r2a += ra[l1] * ra[l1]
                r2ab += dra[l1] * dra[l1]
                t1a += ra[l1] * dra[l1]
                t2a += dipm[l1, j] * dra[l1]
                t3a += ra[l1] * dipm[l1, j]
                for l2 in range(3):
                    ll = idx[l1, l2]
                    dum5a -= qp[ll, j] * dra[l1] * dra[l2]
                    dum5a -= 1.5 * q[j] * dra[l1] * dra[l2] * ra[l1] * ra[l2]
                    if l2 >= l1:
                        continue
                    # qtmp off-diag in **qpint layout** (xx=0, yy=1, zz=2,
                    # xy=3, xz=4, yz=5) — matches xtb's ki = l1+l2+1
                    # (Fortran 1-indexed) which is 4/5/6 = xy/xz/yz.
                    if (l1, l2) == (1, 0): ki = 3        # xy
                    elif (l1, l2) == (2, 0): ki = 4      # xz
                    elif (l1, l2) == (2, 1): ki = 5      # yz
                    else: raise RuntimeError("unreachable")
                    qtmp[ki] -= 3.0 * q[j] * g5 * dra[l2] * dra[l1]
                # diagonal qtmp position in qpint layout (l1 → xx, yy, zz)
                qtmp[l1] -= 1.5 * q[j] * g5 * dra[l1] * dra[l1]
            dum3a = -t1a * q[j] - t2a
            dum5a += t3a * r2ab - 3.0 * t1a * t2a + 0.5 * q[j] * r2a * r2ab
            stmp += dum5a * g5 + dum3a * g3
            for l1 in range(3):
                dum3a = dra[l1] * q[j]
                dum5a = (3.0 * dra[l1] * t2a
                         - r2ab * dipm[l1, j]
                         - q[j] * r2ab * ra[l1]
                         + 3.0 * q[j] * dra[l1] * t1a)
                dtmp[l1] += dum3a * g3 + dum5a * g5
                # diagonal qtmp position in qpint layout (xx, yy, zz at 0/1/2)
                qtmp[l1] += 0.5 * r2ab * q[j] * g5
        vs[i] = stmp
        vd[:, i] = dtmp
        vq[:, i] = qtmp

        # CT correction (per-atom)
        Z = atoms[i]
        p = GFN2_PARAMS[int(Z)]
        qs1 = p.dip_kernel * 2.0
        qs2 = p.quad_kernel * 6.0
        t3a = 0.0
        t2a = 0.0
        # CT correction reads qp in mmompop layout (its native layout)
        # but writes to vq in qpint layout. The mmompop `idx[l1, l2]`
        # gives the right slot for qp[].
        for l1 in range(3):
            t3a += ra[l1] * dipm[l1, i] * qs1
            vd[l1, i] -= qs1 * dipm[l1, i]
            for l2 in range(l1):
                ll = idx[l1, l2]      # qp slot in mmompop layout
                # vq slot in qpint layout: xy=3, xz=4, yz=5
                if (l1, l2) == (1, 0): kvq = 3
                elif (l1, l2) == (2, 0): kvq = 4
                elif (l1, l2) == (2, 1): kvq = 5
                else: raise RuntimeError("unreachable")
                vq[kvq, i] -= qp[ll, i] * qs2
                t3a -= ra[l1] * ra[l2] * qp[ll, i] * qs2
                vd[l1, i] += ra[l2] * qp[ll, i] * qs2
                vd[l2, i] += ra[l1] * qp[ll, i] * qs2
            # diagonal: qp at mmompop position (0, 2, 5); vq at qpint (l1)
            ll_diag_mm = (0, 2, 5)[l1]
            vq[l1, i] -= qp[ll_diag_mm, i] * qs2 * 0.5
            t3a -= ra[l1] * ra[l1] * qp[ll_diag_mm, i] * qs2 * 0.5
            vd[l1, i] += ra[l1] * qp[ll_diag_mm, i] * qs2
            t2a += qp[ll_diag_mm, i]
        vs[i] += t3a
        # trace removal applies to vq diagonal in qpint layout (slots 0, 1, 2)
        t2a *= p.quad_kernel
        for l1 in range(3):
            vq[l1, i] += t2a
            vd[l1, i] -= 2.0 * ra[l1] * t2a
            vs[i] += t2a * ra[l1] * ra[l1]
    return vs, vd, vq


def fockelectro(
    P: np.ndarray,
    S: np.ndarray,
    dpint: np.ndarray,
    qpint: np.ndarray,
    aoat: np.ndarray,
    vs: np.ndarray,
    vd: np.ndarray,
    vq: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Build the AES Fock contribution and the AES energy via fockelectro.

    Returns ``(F_aes, E_aes)`` where ``F_aes`` is what to ADD to the
    base monopole F. The energy returned is

        E_aes = ¼ · Σ_{ij} P[j, i] · ( S(vs_i+vs_j) +
                                       Σ_α dpint_α(vd_α_i+vd_α_j) +
                                       Σ_k qpint_k_lin(vq_k_i+vq_k_j) )

    The translation between qpint's (xx, yy, zz, xy, xz, yz) and vq's
    mmompop (xx, xy, yy, xz, yz, zz) is handled inline.
    """
    nao = S.shape[0]
    F = np.zeros_like(S)
    e_aes = 0.0
    # vq is in qpint layout (xx, yy, zz, xy, xz, yz) — same as qpint —
    # so no translation needed.
    for i in range(nao):
        ii = int(aoat[i])
        for j in range(nao):
            jj = int(aoat[j])
            pji = P[j, i]
            fji = S[j, i] * (vs[ii] + vs[jj])
            for k in range(3):
                fji += dpint[k, i, j] * (vd[k, ii] + vd[k, jj])
            for k in range(6):
                fji += qpint[k, i, j] * (vq[k, ii] + vq[k, jj])
            F[j, i] += 0.5 * fji
            e_aes += pji * fji
    e_aes *= 0.25
    return F, float(e_aes)
