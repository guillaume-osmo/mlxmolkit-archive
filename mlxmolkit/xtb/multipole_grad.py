# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Cartesian-Gaussian dipole and quadrupole AO integral derivatives.

Companion to :mod:`multipole_integrals`. For the AES band-energy
gradient (xtb's ``dtmp`` / ``qtmp`` pieces in build_dSDQH0_gpu.f90),
we need

    ∂dpint_α[μ, ν] / ∂A_β   and   ∂dpint_α[μ, ν] / ∂B_β
    ∂qpint_k[μ, ν] / ∂A_β   and   ∂qpint_k[μ, ν] / ∂B_β

for k ∈ {xx, yy, zz, xy, xz, yz} (xtb order).

Derivative formula (shift-recurrence, identical structure to
:mod:`overlap_grad`):

    ∂φ_a/∂A_β = -la_β · φ_a(la - 1_β)  +  2α_a · φ_a(la + 1_β)

so

    ∂M(la, lb)/∂A_β = -la_β · M(la - 1_β, lb)  +  2α_a · M(la + 1_β, lb)
    ∂M(la, lb)/∂B_β = -lb_β · M(la, lb - 1_β)  +  2α_b · M(la, lb + 1_β)

for any quantity ``M`` that's a linear functional of φ_a · φ_b
(overlap, dipole, quadrupole — all qualify because the multipole
``r_α`` factor doesn't depend on the centers).

Implementation: build per-axis OS tables augmented by +3 on each side
so the M(la+δ, lb+δ') evaluations fit in-table for δ, δ' ∈ {-1, 0,
+1, +2}. For each pair (μ, ν) primitive contraction sum the per-prim
kernel.

Verifies against central-difference of :mod:`multipole_integrals` to
~1e-9 on random small-molecule layouts.
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction
from .multipole_integrals import _augmented_overlap_axis


def _multipole_primitive_grad(
    alpha_a: float, A: np.ndarray, l_xyz_a: tuple[int, int, int],
    alpha_b: float, B: np.ndarray, l_xyz_b: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-primitive ``(∂D/∂A, ∂D/∂B, ∂Q/∂A, ∂Q/∂B)`` derivatives.

    Each output has the leading axes ``(deriv_axis_β, multipole_axis)``,
    so e.g. ``dD_dA[β, α] = ∂D_α / ∂A_β``.

    Returns:
        ``(dD_dA, dD_dB, dQ_dA, dQ_dB)`` with shapes ``(3, 3)``,
        ``(3, 3)``, ``(3, 6)``, ``(3, 6)`` respectively.
    """
    p = alpha_a + alpha_b
    mu = alpha_a * alpha_b / p
    P = (alpha_a * A + alpha_b * B) / p
    R2 = float(np.sum((A - B) ** 2))
    base = (np.pi / p) ** 1.5 * float(np.exp(-mu * R2))

    la = l_xyz_a
    lb = l_xyz_b
    # Augment by +3 on both bra and ket so shifts (la+δ, lb+δ') for
    # δ, δ' ∈ {-1, 0, +1, +2} are all reachable. la+0 → la+3 covers
    # the diagonal-Q ``+2`` plus the bra-side derivative ``+1`` shift.
    Sx = _augmented_overlap_axis(
        p, P[0], A[0], B[0], la[0] + 3, lb[0] + 3,
    )
    Sy = _augmented_overlap_axis(
        p, P[1], A[1], B[1], la[1] + 3, lb[1] + 3,
    )
    Sz = _augmented_overlap_axis(
        p, P[2], A[2], B[2], la[2] + 3, lb[2] + 3,
    )

    def _M_at(da: tuple[int, int, int], db: tuple[int, int, int]
              ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate (S, D[3], Q[6]) at shifted angular indices.

        ``da, db`` are *deltas* on top of (la, lb). Returns zero if any
        shifted index falls below 0.
        """
        ila = (la[0] + da[0], la[1] + da[1], la[2] + da[2])
        ilb = (lb[0] + db[0], lb[1] + db[1], lb[2] + db[2])
        if any(v < 0 for v in ila) or any(v < 0 for v in ilb):
            return 0.0, np.zeros(3), np.zeros(6)

        sx = float(Sx[ila[0],     ilb[0]])
        sy = float(Sy[ila[1],     ilb[1]])
        sz = float(Sz[ila[2],     ilb[2]])
        S = base * sx * sy * sz

        # bra-shift +1 entries on each axis (raise la by 1)
        sx1 = float(Sx[ila[0] + 1, ilb[0]])
        sy1 = float(Sy[ila[1] + 1, ilb[1]])
        sz1 = float(Sz[ila[2] + 1, ilb[2]])
        Dx = base * (sx1 + A[0] * sx) * sy * sz
        Dy = base * sx * (sy1 + A[1] * sy) * sz
        Dz = base * sx * sy * (sz1 + A[2] * sz)
        D = np.array([Dx, Dy, Dz], dtype=np.float64)

        sx2 = float(Sx[ila[0] + 2, ilb[0]])
        sy2 = float(Sy[ila[1] + 2, ilb[1]])
        sz2 = float(Sz[ila[2] + 2, ilb[2]])
        Qxx = base * (sx2 + 2.0 * A[0] * sx1 + A[0] ** 2 * sx) * sy * sz
        Qyy = base * sx * (sy2 + 2.0 * A[1] * sy1 + A[1] ** 2 * sy) * sz
        Qzz = base * sx * sy * (sz2 + 2.0 * A[2] * sz1 + A[2] ** 2 * sz)
        Qxy = base * (sx1 + A[0] * sx) * (sy1 + A[1] * sy) * sz
        Qxz = base * (sx1 + A[0] * sx) * sy * (sz1 + A[2] * sz)
        Qyz = base * sx * (sy1 + A[1] * sy) * (sz1 + A[2] * sz)
        Q = np.array([Qxx, Qyy, Qzz, Qxy, Qxz, Qyz], dtype=np.float64)
        return S, D, Q

    dD_dA = np.zeros((3, 3), dtype=np.float64)
    dD_dB = np.zeros((3, 3), dtype=np.float64)
    dQ_dA = np.zeros((3, 6), dtype=np.float64)
    dQ_dB = np.zeros((3, 6), dtype=np.float64)

    twoa = 2.0 * alpha_a
    twob = 2.0 * alpha_b
    for beta in range(3):
        # ∂/∂A_β = 2α_a · M(la + 1_β, lb)  −  la_β · M(la − 1_β, lb)
        dap = [0, 0, 0]; dap[beta] = +1
        dam = [0, 0, 0]; dam[beta] = -1
        _, Dp, Qp = _M_at(tuple(dap), (0, 0, 0))
        _, Dm, Qm = _M_at(tuple(dam), (0, 0, 0))
        dD_dA[beta] = twoa * Dp - la[beta] * Dm
        dQ_dA[beta] = twoa * Qp - la[beta] * Qm

        # ∂/∂B_β = 2α_b · M(la, lb + 1_β)  −  lb_β · M(la, lb − 1_β)
        dbp = [0, 0, 0]; dbp[beta] = +1
        dbm = [0, 0, 0]; dbm[beta] = -1
        _, Dpb, Qpb = _M_at((0, 0, 0), tuple(dbp))
        _, Dmb, Qmb = _M_at((0, 0, 0), tuple(dbm))
        dD_dB[beta] = twob * Dpb - lb[beta] * Dmb
        dQ_dB[beta] = twob * Qpb - lb[beta] * Qmb

    return dD_dA, dD_dB, dQ_dA, dQ_dB


def shift_multipole_grad(
    dS_d: np.ndarray, dD_d: np.ndarray, dQ_d: np.ndarray,
    S: np.ndarray, D: np.ndarray, Q: np.ndarray,
    r: np.ndarray, is_deriv_atom: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift the origin of multipole-integral derivatives from 0 to r.

    Verbatim port of xtb's ``shiftintg`` (intgrad.f90:703-728). Given
    frame-origin (origin=0) derivatives, returns derivatives at the
    new origin = r.

    For ``is_deriv_atom=True``, ``r`` is itself the position of the
    atom whose coordinate we're differentiating w.r.t. — additional
    ``δ_αβ`` correction terms fire because ``∂r_α/∂R_β = δ_αβ``. For
    ``is_deriv_atom=False`` (e.g., shifting the bra-side derivative
    by the ket atom's position), those correction terms vanish.

    Args:
        dS_d: ``(3, n, n)`` ∂S/∂R_β, frame origin.
        dD_d: ``(3, 3, n, n)`` ∂dpint_α/∂R_β, frame origin
            (axes ``[β, α, μ, ν]``).
        dQ_d: ``(3, 6, n, n)`` ∂qpint_k/∂R_β, frame origin
            (xtb k-order: xx, yy, zz, xy, xz, yz).
        S: ``(n, n)`` overlap, frame origin.
        D: ``(3, n, n)`` dpint_α, frame origin.
        Q: ``(6, n, n)`` qpint_k, frame origin.
        r: ``(3,)`` shift vector (typically an atom position in Bohr).
        is_deriv_atom: see above.

    Returns:
        ``(shifted_dD, shifted_dQ)`` at origin = r.
    """
    shifted_dD = dD_d.copy()
    shifted_dQ = dQ_d.copy()

    # Dipole shift: shifted_dpint_α = dpint_α - r_α · S
    # ⇒ ∂(...)/∂R_β = ∂dpint_α/∂R_β - r_α · ∂S/∂R_β
    # plus -δ_αβ · S if r is the deriv atom's position.
    for alpha in range(3):
        shifted_dD[:, alpha] -= r[alpha] * dS_d
    if is_deriv_atom:
        for ax in range(3):
            shifted_dD[ax, ax] -= S

    # Quadrupole diagonal shift: shifted_qpint_αα = qpint_αα - 2 r_α · dpint_α + r_α² · S
    # k-slot for diagonal αα is just α (xtb order: 0=xx, 1=yy, 2=zz)
    for alpha in range(3):
        shifted_dQ[:, alpha] -= 2.0 * r[alpha] * dD_d[:, alpha]
        shifted_dQ[:, alpha] += (r[alpha] ** 2) * dS_d
        if is_deriv_atom:
            shifted_dQ[alpha, alpha] -= 2.0 * D[alpha]
            shifted_dQ[alpha, alpha] += 2.0 * r[alpha] * S

    # Quadrupole off-diagonal (xy=3, xz=4, yz=5):
    # shifted_qpint_αβ = qpint_αβ - r_α · dpint_β - r_β · dpint_α + r_α·r_β · S
    for a, b, kk in [(0, 1, 3), (0, 2, 4), (1, 2, 5)]:
        shifted_dQ[:, kk] -= r[a] * dD_d[:, b]
        shifted_dQ[:, kk] -= r[b] * dD_d[:, a]
        shifted_dQ[:, kk] += r[a] * r[b] * dS_d
        if is_deriv_atom:
            shifted_dQ[a, kk] -= D[b]
            shifted_dQ[a, kk] += r[b] * S
            shifted_dQ[b, kk] -= D[a]
            shifted_dQ[b, kk] += r[a] * S

    return shifted_dD, shifted_dQ


def multipole_gradient(
    basis: list[BasisFunction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``∂dpint`` and ``∂qpint`` w.r.t. bra and ket centers.

    Returns:
        ``(dD_dA, dD_dB, dQ_dA, dQ_dB)`` with shapes
        ``(3, 3, n, n)``, ``(3, 3, n, n)``,
        ``(3, 6, n, n)``, ``(3, 6, n, n)`` respectively. Index layout:
        ``dD_dA[deriv_axis_β, multipole_axis_α, μ, ν]``.

    For nuclear gradient on atom ``a``:
        ``∂dpint_α[μ, ν] / ∂R_a = δ_{a, A_μ} · dD_dA[β, α, μ, ν]
                                + δ_{a, A_ν} · dD_dB[β, α, μ, ν]``
    """
    n = len(basis)
    dD_dA = np.zeros((3, 3, n, n), dtype=np.float64)
    dD_dB = np.zeros((3, 3, n, n), dtype=np.float64)
    dQ_dA = np.zeros((3, 6, n, n), dtype=np.float64)
    dQ_dB = np.zeros((3, 6, n, n), dtype=np.float64)
    for mu in range(n):
        bm = basis[mu]
        for nu in range(n):
            if mu == nu:
                continue
            bn = basis[nu]
            for i in range(len(bm.alphas)):
                for j in range(len(bn.alphas)):
                    dDA, dDB, dQA, dQB = _multipole_primitive_grad(
                        bm.alphas[i], bm.center, bm.l_xyz,
                        bn.alphas[j], bn.center, bn.l_xyz,
                    )
                    c = bm.coeffs[i] * bn.coeffs[j]
                    dD_dA[:, :, mu, nu] += c * dDA
                    dD_dB[:, :, mu, nu] += c * dDB
                    dQ_dA[:, :, mu, nu] += c * dQA
                    dQ_dB[:, :, mu, nu] += c * dQB
    return dD_dA, dD_dB, dQ_dA, dQ_dB
