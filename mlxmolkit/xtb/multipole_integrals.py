# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Cartesian Gaussian dipole and quadrupole AO integrals.

These are the extra integrals GFN2 needs (alongside the overlap S)
for the AES (anisotropic electrostatics) machinery: ``dpint`` and
``qpint`` in xtb's terminology (xtb/src/aespot.F90 + intgrad.f90).

Reference convention (matching xtb):
    dpint_α[μ, ν] = ∫ φ_μ(r) · r_α · φ_ν(r) dr,   α ∈ {x, y, z}
    qpint_k[μ, ν] = ∫ φ_μ(r) · (r_α r_β) · φ_ν(r) dr
where the 6 quadrupole components are stored in xtb's order
``(xx, yy, zz, xy, xz, yz)`` (see aespot.F90:30-32).

Origin: r is measured from the *Cartesian frame origin* (typically the
COM or just the input frame origin) — matching xtb's mmompop:
``dipm_omp(k,jj) = dipm_omp(k,jj) + xyz(k,jj)*ps - pdmk`` where the
``xyz(k, jj)*ps`` term is the atom-position shift and ``pdmk`` is the
dipole-int contribution.

Implementation: Obara-Saika augmented overlap. For the x-axis,

    Dx(la, lb) = S1(la_x+1, lb_x) * Sy * Sz   +   A_x * S(la, lb)

(equivalently shifting on b: Dx = S(la_x, lb_x+1) + B_x · S; we use
the on-bra form for symmetry with overlap_matrix's loop convention).
S1 is the same OS table as overlap, computed up through la_max+2.

Each component-pair (μ, ν) primitive contraction is summed over all
primitives of μ and ν.

We do not apply the CAO→SAO transform here; the SCF orchestrator does
that (the transform applies linearly to dpint and qpint just like S,
since they are linear functionals of the basis).
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction


def _augmented_overlap_axis(p: float, P: float, A: float, B: float,
                            la_max: int, lb_max: int) -> np.ndarray:
    """Per-axis 1D Obara-Saika overlap table ``S[la, lb]`` for
    ``la ∈ [0, la_max]``, ``lb ∈ [0, lb_max]``.

    Returns the *unnormalized* axis integrals
    ``∫ (x-A)^la · (x-B)^lb · exp(-p (x-P)²) dx`` divided by the
    Gaussian prefactor (so S[0, 0] = 1). Caller multiplies by
    ``(π/p)^(3/2) · exp(-μ R²)`` over all three axes.
    """
    PA = P - A
    PB = P - B
    inv2p = 1.0 / (2.0 * p)
    n = la_max + 1
    m = lb_max + 1
    S = np.zeros((n, m), dtype=np.float64)
    S[0, 0] = 1.0
    for i in range(1, n):
        S[i, 0] = PA * S[i - 1, 0]
        if i >= 2:
            S[i, 0] += inv2p * (i - 1) * S[i - 2, 0]
    for j in range(1, m):
        for i in range(n):
            term = PB * S[i, j - 1]
            if j >= 2:
                term += inv2p * (j - 1) * S[i, j - 2]
            if i >= 1:
                term += inv2p * i * S[i - 1, j - 1]
            S[i, j] = term
    return S


def _multipole_primitive(
    alpha_a: float, A: np.ndarray, l_xyz_a: tuple[int, int, int],
    alpha_b: float, B: np.ndarray, l_xyz_b: tuple[int, int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Primitive overlap S, dipole D[3], quadrupole Q[6] in xtb order
    ``(xx, yy, zz, xy, xz, yz)``.

    Origin for the multipoles is the Cartesian frame origin (0, 0, 0).
    Returns the *unnormalized* integrals (same convention as
    :func:`mlxmolkit.xtb.basis._primitive_overlap`).
    """
    p = alpha_a + alpha_b
    mu = alpha_a * alpha_b / p
    P = (alpha_a * A + alpha_b * B) / p
    R2 = float(np.sum((A - B) ** 2))
    base = (np.pi / p) ** 1.5 * float(np.exp(-mu * R2))

    # Per-axis OS tables augmented through (la+2, lb+2) so dipole and
    # diagonal-quadrupole on either axis is reachable.
    Sx = _augmented_overlap_axis(p, P[0], A[0], B[0], l_xyz_a[0] + 2, l_xyz_b[0])
    Sy = _augmented_overlap_axis(p, P[1], A[1], B[1], l_xyz_a[1] + 2, l_xyz_b[1])
    Sz = _augmented_overlap_axis(p, P[2], A[2], B[2], l_xyz_a[2] + 2, l_xyz_b[2])

    la = l_xyz_a
    lb = l_xyz_b
    sx = float(Sx[la[0],     lb[0]])
    sy = float(Sy[la[1],     lb[1]])
    sz = float(Sz[la[2],     lb[2]])

    # Overlap S = base · sx · sy · sz
    S = base * sx * sy * sz

    # Dipole on bra-side: D_α = (S[la_α + 1] + A_α · S[la_α]) · S(other axes)
    sx1 = float(Sx[la[0] + 1, lb[0]])
    sy1 = float(Sy[la[1] + 1, lb[1]])
    sz1 = float(Sz[la[2] + 1, lb[2]])
    Dx = base * (sx1 + A[0] * sx) * sy * sz
    Dy = base * sx * (sy1 + A[1] * sy) * sz
    Dz = base * sx * sy * (sz1 + A[2] * sz)
    D = np.array([Dx, Dy, Dz], dtype=np.float64)

    # Quadrupole: r_α r_β = (P-A+A)_α · (P-A+A)_β.
    # On bra side:
    #   r_α r_β = (x-A+A_α)(x-A+A_β)
    # so its integral is
    #   Q_αβ = S[la_α+δαβ_α + δαβ_β, lb] + ... cross terms with A_α, A_β.
    # We compute it explicitly for diagonal αα and off-diagonal αβ.
    sx2 = float(Sx[la[0] + 2, lb[0]])
    sy2 = float(Sy[la[1] + 2, lb[1]])
    sz2 = float(Sz[la[2] + 2, lb[2]])

    # Q_xx = ∫ x² φ_a φ_b dr = ∫ ((x-A)+A_x)² φ_a φ_b dr
    #      = S[la_x+2] + 2·A_x · S[la_x+1] + A_x² · S[la_x]   (per x-axis), all × Sy · Sz
    Qxx = base * (sx2 + 2.0 * A[0] * sx1 + A[0] ** 2 * sx) * sy * sz
    Qyy = base * sx * (sy2 + 2.0 * A[1] * sy1 + A[1] ** 2 * sy) * sz
    Qzz = base * sx * sy * (sz2 + 2.0 * A[2] * sz1 + A[2] ** 2 * sz)

    # Q_αβ for α ≠ β: separable per axis.
    # Q_xy = (Sx[la_x+1] + A_x·Sx[la_x]) · (Sy[la_y+1] + A_y·Sy[la_y]) · Sz[la_z]
    Qxy = base * (sx1 + A[0] * sx) * (sy1 + A[1] * sy) * sz
    Qxz = base * (sx1 + A[0] * sx) * sy * (sz1 + A[2] * sz)
    Qyz = base * sx * (sy1 + A[1] * sy) * (sz1 + A[2] * sz)

    Q = np.array([Qxx, Qyy, Qzz, Qxy, Qxz, Qyz], dtype=np.float64)
    return S, D, Q


def multipole_matrices(
    basis: list[BasisFunction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute S, dpint(3, n, n), qpint(6, n, n) in the CAO basis.

    The xtb convention for ``qpint`` axis order is
    ``(xx, yy, zz, xy, xz, yz)``.

    These are the *raw* AO multipole integrals — the SCF orchestrator
    must apply the CAO→SAO transform afterward (T · M · T^T for each
    of the 9 + 1 matrices).

    Returns:
        ``(S, dpint, qpint)`` — S has shape (n, n); dpint (3, n, n);
        qpint (6, n, n).
    """
    n = len(basis)
    S = np.zeros((n, n), dtype=np.float64)
    dpint = np.zeros((3, n, n), dtype=np.float64)
    qpint = np.zeros((6, n, n), dtype=np.float64)
    for mu in range(n):
        bm = basis[mu]
        for nu in range(mu, n):
            bn = basis[nu]
            S_mn = 0.0
            D_mn = np.zeros(3, dtype=np.float64)
            Q_mn = np.zeros(6, dtype=np.float64)
            for i in range(len(bm.alphas)):
                for j in range(len(bn.alphas)):
                    s, d, q = _multipole_primitive(
                        bm.alphas[i], bm.center, bm.l_xyz,
                        bn.alphas[j], bn.center, bn.l_xyz,
                    )
                    c = bm.coeffs[i] * bn.coeffs[j]
                    S_mn += c * s
                    D_mn += c * d
                    Q_mn += c * q
            S[mu, nu] = S_mn
            S[nu, mu] = S_mn
            dpint[:, mu, nu] = D_mn
            dpint[:, nu, mu] = D_mn
            qpint[:, mu, nu] = Q_mn
            qpint[:, nu, mu] = Q_mn
    return S, dpint, qpint
