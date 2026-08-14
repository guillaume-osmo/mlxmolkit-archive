# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Cartesian-Gaussian overlap derivative ``∂S_μν / ∂R_A`` and
``∂S_μν / ∂R_B`` via Obara-Saika.

For a primitive overlap

    s(la, lb) = ∫ (x - A)^la · (x - B)^lb · exp(-α(x-A)²) · exp(-β(x-B)²) dx

the derivative w.r.t. A is obtained by acting d/dA on the integrand:

    d/dA [(x-A)^la · exp(-α(x-A)²)]
    = -la · (x-A)^(la-1) · exp + 2α · (x-A)^(la+1) · exp

So
    ∂s(la, lb) / ∂A = 2α · s(la+1, lb)  −  la · s(la-1, lb)
    ∂s(la, lb) / ∂B = 2β · s(la, lb+1)  −  lb · s(la, lb-1)

The total Cartesian derivative on a per-primitive basis is the sum of
these two (one per axis). For a contracted Gaussian basis function we
sum over primitives with their contraction coefficients.

Returns: ``dS_dA[3, n, n]`` and ``dS_dB[3, n, n]`` shaped tensors —
the gradient of every overlap matrix element with respect to the bra
center A and the ket center B respectively.

For nuclear gradient assembly, ``∂S_μν / ∂R_A_atom = (dS_dA[μ, ν] if
atom_μ == atom else 0) + (dS_dB[μ, ν] if atom_ν == atom else 0)`` —
i.e. sum the contribution if the atom is the center of either μ or ν.
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction


def _augmented_overlap_axis(p: float, P: float, A: float, B: float,
                            la_max: int, lb_max: int) -> np.ndarray:
    """1D OS overlap table ``S[la, lb]`` for la∈[0..la_max], lb∈[0..lb_max].

    Same recurrence as in :mod:`multipole_integrals` — kept here for
    locality and to allow extending la/lb by 1 on either side for the
    derivative without reaching across modules.
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


def _primitive_overlap_grad(
    alpha_a: float, A: np.ndarray, l_xyz_a: tuple[int, int, int],
    alpha_b: float, B: np.ndarray, l_xyz_b: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Primitive ``(∂s/∂A, ∂s/∂B)`` — each a 3-vector.

    Uses the OS recurrence to extend la / lb by 1 on each axis, then
    applies the formulas
        ∂s/∂A_α = 2α · s(la+1, lb, axis_α) · s(other axes)
                  − la_α · s(la-1, lb, axis_α) · s(other axes)
    and analogously for B. The "other axes" overlaps reuse the la, lb
    base table to avoid recomputation.
    """
    p = alpha_a + alpha_b
    mu = alpha_a * alpha_b / p
    P = (alpha_a * A + alpha_b * B) / p
    R2 = float(np.sum((A - B) ** 2))
    base = (np.pi / p) ** 1.5 * float(np.exp(-mu * R2))

    # Build per-axis tables augmented through (la+1, lb+1).
    Sx = _augmented_overlap_axis(
        p, P[0], A[0], B[0], l_xyz_a[0] + 1, l_xyz_b[0] + 1,
    )
    Sy = _augmented_overlap_axis(
        p, P[1], A[1], B[1], l_xyz_a[1] + 1, l_xyz_b[1] + 1,
    )
    Sz = _augmented_overlap_axis(
        p, P[2], A[2], B[2], l_xyz_a[2] + 1, l_xyz_b[2] + 1,
    )

    la = l_xyz_a
    lb = l_xyz_b

    sx = float(Sx[la[0], lb[0]])
    sy = float(Sy[la[1], lb[1]])
    sz = float(Sz[la[2], lb[2]])

    # Per-axis components of ∂s/∂A_α.
    sxp = float(Sx[la[0] + 1, lb[0]])    # raise la_x
    syp = float(Sy[la[1] + 1, lb[1]])
    szp = float(Sz[la[2] + 1, lb[2]])
    sxm = float(Sx[la[0] - 1, lb[0]]) if la[0] >= 1 else 0.0
    sym = float(Sy[la[1] - 1, lb[1]]) if la[1] >= 1 else 0.0
    szm = float(Sz[la[2] - 1, lb[2]]) if la[2] >= 1 else 0.0

    # Per-axis components of ∂s/∂B_α.
    sxbp = float(Sx[la[0], lb[0] + 1])
    sybp = float(Sy[la[1], lb[1] + 1])
    szbp = float(Sz[la[2], lb[2] + 1])
    sxbm = float(Sx[la[0], lb[0] - 1]) if lb[0] >= 1 else 0.0
    sybm = float(Sy[la[1], lb[1] - 1]) if lb[1] >= 1 else 0.0
    szbm = float(Sz[la[2], lb[2] - 1]) if lb[2] >= 1 else 0.0

    # ∂s/∂A_x = base · [2α · sxp − la_x · sxm] · sy · sz
    twoa = 2.0 * alpha_a
    twob = 2.0 * alpha_b
    dS_dA = np.array([
        base * (twoa * sxp - la[0] * sxm) * sy * sz,
        base * sx * (twoa * syp - la[1] * sym) * sz,
        base * sx * sy * (twoa * szp - la[2] * szm),
    ], dtype=np.float64)
    dS_dB = np.array([
        base * (twob * sxbp - lb[0] * sxbm) * sy * sz,
        base * sx * (twob * sybp - lb[1] * sybm) * sz,
        base * sx * sy * (twob * szbp - lb[2] * szbm),
    ], dtype=np.float64)
    return dS_dA, dS_dB


def overlap_gradient(
    basis: list[BasisFunction],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(dS/dA, dS/dB)`` arrays of shape ``(3, n_basis, n_basis)``.

    ``dS_dA[α, μ, ν]`` is ``∂S[μ, ν] / ∂A_α`` where A is the bra
    center. ``dS_dB[α, μ, ν]`` is the analog for the ket center.

    For nuclear gradient on atom ``a`` of element matrix ``M``:

        ``∂Σ M·X/∂R_a = Σ_{μ ∈ atom_μ == a} (dX_dA[α, μ, ν] · M[μ, ν])
                       + Σ_{ν ∈ atom_ν == a} (dX_dB[α, μ, ν] · M[μ, ν])``

    where ``X`` is the overlap matrix (or any quantity built linearly
    on top of S).
    """
    n = len(basis)
    dSA = np.zeros((3, n, n), dtype=np.float64)
    dSB = np.zeros((3, n, n), dtype=np.float64)
    for mu in range(n):
        bm = basis[mu]
        for nu in range(n):
            if mu == nu:
                # Same primitive on same atom: ∂S/∂A + ∂S/∂B = 0
                # (overlap is translation-invariant). Skip — diagonal
                # is unchanged and any same-atom contribution cancels.
                continue
            bn = basis[nu]
            for i in range(len(bm.alphas)):
                for j in range(len(bn.alphas)):
                    dA, dB = _primitive_overlap_grad(
                        bm.alphas[i], bm.center, bm.l_xyz,
                        bn.alphas[j], bn.center, bn.l_xyz,
                    )
                    c = bm.coeffs[i] * bn.coeffs[j]
                    dSA[:, mu, nu] += c * dA
                    dSB[:, mu, nu] += c * dB
    return dSA, dSB
