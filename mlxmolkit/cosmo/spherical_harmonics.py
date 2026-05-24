"""
Real spherical harmonics Y_l^m for ddCOSMO basis expansion.

The surface charge density σ on each atom's sphere is expanded as:
  σ(θ,φ) = Σ_{l=0}^{lmax} Σ_{m=-l}^{l} σ_{lm} · Y_l^m(θ,φ)

For lmax=6: 49 basis functions per atom (vs 194+ raw Lebedev points).
This reduces the linear system from n_seg × n_seg to n_basis × n_basis
where n_basis = n_atoms × (lmax+1)².

Uses real spherical harmonics (Condon-Shortley convention).
"""
from __future__ import annotations

import numpy as np
from math import factorial


def _associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Compute associated Legendre polynomial P_l^m(x).

    Uses recurrence relation. x = cos(theta).
    """
    m_abs = abs(m)

    if m_abs > l:
        return np.zeros_like(x)

    # Start with P_m^m
    pmm = np.ones_like(x)
    if m_abs > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m_abs + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m_abs:
        return pmm

    # P_{m+1}^m
    pmmp1 = x * (2 * m_abs + 1) * pmm

    if l == m_abs + 1:
        return pmmp1

    # Recurrence for P_l^m
    pll = np.zeros_like(x)
    for ll in range(m_abs + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m_abs - 1) * pmm) / (ll - m_abs)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def real_spherical_harmonics(lmax: int, points: np.ndarray) -> np.ndarray:
    """Compute real spherical harmonics at given points.

    Args:
        lmax: maximum angular momentum (6 → 49 basis functions)
        points: (n_pts, 3) unit vectors on sphere

    Returns:
        Y: (n_basis, n_pts) where n_basis = (lmax+1)²
    """
    n_pts = len(points)
    n_basis = (lmax + 1) ** 2

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    r = np.maximum(r, 1e-30)

    cos_theta = z / r
    phi = np.arctan2(y, x)

    Y = np.zeros((n_basis, n_pts))

    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            m_abs = abs(m)

            # Normalization factor
            norm = np.sqrt((2*l + 1) / (4 * np.pi) *
                          factorial(l - m_abs) / factorial(l + m_abs))

            # Associated Legendre
            P_lm = _associated_legendre(l, m_abs, cos_theta)

            # Real spherical harmonic
            if m > 0:
                Y[idx] = norm * np.sqrt(2) * P_lm * np.cos(m * phi)
            elif m < 0:
                Y[idx] = norm * np.sqrt(2) * P_lm * np.sin(m_abs * phi)
            else:
                Y[idx] = norm * P_lm

            idx += 1

    return Y


def project_to_harmonics(
    values: np.ndarray,
    weights: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Project function values on sphere to spherical harmonic coefficients.

    f_{lm} = ∫ f(θ,φ) · Y_{lm}(θ,φ) dΩ ≈ Σ_i w_i · f_i · Y_{lm,i}

    Args:
        values: (n_pts,) function values at quadrature points
        weights: (n_pts,) quadrature weights
        Y: (n_basis, n_pts) spherical harmonic values

    Returns:
        coeffs: (n_basis,) harmonic expansion coefficients
    """
    return Y @ (values * weights)


def expand_from_harmonics(
    coeffs: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Reconstruct function from spherical harmonic coefficients.

    f(θ_i, φ_i) = Σ_{lm} f_{lm} · Y_{lm}(θ_i, φ_i)

    Args:
        coeffs: (n_basis,) harmonic coefficients
        Y: (n_basis, n_pts) spherical harmonic values

    Returns:
        values: (n_pts,) reconstructed function values
    """
    return coeffs @ Y
