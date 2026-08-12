"""Exact two-center Slater overlap by prolate-spheroidal numerical integration.

Reference/oracle for the analytic diatomic-overlap path. Correct for ANY principal
quantum number, so it fixes the n=5 (iodine) breakdown in the vendored PYSEQM port
(which mis-transcribes the hardcoded qn>=5 reduced-overlap coefficients).

Overlap of two normalized real Slater AOs on atoms A (origin) and B (+R z), each with
the same magnetic quantum number m about the bond axis:

    chi_{n,l,m} = N_n r^{n-1} e^{-z r} * Y_{l,m}(theta, phi)

In spheroidal coords (xi in [1, inf), eta in [-1, 1]):
    r_a = (R/2)(xi+eta),  r_b = (R/2)(xi-eta)
    cos th_a = (1+xi*eta)/(xi+eta),  cos th_b = (1-xi*eta)/(xi-eta)
    dV = (R/2)^3 (xi^2-eta^2) dxi deta dphi
The phi integral is nonzero only when m_a == m_b, giving 2*pi (m=0) or pi (m>0).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate
from scipy.special import lpmv


def _radial_norm(n: int, zeta: float) -> float:
    # N such that ∫_0^∞ (N r^{n-1} e^{-z r})^2 r^2 dr = 1
    return (2.0 * zeta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


def _ang_const(l: int, m: int) -> float:
    # |Y_{l,m}| = K * P_l^m(cos th) * {cos(m phi) or 1}; K below
    if m == 0:
        return math.sqrt((2 * l + 1) / (4.0 * math.pi))
    return math.sqrt((2 * l + 1) / (2.0 * math.pi)
                     * math.factorial(l - m) / math.factorial(l + m))


def reduced_overlap(na, la, nb, lb, m, za, zb, R):
    """Local-frame overlap <na la m | nb lb m> for STOs separated by R (bohr), bond axis z.

    m is the shared magnetic quantum number (0=sigma, 1=pi, 2=delta). Returns 0 if the
    Condon-Shortley angular factor vanishes (|m| > min(la, lb)).
    """
    if m > la or m > lb:
        return 0.0
    Na, Nb = _radial_norm(na, za), _radial_norm(nb, zb)
    Ka, Kb = _ang_const(la, m), _ang_const(lb, m)
    phi_int = 2.0 * math.pi if m == 0 else math.pi
    half = 0.5 * R

    def integrand(eta, xi):
        ra = half * (xi + eta)
        rb = half * (xi - eta)
        cta = (1.0 + xi * eta) / (xi + eta)
        ctb = (1.0 - xi * eta) / (xi - eta)
        # clip tiny numerical excursions outside [-1,1]
        cta = min(1.0, max(-1.0, cta))
        ctb = min(1.0, max(-1.0, ctb))
        radial = (Na * ra ** (na - 1) * math.exp(-za * ra)) * (Nb * rb ** (nb - 1) * math.exp(-zb * rb))
        ang = lpmv(m, la, cta) * lpmv(m, lb, ctb)
        return radial * ang * (xi * xi - eta * eta)

    # xi from 1 to a cutoff where e^{-(za+zb)*half*xi} is negligible
    xi_max = 1.0 + 40.0 / max(1e-6, (za + zb) * half)
    val, _ = integrate.dblquad(integrand, 1.0, xi_max, -1.0, 1.0, epsabs=1e-12, epsrel=1e-10)
    return Ka * Kb * phi_int * (half ** 3) * val
