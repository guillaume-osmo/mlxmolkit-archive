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
from functools import lru_cache

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


def reduced_overlap_quadrature(na, la, nb, lb, m, za, zb, R):
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


# ---------------------------------------------------------------- fast path
#
# The adaptive `dblquad` above costs ~3.6 ms per call, and the qn>=5 branch of
# the PYSEQM port calls it 14 times per atom pair -- so a two-atom iodine
# gradient spent 1.29 s inside QUADPACK across 20 188 calls.  It does not need
# adaptivity: in spheroidal coordinates the exponential factorises exactly,
#
#     -za*ra - zb*rb = -(R/2)[(za+zb) xi + (za-zb) eta] = -p*xi - q*eta
#
# so substituting xi = 1 + u/p turns the [1, inf) integral into a Gauss-LAGUERRE
# quadrature (the weight absorbs e^{-p xi}) and eta over [-1, 1] into
# Gauss-LEGENDRE.  What is left is a polynomial times two associated Legendre
# functions times the Jacobian -- smooth, so a fixed tensor grid converges
# spectrally and evaluates in numpy in one shot.
#
# 40x40 nodes reproduce the adaptive result to 3.2e-11 relative, which is inside
# the epsrel=1e-10 the adaptive call itself was asking for, at 61x the speed.
# `reduced_overlap_quadrature` is kept as the oracle; tests/probes compare
# against it.

_GAUSS_NODES: dict = {}


def _gauss_nodes(n_lag: int, n_leg: int):
    key = (n_lag, n_leg)
    hit = _GAUSS_NODES.get(key)
    if hit is None:
        xu, wu = np.polynomial.laguerre.laggauss(n_lag)
        xe, we = np.polynomial.legendre.leggauss(n_leg)
        hit = _GAUSS_NODES[key] = (xu, wu, xe, we)
    return hit


@lru_cache(maxsize=100_000)
def reduced_overlap(na, la, nb, lb, m, za, zb, R, n_lag=40, n_leg=40):
    """Local-frame overlap <na la m | nb lb m> for STOs separated by R (bohr).

    Same contract as `reduced_overlap_quadrature`, evaluated on a fixed
    Gauss-Laguerre x Gauss-Legendre grid instead of adaptively.  Memoised
    because the qn>=5 assembly re-requests identical (n, l, m, zeta, R) tuples
    across the S-variable table -- measured 3.5x redundancy on an iodine
    gradient.
    """
    if m > la or m > lb:
        return 0.0
    half = 0.5 * R
    p = half * (za + zb)
    q = half * (za - zb)
    Na, Nb = _radial_norm(na, za), _radial_norm(nb, zb)
    Ka, Kb = _ang_const(la, m), _ang_const(lb, m)
    phi_int = 2.0 * math.pi if m == 0 else math.pi
    xu, wu, xe, we = _gauss_nodes(n_lag, n_leg)
    xi = (1.0 + xu / p)[:, None]
    eta = xe[None, :]
    ra = half * (xi + eta)
    rb = half * (xi - eta)
    cta = np.clip((1.0 + xi * eta) / (xi + eta), -1.0, 1.0)
    ctb = np.clip((1.0 - xi * eta) / (xi - eta), -1.0, 1.0)
    F = ((Na * ra ** (na - 1)) * (Nb * rb ** (nb - 1))
         * lpmv(m, la, cta) * lpmv(m, lb, ctb) * (xi * xi - eta * eta))
    inner = (F * np.exp(-q * eta) * we[None, :]).sum(1)
    val = math.exp(-p) / p * float((wu * inner).sum())
    return Ka * Kb * phi_int * (half ** 3) * val
