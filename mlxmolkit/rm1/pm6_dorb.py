"""
PM6 d-orbital extension.

Adds d-orbital (l=2) basis functions to the sp NDDO framework.
Basis order: [s, px, py, pz, dz², dxz, dyz, dx²-y², dxy] = 9 per atom.

For PM6, d-orbitals on P, S, Cl, Br contribute through:
1. Overlap integrals (Slater d-orbital overlaps)
2. Resonance: H_μν = 0.5*(β_μ + β_ν)*S_μν where β_d is d-orbital beta
3. Core Hamiltonian diagonal: Udd
4. One-center: minimal (F0SD=G2SD=0 for most elements)

The d-orbital overlap uses a separate zeta_d exponent and
the d-orbital Slater functions.
"""
from __future__ import annotations

import numpy as np
from .params import ElementParams, ANG_TO_BOHR


# d-orbital indices in the 9-basis set
# 0=s, 1=px, 2=py, 3=pz, 4=dz², 5=dxz, 6=dyz, 7=dx²-y², 8=dxy
D_START = 4
N_D = 5  # 5 d-orbitals


def get_pm6_n_basis(p: ElementParams) -> int:
    """Get basis count for PM6: 1 (H/He), 4 (sp), or 9 (spd)."""
    if p.Z <= 2:
        return 1
    if p.has_d and p.zeta_d > 0:
        return 9
    return 4


def get_orbital_type_pm6(basis_idx: int, n_basis: int) -> int:
    """Get orbital type: 0=s, 1=p, 2=d."""
    if basis_idx == 0:
        return 0
    if basis_idx <= 3:
        return 1
    return 2


def get_beta_pm6(p: ElementParams, orbital_type: int) -> float:
    """Get beta for orbital type: 0=s, 1=p, 2=d."""
    if orbital_type == 0:
        return p.beta_s
    elif orbital_type == 1:
        return p.beta_p
    else:
        return p.beta_d


def overlap_d_local_frame(pA: ElementParams, pB: ElementParams, R_bohr: float) -> np.ndarray:
    """Compute overlap matrix including d-orbitals in local frame (bond=x).

    Returns (nA, nB) overlap matrix where nA, nB are 1, 4, or 9.
    Only computes elements involving d-orbitals; sp elements from standard overlap.

    For d-orbital overlaps, uses Slater exponents zeta_d.
    Local frame d-orbital indices: dσ, dπ, dδ (along bond axis).
    """
    from .overlap import overlap_molecular_frame as sp_overlap

    nA = get_pm6_n_basis(pA)
    nB = get_pm6_n_basis(pB)

    # Start with sp overlap (handles 1×1, 4×1, 4×4 blocks)
    sp_nA = min(nA, 4)
    sp_nB = min(nB, 4)
    S_sp = sp_overlap(pA, pB, np.array([0.0, 0.0, 0.0]), np.array([R_bohr * 0.529167, 0.0, 0.0]))

    S = np.zeros((nA, nB))
    S[:sp_nA, :sp_nB] = S_sp

    # d-orbital overlaps (simplified: zero for now — will add Slater d-overlap)
    # In PM6, d-orbital overlap with s and p is small for main-group elements
    # Full implementation needs d-orbital Slater integrals from diat_overlapD.py

    # For a minimal working PM6: use approximate d-d overlap
    if nA == 9 and nB == 9:
        # d-d overlap (same element): approximate with exp(-zeta*R)
        za = pA.zeta_d
        zb = pB.zeta_d
        if za > 0 and zb > 0:
            rho = 0.5 * (za + zb) * R_bohr
            # Simple Slater approximation for d-d sigma overlap
            s_dd = np.exp(-rho) * (1.0 + rho + 0.4 * rho**2 + rho**3/15.0) if rho < 20 else 0.0
            # dσ-dσ overlap (dz² along bond)
            S[4, 4] = s_dd * 0.5
            # dπ overlaps
            S[5, 5] = s_dd * 0.3
            S[6, 6] = s_dd * 0.3
            # dδ overlaps
            S[7, 7] = s_dd * 0.1
            S[8, 8] = s_dd * 0.1

    return S


def build_hcore_d_block(
    pA: ElementParams,
    pB: ElementParams,
    S: np.ndarray,
    nA: int,
    nB: int,
) -> np.ndarray:
    """Build H_core off-diagonal block including d-orbital resonance.

    H_μν = 0.5 * (β_μ + β_ν) * S_μν
    where β is beta_s, beta_p, or beta_d depending on orbital type.
    """
    H = np.zeros((nA, nB))

    for mu in range(nA):
        ot_mu = get_orbital_type_pm6(mu, nA)
        beta_mu = get_beta_pm6(pA, ot_mu)

        for nu in range(nB):
            ot_nu = get_orbital_type_pm6(nu, nB)
            beta_nu = get_beta_pm6(pB, ot_nu)

            H[mu, nu] = 0.5 * (beta_mu + beta_nu) * S[mu, nu]

    return H


def fock_one_center_d(
    F: np.ndarray,
    P: np.ndarray,
    p: ElementParams,
    idx: np.ndarray,
    btype: np.ndarray,
) -> np.ndarray:
    """Add d-orbital one-center contributions to Fock matrix.

    For PM6 main-group elements (F0SD=G2SD=0):
    - d-d diagonal: Udd + P_dd * gdd (approximate)
    - s-d, p-d cross terms: minimal

    Full d-orbital one-center integrals (W tensor, 243 components)
    would be needed for transition metals. Placeholder for now.
    """
    if not p.has_d or p.n_basis < 9:
        return F

    s = idx[0]

    # For main-group (F0SD=G2SD=0): d-orbitals are nearly non-interacting
    # Just add the diagonal Pdd*gss approximation
    Pdd_total = sum(P[idx[4+k], idx[4+k]] for k in range(5))

    # d-d self-interaction (approximate with gss-like term)
    # In PM6, F0SD=G2SD=0 means minimal d-d repulsion on main-group
    # Use a small fraction of gss as placeholder
    gdd_approx = 0.3 * p.gss  # rough approximation

    for k in range(5):
        dk = idx[4 + k]
        F[dk, dk] += P[dk, dk] * gdd_approx * 0.5

    return F
