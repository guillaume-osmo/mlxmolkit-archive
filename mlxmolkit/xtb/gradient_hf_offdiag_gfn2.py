# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Off-diagonal H0 Hellmann-Feynman gradient — GFN2 variant.

GFN2's H0 differs from GFN1 in three places (cf. :mod:`hcore_gfn2`):

    H0[μ, ν] = K_AB · ζ_ij · h_avg · Π · S[μ, ν]

* ``K_AB`` uses ``ksd`` / ``kpd`` overrides (s-d, p-d) and no
  ``pairParam`` per-pair tuning. ``enscale`` flips sign vs GFN1.
* ``ζ_ij = (2·sqrt(ζ_i·ζ_j) / (ζ_i + ζ_j))^wExp`` with ``wExp = 0.5``
  multiplies ``h_avg`` (geometry-independent — ζ are basis exponents,
  not coordinates).
* ``selfE_eV[μ] = h_μ − kCN_μ · CN_{A_μ}`` with ``kCN`` already in
  eV/CN (no ``× 0.01`` factor; reverse from GFN1).

Gradient chain rule pieces (same three terms as GFN1):

    Term 1 (CN):    K_AB · ζ_ij · ∂h_avg/∂r · Π · S
    Term 2 (Π):     K_AB · ζ_ij · h_avg · ∂Π/∂r · S
    Term 3 (S):     K_AB · ζ_ij · h_avg · Π · ∂S/∂r

Returns ``(n_atoms, 3)`` in **Hartree per Bohr** (matches the rest of
the band-energy gradient assembly in :mod:`gradient_gfn2`).
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction
from .hcore import _ATOMIC_RAD_MANTINA_ANG
from .hcore_gfn2 import _shell_kscale
from .params_gfn2 import GFN2_GLOBALS, GFN2_PARAMS, GFN2Shell


_HARTREE_PER_EV = 1.0 / 27.211386245988
_ANG_TO_BOHR = 1.8897259886


def hf_offdiag_gradient_gfn2(
    atoms_list: list[int],
    coords_ang: np.ndarray,
    cao_basis: list[BasisFunction],
    cao_bf_shells: list[GFN2Shell],
    P_cao: np.ndarray,
    S_cao: np.ndarray,
    cn: np.ndarray,
    dcn_dr: np.ndarray,
    dSA: np.ndarray,
    dSB: np.ndarray,
) -> np.ndarray:
    """GFN2 off-diagonal H0 chain rule on ``K · ζ · h_avg · Π · S``."""
    n_atoms = len(atoms_list)
    n_basis = len(cao_basis)
    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords * _ANG_TO_BOHR

    # Per-CAO-BF self-energy + ∂selfE/∂CN. GFN2's kCN is already eV/CN.
    selfE_eV = np.zeros(n_basis, dtype=np.float64)
    d_selfE_d_cn = np.zeros(n_basis, dtype=np.float64)
    for mu in range(n_basis):
        s_mu = cao_bf_shells[mu]
        A = cao_basis[mu].atom_idx
        selfE_eV[mu] = s_mu.h - s_mu.kcn * cn[A]
        d_selfE_d_cn[mu] = -s_mu.kcn  # ∂selfE_μ / ∂CN_A in eV/CN

    g_glob = GFN2_GLOBALS
    en_atoms = np.array(
        [GFN2_PARAMS[int(Z)].en for Z in atoms_list], dtype=np.float64
    )
    r_A_bohr = (
        np.array(
            [_ATOMIC_RAD_MANTINA_ANG[int(Z)] for Z in atoms_list],
            dtype=np.float64,
        )
        * _ANG_TO_BOHR
    )
    enscale_ij = 0.005 * (g_glob.enshell + g_glob.enshell)

    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    for mu in range(n_basis):
        bm = cao_basis[mu]
        s_mu = cao_bf_shells[mu]
        A = bm.atom_idx
        for nu in range(n_basis):
            if mu == nu:
                continue
            bn = cao_basis[nu]
            B = bn.atom_idx
            if A == B:
                continue
            s_nu = cao_bf_shells[nu]
            l_mu = s_mu.l
            l_nu = s_nu.l

            # K_AB · ζ_ij — both geometry-independent.
            d_chi = en_atoms[A] - en_atoms[B]
            den2 = d_chi * d_chi
            enpoly = 1.0 + enscale_ij * den2
            K_AB = _shell_kscale(l_mu, l_nu) * enpoly
            zi = s_mu.zeta
            zj = s_nu.zeta
            zeta_ij = (2.0 * np.sqrt(zi * zj) / (zi + zj)) ** g_glob.wexp
            K_eff = K_AB * zeta_ij  # the constant prefactor in front of h_avg·Π·S

            # Geometry: u, Π, R_AB.
            R_vec_bohr = coords_bohr[A] - coords_bohr[B]
            R_AB = float(np.linalg.norm(R_vec_bohr))
            r_sum_bohr = float(r_A_bohr[A] + r_A_bohr[B])
            if r_sum_bohr < 1e-12 or R_AB < 1e-12:
                continue
            u = float(np.sqrt(R_AB / r_sum_bohr))
            alpha_mu = 0.01 * s_mu.k_poly
            alpha_nu = 0.01 * s_nu.k_poly
            pi_A = 1.0 + alpha_mu * u
            pi_B = 1.0 + alpha_nu * u
            Pi = pi_A * pi_B

            h_avg_h = 0.5 * (selfE_eV[mu] + selfE_eV[nu]) * _HARTREE_PER_EV
            S_munu = float(S_cao[mu, nu])
            P_munu = float(P_cao[mu, nu])

            # Term 1: K_eff · ∂h_avg/∂r · Π · S · P
            common = K_eff * Pi * S_munu * P_munu
            half_ha_per_ev = 0.5 * _HARTREE_PER_EV
            coef_A = common * half_ha_per_ev * d_selfE_d_cn[mu]
            coef_B = common * half_ha_per_ev * d_selfE_d_cn[nu]
            grad += coef_A * dcn_dr[A, :, :] / _ANG_TO_BOHR
            grad += coef_B * dcn_dr[B, :, :] / _ANG_TO_BOHR

            # Term 2: K_eff · h_avg · ∂Π/∂r · S · P
            du_dR = 1.0 / (2.0 * u * r_sum_bohr)
            dPi_dR = du_dR * (
                (alpha_mu + alpha_nu) + 2.0 * alpha_mu * alpha_nu * u
            )
            unit_AB = R_vec_bohr / R_AB
            term2_A = K_eff * h_avg_h * dPi_dR * S_munu * P_munu * unit_AB
            grad[A] += term2_A
            grad[B] -= term2_A

            # Term 3: K_eff · h_avg · Π · ∂S/∂r · P
            pref3 = K_eff * h_avg_h * Pi * P_munu
            grad[A, 0] += pref3 * dSA[0, mu, nu]
            grad[A, 1] += pref3 * dSA[1, mu, nu]
            grad[A, 2] += pref3 * dSA[2, mu, nu]
            grad[B, 0] += pref3 * dSB[0, mu, nu]
            grad[B, 1] += pref3 * dSB[1, mu, nu]
            grad[B, 2] += pref3 * dSB[2, mu, nu]

    return grad
