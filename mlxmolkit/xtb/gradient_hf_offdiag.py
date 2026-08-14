# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Off-diagonal H0 Hellmann-Feynman gradient for GFN1.

Companion to :mod:`gradient_hf_diag` (the diagonal piece). Together they
cover the full ``Σ_{μν} P[μ, ν] · ∂H0[μ, ν] / ∂R_a`` band-energy
gradient — what xtb calls the "Hellmann-Feynman" contribution.

Off-diagonal H0 in the CAO basis (xtb's scc_core.f90:50-108 + h0scal at
:644-680) is

    H0[μ, ν] = K_AB · h_avg · Π · S[μ, ν]                  (μ on A, ν on B, A ≠ B)

where

    K_AB     = kScale[l_μ, l_ν] · enpoly · pairParam       (geometry-free)
    h_avg    = ½ (selfE_μ + selfE_ν) · Hartree/eV          (CN-dependent)
    selfE_μ  = h_μ - kCN_μ · CN_{A_μ}                       (eV)
    Π        = (1 + α_μ · u)(1 + α_ν · u)                   (geometry-only)
    α_l      = 0.01 · k_poly_l
    u        = sqrt(R_AB / (r_A + r_B))

Chain rule decomposes ∂H0[μ, ν] / ∂r_a into three pieces:

    Term 1 (CN):    K_AB · ∂h_avg/∂r_a · Π · S
    Term 2 (Π):     K_AB · h_avg · ∂Π/∂r_a · S
    Term 3 (S):     K_AB · h_avg · Π · ∂S/∂r_a

with

    ∂h_avg/∂r_a = ½ (Ha/eV) · ( d_selfE_d_cn_μ · ∂CN_A/∂r_a
                                + d_selfE_d_cn_ν · ∂CN_B/∂r_a )
    d_selfE_d_cn_μ = -kCN_μ                                 (eV/CN unit)

    dΠ/dR_AB = (1/(2u·(r_A+r_B))) · ((α_μ + α_ν) + 2·α_μ·α_ν·u)
    ∂Π/∂R_A  = dΠ/dR_AB · (R_A − R_B) / R_AB
    ∂Π/∂R_B  = -∂Π/∂R_A

    ∂S[μ,ν]/∂R_{A_μ} = dSA[ax, μ, ν]                        (only when ax∈{A_μ})
    ∂S[μ,ν]/∂R_{A_ν} = dSB[ax, μ, ν]
    (zero for atoms ≠ A_μ, A_ν — AOs are atom-localized.)

CAO ↔ SAO. The H0_off matrix is built in CAO and then projected via
``T = sao_basis_metadata`` to SAO before the SCF runs:

    H0_off_sao = T · H0_off_cao · T^T

so

    Σ_{μν,SAO} P_sao[μ, ν] · H0_off_sao[μ, ν]
        = Σ_{μν,CAO} P_cao_eff[μ, ν] · H0_off_cao[μ, ν]

where P_cao_eff = T^T · P_sao · T is the back-projected effective
density. Since T is geometry-independent, the gradient only sees
P_cao_eff (and the CAO-side ∂H0_off_cao). The SAO diagonal piece
(``H0_diag_sao = diag(selfE_sao · Ha/eV)``) couples through P_sao
directly and is handled by :mod:`gradient_hf_diag`.

Units. The returned gradient is in **Hartree per Bohr**, matching the
convention used by :mod:`gradient_pulay`. Note that ``dcn_dr`` (the CN
gradient) is in 1/Å — we divide by ``ANG_TO_BOHR`` internally to align
with the per-Bohr basis-derivative pieces.
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction
from .hcore import _ATOMIC_RAD_MANTINA_ANG
from .hcore_gfn1 import (
    _gfn1_pair_param,
    _set_gfn1_kcn,
    _shell_kscale,
)
from .params_gfn1 import GFN1_GLOBALS, GFN1_PARAMS, GFN1Shell


_HARTREE_PER_EV = 1.0 / 27.211386245988
_ANG_TO_BOHR = 1.8897259886


def hf_offdiag_gradient(
    atoms_list: list[int],
    coords_ang: np.ndarray,
    cao_basis: list[BasisFunction],
    cao_bf_shells: list[GFN1Shell],
    P_cao: np.ndarray,
    S_cao: np.ndarray,
    cn: np.ndarray,
    dcn_dr: np.ndarray,
    dSA: np.ndarray,
    dSB: np.ndarray,
) -> np.ndarray:
    """Off-diagonal H0 Hellmann-Feynman gradient for GFN1.

    Args:
        atoms_list: ``(n_atoms,)`` atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom positions.
        cao_basis: CAO basis used by :func:`build_hcore_gfn1`.
        cao_bf_shells: per-CAO-BF shell metadata (h, l, k_poly).
        P_cao: ``(n_cao, n_cao)`` back-projected density
            ``T^T · P_sao · T`` — the effective density that couples to
            the CAO-basis off-diagonal H0.
        S_cao: ``(n_cao, n_cao)`` CAO overlap matrix.
        cn: ``(n_atoms,)`` GFN0-style erf CN.
        dcn_dr: ``(n_atoms, n_atoms, 3)`` ``∂CN_i/∂r_k`` in 1/Å.
        dSA: ``(3, n_cao, n_cao)`` ``∂S[μ, ν]/∂R_{A_μ}`` (1/Bohr).
        dSB: ``(3, n_cao, n_cao)`` ``∂S[μ, ν]/∂R_{A_ν}`` (1/Bohr).

    Returns:
        ``(n_atoms, 3)`` gradient in **Hartree per Bohr**.
    """
    n_atoms = len(atoms_list)
    n_basis = len(cao_basis)
    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords * _ANG_TO_BOHR

    # Per-CAO-BF kCN and CN-shifted self-energy (matches build_hcore_gfn1).
    selfE_eV = np.zeros(n_basis, dtype=np.float64)
    d_selfE_d_cn = np.zeros(n_basis, dtype=np.float64)  # eV per CN unit
    for mu in range(n_basis):
        s_mu = cao_bf_shells[mu]
        A = cao_basis[mu].atom_idx
        Z = atoms_list[A]
        kcn = -s_mu.h * _set_gfn1_kcn(Z, s_mu.l) * 0.01
        selfE_eV[mu] = s_mu.h - kcn * cn[A]
        d_selfE_d_cn[mu] = -kcn  # ∂selfE_μ / ∂CN_A

    g_glob = GFN1_GLOBALS
    en_atoms = np.array(
        [GFN1_PARAMS[int(Z)].en for Z in atoms_list], dtype=np.float64
    )
    r_A_bohr = (
        np.array(
            [_ATOMIC_RAD_MANTINA_ANG[int(Z)] for Z in atoms_list],
            dtype=np.float64,
        )
        * _ANG_TO_BOHR
    )

    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    for mu in range(n_basis):
        bm = cao_basis[mu]
        s_mu = cao_bf_shells[mu]
        A = bm.atom_idx
        val_mu = bm.is_valence
        for nu in range(n_basis):
            if mu == nu:
                continue
            bn = cao_basis[nu]
            B = bn.atom_idx
            if A == B:
                # Same-atom off-diag H0 is zero in the CAO basis (xtb's
                # hamiltonian.F90:307-369). It contributes nothing here.
                continue
            s_nu = cao_bf_shells[nu]
            val_nu = bn.is_valence
            l_mu = s_mu.l
            l_nu = s_nu.l

            # K_AB — entirely geometry-independent.
            if val_mu and val_nu:
                d_chi = en_atoms[A] - en_atoms[B]
                den2 = d_chi * d_chi
                enscale_ij = 0.005 * (g_glob.enshell + g_glob.enshell)
                enpoly = 1.0 + enscale_ij * den2
                pair_p = _gfn1_pair_param(atoms_list[A], atoms_list[B])
                K_AB = _shell_kscale(l_mu, l_nu) * enpoly * pair_p
            elif (not val_mu) and (not val_nu):
                K_AB = g_glob.kdiff
            elif val_mu and not val_nu:
                K_AB = 0.5 * (_shell_kscale(l_mu, l_mu) + g_glob.kdiff)
            else:
                K_AB = 0.5 * (_shell_kscale(l_nu, l_nu) + g_glob.kdiff)

            # Geometry: u, Π, R_AB (all in Bohr units).
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

            # ===== Term 1: K_AB · ∂h_avg/∂r_a · Π · S · P_munu =====
            # ∂h_avg/∂r_a = ½(Ha/eV)·(d_selfE_d_cn[μ]·∂CN_A/∂r_a
            #                          + d_selfE_d_cn[ν]·∂CN_B/∂r_a)
            # Convert dcn_dr (1/Å) → 1/Bohr by dividing by A2B.
            common = K_AB * Pi * S_munu * P_munu  # Ha
            half_ha_per_ev = 0.5 * _HARTREE_PER_EV
            coef_A = common * half_ha_per_ev * d_selfE_d_cn[mu]  # Ha / CN
            coef_B = common * half_ha_per_ev * d_selfE_d_cn[nu]  # Ha / CN
            # Vectorize over atoms a:
            # grad[a, :] += coef_A · dcn_dr[A, a, :] / A2B
            grad += coef_A * dcn_dr[A, :, :] / _ANG_TO_BOHR
            grad += coef_B * dcn_dr[B, :, :] / _ANG_TO_BOHR

            # ===== Term 2: K_AB · h_avg · ∂Π/∂r_a · S · P_munu =====
            du_dR = 1.0 / (2.0 * u * r_sum_bohr)  # 1/Bohr
            dPi_dR = du_dR * ((alpha_mu + alpha_nu) + 2.0 * alpha_mu * alpha_nu * u)
            unit_AB = R_vec_bohr / R_AB  # dimensionless
            # ∂Π/∂R_A = dPi_dR · unit_AB; ∂Π/∂R_B = -dPi_dR · unit_AB
            term2_A = K_AB * h_avg_h * dPi_dR * S_munu * P_munu * unit_AB
            grad[A] += term2_A
            grad[B] -= term2_A

            # ===== Term 3: K_AB · h_avg · Π · ∂S/∂r_a · P_munu =====
            pref3 = K_AB * h_avg_h * Pi * P_munu  # Ha
            grad[A, 0] += pref3 * dSA[0, mu, nu]
            grad[A, 1] += pref3 * dSA[1, mu, nu]
            grad[A, 2] += pref3 * dSA[2, mu, nu]
            grad[B, 0] += pref3 * dSB[0, mu, nu]
            grad[B, 1] += pref3 * dSB[1, mu, nu]
            grad[B, 2] += pref3 * dSB[2, mu, nu]

    return grad
