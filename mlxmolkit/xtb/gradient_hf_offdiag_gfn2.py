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
    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords * _ANG_TO_BOHR

    bf_atom = np.array([bf.atom_idx for bf in cao_basis], dtype=np.int64)
    bf_l = np.array([sh.l for sh in cao_bf_shells], dtype=np.int64)
    bf_h = np.array([sh.h for sh in cao_bf_shells], dtype=np.float64)
    bf_kcn = np.array([sh.kcn for sh in cao_bf_shells], dtype=np.float64)
    bf_zeta = np.array([sh.zeta for sh in cao_bf_shells], dtype=np.float64)
    bf_kpoly = np.array([sh.k_poly for sh in cao_bf_shells], dtype=np.float64)

    # Per-CAO-BF self-energy + ∂selfE/∂CN. GFN2's kCN is already eV/CN.
    selfE_eV = bf_h - bf_kcn * cn[bf_atom]
    d_selfE_d_cn = -bf_kcn  # ∂selfE_μ / ∂CN_A in eV/CN

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

    A = bf_atom[:, None]
    B = bf_atom[None, :]
    mask = A != B

    k_lut = np.empty((4, 4), dtype=np.float64)
    for l1 in range(4):
        for l2 in range(4):
            k_lut[l1, l2] = _shell_kscale(l1, l2)

    # K_AB · ζ_ij — both geometry-independent except for atom labels.
    d_chi = en_atoms[A] - en_atoms[B]
    enpoly = 1.0 + enscale_ij * d_chi * d_chi
    zi = bf_zeta[:, None]
    zj = bf_zeta[None, :]
    zeta_ij = (2.0 * np.sqrt(zi * zj) / (zi + zj)) ** g_glob.wexp
    K_eff = k_lut[bf_l[:, None], bf_l[None, :]] * enpoly * zeta_ij

    # Geometry: u, Π, R_AB.
    bf_coords = coords_bohr[bf_atom]
    R_vec = bf_coords[:, None, :] - bf_coords[None, :, :]
    R = np.linalg.norm(R_vec, axis=2)
    r_sum = r_A_bohr[A] + r_A_bohr[B]
    safe = mask & (R > 1e-12) & (r_sum > 1e-12)
    R_safe = np.where(safe, R, 1.0)
    r_sum_safe = np.where(safe, r_sum, 1.0)
    u = np.sqrt(R_safe / r_sum_safe)
    alpha_mu = 0.01 * bf_kpoly[:, None]
    alpha_nu = 0.01 * bf_kpoly[None, :]
    Pi = (1.0 + alpha_mu * u) * (1.0 + alpha_nu * u)

    h_avg_h = 0.5 * (selfE_eV[:, None] + selfE_eV[None, :]) * _HARTREE_PER_EV
    pair_sp = np.where(safe, S_cao * P_cao, 0.0)

    # Term 1: K_eff · ∂h_avg/∂r · Π · S · P
    common = K_eff * Pi * pair_sp
    half_ha_per_ev = 0.5 * _HARTREE_PER_EV
    cn_coef = np.bincount(
        bf_atom,
        weights=np.sum(common, axis=1) * half_ha_per_ev * d_selfE_d_cn,
        minlength=n_atoms,
    )
    cn_coef += np.bincount(
        bf_atom,
        weights=np.sum(common, axis=0) * half_ha_per_ev * d_selfE_d_cn,
        minlength=n_atoms,
    )

    # Term 2: K_eff · h_avg · ∂Π/∂r · S · P
    dPi_dR = (
        ((alpha_mu + alpha_nu) + 2.0 * alpha_mu * alpha_nu * u)
        / (2.0 * u * r_sum_safe)
    )
    unit = R_vec / R_safe[:, :, None]
    term2 = (K_eff * h_avg_h * dPi_dR * pair_sp)[:, :, None] * unit
    row_term2 = np.sum(term2, axis=1)
    col_term2 = np.sum(term2, axis=0)
    for ax in range(3):
        grad[:, ax] += np.bincount(
            bf_atom,
            weights=row_term2[:, ax] - col_term2[:, ax],
            minlength=n_atoms,
        )

    # Term 3: K_eff · h_avg · Π · ∂S/∂r · P
    pref3 = np.where(safe, K_eff * h_avg_h * Pi * P_cao, 0.0)
    for ax in range(3):
        row = np.sum(pref3 * dSA[ax], axis=1)
        col = np.sum(pref3 * dSB[ax], axis=0)
        grad[:, ax] += np.bincount(bf_atom, weights=row, minlength=n_atoms)
        grad[:, ax] += np.bincount(bf_atom, weights=col, minlength=n_atoms)

    grad += np.tensordot(cn_coef, dcn_dr, axes=(0, 0)) / _ANG_TO_BOHR
    return grad
