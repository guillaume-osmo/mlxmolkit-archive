# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN2-xTB core Hamiltonian H0 (no SCF dependence).

Verbatim port of the GFN2 H0 build (xtb/src/scc_core.f90:50-108 +
:644-680, with the slater-exponent ratio factor applied at line 91 +
the explicit ksd / kpd cross-couplings from gfn2.f90:830-833).

Differences from :mod:`hcore_gfn1`:

- ``wExp = 0.5`` (vs 0 in GFN1): the Slater-exponent ratio
  ``ζ_ij = (2·sqrt(ζ_i ζ_j) / (ζ_i + ζ_j))^wExp`` actually multiplies
  ``hav`` (it's a unit-or-less factor, smaller than 1 when the two
  exponents differ).
- ``kScale[s, d] = ksd = 2.0`` and ``kScale[p, d] = kpd = 2.0`` are
  *explicit overrides*, not the default ``½(k_i + k_j)``.
  ``kScale[s, p] = ksp = ½(1.85 + 2.23) = 2.04`` (GFN2 doesn't override
  ksp like GFN1 does — there's no per-pair list).
- ``enscale_ij = 0.005 · (enshell + enshell) = 0.020`` and is
  *positive* (vs ``-0.007`` in GFN1) — sign flips.
- ``pairParam(Z_i, Z_j) = 1.0`` for all pairs (no per-pair tuning, in
  contrast to GFN1).
- CN-shift: ``selfE_eff = h_l − kCN[l, Z] · CN``, where ``kCN`` is
  already stored in *eV per CN unit* and used directly (no extra
  ``× 0.01`` like GFN1's ``kcn = -h · cnshell · 0.01`` reverse-
  engineered form).
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction
from .params_gfn2 import GFN2_GLOBALS, GFN2_PARAMS, GFN2Shell

_HARTREE_PER_EV = 1.0 / 27.211386245988
_ANG_TO_BOHR = 1.8897259886

from .hcore import _ATOMIC_RAD_MANTINA_ANG


def _shell_kscale(l1: int, l2: int) -> float:
    """GFN2 K-shell prefactor with explicit ksd/kpd overrides
    (gfn2.f90:830-833). All other (l_i, l_j) use ``½(k_i + k_j)``.
    """
    g = GFN2_GLOBALS
    if {l1, l2} == {0, 2}:
        return g.ksd
    if {l1, l2} == {1, 2}:
        return g.kpd
    k = [g.ks, g.kp, g.kd, g.kf]
    return 0.5 * (k[l1] + k[l2])


def build_hcore_gfn2(
    atoms: list[int],
    coords_ang: np.ndarray,
    basis: list[BasisFunction],
    S: np.ndarray,
    cn: np.ndarray,
    bf_shells: list[GFN2Shell],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the GFN2 core Hamiltonian and per-BF self-energy vector.

    Args:
        atoms: list of atomic numbers.
        coords_ang: (n_atoms, 3) Angstrom positions.
        basis: AO basis (CAO; the SCF orchestrator handles dtrf2).
        S: (n_basis, n_basis) overlap matrix (CAO).
        cn: (n_atoms,) erf-CN array (same as GFN0/1).
        bf_shells: per-BF GFN2Shell tag.

    Returns:
        ``(H0, selfE_eV)`` — H0 in Hartree, selfE_eV the per-BF
        CN-shifted level energy in eV.
    """
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    cn = np.asarray(cn, dtype=np.float64)
    n_basis = len(basis)
    g = GFN2_GLOBALS

    bf_atom = np.array([bf.atom_idx for bf in basis], dtype=np.int64)
    bf_l = np.array([sh.l for sh in bf_shells], dtype=np.int64)
    bf_h = np.array([sh.h for sh in bf_shells], dtype=np.float64)
    bf_kcn = np.array([sh.kcn for sh in bf_shells], dtype=np.float64)
    bf_zeta = np.array([sh.zeta for sh in bf_shells], dtype=np.float64)
    bf_kpoly = np.array([sh.k_poly for sh in bf_shells], dtype=np.float64)

    # Per-BF CN-shifted self-energy.
    # GFN2: selfE_eff = h - kCN · CN  (kCN already in eV/CN).
    selfE_eV = bf_h - bf_kcn * cn[bf_atom]

    en_atoms = np.array(
        [GFN2_PARAMS[int(Z)].en for Z in atoms], dtype=np.float64
    )
    r_A_ang = np.array(
        [_ATOMIC_RAD_MANTINA_ANG[int(Z)] for Z in atoms], dtype=np.float64
    )
    r_A_bohr = r_A_ang * _ANG_TO_BOHR

    enscale_ij = 0.005 * (g.enshell + g.enshell)   # = +0.020 for GFN2

    A = bf_atom[:, None]
    B = bf_atom[None, :]
    mask = A != B

    k_lut = np.empty((4, 4), dtype=np.float64)
    for l1 in range(4):
        for l2 in range(4):
            k_lut[l1, l2] = _shell_kscale(l1, l2)

    # h0scal — GFN2 has pairParam = 1 everywhere.
    d_chi = en_atoms[A] - en_atoms[B]
    enpoly = 1.0 + enscale_ij * d_chi * d_chi
    K_AB = k_lut[bf_l[:, None], bf_l[None, :]] * enpoly

    # Slater-exponent ratio with wExp = 0.5.
    zi = bf_zeta[:, None]
    zj = bf_zeta[None, :]
    zeta_ij = (2.0 * np.sqrt(zi * zj) / (zi + zj)) ** g.wexp

    bf_coords = coords[bf_atom]
    R_AB_bohr = np.linalg.norm(bf_coords[:, None, :] - bf_coords[None, :, :], axis=2)
    r_sum_bohr = r_A_bohr[A] + r_A_bohr[B]
    sqrt_term = np.sqrt(R_AB_bohr / np.maximum(r_sum_bohr, 1e-12))
    pi_A = 1.0 + 0.01 * bf_kpoly[:, None] * sqrt_term
    pi_B = 1.0 + 0.01 * bf_kpoly[None, :] * sqrt_term
    Pi = pi_A * pi_B

    h_avg = 0.5 * (selfE_eV[:, None] + selfE_eV[None, :]) * _HARTREE_PER_EV
    H0 = np.where(mask, K_AB * zeta_ij * h_avg * Pi * S, 0.0)
    np.fill_diagonal(H0, selfE_eV * _HARTREE_PER_EV)

    return H0, selfE_eV
