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

    # Per-BF CN-shifted self-energy.
    selfE_eV = np.zeros(n_basis, dtype=np.float64)
    for mu in range(n_basis):
        s_mu = bf_shells[mu]
        A = basis[mu].atom_idx
        # GFN2: selfE_eff = h - kCN · CN  (kCN already in eV/CN).
        selfE_eV[mu] = s_mu.h - s_mu.kcn * cn[A]

    H0 = np.zeros((n_basis, n_basis), dtype=np.float64)
    for mu in range(n_basis):
        H0[mu, mu] = selfE_eV[mu] * _HARTREE_PER_EV

    en_atoms = np.array(
        [GFN2_PARAMS[int(Z)].en for Z in atoms], dtype=np.float64
    )
    r_A_ang = np.array(
        [_ATOMIC_RAD_MANTINA_ANG[int(Z)] for Z in atoms], dtype=np.float64
    )
    r_A_bohr = r_A_ang * _ANG_TO_BOHR

    enscale_ij = 0.005 * (g.enshell + g.enshell)   # = +0.020 for GFN2

    for mu in range(n_basis):
        bm = basis[mu]
        s_mu = bf_shells[mu]
        A = bm.atom_idx
        for nu in range(mu + 1, n_basis):
            bn = basis[nu]
            s_nu = bf_shells[nu]
            B = bn.atom_idx
            if A == B:
                # Same-atom CAO off-diagonal stays zero (xtb's
                # hamiltonian.F90:307+ only sets SAO diagonals on the
                # same-atom pass).
                continue
            l_mu = s_mu.l
            l_nu = s_nu.l
            # h0scal — GFN2 has pairParam = 1 everywhere.
            d_chi = en_atoms[A] - en_atoms[B]
            den2 = d_chi * d_chi
            enpoly = 1.0 + enscale_ij * den2
            K_AB = _shell_kscale(l_mu, l_nu) * enpoly
            # Slater-exponent ratio with wExp = 0.5.
            zi = s_mu.zeta
            zj = s_nu.zeta
            zeta_ij = (2.0 * np.sqrt(zi * zj) / (zi + zj)) ** g.wexp

            R_AB_bohr = float(np.linalg.norm(coords[A] - coords[B]))
            r_sum_bohr = r_A_bohr[A] + r_A_bohr[B]
            sqrt_term = float(np.sqrt(R_AB_bohr / max(r_sum_bohr, 1e-12)))
            pi_A = 1.0 + 0.01 * s_mu.k_poly * sqrt_term
            pi_B = 1.0 + 0.01 * s_nu.k_poly * sqrt_term
            Pi = pi_A * pi_B

            h_avg = 0.5 * (selfE_eV[mu] + selfE_eV[nu]) * _HARTREE_PER_EV
            H0[mu, nu] = K_AB * zeta_ij * h_avg * Pi * float(S[mu, nu])
            H0[nu, mu] = H0[mu, nu]

    return H0, selfE_eV
