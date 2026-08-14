# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN1-xTB core Hamiltonian H0 (no SCF dependence).

Verbatim port of the GFN1 H0 build (xtb/src/scc_core.f90:50-108 +
:644-680). The off-diagonal scaling differs from GFN0's peeq driver:

    H_μν = ½ K_AB · (selfE_μ + selfE_ν) · Π · S_μν

with K_AB from h0scal:
    val-val:   km = kScale[il, jl] · enpoly · pairParam
    aux-aux:   km = kDiff                    (NOT zero, unlike GFN0!)
    val-aux:   km = ½(kScale[l_val, l_val] + kDiff)   (uses the VALENCE l)

GFN1 specifics:
- ``kScale[s,p]`` and ``kScale[p,s]`` are overridden by the global
  ``ksp = 2.08`` (gfn1.f90:53). Other entries: 0.5·(kshell[i]+kshell[j]).
- ``enscale = 0.005 · (enshell_i + enshell_j)`` with ``enshell = -0.7``
  (scalar) and ``enscale4 = 0``, so ``enpoly = 1 + enscale·den²``.
- ``wExp = 0`` (gfn1.f90:658) — no Slater-exponent ratio factor.
- The ``selfEnergy`` for GFN1's diagonal does include CN-shift (via
  ``setGFN1kCN``) but NOT a charge correction (q-shift comes from
  the SCF Coulomb potential, not directly from a kQ multiplier).

Phase B0 scope (no SCF yet): just the bare H0 with CN-shift on the
diagonal. The SCF wrapper will then add the Coulomb potential V(P)
on top.
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction
from .params_gfn1 import GFN1_GLOBALS, GFN1_PARAMS, GFN1Shell

_HARTREE_PER_EV = 1.0 / 27.211386245988
_ANG_TO_BOHR = 1.8897259886

# Atomic radii for the shellPoly Π factor — same Mantina/Truhlar table
# as GFN0; the GFN1 shellPoly k_poly values differ but the radii table
# is the canonical xtb atomicrad.f90 (used for all xTB methods).
from .hcore import _ATOMIC_RAD_MANTINA_ANG


def _shell_kscale(l1: int, l2: int) -> float:
    """GFN1's K-shell prefactor with the explicit s-p override (ksp).

    ``kScale[i, j] = ½(k_shell[i] + k_shell[j])`` for all (i, j) except
    the s-p pair, which is overridden to ``ksp`` (gfn1.f90:53,
    read_gfn_param.f90:189-192).
    """
    g = GFN1_GLOBALS
    if (l1, l2) == (0, 1) or (l1, l2) == (1, 0):
        return g.ksp
    k = [g.ks, g.kp, g.kd, g.kf]
    return 0.5 * (k[l1] + k[l2])


# Per-element-pair scaling on the val-val H0 off-diagonal (gfn1.f90:680-696
# + setGFN1PairParam at gfn1.f90:774). Default = 1.0 for non-listed pairs.
# Transition-metal block: kp = [1.1, 1.2, 1.2] for rows (Sc-Cu, Y-Ag, La-Au).
_GFN1_PAIRPARAM_OVERRIDES: dict[tuple[int, int], float] = {
    (1, 1):   0.96,
    (5, 1):   0.95, (1, 5):   0.95,
    (7, 1):   1.04, (1, 7):   1.04,
    (28, 1):  0.90, (1, 28):  0.90,
    (75, 1):  0.80, (1, 75):  0.80,
    (78, 1):  0.80, (1, 78):  0.80,
    (15, 5):  0.97, (5, 15):  0.97,
    (14, 7):  1.01, (7, 14):  1.01,
}


def _dblock_row(Z: int) -> int:
    """Map Z to xtb's transition-metal row index (1-3) or 0 if not in d-block."""
    if 21 <= Z <= 29:
        return 1
    if 39 <= Z <= 47:
        return 2
    if 57 <= Z <= 79:
        return 3
    return 0


def _gfn1_pair_param(Zi: int, Zj: int) -> float:
    """``pairParam(izp, jzp)`` per gfn1.f90:680-696 + setGFN1PairParam."""
    key = (int(Zi), int(Zj))
    if key in _GFN1_PAIRPARAM_OVERRIDES:
        return _GFN1_PAIRPARAM_OVERRIDES[key]
    iTr = _dblock_row(int(Zi))
    jTr = _dblock_row(int(Zj))
    if iTr > 0 and jTr > 0:
        kp = (1.1, 1.2, 1.2)
        return 0.5 * (kp[iTr - 1] + kp[jTr - 1])
    return 1.0


def _set_gfn1_kcn(Z: int, l: int) -> float:
    """GFN1's per-shell CN-shift coefficient. From setGFN1kCN
    (gfn1.f90:765+) — for GFN1 it depends on selfEnergy and a
    cnshell(2,0:3) global table:

        kCN(ish, izp) = - selfEnergy(ish, izp) · cnshell(kind, l)

    where kind=1 for s/p of light atoms, kind=2 for transition-metal
    d. We simplify via gfn1Kinds(Z) ∈ {0, 1, 2} (gfn1.f90:69+); for
    main-group sp valence kind=1.

    Returns kCN in eV per CN unit (signed).
    """
    g = GFN1_GLOBALS
    # cnshell layout: row 0 (kind=1) for sp; row 1 (kind=2) for d-block.
    # gfn1.f90:43-45: cnshell(2, 0:3) = reshape(
    #   [0.6,0.6, -0.3,-0.3, -0.5,0.5, -0.5,0.5], shape(cnshell))
    # i.e. cnshell[1, l=0]=0.6, cnshell[1, l=1]=-0.3, cnshell[1, l=2]=-0.5
    #      cnshell[2, l=0]=0.6, cnshell[2, l=1]=-0.3, cnshell[2, l=2]= 0.5
    cnshell = (
        (0.6, -0.3, -0.5, -0.5),
        (0.6, -0.3,  0.5,  0.5),
    )
    kind = _gfn1_kind(Z)
    if kind == 0:
        return 0.0
    # Pick row by kind (1 → row 0; 2 → row 1).
    return cnshell[kind - 1][l]


_GFN1_KINDS = (
    0,  # sentinel
    1, 1,                                                        # H He
    0, 0,                               0, 1, 1, 1, 1, 1,        # Li-Ne
    0, 0,                               0, 1, 1, 1, 1, 1,        # Na-Ar
    0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,        # K-Kr
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,        # Rb-Xe
    0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
)
assert len(_GFN1_KINDS) == 87


def _gfn1_kind(Z: int) -> int:
    return _GFN1_KINDS[Z]


def build_hcore_gfn1(
    atoms: list[int],
    coords_ang: np.ndarray,
    basis: list[BasisFunction],
    S: np.ndarray,
    cn: np.ndarray,
    bf_shells: list[GFN1Shell],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the GFN1 core Hamiltonian and per-BF self-energy vector.

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` Angstrom positions.
        basis: AO basis from :func:`mlxmolkit.xtb.basis.build_basis`.
        S: ``(n_basis, n_basis)`` overlap matrix.
        cn: ``(n_atoms,)`` GFN0-style erf-CN array (same as GFN0).
        bf_shells: per-BF GFN1Shell tag (same length as ``basis``).

    Returns:
        ``(H0, selfE_eV)`` — H0 in Hartree, selfE_eV the per-BF
        CN-shifted level energy (used later by the SCF Coulomb step).
    """
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    cn = np.asarray(cn, dtype=np.float64)
    n_basis = len(basis)

    # Per-BF CN-shifted self-energy.
    selfE_eV = np.zeros(n_basis, dtype=np.float64)
    for mu in range(n_basis):
        s_mu = bf_shells[mu]
        A = basis[mu].atom_idx
        Z = atoms[A]
        # gfn1.f90:765+: kCN = -h · cnshell[kind, l] · 0.01.
        kcn = -s_mu.h * _set_gfn1_kcn(Z, s_mu.l) * 0.01
        selfE_eV[mu] = s_mu.h - kcn * cn[A]

    H0 = np.zeros((n_basis, n_basis), dtype=np.float64)
    for mu in range(n_basis):
        H0[mu, mu] = selfE_eV[mu] * _HARTREE_PER_EV

    g = GFN1_GLOBALS
    en_atoms = np.array([GFN1_PARAMS[int(Z)].en for Z in atoms], dtype=np.float64)

    r_A_ang = np.array(
        [_ATOMIC_RAD_MANTINA_ANG[int(Z)] for Z in atoms], dtype=np.float64
    )
    r_A_bohr = r_A_ang * _ANG_TO_BOHR

    for mu in range(n_basis):
        bm = basis[mu]
        s_mu = bf_shells[mu]
        A = bm.atom_idx
        val_mu = bm.is_valence
        for nu in range(mu + 1, n_basis):
            bn = basis[nu]
            s_nu = bf_shells[nu]
            B = bn.atom_idx
            val_nu = bn.is_valence
            if A == B:
                # xtb's hamiltonian.F90:307-369 only sets the SAO
                # diagonal on same-atom shells (selfE per shell), and
                # accumulates dipole/quadrupole integrals — NOT H0
                # off-diagonals. Same-atom off-diag H0 is zero in the
                # CAO basis; the d-shell diagonal-energy correction
                # is handled afterwards by the SAO-side diagonal patch
                # (see scf_gfn1.gfn1_energy).
                continue
            l_mu = s_mu.l
            l_nu = s_nu.l
            # h0scal — scc_core.f90:644-680.
            if val_mu and val_nu:
                d_chi = en_atoms[A] - en_atoms[B]
                den2 = d_chi * d_chi
                # GFN1: enscale[i,j] = 0.005·(enshell+enshell), enscale4=0.
                enscale_ij = 0.005 * (g.enshell + g.enshell)
                enpoly = 1.0 + enscale_ij * den2
                pair_p = _gfn1_pair_param(atoms[A], atoms[B])
                K_AB = _shell_kscale(l_mu, l_nu) * enpoly * pair_p
            elif (not val_mu) and (not val_nu):
                K_AB = g.kdiff
            elif val_mu and not val_nu:
                # val on µ, aux on ν → use µ's (valence) l.
                K_AB = 0.5 * (_shell_kscale(l_mu, l_mu) + g.kdiff)
            else:
                # aux on µ, val on ν → use ν's (valence) l.
                K_AB = 0.5 * (_shell_kscale(l_nu, l_nu) + g.kdiff)
            # GFN1 wExp = 0, so no slater-exponent ratio factor.

            R_AB_bohr = float(np.linalg.norm(coords[A] - coords[B]))
            r_sum_bohr = r_A_bohr[A] + r_A_bohr[B]
            sqrt_term = float(np.sqrt(R_AB_bohr / max(r_sum_bohr, 1e-12)))
            pi_A = 1.0 + 0.01 * s_mu.k_poly * sqrt_term
            pi_B = 1.0 + 0.01 * s_nu.k_poly * sqrt_term
            Pi = pi_A * pi_B

            h_avg = 0.5 * (selfE_eV[mu] + selfE_eV[nu]) * _HARTREE_PER_EV
            H0[mu, nu] = K_AB * h_avg * Pi * float(S[mu, nu])
            H0[nu, mu] = H0[mu, nu]

    return H0, selfE_eV
