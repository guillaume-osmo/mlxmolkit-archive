# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Experimental g-xTB H0 builder backed by recovered binary parameters."""

from __future__ import annotations

import numpy as np

from .gxtb_basis import ANG_TO_BOHR, GXTBQVSZPBasis
from .params_gxtb import GXTB_PARAMS
from .qvszp_params import QVSZP_PARAMS


GXTB_D_H0_SCALE_DAMP = 0.1


# --- Carbon-p environment self-energy correction --------------------------------
#
# The recovered CN-only H0 self-energy systematically UNDER-polarizes carbon toward
# bonded OXYGEN.  Across a 64-molecule --gxtb oracle set the per-carbon Mulliken
# error (port - oracle) is essentially  err ~= -0.044 - 0.094 * (#bonded O):
#   corr(err, n_O) = -0.87   (dominant driver: carbonyls/esters/CO2 most affected)
#   corr(err, n_N) = -0.09   (weak)
#   corr(err, n_F) = +0.03   (none -- fluorine does NOT drive it)
# The flat carbon-p shift only reaches MAE ~0.0205 and leaves carbonyls/CO2 off; the
# real g-xTB level alignment is set downstream (q-vSZP basis + SCC electrostatics).
# We restore the missing oxygen-driven charge transfer directly on the carbon p
# on-site level:
#     d_eps_p(C) = A + B * sum_{j bonded} w[Zj] * (rcov_i + rcov_j) / r_ij
# with EMPIRICAL per-element weights w[O]=1.0, w[N]=0.25, w[F]=0.0.  (An earlier
# Pauling-electronegativity weighting put the LARGEST weight on F and catastrophically
# over-polarized CF2/CF3 carbons by +0.37 e on a held-out set; the oracle data show F
# has no effect, so it is excluded.)  The (rcov_i+rcov_j)/r_ij ratio makes the term a
# smooth, bond-order-sensitive function of geometry.
# Tuned on the combined set (A=0.040, B=0.015 Ha): broad-set MAE 0.0444 -> 0.0167,
# C-MAE 0.067 -> 0.021, carbonyls/CO2/esters fixed (CO2 0.27 -> ~0), fluoro carbons
# unharmed (CHF3 0.18 -> 0.02), non-carbon (water/NH3/HF) unchanged (carbon-p only).
# The minimum is flat over A in [0.035, 0.045], B in [0.010, 0.020].
GXTB_C_PLEVEL_A = 0.040
GXTB_C_PLEVEL_B = 0.015
GXTB_C_PLEVEL_BOND_FACTOR = 1.30
# Bonded-neighbour weights (by atomic number) that drive carbon charge transfer:
# oxygen dominant, nitrogen weak, fluorine excluded.
GXTB_C_PLEVEL_NEIGHBOR_W = {7: 0.25, 8: 1.0}


def _carbon_plevel_shift(
    atomic_numbers: np.ndarray,
    coords_bohr: np.ndarray,
) -> np.ndarray:
    """Per-atom carbon-p H0 self-energy shift (Ha) from bonded O/N environment.

    Zero for every non-carbon atom.  For each carbon, ``A + B * sum_j w[Zj] *
    (rcov_i + rcov_j)/r_ij`` over bonded O/N neighbours ``j`` (fluorine excluded;
    see ``GXTB_C_PLEVEL_*`` above).  The covalent radius/distance ratio is unitless,
    so any consistent length unit works; we use the H0 builder's Bohr coordinates.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    nat = atoms.size
    shift = np.zeros(nat, dtype=np.float64)
    if nat == 0:
        return shift
    rcov = np.asarray(QVSZP_PARAMS["cov_radii"][atoms - 1], dtype=np.float64) * ANG_TO_BOHR
    for i in range(nat):
        if int(atoms[i]) != 6:
            continue
        acc = 0.0
        for j in range(nat):
            if j == i:
                continue
            w = GXTB_C_PLEVEL_NEIGHBOR_W.get(int(atoms[j]), 0.0)
            if w == 0.0:
                continue
            rij = float(np.linalg.norm(coords_bohr[i] - coords_bohr[j]))
            rsum = float(rcov[i] + rcov[j])
            if rij < 1.0e-9 or rij > GXTB_C_PLEVEL_BOND_FACTOR * rsum:
                continue
            acc += w * (rsum / rij)
        shift[i] = GXTB_C_PLEVEL_A + GXTB_C_PLEVEL_B * acc
    return shift


def _diat_scale(Za: int, Zb: int, interaction: int) -> float:
    """Harmonic atom-pair diatomic-frame overlap scale for sigma/pi/delta."""

    table = np.asarray(GXTB_PARAMS["ps_h0_diat_scale"], dtype=np.float64)
    ka = float(table[(int(Za) - 1) * 3 + int(interaction)])
    kb = float(table[(int(Zb) - 1) * 3 + int(interaction)])
    if ka <= 0.0 or kb <= 0.0:
        return 1.0
    return 2.0 / (1.0 / ka + 1.0 / kb)


def _h0_shell_kscale(l1: int, l2: int) -> float:
    kshell = np.asarray(GXTB_PARAMS["pg_h0_kshell"], dtype=np.float64)
    return 0.5 * (float(kshell[int(l1)]) + float(kshell[int(l2)]))


def _diat_interaction_index(l_a: int, l_b: int) -> int | None:
    """Return the scalar sigma/pi channel used for mixed d-shell fallback.

    The exact g-xTB H0 contains a richer anisotropic term.  Until that full
    Slater-Koster-style d-sector is decoded, this fallback at least applies the
    recovered binary diatomic scale to mixed s-d/p-d overlaps instead of
    leaving sulfur d couplings entirely at the raw overlap value.  The scalar
    fallback is damped by ``GXTB_D_H0_SCALE_DAMP`` because the true binary path
    decomposes these blocks anisotropically; a full scalar scale destabilizes
    sulfur-rich conjugated systems.

    Pure d-d blocks are intentionally left raw: applying the scalar delta
    channel to multi-sulfur d-d blocks is much too aggressive without the full
    angular decomposition.
    """

    if l_a == 0 or l_b == 0:
        return 0
    if l_a == 1 or l_b == 1:
        return 1
    return None


def _element_has_active_d_shell(Z: int) -> bool:
    return int(GXTB_PARAMS["pa_nshell"][int(Z) - 1]) >= 3


def gxtb_shell_selfenergies(
    atomic_numbers: np.ndarray | list[int],
    basis: GXTBQVSZPBasis,
    cn: np.ndarray | None = None,
    carbon_plevel_shift: np.ndarray | None = None,
) -> np.ndarray:
    """CN-shifted per-shell H0 self energies in Hartree.

    When ``carbon_plevel_shift`` (per-atom, from :func:`_carbon_plevel_shift`) is
    supplied, it is added to each carbon p shell to restore the oxygen-driven
    charge transfer missing from the CN-only recovered level.  It is optional so
    callers without geometry still get the plain CN-shifted levels.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    cn_arr = basis.cn if cn is None else np.asarray(cn, dtype=np.float64)
    cps = None if carbon_plevel_shift is None else np.asarray(carbon_plevel_shift, dtype=np.float64)
    out = np.zeros(basis.shell_atom.size, dtype=np.float64)
    for ish, atom_idx in enumerate(basis.shell_atom):
        ai = int(atom_idx)
        Z = int(atoms[ai])
        l = int(basis.shell_l[ish])
        h0 = float(GXTB_PARAMS["ps_h0_selfenergy"][Z - 1, l])
        kcn = float(GXTB_PARAMS["ps_h0_selfenergy_cn"][Z - 1, l])
        level = h0 - kcn * float(cn_arr[ai])
        if cps is not None and Z == 6 and l == 1:
            level += float(cps[ai])
        out[ish] = level
    return out


def _shell_index_groups(basis: GXTBQVSZPBasis) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for mu, bf in enumerate(basis.cao_basis):
        groups.setdefault(int(bf.shell_id), []).append(mu)
    return groups


def _diatomic_scaled_overlap_cao(
    atomic_numbers: np.ndarray,
    coords_bohr: np.ndarray,
    basis: GXTBQVSZPBasis,
) -> np.ndarray:
    """Apply the SI Eq. 31/32 sigma/pi scaling for active s/p CAO blocks."""

    S = np.asarray(basis.S_cao, dtype=np.float64).copy()
    groups = _shell_index_groups(basis)
    shell_ids = sorted(groups)
    shell_atom: dict[int, int] = {}
    shell_l: dict[int, int] = {}
    for sid, indices in groups.items():
        bf = basis.cao_basis[indices[0]]
        shell_atom[sid] = int(bf.atom_idx)
        shell_l[sid] = int(bf.l_total)

    for pos, sid_a in enumerate(shell_ids[:-1]):
        atom_a = shell_atom[sid_a]
        l_a = shell_l[sid_a]
        for sid_b in shell_ids[pos + 1 :]:
            atom_b = shell_atom[sid_b]
            if atom_a == atom_b:
                continue
            l_b = shell_l[sid_b]
            Za = int(atomic_numbers[atom_a])
            Zb = int(atomic_numbers[atom_b])
            ia = groups[sid_a]
            ib = groups[sid_b]
            if l_a == 0 and l_b == 0:
                block = S[np.ix_(ia, ib)] * _diat_scale(Za, Zb, 0)
            elif l_a == 0 and l_b == 1:
                block = S[np.ix_(ia, ib)] * _diat_scale(Za, Zb, 0)
            elif l_a == 1 and l_b == 0:
                block = S[np.ix_(ia, ib)] * _diat_scale(Za, Zb, 0)
            elif l_a == 1 and l_b == 1:
                rab = coords_bohr[atom_b] - coords_bohr[atom_a]
                r = float(np.linalg.norm(rab))
                if r < 1.0e-14:
                    continue
                u = rab / r
                psigma = np.outer(u, u)
                ppi = np.eye(3) - psigma
                ksigma = _diat_scale(Za, Zb, 0)
                kpi = _diat_scale(Za, Zb, 1)
                raw = S[np.ix_(ia, ib)]
                block = kpi * (ppi @ raw @ ppi) + ksigma * (psigma @ raw @ psigma)
            else:
                interaction = _diat_interaction_index(l_a, l_b)
                block = S[np.ix_(ia, ib)]
                if (
                    interaction is not None
                    and not (_element_has_active_d_shell(Za) and _element_has_active_d_shell(Zb))
                ):
                    scale = _diat_scale(Za, Zb, interaction)
                    block = block * (1.0 + GXTB_D_H0_SCALE_DAMP * (scale - 1.0))
            S[np.ix_(ia, ib)] = block
            S[np.ix_(ib, ia)] = block.T
    return S


def build_hcore_gxtb(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
    basis: GXTBQVSZPBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the current reconstructed g-xTB H0 in the SAO basis.

    This covers the shell-charge/CN H0 level shifts, q-vSZP overlap, global
    shell K factors, and the recovered atom/shell ``shpoly2`` distance factor.
    The additional anisotropic H0 block remains separate and is not yet applied.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords * ANG_TO_BOHR
    S_cao = _diatomic_scaled_overlap_cao(atoms, coords_bohr, basis)
    n_cao = S_cao.shape[0]

    c_plevel = _carbon_plevel_shift(atoms, coords_bohr)
    shell_self = gxtb_shell_selfenergies(atoms, basis, carbon_plevel_shift=c_plevel)
    cao_shell = np.asarray(basis.cao_bf_to_shell, dtype=np.int64)
    bf_atom = np.array([bf.atom_idx for bf in basis.cao_basis], dtype=np.int64)
    bf_shell_l = basis.shell_l[cao_shell]
    bf_self = shell_self[cao_shell]

    atom_cov = np.asarray(QVSZP_PARAMS["cov_radii"][atoms - 1], dtype=np.float64) * ANG_TO_BOHR
    shpoly_atom = np.asarray(GXTB_PARAMS["pa_h0_shpoly2"][atoms - 1], dtype=np.float64)
    shpoly_shell = np.asarray(GXTB_PARAMS["pg_h0_shpoly2"], dtype=np.float64)

    H0_cao = np.zeros((n_cao, n_cao), dtype=np.float64)
    for mu in range(n_cao):
        atom_mu = int(bf_atom[mu])
        l_mu = int(bf_shell_l[mu])
        for nu in range(mu + 1, n_cao):
            atom_nu = int(bf_atom[nu])
            if atom_mu == atom_nu:
                continue
            l_nu = int(bf_shell_l[nu])
            rij = float(np.linalg.norm(coords_bohr[atom_mu] - coords_bohr[atom_nu]))
            # gp3 source (xtb/h0.f90 get_hamiltonian) + binary: rr = sqrt(R/(rad_i+rad_j)),
            # the FULL radius sum and a SQRT distance term — NOT the linear rij/(0.5*sum) used before.
            rcov = max(float(atom_cov[atom_mu] + atom_cov[atom_nu]), 1.0e-12)
            rr = float(np.sqrt(rij / rcov))
            pi_mu = 1.0 + shpoly_atom[atom_mu] * shpoly_shell[l_mu] * rr
            pi_nu = 1.0 + shpoly_atom[atom_nu] * shpoly_shell[l_nu] * rr
            hscale = _h0_shell_kscale(l_mu, l_nu)
            h_avg = 0.5 * (bf_self[mu] + bf_self[nu])
            value = hscale * h_avg * pi_mu * pi_nu * S_cao[mu, nu]
            H0_cao[mu, nu] = value
            H0_cao[nu, mu] = value

    T = basis.T_cao_to_sao
    H0 = T @ H0_cao @ T.T
    np.fill_diagonal(H0, shell_self[basis.bf_to_shell])
    return H0, shell_self
