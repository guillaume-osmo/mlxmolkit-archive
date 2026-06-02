# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Experimental g-xTB H0 builder backed by recovered binary parameters."""

from __future__ import annotations

import numpy as np

from .gxtb_basis import ANG_TO_BOHR, GXTBQVSZPBasis
from .params_gxtb import GXTB_PARAMS
from .qvszp_params import QVSZP_PARAMS


GXTB_D_H0_SCALE_DAMP = 0.1


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
) -> np.ndarray:
    """CN-shifted per-shell H0 self energies in Hartree."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    cn_arr = basis.cn if cn is None else np.asarray(cn, dtype=np.float64)
    out = np.zeros(basis.shell_atom.size, dtype=np.float64)
    for ish, atom_idx in enumerate(basis.shell_atom):
        Z = int(atoms[int(atom_idx)])
        l = int(basis.shell_l[ish])
        h0 = float(GXTB_PARAMS["ps_h0_selfenergy"][Z - 1, l])
        kcn = float(GXTB_PARAMS["ps_h0_selfenergy_cn"][Z - 1, l])
        out[ish] = h0 - kcn * float(cn_arr[int(atom_idx)])
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

    shell_self = gxtb_shell_selfenergies(atoms, basis)
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
            rcov = max(0.5 * float(atom_cov[atom_mu] + atom_cov[atom_nu]), 1.0e-12)
            rr = rij / rcov
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
