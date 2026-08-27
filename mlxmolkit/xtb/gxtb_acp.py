# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Small p-ACP scaffold for the reconstructed g-xTB driver.

The exact p-ACP term in tblite is a one-electron potential assembled from the
``ps_acp_*`` and ``pa_l_acp`` tables.  Until the projector assembly is decoded,
this module exposes a conservative pair-energy proxy so the driver can keep
the term isolated and measurable without folding it into unrelated SCC code.
"""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction, overlap_matrix, primitive_norm_d, primitive_norm_p, primitive_norm_s
from .gxtb_basis import ANG_TO_BOHR, GXTBQVSZPBasis
from .params_gxtb import GXTB_PARAMS


_D_LXYZ = (
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)
GXTB_ACP_PROJECTOR_SCALE = 0.2


def _primitive_norm(l: int, alpha: float) -> float:
    arr = np.asarray([alpha], dtype=np.float64)
    if l == 0:
        return float(primitive_norm_s(arr)[0])
    if l == 1:
        return float(primitive_norm_p(arr)[0])
    if l == 2:
        return float(primitive_norm_d(arr)[0])
    raise NotImplementedError("ACP auxiliary f projectors are not implemented in the native scaffold yet")


def build_gxtb_acp_hamiltonian_reference(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    basis: GXTBQVSZPBasis,
    *,
    enabled: bool = True,
    scale: float = GXTB_ACP_PROJECTOR_SCALE,
) -> np.ndarray:
    """Build the reduced non-local ACP Hamiltonian from SI Eq. 78.

    The binary/projector tables expose one level and exponent per atom ACP
    channel.  We use normalized cartesian Gaussian projectors and assemble the
    one-electron matrix as ``H = S_AO,aux * level * S_aux,AO`` in the CAO basis
    before applying the existing CAO→SAO transform.
    """

    n_cao = len(basis.cao_basis)
    if not enabled:
        return np.zeros_like(basis.S)

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    aux_basis: list[BasisFunction] = []
    aux_level: list[float] = []

    for atom_idx, Z0 in enumerate(atoms):
        Z = int(Z0)
        n_acp = int(GXTB_PARAMS["pa_nacp"][Z - 1])
        for iproj in range(n_acp):
            l = int(GXTB_PARAMS["pa_l_acp"][Z - 1, iproj])
            if l > 2:
                continue
            level = float(GXTB_PARAMS["ps_acp_level"][Z - 1, iproj])
            alpha = float(GXTB_PARAMS["ps_acp_exp"][Z - 1, iproj])
            if alpha <= 0.0 or level == 0.0:
                continue
            coeff = np.asarray([_primitive_norm(l, alpha)], dtype=np.float64)
            alphas = np.asarray([alpha], dtype=np.float64)
            if l == 0:
                lxyz_iter = [(0, 0, 0)]
            elif l == 1:
                lxyz_iter = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            else:
                lxyz_iter = list(_D_LXYZ)
            for l_xyz in lxyz_iter:
                aux_basis.append(
                    BasisFunction(
                        atom_idx=atom_idx,
                        l_total=l,
                        l_xyz=l_xyz,
                        center=coords_bohr[atom_idx],
                        alphas=alphas,
                        coeffs=coeff,
                        is_valence=False,
                    )
                )
                aux_level.append(level)

    if not aux_basis:
        return np.zeros_like(basis.S)

    combined = list(basis.cao_basis) + aux_basis
    S_full = overlap_matrix(combined)
    B = S_full[:n_cao, n_cao:]
    levels = np.asarray(aux_level, dtype=np.float64)
    H_cao = float(scale) * ((B * levels[None, :]) @ B.T)
    H_cao = 0.5 * (H_cao + H_cao.T)
    T = basis.T_cao_to_sao
    H_sao = T @ H_cao @ T.T
    return 0.5 * (H_sao + H_sao.T)


def gxtb_pacp_proxy_energy(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    enabled: bool = True,
) -> float:
    """Return a bounded H-F p-ACP proxy energy in Hartree.

    This is intentionally not used inside the Fock matrix.  It is a placeholder
    component with the right parameter tables and atom domain (H-F), useful for
    smoke-testing the full driver while the exact projector kernel is recovered.
    """

    if not enabled:
        return 0.0
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    e = 0.0
    for i in range(atoms.size - 1):
        Zi = int(atoms[i])
        if Zi > 9:
            continue
        for j in range(i + 1, atoms.size):
            Zj = int(atoms[j])
            if Zj > 9:
                continue
            r2 = float(np.sum((coords[i] - coords[j]) ** 2))
            ni = int(GXTB_PARAMS["pa_nacp"][Zi - 1])
            nj = int(GXTB_PARAMS["pa_nacp"][Zj - 1])
            li = GXTB_PARAMS["ps_acp_level"][Zi - 1, :ni]
            lj = GXTB_PARAMS["ps_acp_level"][Zj - 1, :nj]
            ei = GXTB_PARAMS["ps_acp_exp"][Zi - 1, :ni]
            ej = GXTB_PARAMS["ps_acp_exp"][Zj - 1, :nj]
            amp = 0.5 * (float(np.sum(li)) + float(np.sum(lj)))
            decay = 0.5 * (float(np.mean(ei)) + float(np.mean(ej)))
            e += 0.01 * amp * float(np.exp(-max(decay, 1.0e-8) * r2))
    return float(e)



# --------------------------------------------------------------------------
# Fast path: build the AO x aux block directly.
#
# The reference above concatenates the CAO basis with the ACP auxiliary basis
# and takes one off-diagonal block of the resulting overlap.  For a 44-atom
# molecule that is 92 AO + 572 aux = 664 functions, and the compiled multipole
# kernel it routes to returns TEN matrices -- about 84x the arithmetic the term
# actually consumes, for a 92 x 572 block of one of them.
#
# Build that block instead.  Cartesian components of a shell share centre,
# exponents and contraction coefficients, so R^2, the exponential and the
# Obara-Saika tables are evaluated once per SHELL pair; blocking by angular type
# makes the per-component work three multiplies with no gather.  Two smaller
# points that measured larger than they look: (pi/p)**1.5 calls `pow`, 9.6x
# slower than t*sqrt(t) for a one-ulp difference, and keeping the displacements
# as three contiguous per-axis arrays rather than one (P, M, 3) array avoids a
# stride-3 read in the table recurrence.
#
# 6.0x on the large molecules; agrees with the reference to 2.9e-16.

_ACP_NCOMP = (1, 3, 6)


_ACP_P_LXYZ = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


_ACP_D_LXYZ = ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1))


_ACP_ELEM_CACHE: dict = {}


def _acp_cao_shells(cao_basis):
    """Group the CAO list into shells, keyed by angular momentum."""
    out: dict = {}
    i, n = 0, len(cao_basis)
    while i < n:
        b = cao_basis[i]
        l = int(b.l_total)
        nc = _ACP_NCOMP[l]
        idxs = list(range(i, i + nc))
        for k in idxs:
            bk = cao_basis[k]
            # The builder shares one alphas/coeffs object across a shell's
            # components (measured: 92/92), so identity is the fast path; the
            # array compare stays as the fallback, keeping the check as strong.
            if (int(bk.l_total) != l or bk.atom_idx != b.atom_idx
                    or not (bk.alphas is b.alphas
                            or np.array_equal(bk.alphas, b.alphas))
                    or not (bk.coeffs is b.coeffs
                            or np.array_equal(bk.coeffs, b.coeffs))):
                raise RuntimeError("CAO shell grouping assumption violated")
        out.setdefault(l, []).append(
            (np.asarray(b.center, dtype=np.float64),
             np.asarray(b.alphas, dtype=np.float64),
             np.asarray(b.coeffs, dtype=np.float64),
             [tuple(int(v) for v in cao_basis[k].l_xyz) for k in idxs], idxs))
        i += nc
    return out


def _acp_elem(Z: int):
    """Per-element ACP projector template (l, alpha, norm, level)."""
    hit = _ACP_ELEM_CACHE.get(Z)
    if hit is not None:
        return hit
    from mlxmolkit.xtb.basis import (
        primitive_norm_d, primitive_norm_p, primitive_norm_s,
    )
    rows = []
    for iproj in range(int(GXTB_PARAMS["pa_nacp"][Z - 1])):
        l = int(GXTB_PARAMS["pa_l_acp"][Z - 1, iproj])
        if l > 2:
            continue
        level = float(GXTB_PARAMS["ps_acp_level"][Z - 1, iproj])
        alpha = float(GXTB_PARAMS["ps_acp_exp"][Z - 1, iproj])
        if alpha <= 0.0 or level == 0.0:
            continue
        nrm = float((primitive_norm_s, primitive_norm_p, primitive_norm_d)[l](
            np.asarray([alpha], dtype=np.float64))[0])
        rows.append((l, alpha, nrm, level))
    _ACP_ELEM_CACHE[Z] = rows
    return rows


def _acp_aux_shells(atoms, coords_ang):
    """ACP projectors as shells, keyed by angular momentum."""
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    out: dict = {}
    col = 0
    for ia, Z0 in enumerate(np.asarray(atoms, dtype=np.intp)):
        for l, alpha, nrm, level in _acp_elem(int(Z0)):
            lxyz = ((0, 0, 0),) if l == 0 else (
                _ACP_P_LXYZ if l == 1 else _ACP_D_LXYZ)
            nc = _ACP_NCOMP[l]
            out.setdefault(l, []).append(
                (coords_bohr[ia], alpha, nrm, level, lxyz,
                 list(range(col, col + nc))))
            col += nc
    return out, col


def _acp_os_tables(PA, PB, inv2p, la, lb):
    """Obara-Saika 1D overlap tables S[i][j] for i <= la, j <= lb.

    ``None`` stands for the implicit S[0][0] = 1, so the s-s case costs nothing.
    """
    S = [[None] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        t = PA if S[i - 1][0] is None else PA * S[i - 1][0]
        if i >= 2:
            t = t + inv2p * (i - 1) * (1.0 if S[i - 2][0] is None else S[i - 2][0])
        S[i][0] = t
    for j in range(1, lb + 1):
        for i in range(la + 1):
            t = PB if S[i][j - 1] is None else PB * S[i][j - 1]
            if j >= 2:
                t = t + inv2p * (j - 1) * (1.0 if S[i][j - 2] is None else S[i][j - 2])
            if i >= 1:
                t = t + inv2p * i * (1.0 if S[i - 1][j - 1] is None else S[i - 1][j - 1])
            S[i][j] = t
    return S


def _acp_cross_overlap(cao_basis, atoms, coords_ang):
    """<AO_mu | aux_j> and the projector levels, assembled shell-pair-wise."""
    ao = _acp_cao_shells(cao_basis)
    aux, n_aux = _acp_aux_shells(atoms, coords_ang)
    if n_aux == 0:
        return None, None
    B = np.zeros((len(cao_basis), n_aux), dtype=np.float64)
    levels = np.zeros(n_aux, dtype=np.float64)
    for blocks in aux.values():
        for _, _, _, lev, _, cols in blocks:
            levels[cols] = lev
    for la, a_blocks in ao.items():
        a_cen = np.concatenate([np.repeat(s[0][None, :], len(s[1]), 0) for s in a_blocks])
        a_cx = np.ascontiguousarray(a_cen[:, 0])
        a_cy = np.ascontiguousarray(a_cen[:, 1])
        a_cz = np.ascontiguousarray(a_cen[:, 2])
        a_al = np.concatenate([s[1] for s in a_blocks])
        a_cf = np.concatenate([s[2] for s in a_blocks])
        offs = np.cumsum([0] + [len(s[1]) for s in a_blocks])[:-1]
        a_rows = np.asarray([s[4] for s in a_blocks], dtype=np.intp)
        a_lxyz = a_blocks[0][3]
        for lb, b_blocks in aux.items():
            b_cen = np.asarray([s[0] for s in b_blocks], dtype=np.float64)
            b_al = np.asarray([s[1] for s in b_blocks], dtype=np.float64)
            b_cf = np.asarray([s[2] for s in b_blocks], dtype=np.float64)
            b_cols = np.asarray([s[5] for s in b_blocks], dtype=np.intp)
            b_lxyz = b_blocks[0][4]
            p = a_al[:, None] + b_al[None, :]
            inv2p = 0.5 / p
            # Per-axis contiguous displacements rather than one (Pa, Mb, 3)
            # array: the Obara-Saika tables below read one axis at a time, and
            # a stride-3 view costs ~1.5x a contiguous one.
            dxyz = (a_cx[:, None] - b_cen[:, 0][None, :],
                    a_cy[:, None] - b_cen[:, 1][None, :],
                    a_cz[:, None] - b_cen[:, 2][None, :])
            R2 = dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2]
            # t*sqrt(t) rather than t**1.5: pow is a transcendental and is 9.6x
            # slower here, for a one-ulp difference (2.2e-16 relative).
            t = np.pi / p
            base = ((t * np.sqrt(t))
                    * np.exp(-(a_al[:, None] * b_al[None, :]) / p * R2)
                    * (a_cf[:, None] * b_cf[None, :]))
            rb = b_al[None, :] / p
            ra = a_al[:, None] / p
            tabs = [_acp_os_tables(-rb * dxyz[ax], ra * dxyz[ax], inv2p, la, lb)
                    for ax in range(3)]
            for ca, lxa in enumerate(a_lxyz):
                for cb, lxb in enumerate(b_lxyz):
                    f = base
                    for ax in range(3):
                        t = tabs[ax][lxa[ax]][lxb[ax]]
                        if t is not None:
                            f = f * t
                    B[np.ix_(a_rows[:, ca], b_cols[:, cb])] = np.add.reduceat(f, offs, axis=0)
    return B, levels


def build_gxtb_acp_hamiltonian(atomic_numbers, coords_ang, basis, *,
                               enabled: bool = True,
                               scale: float = GXTB_ACP_PROJECTOR_SCALE):
    """Reduced non-local ACP Hamiltonian (SI Eq. 78), via a direct AO x aux block."""
    if not enabled:
        return np.zeros_like(basis.S)
    B, levels = _acp_cross_overlap(basis.cao_basis, atomic_numbers, coords_ang)
    if B is None:
        return np.zeros_like(basis.S)
    H_cao = float(scale) * ((B * levels[None, :]) @ B.T)
    T = basis.T_cao_to_sao
    H_sao = T @ (0.5 * (H_cao + H_cao.T)) @ T.T
    return 0.5 * (H_sao + H_sao.T)
