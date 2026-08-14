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


def build_gxtb_acp_hamiltonian(
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
