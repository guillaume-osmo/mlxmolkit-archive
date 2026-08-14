# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Binary q-vSZP basis and overlap builder for the reconstructed g-xTB path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import (
    BasisFunction,
    _contraction_norm,
    overlap_matrix,
    primitive_norm_d,
    primitive_norm_p,
    primitive_norm_s,
    sao_basis_metadata,
)
from .eeqbc import eeqbc_solve
from .gxtb_reconstructed import _gxtb_erf_coordination_number
from .params_gxtb import GXTB_PARAMS
from .qvszp_params import QVSZP_PARAMS


ANG_TO_BOHR = 1.8897259886
_D_LXYZ = (
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)


@dataclass(frozen=True)
class GXTBQVSZPBasis:
    """q-vSZP CAO/SAO basis plus shell metadata used by H0/SCC."""

    cao_basis: list[BasisFunction]
    sao_basis: list[BasisFunction]
    T_cao_to_sao: np.ndarray
    S_cao: np.ndarray
    S: np.ndarray
    cao_bf_to_shell: np.ndarray
    bf_to_shell: np.ndarray
    shell_atom: np.ndarray
    shell_l: np.ndarray
    shell_zref: np.ndarray
    shell_hardness: np.ndarray
    shell_third: np.ndarray
    shell_fourth: np.ndarray
    shell_exchange: np.ndarray
    cn: np.ndarray
    eeqbc_charges: np.ndarray
    qeff: np.ndarray


def qvszp_qeff(q: np.ndarray, cn: np.ndarray, k0: np.ndarray, k1: np.ndarray, k2: np.ndarray, k3: np.ndarray) -> np.ndarray:
    """Charge/CN response variable used by q-vSZP coefficient updates."""

    q_arr = np.asarray(q, dtype=np.float64)
    cn_arr = np.asarray(cn, dtype=np.float64)
    return k0 * (q_arr - k1 * q_arr * q_arr) + k2 * np.sqrt(np.maximum(cn_arr, 0.0)) + k3 * cn_arr * q_arr


def qvszp_qeff_derivatives(
    q: np.ndarray,
    cn: np.ndarray,
    k0: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dqeff/dq, dqeff/dCN)`` for tests and future gradients."""

    q_arr = np.asarray(q, dtype=np.float64)
    cn_arr = np.asarray(cn, dtype=np.float64)
    safe_cn = np.maximum(cn_arr, 1.0e-14)
    dq = k0 * (1.0 - 2.0 * k1 * q_arr) + k3 * cn_arr
    dcn = 0.5 * k2 / np.sqrt(safe_cn) + k3 * q_arr
    return dq, dcn


def _primitive_norms(l: int, alphas: np.ndarray) -> np.ndarray:
    if l == 0:
        return primitive_norm_s(alphas)
    if l == 1:
        return primitive_norm_p(alphas)
    if l == 2:
        return primitive_norm_d(alphas)
    raise NotImplementedError(f"q-vSZP shell angular momentum l={l} is not supported yet")


def _active_shell_count(Z: int, include_polarization_shells: bool) -> int:
    if include_polarization_shells:
        return int(QVSZP_PARAMS["nshell"][int(Z) - 1])
    return int(GXTB_PARAMS["pa_nshell"][int(Z) - 1])


def _shell_component_count(l: int, *, spherical: bool) -> int:
    if l == 0:
        return 1
    if l == 1:
        return 3
    if l == 2:
        return 5 if spherical else 6
    raise NotImplementedError(f"l={l} shells are not supported")


def build_gxtb_qvszp_basis(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    total_charge: float = 0.0,
    eeqbc_charges: np.ndarray | None = None,
    cn: np.ndarray | None = None,
    include_polarization_shells: bool = False,
) -> GXTBQVSZPBasis:
    """Build the q-vSZP basis used by the experimental native g-xTB driver.

    The binary q-vSZP table contains H p and C/N/O d polarization shells.
    The currently recovered g-xTB H0/SCC parameter tables have active
    occupations and hardnesses only for ``pa_nshell`` shells.  The default
    therefore follows ``pa_nshell``.  ``include_polarization_shells=True`` is
    exposed for probes, but the SCF driver intentionally leaves it off until
    the corresponding H0/ACP coupling is fully decoded.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    if coords.shape != (atoms.size, 3):
        raise ValueError("coords_ang must have shape (nat, 3)")

    if cn is None:
        cn_arr = _gxtb_erf_coordination_number(atoms, coords)
    else:
        cn_arr = np.asarray(cn, dtype=np.float64)
    if eeqbc_charges is None:
        eeqbc_result = eeqbc_solve(atoms, coords, total_charge=total_charge)
        q_basis = np.asarray(eeqbc_result.charges, dtype=np.float64)
    else:
        q_basis = np.asarray(eeqbc_charges, dtype=np.float64)
    if cn_arr.shape != atoms.shape or q_basis.shape != atoms.shape:
        raise ValueError("cn and eeqbc_charges must have one value per atom")

    k0 = QVSZP_PARAMS["p_k0"][atoms - 1] * GXTB_PARAMS["pa_h0_qvszp_k0_scal"][atoms - 1]
    k1 = QVSZP_PARAMS["p_k1"][atoms - 1]
    k2 = QVSZP_PARAMS["p_k2"][atoms - 1] * GXTB_PARAMS["pa_h0_qvszp_k2_scal"][atoms - 1]
    k3 = QVSZP_PARAMS["p_k3"][atoms - 1] * GXTB_PARAMS["pa_h0_qvszp_k3_scal"][atoms - 1]
    qeff_atom = qvszp_qeff(q_basis, cn_arr, k0, k1, k2, k3)

    cao_basis: list[BasisFunction] = []
    cao_bf_to_shell: list[int] = []
    shell_atom: list[int] = []
    shell_l: list[int] = []
    shell_zref: list[float] = []
    shell_hardness: list[float] = []
    shell_third: list[float] = []
    shell_fourth: list[float] = []
    shell_exchange: list[float] = []

    next_shell_id = 0
    coords_bohr = coords * ANG_TO_BOHR
    for atom_idx, Z0 in enumerate(atoms):
        Z = int(Z0)
        n_active = _active_shell_count(Z, include_polarization_shells)
        for ish in range(n_active):
            qshell = QVSZP_PARAMS.shell(Z, ish)
            if qshell.n_prim <= 0:
                continue
            l = qshell.l
            if l > 2:
                continue

            # The q-vSZP overlap/density basis uses BASE exponents; the charge
            # dependence lives in the contraction coefficients below.  The
            # ``ps_h0_qvszp_exp_scal`` table is an H0-only effective-basis
            # scaling and must NOT touch the overlap basis (verified against the
            # released g-xTB molden: base exps reproduce S to machine precision).
            alphas = np.asarray(qshell.exponents, dtype=np.float64)
            raw_coeffs = np.asarray(qshell.coefficients, dtype=np.float64) + (
                np.asarray(qshell.coefficients_env, dtype=np.float64) * qeff_atom[atom_idx]
            )
            norm = _contraction_norm(alphas, raw_coeffs, l)
            coeffs = raw_coeffs * _primitive_norms(l, alphas) * norm

            shell_id = next_shell_id
            next_shell_id += 1
            sh_index = len(shell_atom)
            shell_atom.append(atom_idx)
            shell_l.append(l)
            shell_zref.append(float(GXTB_PARAMS["ps_reference_occ"][Z - 1, ish]))
            base_hard = (
                float(GXTB_PARAMS["ps_tb2_shell_hubbard"][Z - 1, ish])
                * float(GXTB_PARAMS["pa_hubbard_parameter"][Z - 1])
            )
            cn_slope = float(GXTB_PARAMS["pa_tb2_hubbard_cn"][Z - 1])
            shell_hardness.append(max(base_hard * (1.0 + cn_slope * cn_arr[atom_idx]), 1.0e-8))
            shell_third.append(
                float(GXTB_PARAMS["pa_tb3_hubbard_derivs"][Z - 1])
                * float(GXTB_PARAMS["pg_tb3_kshell"][l])
            )
            # 4th-order onsite hardness Gamma4_sh = pg_tb4_kshell[l] * K4TH (=0.036).
            # Binary-exact (add_coulomb 0x41a0b4: ldr d25,[x1,0xbe8]=0.036 ; fmul on
            # pg_tb4_kshell): the per-element pa_tb3_hubbard_derivs factor is NOT
            # present here. The 0.036 lives in scf_gxtb.GXTB_K4TH_SCALE, so this table
            # holds pg_tb4_kshell[l] alone.
            shell_fourth.append(float(GXTB_PARAMS["pg_tb4_kshell"][l]))
            shell_exchange.append(
                float(GXTB_PARAMS["ps_fock_shell_hubbard"][Z - 1, ish])
                * float(GXTB_PARAMS["pg_fock_kq"][l])
            )

            if l == 0:
                cao_basis.append(
                    BasisFunction(
                        atom_idx=atom_idx,
                        l_total=0,
                        l_xyz=(0, 0, 0),
                        center=coords_bohr[atom_idx],
                        alphas=alphas,
                        coeffs=coeffs,
                        is_valence=True,
                        shell_id=shell_id,
                    )
                )
                cao_bf_to_shell.append(sh_index)
            elif l == 1:
                for axis in range(3):
                    l_xyz = [0, 0, 0]
                    l_xyz[axis] = 1
                    cao_basis.append(
                        BasisFunction(
                            atom_idx=atom_idx,
                            l_total=1,
                            l_xyz=tuple(l_xyz),
                            center=coords_bohr[atom_idx],
                            alphas=alphas,
                            coeffs=coeffs,
                            is_valence=True,
                            shell_id=shell_id,
                        )
                    )
                    cao_bf_to_shell.append(sh_index)
            elif l == 2:
                for l_xyz in _D_LXYZ:
                    cao_basis.append(
                        BasisFunction(
                            atom_idx=atom_idx,
                            l_total=2,
                            l_xyz=l_xyz,
                            center=coords_bohr[atom_idx],
                            alphas=alphas,
                            coeffs=coeffs,
                            is_valence=True,
                            shell_id=shell_id,
                        )
                    )
                    cao_bf_to_shell.append(sh_index)

    S_cao = overlap_matrix(cao_basis)
    sao_basis, T = sao_basis_metadata(cao_basis)
    if T.shape[0] == T.shape[1] and np.array_equal(T, np.eye(T.shape[0])):
        S = S_cao
        bf_to_shell = np.asarray(cao_bf_to_shell, dtype=np.int64)
    else:
        S = T @ S_cao @ T.T
        bf_to_shell_list: list[int] = []
        seen: set[int] = set()
        for mu, bf in enumerate(cao_basis):
            if mu in seen:
                continue
            if bf.l_total < 2:
                bf_to_shell_list.append(cao_bf_to_shell[mu])
            else:
                sid = bf.shell_id
                d_idx = [k for k in range(mu, len(cao_basis)) if cao_basis[k].shell_id == sid]
                for k in d_idx:
                    seen.add(k)
                bf_to_shell_list.extend([cao_bf_to_shell[mu]] * 5)
        bf_to_shell = np.asarray(bf_to_shell_list, dtype=np.int64)

    return GXTBQVSZPBasis(
        cao_basis=cao_basis,
        sao_basis=sao_basis,
        T_cao_to_sao=T,
        S_cao=S_cao,
        S=S,
        cao_bf_to_shell=np.asarray(cao_bf_to_shell, dtype=np.int64),
        bf_to_shell=bf_to_shell,
        shell_atom=np.asarray(shell_atom, dtype=np.int64),
        shell_l=np.asarray(shell_l, dtype=np.int64),
        shell_zref=np.asarray(shell_zref, dtype=np.float64),
        shell_hardness=np.asarray(shell_hardness, dtype=np.float64),
        shell_third=np.asarray(shell_third, dtype=np.float64),
        shell_fourth=np.asarray(shell_fourth, dtype=np.float64),
        shell_exchange=np.asarray(shell_exchange, dtype=np.float64),
        cn=cn_arr,
        eeqbc_charges=q_basis,
        qeff=qeff_atom,
    )
