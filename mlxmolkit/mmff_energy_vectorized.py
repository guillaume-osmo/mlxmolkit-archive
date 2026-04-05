"""
Vectorized MMFF94 energy and gradient for batch conformer optimization.

All C conformers are processed simultaneously using numpy broadcasting.
Parameters (topology) are shared across conformers; only positions differ.
This eliminates the RDKit callback bottleneck.

Shapes:
  positions: (C, N, 3)  -- C conformers, N atoms, 3D coordinates
  energies:  (C,)        -- one energy per conformer
  gradients: (C, N, 3)   -- gradient per conformer
"""
from __future__ import annotations

import numpy as np

from mlxmolkit.mmff_params import MMFFParams

MDYNE_A_TO_KCAL = 143.9325
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
ANGLE_BEND_CONST = 0.043844
STRETCH_BEND_CONST = 2.51210
ELE_CONST = 332.0716
ELE_DAMP = 0.05
BOND_CUBIC_CS = -2.0
EPS = 1e-30


def _safe_norm(v: np.ndarray) -> np.ndarray:
    """L2 norm along last axis, with epsilon for safety."""
    return np.sqrt(np.sum(v ** 2, axis=-1, keepdims=True)).clip(min=EPS)


def mmff_energy_grad_batch(
    params: MMFFParams,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MMFF94 energy and gradient for C conformers simultaneously.

    Args:
        params: MMFFParams (shared topology for all conformers).
        positions: shape (C, N, 3) float64 atom positions.

    Returns:
        (energies, gradients) of shapes (C,) and (C, N*3).
    """
    C, N, _ = positions.shape
    pos = positions.astype(np.float64)
    grad = np.zeros_like(pos)
    energies = np.zeros(C, dtype=np.float64)

    # --- 1. Bond stretch ---
    if len(params.bond_idx1) > 0:
        i1 = params.bond_idx1
        i2 = params.bond_idx2
        kb = params.bond_kb.astype(np.float64)
        r0 = params.bond_r0.astype(np.float64)

        # d: (C, n_bonds, 3)
        d = pos[:, i1] - pos[:, i2]
        r = np.sqrt(np.sum(d ** 2, axis=-1)).clip(min=EPS)  # (C, n_bonds)
        d_hat = d / r[..., None]  # (C, n_bonds, 3)
        dr = r - r0[None, :]  # (C, n_bonds)

        cs = BOND_CUBIC_CS
        e_bond = MDYNE_A_TO_KCAL * kb / 2.0 * dr ** 2 * (
            1.0 + cs * dr + 7.0 / 12.0 * cs ** 2 * dr ** 2
        )
        dE_dr = MDYNE_A_TO_KCAL * kb * dr * (
            1.0 + 1.5 * cs * dr + 7.0 / 3.0 * cs ** 2 * dr ** 2 / 2.0
        )
        energies += e_bond.sum(axis=1)

        f = dE_dr[..., None] * d_hat  # (C, n_bonds, 3)
        np.add.at(grad, (np.arange(C)[:, None], i1[None, :]), f)
        np.add.at(grad, (np.arange(C)[:, None], i2[None, :]), -f)

    # --- 2. Angle bend ---
    if len(params.angle_idx1) > 0:
        ia = params.angle_idx1
        ib = params.angle_idx2
        ic = params.angle_idx3
        ka = params.angle_ka.astype(np.float64)
        theta0 = params.angle_theta0.astype(np.float64) * DEG_TO_RAD

        v1 = pos[:, ia] - pos[:, ib]  # (C, n_angles, 3)
        v2 = pos[:, ic] - pos[:, ib]
        r1 = np.sqrt(np.sum(v1 ** 2, axis=-1)).clip(min=EPS)
        r2 = np.sqrt(np.sum(v2 ** 2, axis=-1)).clip(min=EPS)
        cos_theta = np.clip(np.sum(v1 * v2, axis=-1) / (r1 * r2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta).clip(min=EPS)

        dt_dv1 = (cos_theta[..., None] * v1 / r1[..., None] - v2 / r2[..., None]) / (r1[..., None] * sin_theta[..., None])
        dt_dv2 = (cos_theta[..., None] * v2 / r2[..., None] - v1 / r1[..., None]) / (r2[..., None] * sin_theta[..., None])

        d_theta = theta - theta0[None, :]
        cb = -0.007
        e_angle = ANGLE_BEND_CONST * ka / 2.0 * (d_theta * RAD_TO_DEG) ** 2 * (
            1.0 + cb * d_theta * RAD_TO_DEG
        )
        dE_dtheta = ANGLE_BEND_CONST * ka * d_theta * RAD_TO_DEG * (
            1.0 + 1.5 * cb * d_theta * RAD_TO_DEG
        ) * RAD_TO_DEG
        energies += e_angle.sum(axis=1)

        gi = dE_dtheta[..., None] * dt_dv1
        gk = dE_dtheta[..., None] * dt_dv2
        gj = -gi - gk
        np.add.at(grad, (np.arange(C)[:, None], ia[None, :]), gi)
        np.add.at(grad, (np.arange(C)[:, None], ib[None, :]), gj)
        np.add.at(grad, (np.arange(C)[:, None], ic[None, :]), gk)

    # --- 3. Stretch-bend ---
    if len(params.strbend_idx1) > 0:
        si = params.strbend_idx1
        sj = params.strbend_idx2
        sk = params.strbend_idx3
        kba_ijk = params.strbend_kba_ijk.astype(np.float64)
        kba_kji = params.strbend_kba_kji.astype(np.float64)
        r0_ij = params.strbend_r0_ij.astype(np.float64)
        r0_kj = params.strbend_r0_kj.astype(np.float64)
        sb_theta0 = params.strbend_theta0.astype(np.float64) * DEG_TO_RAD

        vij = pos[:, si] - pos[:, sj]
        vkj = pos[:, sk] - pos[:, sj]
        rij = np.sqrt(np.sum(vij ** 2, axis=-1)).clip(min=EPS)
        rkj = np.sqrt(np.sum(vkj ** 2, axis=-1)).clip(min=EPS)
        dij_hat = vij / rij[..., None]
        dkj_hat = vkj / rkj[..., None]

        cos_th = np.clip(np.sum(vij * vkj, axis=-1) / (rij * rkj), -1.0, 1.0)
        theta_sb = np.arccos(cos_th)
        sin_th = np.sin(theta_sb).clip(min=EPS)

        dt_dvi = (cos_th[..., None] * vij / rij[..., None] - vkj / rkj[..., None]) / (rij[..., None] * sin_th[..., None])
        dt_dvk = (cos_th[..., None] * vkj / rkj[..., None] - vij / rij[..., None]) / (rkj[..., None] * sin_th[..., None])

        dr_ij = rij - r0_ij[None, :]
        dr_kj = rkj - r0_kj[None, :]
        dtheta = (theta_sb - sb_theta0[None, :]) * RAD_TO_DEG

        e_sb = STRETCH_BEND_CONST * (kba_ijk * dr_ij + kba_kji * dr_kj) * dtheta
        energies += e_sb.sum(axis=1)

        dE_dth = STRETCH_BEND_CONST * (kba_ijk * dr_ij + kba_kji * dr_kj) * RAD_TO_DEG
        dE_drij = STRETCH_BEND_CONST * kba_ijk * dtheta
        dE_drkj = STRETCH_BEND_CONST * kba_kji * dtheta

        gi_sb = dE_dth[..., None] * dt_dvi + dE_drij[..., None] * dij_hat
        gk_sb = dE_dth[..., None] * dt_dvk + dE_drkj[..., None] * dkj_hat
        gj_sb = -gi_sb - gk_sb
        np.add.at(grad, (np.arange(C)[:, None], si[None, :]), gi_sb)
        np.add.at(grad, (np.arange(C)[:, None], sj[None, :]), gj_sb)
        np.add.at(grad, (np.arange(C)[:, None], sk[None, :]), gk_sb)

    # --- 4. Out-of-plane bending (analytic gradient) ---
    if len(params.oop_idx1) > 0:
        oi1 = params.oop_idx1
        oi2 = params.oop_idx2
        oi3 = params.oop_idx3
        oi4 = params.oop_idx4
        koop = params.oop_koop.astype(np.float64)

        u = pos[:, oi1] - pos[:, oi2]  # (C, n_oop, 3)
        v = pos[:, oi3] - pos[:, oi2]
        w = pos[:, oi4] - pos[:, oi2]

        n_vec = np.cross(u, v)  # (C, n_oop, 3)
        A = np.sqrt(np.sum(n_vec ** 2, axis=-1)).clip(min=EPS)  # (C, n_oop)
        B = np.sqrt(np.sum(w ** 2, axis=-1)).clip(min=EPS)
        n_hat = n_vec / A[..., None]
        w_hat = w / B[..., None]

        f = np.sum(n_vec * w, axis=-1)  # scalar triple product (C, n_oop)
        AB = A * B
        sin_chi = np.clip(f / AB, -1.0, 1.0)
        chi_rad = np.arcsin(sin_chi)
        chi_deg = chi_rad * RAD_TO_DEG
        cos_chi = np.cos(chi_rad).clip(min=EPS)

        e_oop = ANGLE_BEND_CONST * koop / 2.0 * chi_deg ** 2
        energies += e_oop.sum(axis=1)

        # dE/dsin_chi = dE/dchi_deg * dchi_deg/dchi_rad * dchi_rad/dsin_chi
        dE_dS = ANGLE_BEND_CONST * koop * chi_deg * RAD_TO_DEG / cos_chi  # (C, n_oop)

        # Gradients of sin_chi w.r.t. atom positions:
        # dS/dp_i1 = (v×w)/(AB) - S*(v×n_hat)/A
        vxw = np.cross(v, w)  # (C, n_oop, 3)
        wxu = np.cross(w, u)
        vxn = np.cross(v, n_hat)
        nxu = np.cross(n_hat, u)

        dS_dp1 = vxw / AB[..., None] - sin_chi[..., None] * vxn / A[..., None]
        dS_dp3 = wxu / AB[..., None] - sin_chi[..., None] * nxu / A[..., None]
        dS_dp4 = (n_hat - sin_chi[..., None] * w_hat) / B[..., None]
        dS_dp2 = -(dS_dp1 + dS_dp3 + dS_dp4)  # translation invariance

        # Apply chain rule: dE/dp = dE/dS * dS/dp
        g1 = dE_dS[..., None] * dS_dp1
        g2 = dE_dS[..., None] * dS_dp2
        g3 = dE_dS[..., None] * dS_dp3
        g4 = dE_dS[..., None] * dS_dp4

        np.add.at(grad, (np.arange(C)[:, None], oi1[None, :]), g1)
        np.add.at(grad, (np.arange(C)[:, None], oi2[None, :]), g2)
        np.add.at(grad, (np.arange(C)[:, None], oi3[None, :]), g3)
        np.add.at(grad, (np.arange(C)[:, None], oi4[None, :]), g4)

    # --- 5. Torsion ---
    if len(params.torsion_idx1) > 0:
        ti1 = params.torsion_idx1
        ti2 = params.torsion_idx2
        ti3 = params.torsion_idx3
        ti4 = params.torsion_idx4
        tV1 = params.torsion_V1.astype(np.float64)
        tV2 = params.torsion_V2.astype(np.float64)
        tV3 = params.torsion_V3.astype(np.float64)

        b1 = pos[:, ti2] - pos[:, ti1]  # (C, n_torsions, 3)
        b2 = pos[:, ti3] - pos[:, ti2]
        b3 = pos[:, ti4] - pos[:, ti3]
        c1 = np.cross(b1, b2)
        c2 = np.cross(b2, b3)
        b2n = np.sqrt(np.sum(b2 ** 2, axis=-1)).clip(min=EPS)
        m_vec = np.cross(c1, b2 / b2n[..., None])
        n1n2 = np.sqrt(np.sum(c1 ** 2, axis=-1) * np.sum(c2 ** 2, axis=-1)).clip(min=EPS)
        sin_w = np.sum(m_vec * c2, axis=-1) / n1n2
        cos_w = np.sum(c1 * c2, axis=-1) / n1n2
        omega = np.arctan2(sin_w, cos_w)

        e_tor = 0.5 * (
            tV1 * (1.0 + np.cos(omega)) +
            tV2 * (1.0 - np.cos(2.0 * omega)) +
            tV3 * (1.0 + np.cos(3.0 * omega))
        )
        energies += e_tor.sum(axis=1)

        dE_dw = 0.5 * (
            -tV1 * np.sin(omega) +
            2.0 * tV2 * np.sin(2.0 * omega) -
            3.0 * tV3 * np.sin(3.0 * omega)
        )

        # Analytic torsion gradient (GROMACS/CHARMM formula)
        rij = pos[:, ti1] - pos[:, ti2]
        rkj = pos[:, ti3] - pos[:, ti2]
        rkl = pos[:, ti3] - pos[:, ti4]
        m_cr = np.cross(rij, rkj)
        n_cr = np.cross(rkj, rkl)
        m_sq = np.sum(m_cr ** 2, axis=-1).clip(min=EPS)
        n_sq = np.sum(n_cr ** 2, axis=-1).clip(min=EPS)
        rkj_n = np.sqrt(np.sum(rkj ** 2, axis=-1)).clip(min=EPS)
        rkj_n2 = rkj_n ** 2

        rij_dot_rkj = np.sum(rij * rkj, axis=-1)
        rkl_dot_rkj = np.sum(rkl * rkj, axis=-1)

        fi = (-rkj_n / m_sq)[..., None] * m_cr * dE_dw[..., None]
        fl = (rkj_n / n_sq)[..., None] * n_cr * dE_dw[..., None]

        proj_ij = (rij_dot_rkj / rkj_n2)[..., None]
        proj_kl = (rkl_dot_rkj / rkj_n2)[..., None]
        fj = -fi + proj_ij * fi - proj_kl * fl
        fk = -fl - proj_ij * fi + proj_kl * fl

        np.add.at(grad, (np.arange(C)[:, None], ti1[None, :]), fi)
        np.add.at(grad, (np.arange(C)[:, None], ti2[None, :]), fj)
        np.add.at(grad, (np.arange(C)[:, None], ti3[None, :]), fk)
        np.add.at(grad, (np.arange(C)[:, None], ti4[None, :]), fl)

    # --- 6. Van der Waals (Buffered 14-7) ---
    if len(params.vdw_idx1) > 0:
        vi1 = params.vdw_idx1
        vi2 = params.vdw_idx2
        R_star = params.vdw_R_star.astype(np.float64)
        vdw_eps = params.vdw_eps.astype(np.float64)

        d_vdw = pos[:, vi1] - pos[:, vi2]
        r_vdw = np.sqrt(np.sum(d_vdw ** 2, axis=-1)).clip(min=EPS)
        d_hat_vdw = d_vdw / r_vdw[..., None]

        rho = r_vdw / R_star[None, :]
        rho7 = rho ** 7
        t1 = 1.07 / (rho + 0.07)
        t2 = 1.12 / (rho7 + 0.12) - 2.0
        e_vdw = vdw_eps * t1 ** 7 * t2
        energies += e_vdw.sum(axis=1)

        dt1 = -1.07 / (rho + 0.07) ** 2
        dt2 = -1.12 * 7.0 * rho ** 6 / (rho7 + 0.12) ** 2
        dE_drho = vdw_eps * (7.0 * t1 ** 6 * dt1 * t2 + t1 ** 7 * dt2)
        dE_dr_vdw = dE_drho / R_star[None, :]

        f_vdw = dE_dr_vdw[..., None] * d_hat_vdw
        np.add.at(grad, (np.arange(C)[:, None], vi1[None, :]), f_vdw)
        np.add.at(grad, (np.arange(C)[:, None], vi2[None, :]), -f_vdw)

    # --- 7. Electrostatic (constant dielectric) ---
    if len(params.ele_idx1) > 0:
        ei1 = params.ele_idx1
        ei2 = params.ele_idx2
        ct = params.ele_charge_term.astype(np.float64)
        is14 = params.ele_is_1_4.astype(np.float64)
        scale = np.where(is14, 0.75, 1.0)

        d_ele = pos[:, ei1] - pos[:, ei2]
        r_ele = np.sqrt(np.sum(d_ele ** 2, axis=-1)).clip(min=EPS)
        d_hat_ele = d_ele / r_ele[..., None]

        denom = r_ele + ELE_DAMP
        e_ele = scale * ELE_CONST * ct / denom
        energies += e_ele.sum(axis=1)

        dE_dr_ele = -scale * ELE_CONST * ct / denom ** 2
        f_ele = dE_dr_ele[..., None] * d_hat_ele
        np.add.at(grad, (np.arange(C)[:, None], ei1[None, :]), f_ele)
        np.add.at(grad, (np.arange(C)[:, None], ei2[None, :]), -f_ele)

    return energies, grad.reshape(C, -1).astype(np.float32)
