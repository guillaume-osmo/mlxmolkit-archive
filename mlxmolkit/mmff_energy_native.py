"""
MMFF94 energy and gradient computation in pure numpy.

No RDKit callback during optimization. Parameters come from mmff_params.py.
This implements all 7 MMFF energy terms with analytic gradients.

References:
  - Halgren, J. Comput. Chem. 1996, 17, 490-519
  - https://www.charmm-gui.org/charmmdoc/mmff.html
  - nvMolKit mmff_kernels_device.cuh
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


def _dist_and_grad(pos: np.ndarray, i: int, j: int):
    """Distance r_ij and gradient dr/dx for atoms i, j."""
    d = pos[i] - pos[j]
    r = np.linalg.norm(d) + 1e-30
    return r, d / r


def _angle_and_grad(pos: np.ndarray, i: int, j: int, k: int):
    """
    Angle i-j-k in radians and gradients dtheta/dx_i, dtheta/dx_j, dtheta/dx_k.
    """
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    r1 = np.linalg.norm(v1) + 1e-30
    r2 = np.linalg.norm(v2) + 1e-30
    cos_theta = np.clip(np.dot(v1, v2) / (r1 * r2), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta) + 1e-30

    dt_dv1 = (cos_theta * v1 / r1 - v2 / r2) / (r1 * sin_theta)
    dt_dv2 = (cos_theta * v2 / r2 - v1 / r1) / (r2 * sin_theta)
    return theta, dt_dv1, -dt_dv1 - dt_dv2, dt_dv2


def _torsion_angle(pos: np.ndarray, i1: int, i2: int, i3: int, i4: int) -> float:
    """Compute torsion (dihedral) angle for atoms i1-i2-i3-i4."""
    b1 = pos[i2] - pos[i1]
    b2 = pos[i3] - pos[i2]
    b3 = pos[i4] - pos[i3]
    c1 = np.cross(b1, b2)
    c2 = np.cross(b2, b3)
    b2n = np.linalg.norm(b2) + 1e-30
    m = np.cross(c1, b2 / b2n)
    n1n2 = np.sqrt(np.dot(c1, c1) * np.dot(c2, c2)) + 1e-30
    return np.arctan2(np.dot(m, c2) / n1n2, np.dot(c1, c2) / n1n2)


def _torsion_grad_accumulate(
    pos: np.ndarray,
    i1: int, i2: int, i3: int, i4: int,
    dE_domega: float,
    grad: np.ndarray,
) -> None:
    """
    Accumulate torsion gradient into grad array using the GROMACS/CHARMM
    standard analytic formula.

    Reference: Bekker et al., J. Comput. Chem. 1995; GROMACS manual Appendix.
    """
    rij = pos[i1] - pos[i2]
    rkj = pos[i3] - pos[i2]
    rkl = pos[i3] - pos[i4]

    m = np.cross(rij, rkj)
    n = np.cross(rkj, rkl)

    m_sq = np.dot(m, m) + 1e-30
    n_sq = np.dot(n, n) + 1e-30
    rkj_norm = np.linalg.norm(rkj) + 1e-30

    fi = -rkj_norm / m_sq * m * dE_domega
    fl = rkj_norm / n_sq * n * dE_domega

    uvec = rij - np.dot(rij, rkj) / (rkj_norm ** 2) * rkj
    wvec = rkl - np.dot(rkl, rkj) / (rkj_norm ** 2) * rkj

    fj = -fi + np.dot(rij, rkj) / (rkj_norm ** 2) * fi - np.dot(rkl, rkj) / (rkj_norm ** 2) * fl
    fk = -fl - np.dot(rij, rkj) / (rkj_norm ** 2) * fi + np.dot(rkl, rkj) / (rkj_norm ** 2) * fl

    grad[i1] += fi
    grad[i2] += fj
    grad[i3] += fk
    grad[i4] += fl


def _wilson_angle_and_grad(pos: np.ndarray, i1: int, i2: int, i3: int, i4: int):
    """
    Wilson out-of-plane angle chi for i4 relative to plane i1-i2-i3.
    i2 is the central atom. chi = arcsin(n . (r24) / |r24|).
    Returns (chi_deg, numerical gradients).
    """
    v21 = pos[i1] - pos[i2]
    v23 = pos[i3] - pos[i2]
    n = np.cross(v21, v23)
    n_norm = np.linalg.norm(n) + 1e-30
    n_hat = n / n_norm

    v24 = pos[i4] - pos[i2]
    r24 = np.linalg.norm(v24) + 1e-30
    sin_chi = np.clip(np.dot(n_hat, v24) / r24, -1.0, 1.0)
    chi = np.arcsin(sin_chi) * RAD_TO_DEG

    eps = 1e-5
    grads = []
    for idx in [i1, i2, i3, i4]:
        g = np.zeros(3)
        for c in range(3):
            pos_p, pos_m = pos.copy(), pos.copy()
            pos_p[idx, c] += eps
            pos_m[idx, c] -= eps

            def _chi(p):
                _v21 = p[i1] - p[i2]
                _v23 = p[i3] - p[i2]
                _n = np.cross(_v21, _v23)
                _nn = np.linalg.norm(_n) + 1e-30
                _v24 = p[i4] - p[i2]
                _r24 = np.linalg.norm(_v24) + 1e-30
                return np.arcsin(np.clip(np.dot(_n / _nn, _v24) / _r24, -1, 1)) * RAD_TO_DEG

            g[c] = (_chi(pos_p) - _chi(pos_m)) / (2 * eps)
        grads.append(g)

    return chi, grads[0], grads[1], grads[2], grads[3]


def mmff_energy_grad(
    params: MMFFParams,
    pos_flat: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Compute MMFF94 energy and gradient from pre-extracted parameters.

    Args:
        params: MMFFParams from extract_mmff_params().
        pos_flat: atom positions, shape (n_atoms*3,), float64.

    Returns:
        (energy, gradient) where energy is float and gradient is (n_atoms*3,) float64.
    """
    pos = pos_flat.reshape(-1, 3).astype(np.float64)
    grad = np.zeros_like(pos)
    energy = 0.0

    # --- 1. Bond stretch ---
    # E = MDYNE_A_TO_KCAL * kb/2 * dr^2 * (1 + cs*dr + 7/12*cs^2*dr^2)
    cs = BOND_CUBIC_CS
    for t in range(len(params.bond_idx1)):
        i, j = int(params.bond_idx1[t]), int(params.bond_idx2[t])
        kb, r0 = float(params.bond_kb[t]), float(params.bond_r0[t])
        r, dr_dx = _dist_and_grad(pos, i, j)
        dr = r - r0
        e = MDYNE_A_TO_KCAL * kb / 2.0 * dr ** 2 * (1.0 + cs * dr + 7.0 / 12.0 * cs ** 2 * dr ** 2)
        dE_dr = MDYNE_A_TO_KCAL * kb * dr * (1.0 + 1.5 * cs * dr + 7.0 / 3.0 * cs ** 2 * dr ** 2 / 2.0)
        energy += e
        grad[i] += dE_dr * dr_dx
        grad[j] -= dE_dr * dr_dx

    # --- 2. Angle bend ---
    for t in range(len(params.angle_idx1)):
        i = int(params.angle_idx1[t])
        j = int(params.angle_idx2[t])
        k = int(params.angle_idx3[t])
        ka = float(params.angle_ka[t])
        theta0 = float(params.angle_theta0[t]) * DEG_TO_RAD
        is_linear = int(params.angle_is_linear[t])

        theta, dt_di, dt_dj, dt_dk = _angle_and_grad(pos, i, j, k)

        if is_linear:
            e = MDYNE_A_TO_KCAL * ka * (1.0 + np.cos(theta))
            dE_dtheta = -MDYNE_A_TO_KCAL * ka * np.sin(theta)
        else:
            d_theta = theta - theta0
            cb = -0.007
            e = ANGLE_BEND_CONST * ka / 2.0 * (d_theta * RAD_TO_DEG) ** 2 * (1.0 + cb * d_theta * RAD_TO_DEG)
            dE_dtheta = ANGLE_BEND_CONST * ka * d_theta * RAD_TO_DEG * (
                1.0 + 1.5 * cb * d_theta * RAD_TO_DEG
            ) * RAD_TO_DEG

        energy += e
        grad[i] += dE_dtheta * dt_di
        grad[j] += dE_dtheta * dt_dj
        grad[k] += dE_dtheta * dt_dk

    # --- 3. Stretch-bend ---
    for t in range(len(params.strbend_idx1)):
        i = int(params.strbend_idx1[t])
        j = int(params.strbend_idx2[t])
        k = int(params.strbend_idx3[t])
        kba_ijk = float(params.strbend_kba_ijk[t])
        kba_kji = float(params.strbend_kba_kji[t])
        r0_ij = float(params.strbend_r0_ij[t])
        r0_kj = float(params.strbend_r0_kj[t])
        theta0 = float(params.strbend_theta0[t]) * DEG_TO_RAD

        r_ij, dr_ij_dx = _dist_and_grad(pos, i, j)
        r_kj, dr_kj_dx = _dist_and_grad(pos, k, j)
        theta, dt_di, dt_dj, dt_dk = _angle_and_grad(pos, i, j, k)

        delta_r_ij = r_ij - r0_ij
        delta_r_kj = r_kj - r0_kj
        delta_theta = (theta - theta0) * RAD_TO_DEG

        e = STRETCH_BEND_CONST * (kba_ijk * delta_r_ij + kba_kji * delta_r_kj) * delta_theta
        energy += e

        dE_d_delta_theta = STRETCH_BEND_CONST * (kba_ijk * delta_r_ij + kba_kji * delta_r_kj) * RAD_TO_DEG
        dE_d_dr_ij = STRETCH_BEND_CONST * kba_ijk * delta_theta
        dE_d_dr_kj = STRETCH_BEND_CONST * kba_kji * delta_theta

        grad[i] += dE_d_delta_theta * dt_di + dE_d_dr_ij * dr_ij_dx
        grad[j] += dE_d_delta_theta * dt_dj - dE_d_dr_ij * dr_ij_dx - dE_d_dr_kj * dr_kj_dx
        grad[k] += dE_d_delta_theta * dt_dk + dE_d_dr_kj * dr_kj_dx

    # --- 4. Out-of-plane bending ---
    for t in range(len(params.oop_idx1)):
        i1 = int(params.oop_idx1[t])
        i2 = int(params.oop_idx2[t])
        i3 = int(params.oop_idx3[t])
        i4 = int(params.oop_idx4[t])
        koop = float(params.oop_koop[t])

        chi, g1, g2, g3, g4 = _wilson_angle_and_grad(pos, i1, i2, i3, i4)
        e = ANGLE_BEND_CONST * koop / 2.0 * chi ** 2
        dE_dchi = ANGLE_BEND_CONST * koop * chi
        energy += e
        grad[i1] += dE_dchi * g1
        grad[i2] += dE_dchi * g2
        grad[i3] += dE_dchi * g3
        grad[i4] += dE_dchi * g4

    # --- 5. Torsion ---
    for t in range(len(params.torsion_idx1)):
        i1 = int(params.torsion_idx1[t])
        i2 = int(params.torsion_idx2[t])
        i3 = int(params.torsion_idx3[t])
        i4 = int(params.torsion_idx4[t])
        V1 = float(params.torsion_V1[t])
        V2 = float(params.torsion_V2[t])
        V3 = float(params.torsion_V3[t])

        omega = _torsion_angle(pos, i1, i2, i3, i4)
        e = 0.5 * (V1 * (1.0 + np.cos(omega)) + V2 * (1.0 - np.cos(2.0 * omega)) + V3 * (1.0 + np.cos(3.0 * omega)))
        energy += e

        dE_domega = 0.5 * (-V1 * np.sin(omega) + 2.0 * V2 * np.sin(2.0 * omega) - 3.0 * V3 * np.sin(3.0 * omega))
        _torsion_grad_accumulate(pos, i1, i2, i3, i4, dE_domega, grad)

    # --- 6. Van der Waals (Buffered 14-7) ---
    for t in range(len(params.vdw_idx1)):
        i = int(params.vdw_idx1[t])
        j = int(params.vdw_idx2[t])
        R_star = float(params.vdw_R_star[t])
        eps = float(params.vdw_eps[t])

        r, dr_dx = _dist_and_grad(pos, i, j)
        rho = r / R_star
        rho7 = rho ** 7
        term1 = 1.07 / (rho + 0.07)
        term2 = 1.12 / (rho7 + 0.12) - 2.0
        e = eps * term1 ** 7 * term2
        energy += e

        dt1_drho = -1.07 / (rho + 0.07) ** 2
        dt2_drho = -1.12 * 7.0 * rho ** 6 / (rho7 + 0.12) ** 2
        dE_drho = eps * (7.0 * term1 ** 6 * dt1_drho * term2 + term1 ** 7 * dt2_drho)
        dE_dr = dE_drho / R_star
        grad[i] += dE_dr * dr_dx
        grad[j] -= dE_dr * dr_dx

    # --- 7. Electrostatic (constant dielectric, MMFF94 default) ---
    for t in range(len(params.ele_idx1)):
        i = int(params.ele_idx1[t])
        j = int(params.ele_idx2[t])
        charge_term = float(params.ele_charge_term[t])
        is_1_4 = int(params.ele_is_1_4[t])

        r, dr_dx = _dist_and_grad(pos, i, j)
        scale = 0.75 if is_1_4 else 1.0

        denom = r + ELE_DAMP
        e = scale * ELE_CONST * charge_term / denom
        energy += e

        dE_dr = -scale * ELE_CONST * charge_term / denom ** 2
        grad[i] += dE_dr * dr_dx
        grad[j] -= dE_dr * dr_dx

    return energy, grad.flatten().astype(np.float32)


def make_mmff_native_energy_grad(params: MMFFParams):
    """
    Create energy/gradient function from pre-extracted MMFF parameters.

    Returns a callable(pos_flat_f32) -> (energy_f64, grad_f32) that
    does NOT call RDKit. Can be used directly with the BFGS optimizer
    by wrapping in mx.array.
    """
    def fn(pos_flat: np.ndarray) -> tuple[float, np.ndarray]:
        return mmff_energy_grad(params, pos_flat)
    return fn
