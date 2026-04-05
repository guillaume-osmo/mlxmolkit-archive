"""
MMFF94 energy computation in MLX for Metal GPU acceleration.

Only the forward pass (energy) is implemented. Gradients are obtained
via mx.value_and_grad (automatic differentiation), which compiles to
fused Metal kernels. This replaces 324 lines of manual gradient code
with ~120 lines of pure forward-pass energy.

Architecture (mirrors nvMolKit):
  1. Parameters extracted once from RDKit (mmff_params.py)
  2. Converted to MLX arrays (MMFFParamsMLX)
  3. Energy computed on Metal via this module
  4. Gradient via mx.grad → compiled Metal backward pass
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlxmolkit.mmff_params import MMFFParams

MDYNE_A_TO_KCAL = 143.9325
DEG_TO_RAD = mx.array(3.14159265358979323846 / 180.0, dtype=mx.float32)
RAD_TO_DEG = mx.array(180.0 / 3.14159265358979323846, dtype=mx.float32)
ANGLE_BEND_CONST = 0.043844
STRETCH_BEND_CONST = 2.51210
ELE_CONST = 332.0716
ELE_DAMP = 0.05
BOND_CUBIC_CS = -2.0
EPS = 1e-12


@dataclass
class MMFFParamsMLX:
    """MMFF parameters as MLX arrays, ready for Metal compute."""

    n_atoms: int

    bond_idx1: mx.array
    bond_idx2: mx.array
    bond_kb: mx.array
    bond_r0: mx.array

    angle_idx1: mx.array
    angle_idx2: mx.array
    angle_idx3: mx.array
    angle_ka: mx.array
    angle_theta0_rad: mx.array

    strbend_idx1: mx.array
    strbend_idx2: mx.array
    strbend_idx3: mx.array
    strbend_kba_ijk: mx.array
    strbend_kba_kji: mx.array
    strbend_r0_ij: mx.array
    strbend_r0_kj: mx.array
    strbend_theta0_rad: mx.array

    oop_idx1: mx.array
    oop_idx2: mx.array
    oop_idx3: mx.array
    oop_idx4: mx.array
    oop_koop: mx.array

    torsion_idx1: mx.array
    torsion_idx2: mx.array
    torsion_idx3: mx.array
    torsion_idx4: mx.array
    torsion_V1: mx.array
    torsion_V2: mx.array
    torsion_V3: mx.array

    vdw_idx1: mx.array
    vdw_idx2: mx.array
    vdw_R_star: mx.array
    vdw_eps: mx.array

    ele_idx1: mx.array
    ele_idx2: mx.array
    ele_charge_term: mx.array
    ele_scale: mx.array


def params_to_mlx(p: MMFFParams) -> MMFFParamsMLX:
    """Convert numpy MMFFParams to MLX arrays (one-time GPU upload)."""
    f = mx.float32

    ele_is14 = mx.array(p.ele_is_1_4, dtype=f)
    ele_scale = mx.where(ele_is14 > 0.5, mx.array(0.75, dtype=f), mx.array(1.0, dtype=f))

    return MMFFParamsMLX(
        n_atoms=p.n_atoms,
        bond_idx1=mx.array(p.bond_idx1),
        bond_idx2=mx.array(p.bond_idx2),
        bond_kb=mx.array(p.bond_kb, dtype=f),
        bond_r0=mx.array(p.bond_r0, dtype=f),
        angle_idx1=mx.array(p.angle_idx1),
        angle_idx2=mx.array(p.angle_idx2),
        angle_idx3=mx.array(p.angle_idx3),
        angle_ka=mx.array(p.angle_ka, dtype=f),
        angle_theta0_rad=mx.array(p.angle_theta0, dtype=f) * DEG_TO_RAD,
        strbend_idx1=mx.array(p.strbend_idx1),
        strbend_idx2=mx.array(p.strbend_idx2),
        strbend_idx3=mx.array(p.strbend_idx3),
        strbend_kba_ijk=mx.array(p.strbend_kba_ijk, dtype=f),
        strbend_kba_kji=mx.array(p.strbend_kba_kji, dtype=f),
        strbend_r0_ij=mx.array(p.strbend_r0_ij, dtype=f),
        strbend_r0_kj=mx.array(p.strbend_r0_kj, dtype=f),
        strbend_theta0_rad=mx.array(p.strbend_theta0, dtype=f) * DEG_TO_RAD,
        oop_idx1=mx.array(p.oop_idx1),
        oop_idx2=mx.array(p.oop_idx2),
        oop_idx3=mx.array(p.oop_idx3),
        oop_idx4=mx.array(p.oop_idx4),
        oop_koop=mx.array(p.oop_koop, dtype=f),
        torsion_idx1=mx.array(p.torsion_idx1),
        torsion_idx2=mx.array(p.torsion_idx2),
        torsion_idx3=mx.array(p.torsion_idx3),
        torsion_idx4=mx.array(p.torsion_idx4),
        torsion_V1=mx.array(p.torsion_V1, dtype=f),
        torsion_V2=mx.array(p.torsion_V2, dtype=f),
        torsion_V3=mx.array(p.torsion_V3, dtype=f),
        vdw_idx1=mx.array(p.vdw_idx1),
        vdw_idx2=mx.array(p.vdw_idx2),
        vdw_R_star=mx.array(p.vdw_R_star, dtype=f),
        vdw_eps=mx.array(p.vdw_eps, dtype=f),
        ele_idx1=mx.array(p.ele_idx1),
        ele_idx2=mx.array(p.ele_idx2),
        ele_charge_term=mx.array(p.ele_charge_term, dtype=f),
        ele_scale=ele_scale,
    )


def _cross(a: mx.array, b: mx.array) -> mx.array:
    """Cross product along last axis. Works for any leading batch dims."""
    return mx.linalg.cross(a, b)


def _norm(v: mx.array) -> mx.array:
    """L2 norm along last axis, with epsilon."""
    return mx.sqrt(mx.sum(v * v, axis=-1) + EPS)


def mmff_energy_batch(params: MMFFParamsMLX, positions: mx.array) -> mx.array:
    """
    Compute MMFF94 total energy for a batch of conformers.

    Args:
        params: MMFFParamsMLX (shared topology, on GPU).
        positions: (C, N, 3) float32 atom positions.

    Returns:
        (C,) float32 per-conformer energies.
    """
    C = positions.shape[0]
    energies = mx.zeros((C,), dtype=mx.float32)

    # --- 1. Bond stretch ---
    if params.bond_idx1.size > 0:
        p1 = positions[:, params.bond_idx1]
        p2 = positions[:, params.bond_idx2]
        d = p1 - p2
        r = _norm(d)
        dr = r - params.bond_r0
        cs = BOND_CUBIC_CS
        e_bond = MDYNE_A_TO_KCAL * params.bond_kb * 0.5 * dr * dr * (
            1.0 + cs * dr + (7.0 / 12.0) * cs * cs * dr * dr
        )
        energies = energies + mx.sum(e_bond, axis=1)

    # --- 2. Angle bend ---
    if params.angle_idx1.size > 0:
        v1 = positions[:, params.angle_idx1] - positions[:, params.angle_idx2]
        v2 = positions[:, params.angle_idx3] - positions[:, params.angle_idx2]
        r1 = _norm(v1)
        r2 = _norm(v2)
        cos_theta = mx.clip(
            mx.sum(v1 * v2, axis=-1) / (r1 * r2), -1.0 + 1e-7, 1.0 - 1e-7
        )
        theta = mx.arccos(cos_theta)
        dt = theta - params.angle_theta0_rad
        dt_deg = dt * RAD_TO_DEG
        cb = -0.007
        e_angle = ANGLE_BEND_CONST * params.angle_ka * 0.5 * dt_deg * dt_deg * (
            1.0 + cb * dt_deg
        )
        energies = energies + mx.sum(e_angle, axis=1)

    # --- 3. Stretch-bend ---
    if params.strbend_idx1.size > 0:
        vij = positions[:, params.strbend_idx1] - positions[:, params.strbend_idx2]
        vkj = positions[:, params.strbend_idx3] - positions[:, params.strbend_idx2]
        rij = _norm(vij)
        rkj = _norm(vkj)
        cos_th = mx.clip(
            mx.sum(vij * vkj, axis=-1) / (rij * rkj), -1.0 + 1e-7, 1.0 - 1e-7
        )
        theta_sb = mx.arccos(cos_th)
        dr_ij = rij - params.strbend_r0_ij
        dr_kj = rkj - params.strbend_r0_kj
        dtheta_deg = (theta_sb - params.strbend_theta0_rad) * RAD_TO_DEG
        e_sb = STRETCH_BEND_CONST * (
            params.strbend_kba_ijk * dr_ij + params.strbend_kba_kji * dr_kj
        ) * dtheta_deg
        energies = energies + mx.sum(e_sb, axis=1)

    # --- 4. Out-of-plane bending ---
    if params.oop_idx1.size > 0:
        u = positions[:, params.oop_idx1] - positions[:, params.oop_idx2]
        v = positions[:, params.oop_idx3] - positions[:, params.oop_idx2]
        w = positions[:, params.oop_idx4] - positions[:, params.oop_idx2]
        n_vec = _cross(u, v)
        A = _norm(n_vec)
        B = _norm(w)
        sin_chi = mx.clip(mx.sum(n_vec * w, axis=-1) / (A * B), -1.0, 1.0)
        chi_deg = mx.arcsin(sin_chi) * RAD_TO_DEG
        e_oop = ANGLE_BEND_CONST * params.oop_koop * 0.5 * chi_deg * chi_deg
        energies = energies + mx.sum(e_oop, axis=1)

    # --- 5. Torsion ---
    if params.torsion_idx1.size > 0:
        b1 = positions[:, params.torsion_idx2] - positions[:, params.torsion_idx1]
        b2 = positions[:, params.torsion_idx3] - positions[:, params.torsion_idx2]
        b3 = positions[:, params.torsion_idx4] - positions[:, params.torsion_idx3]
        c1 = _cross(b1, b2)
        c2 = _cross(b2, b3)
        b2n = _norm(b2)
        m_vec = _cross(c1, b2 / mx.expand_dims(b2n, -1))
        n1n2 = mx.sqrt(mx.sum(c1 * c1, axis=-1) * mx.sum(c2 * c2, axis=-1) + EPS)
        sin_w = mx.sum(m_vec * c2, axis=-1) / n1n2
        cos_w = mx.sum(c1 * c2, axis=-1) / n1n2
        omega = mx.arctan2(sin_w, cos_w)
        e_tor = 0.5 * (
            params.torsion_V1 * (1.0 + mx.cos(omega))
            + params.torsion_V2 * (1.0 - mx.cos(2.0 * omega))
            + params.torsion_V3 * (1.0 + mx.cos(3.0 * omega))
        )
        energies = energies + mx.sum(e_tor, axis=1)

    # --- 6. Van der Waals (Buffered 14-7) ---
    if params.vdw_idx1.size > 0:
        d_vdw = positions[:, params.vdw_idx1] - positions[:, params.vdw_idx2]
        r_vdw = _norm(d_vdw)
        rho = r_vdw / params.vdw_R_star
        rho7 = rho ** 7
        t1 = 1.07 / (rho + 0.07)
        t2 = 1.12 / (rho7 + 0.12) - 2.0
        e_vdw = params.vdw_eps * t1 ** 7 * t2
        energies = energies + mx.sum(e_vdw, axis=1)

    # --- 7. Electrostatic (constant dielectric) ---
    if params.ele_idx1.size > 0:
        d_ele = positions[:, params.ele_idx1] - positions[:, params.ele_idx2]
        r_ele = _norm(d_ele)
        e_ele = params.ele_scale * ELE_CONST * params.ele_charge_term / (r_ele + ELE_DAMP)
        energies = energies + mx.sum(e_ele, axis=1)

    return energies


def make_energy_and_grad_fn(
    params: MMFFParamsMLX,
) -> callable:
    """
    Create a compiled energy+gradient function for the given parameters.

    Returns a function: (C, N, 3) -> ((C,) energies, (C, N, 3) gradients)
    that runs entirely on Metal.

    The gradient is computed via mx.value_and_grad on the total (summed)
    energy. Since E_total = sum_c E_c and each E_c only depends on
    positions[c], the resulting gradient equals the per-conformer gradients.
    """

    def total_energy(positions: mx.array) -> mx.array:
        return mx.sum(mmff_energy_batch(params, positions))

    compiled_vg = mx.compile(mx.value_and_grad(total_energy))

    def energy_and_grad(positions: mx.array) -> tuple[mx.array, mx.array]:
        """Returns (per_conf_energies, per_conf_gradients)."""
        _, grad = compiled_vg(positions)
        per_conf_e = mmff_energy_batch(params, positions)
        return per_conf_e, grad

    return energy_and_grad
