"""
3D ETK (Experimental Torsion Knowledge) energy and gradient on Metal.

Three energy terms for stage 5 of ETKDG:
  1. CSD torsion preferences: 6-term Fourier series
     E = Σ_{k=1}^{6} V_k * (1 + sign_k * cos(k * φ)) / 2
  2. Improper torsion (planarity): E = w * (1 - cos(2ω))
  3. 1-4 distance constraints: flat-bottom with harmonic penalty

All threads compute both energy AND gradient (parallel scatter).
One thread per (global_atom, 3D_coord).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from mlxmolkit.etk_extract import BatchedETKSystem

# fmt: off
_ETK_ENERGY_GRAD_SOURCE = """
uint tid = thread_position_in_grid.x;
uint n_atoms_total = params_buf[0];
uint n_mols = params_buf[1];

if (tid >= n_atoms_total * 3u) {{ return; }}

uint global_atom = tid / 3u;
uint coord = tid % 3u;

// Find molecule
uint mol_id = 0;
for (uint mm = 0; mm < n_mols; mm++) {{
    if (global_atom < (uint)atom_starts[mm + 1]) {{
        mol_id = mm;
        break;
    }}
}}

float grad_val = 0.0f;
float e_contrib = 0.0f;

// ======== CSD Torsion terms ========
uint tor_start = (uint)torsion_starts[mol_id];
uint tor_end   = (uint)torsion_starts[mol_id + 1];

for (uint ti = tor_start; ti < tor_end; ti++) {{
    uint i0 = (uint)torsion_idx[ti * 4 + 0];
    uint i1 = (uint)torsion_idx[ti * 4 + 1];
    uint i2 = (uint)torsion_idx[ti * 4 + 2];
    uint i3 = (uint)torsion_idx[ti * 4 + 3];

    bool involved = (global_atom == i0 || global_atom == i1 ||
                     global_atom == i2 || global_atom == i3);
    if (!involved) continue;

    // Vectors along the torsion
    float b1x = pos[i1*3+0] - pos[i0*3+0];
    float b1y = pos[i1*3+1] - pos[i0*3+1];
    float b1z = pos[i1*3+2] - pos[i0*3+2];

    float b2x = pos[i2*3+0] - pos[i1*3+0];
    float b2y = pos[i2*3+1] - pos[i1*3+1];
    float b2z = pos[i2*3+2] - pos[i1*3+2];

    float b3x = pos[i3*3+0] - pos[i2*3+0];
    float b3y = pos[i3*3+1] - pos[i2*3+1];
    float b3z = pos[i3*3+2] - pos[i2*3+2];

    // n1 = b1 × b2, n2 = b2 × b3
    float n1x = b1y*b2z - b1z*b2y;
    float n1y = b1z*b2x - b1x*b2z;
    float n1z = b1x*b2y - b1y*b2x;

    float n2x = b2y*b3z - b2z*b3y;
    float n2y = b2z*b3x - b2x*b3z;
    float n2z = b2x*b3y - b2y*b3x;

    float n1_len = sqrt(n1x*n1x + n1y*n1y + n1z*n1z + 1e-12f);
    float n2_len = sqrt(n2x*n2x + n2y*n2y + n2z*n2z + 1e-12f);
    float b2_len = sqrt(b2x*b2x + b2y*b2y + b2z*b2z + 1e-12f);

    float cos_phi = (n1x*n2x + n1y*n2y + n1z*n2z) / (n1_len * n2_len);
    cos_phi = clamp(cos_phi, -1.0f, 1.0f);

    // m1 = n1 × b2_hat
    float b2hx = b2x / b2_len, b2hy = b2y / b2_len, b2hz = b2z / b2_len;
    float m1x = n1y*b2hz - n1z*b2hy;
    float m1y = n1z*b2hx - n1x*b2hz;
    float m1z = n1x*b2hy - n1y*b2hx;

    float sin_phi = (m1x*n2x + m1y*n2y + m1z*n2z) / (n1_len * n2_len);
    float phi = atan2(sin_phi, cos_phi);

    // 6-term Fourier: E = Σ V_k * (1 + sign_k * cos(k*φ)) / 2
    float E_tor = 0.0f;
    float dE_dphi = 0.0f;
    for (uint k = 0; k < 6u; k++) {{
        float Vk = torsion_V[ti * 6 + k];
        if (Vk == 0.0f) continue;
        float sk = (float)torsion_signs[ti * 6 + k];
        float kf = (float)(k + 1);
        E_tor += Vk * (1.0f + sk * cos(kf * phi)) * 0.5f;
        dE_dphi += Vk * (-sk * kf * sin(kf * phi)) * 0.5f;
    }}

    if (coord == 0u && global_atom == i0) e_contrib += E_tor;

    // Gradient: dE/dpos = dE/dphi * dphi/dpos
    // Using the standard torsion gradient formulas
    float n1_sq = n1x*n1x + n1y*n1y + n1z*n1z + 1e-12f;
    float n2_sq = n2x*n2x + n2y*n2y + n2z*n2z + 1e-12f;

    // dphi/dp0 = -b2_len / n1²  * n1
    // dphi/dp3 =  b2_len / n2²  * n2
    // dphi/dp1 = (b1·b2/(b2²)-1) * dphi/dp0 - (b3·b2/b2²) * dphi/dp3
    // dphi/dp2 = (b3·b2/(b2²)-1) * dphi/dp3 - (b1·b2/b2²) * dphi/dp0

    float b2_sq = b2x*b2x + b2y*b2y + b2z*b2z + 1e-12f;
    float b1_dot_b2 = b1x*b2x + b1y*b2y + b1z*b2z;
    float b3_dot_b2 = b3x*b2x + b3y*b2y + b3z*b2z;

    float f0 = -b2_len / n1_sq;
    float f3 =  b2_len / n2_sq;
    float f1a = b1_dot_b2 / b2_sq - 1.0f;
    float f1b = -b3_dot_b2 / b2_sq;
    float f2a = b3_dot_b2 / b2_sq - 1.0f;
    float f2b = -b1_dot_b2 / b2_sq;

    float dp0, dp1, dp2, dp3;
    if (coord == 0u) {{
        dp0 = f0 * n1x; dp3 = f3 * n2x;
    }} else if (coord == 1u) {{
        dp0 = f0 * n1y; dp3 = f3 * n2y;
    }} else {{
        dp0 = f0 * n1z; dp3 = f3 * n2z;
    }}
    dp1 = f1a * dp0 + f1b * dp3;
    dp2 = f2a * dp3 + f2b * dp0;

    if (global_atom == i0) grad_val += dE_dphi * dp0;
    else if (global_atom == i1) grad_val += dE_dphi * dp1;
    else if (global_atom == i2) grad_val += dE_dphi * dp2;
    else grad_val += dE_dphi * dp3;
}}

// ======== Improper torsion (planarity) ========
uint imp_start = (uint)improper_starts[mol_id];
uint imp_end   = (uint)improper_starts[mol_id + 1];

for (uint ii = imp_start; ii < imp_end; ii++) {{
    uint ic = (uint)improper_i[ii * 4 + 0];
    uint i0 = (uint)improper_i[ii * 4 + 1];
    uint i1 = (uint)improper_i[ii * 4 + 2];
    uint i2 = (uint)improper_i[ii * 4 + 3];

    bool involved = (global_atom == ic || global_atom == i0 ||
                     global_atom == i1 || global_atom == i2);
    if (!involved) continue;

    // Improper dihedral: ic-i0-i1-i2
    float b1x = pos[i0*3+0]-pos[ic*3+0], b1y = pos[i0*3+1]-pos[ic*3+1], b1z = pos[i0*3+2]-pos[ic*3+2];
    float b2x = pos[i1*3+0]-pos[i0*3+0], b2y = pos[i1*3+1]-pos[i0*3+1], b2z = pos[i1*3+2]-pos[i0*3+2];
    float b3x = pos[i2*3+0]-pos[i1*3+0], b3y = pos[i2*3+1]-pos[i1*3+1], b3z = pos[i2*3+2]-pos[i1*3+2];

    float n1x = b1y*b2z - b1z*b2y;
    float n1y = b1z*b2x - b1x*b2z;
    float n1z = b1x*b2y - b1y*b2x;

    float n2x = b2y*b3z - b2z*b3y;
    float n2y = b2z*b3x - b2x*b3z;
    float n2z = b2x*b3y - b2y*b3x;

    float n1_len = sqrt(n1x*n1x + n1y*n1y + n1z*n1z + 1e-12f);
    float n2_len = sqrt(n2x*n2x + n2y*n2y + n2z*n2z + 1e-12f);
    float b2_len = sqrt(b2x*b2x + b2y*b2y + b2z*b2z + 1e-12f);

    float cos_w = (n1x*n2x + n1y*n2y + n1z*n2z) / (n1_len * n2_len);
    cos_w = clamp(cos_w, -1.0f, 1.0f);

    float b2hx = b2x/b2_len, b2hy = b2y/b2_len, b2hz = b2z/b2_len;
    float m1x = n1y*b2hz - n1z*b2hy;
    float m1y = n1z*b2hx - n1x*b2hz;
    float m1z = n1x*b2hy - n1y*b2hx;
    float sin_w = (m1x*n2x + m1y*n2y + m1z*n2z) / (n1_len * n2_len);
    float omega = atan2(sin_w, cos_w);

    float w = improper_w[ii];

    // E = w * (1 - cos(2ω))
    float E_imp = w * (1.0f - cos(2.0f * omega));
    if (coord == 0u && global_atom == ic) e_contrib += E_imp;

    float dE_dw = w * 2.0f * sin(2.0f * omega);

    float n1_sq = n1x*n1x + n1y*n1y + n1z*n1z + 1e-12f;
    float n2_sq = n2x*n2x + n2y*n2y + n2z*n2z + 1e-12f;
    float b2_sq = b2x*b2x + b2y*b2y + b2z*b2z + 1e-12f;

    float f0 = -b2_len / n1_sq;
    float f3 =  b2_len / n2_sq;
    float b1_dot_b2 = b1x*b2x + b1y*b2y + b1z*b2z;
    float b3_dot_b2 = b3x*b2x + b3y*b2y + b3z*b2z;
    float f1a = b1_dot_b2 / b2_sq - 1.0f;
    float f1b = -b3_dot_b2 / b2_sq;
    float f2a = b3_dot_b2 / b2_sq - 1.0f;
    float f2b = -b1_dot_b2 / b2_sq;

    float dp0, dp1, dp2, dp3;
    if (coord == 0u) {{ dp0 = f0*n1x; dp3 = f3*n2x; }}
    else if (coord == 1u) {{ dp0 = f0*n1y; dp3 = f3*n2y; }}
    else {{ dp0 = f0*n1z; dp3 = f3*n2z; }}
    dp1 = f1a * dp0 + f1b * dp3;
    dp2 = f2a * dp3 + f2b * dp0;

    if (global_atom == ic)  grad_val += dE_dw * dp0;
    else if (global_atom == i0) grad_val += dE_dw * dp1;
    else if (global_atom == i1) grad_val += dE_dw * dp2;
    else grad_val += dE_dw * dp3;
}}

// ======== 1-4 Distance constraints ========
uint d14_start = (uint)dist14_starts[mol_id];
uint d14_end   = (uint)dist14_starts[mol_id + 1];

for (uint di = d14_start; di < d14_end; di++) {{
    uint a = (uint)d14_i1[di];
    uint b = (uint)d14_i2[di];

    bool is_a = (a == global_atom);
    bool is_b = (b == global_atom);
    if (!is_a && !is_b) continue;

    float dx = pos[a*3+0] - pos[b*3+0];
    float dy = pos[a*3+1] - pos[b*3+1];
    float dz = pos[a*3+2] - pos[b*3+2];
    float d = sqrt(dx*dx + dy*dy + dz*dz + 1e-12f);

    float lb = d14_lb_arr[di];
    float ub = d14_ub_arr[di];
    float w = d14_w[di];

    float my_diff;
    if (coord == 0u) my_diff = dx;
    else if (coord == 1u) my_diff = dy;
    else my_diff = dz;
    if (!is_a) my_diff = -my_diff;

    if (d < lb) {{
        float diff = d - lb;
        if (coord == 0u && is_a) e_contrib += w * diff * diff;
        grad_val += w * 2.0f * diff * my_diff / d;
    }} else if (d > ub) {{
        float diff = d - ub;
        if (coord == 0u && is_a) e_contrib += w * diff * diff;
        grad_val += w * 2.0f * diff * my_diff / d;
    }}
}}

grad_out[tid] = grad_val;
if (coord == 0u) {{
    energy_parts[global_atom] = e_contrib;
}}
"""
# fmt: on

_etk_kernel = None


def _get_etk_kernel():
    global _etk_kernel
    if _etk_kernel is None:
        _etk_kernel = mx.fast.metal_kernel(
            name="etk_energy_grad",
            input_names=[
                "pos", "params_buf",
                "atom_starts",
                "torsion_idx", "torsion_V", "torsion_signs", "torsion_starts",
                "improper_i", "improper_w", "improper_starts",
                "d14_i1", "d14_i2", "d14_lb_arr", "d14_ub_arr", "d14_w",
                "dist14_starts",
            ],
            output_names=["grad_out", "energy_parts"],
            source=_ETK_ENERGY_GRAD_SOURCE,
            ensure_row_contiguous=True,
        )
    return _etk_kernel


def _ensure_nonempty_1d(arr, dtype):
    """Return at least a 1-element array for Metal kernel (avoids empty buffer)."""
    if len(arr) == 0:
        return mx.zeros((1,), dtype=dtype)
    return mx.array(arr.ravel(), dtype=dtype)


def _ensure_nonempty_2d(arr, dtype, cols):
    """Return at least a 1-row array for Metal kernel, flattened."""
    if len(arr) == 0:
        return mx.zeros((cols,), dtype=dtype)
    return mx.array(arr.reshape(-1), dtype=dtype)


def make_etk_energy_grad(
    system: BatchedETKSystem,
):
    """
    Build a batched 3D ETK energy+gradient function backed by a single Metal kernel.

    Returns:
        fn(pos_flat) → (energy_parts, grad)
        where pos_flat is (n_atoms_total * 3,) float32.

        Also returns (atom_starts, n_atoms_total, n_mols) for the caller.
    """
    n_atoms_total = system.n_atoms_total
    n_mols = system.n_mols
    total_coords = n_atoms_total * 3

    params_buf = mx.array([n_atoms_total, n_mols], dtype=mx.uint32)
    atom_starts_mx = mx.array(system.atom_starts, dtype=mx.int32)

    # Torsion terms (flattened 2D arrays)
    torsion_idx_mx = _ensure_nonempty_2d(system.torsion_idx, mx.int32, 4)
    torsion_V_mx = _ensure_nonempty_2d(system.torsion_V, mx.float32, 6)
    torsion_signs_mx = _ensure_nonempty_2d(system.torsion_signs, mx.int32, 6)
    torsion_starts_mx = mx.array(system.torsion_term_starts, dtype=mx.int32)

    # Improper torsion terms
    improper_idx_mx = _ensure_nonempty_2d(system.improper_idx, mx.int32, 4)
    improper_w_mx = _ensure_nonempty_1d(system.improper_weight, mx.float32)
    improper_starts_mx = mx.array(system.improper_term_starts, dtype=mx.int32)

    # 1-4 distance terms
    d14_i1_mx = _ensure_nonempty_1d(system.dist14_idx1, mx.int32)
    d14_i2_mx = _ensure_nonempty_1d(system.dist14_idx2, mx.int32)
    d14_lb_mx = _ensure_nonempty_1d(system.dist14_lb, mx.float32)
    d14_ub_mx = _ensure_nonempty_1d(system.dist14_ub, mx.float32)
    d14_w_mx = _ensure_nonempty_1d(system.dist14_weight, mx.float32)
    dist14_starts_mx = mx.array(system.dist14_term_starts, dtype=mx.int32)

    kernel = _get_etk_kernel()

    def energy_grad_fn(pos_flat: mx.array) -> tuple[mx.array, mx.array]:
        grad_out, energy_parts = kernel(
            inputs=[
                pos_flat, params_buf,
                atom_starts_mx,
                torsion_idx_mx, torsion_V_mx, torsion_signs_mx, torsion_starts_mx,
                improper_idx_mx, improper_w_mx, improper_starts_mx,
                d14_i1_mx, d14_i2_mx, d14_lb_mx, d14_ub_mx, d14_w_mx,
                dist14_starts_mx,
            ],
            grid=(total_coords, 1, 1),
            threadgroup=(min(256, total_coords), 1, 1),
            output_shapes=[(total_coords,), (n_atoms_total,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        return energy_parts, grad_out

    return energy_grad_fn, system.atom_starts, n_atoms_total, n_mols
