"""
4D Distance Geometry energy and gradient on Metal.

nvMolKit-style distance violation energy with three terms:
  1. Distance violation: asymmetric upper/lower bounds
     - Upper: E = w * (d²/ub² - 1)²  when d² > ub²
     - Lower: E = w * (2·lb²/(lb²+d²) - 1)²  when d² < lb²
  2. Chiral volume constraint: E = (V - bound)²
  3. Fourth dimension penalty: E = w4d * x₄²  (drives 4D→3D collapse)

All threads compute both energy AND gradient (parallel gradient scatter).
One thread per (global_atom, coord).  CSR indexing ensures each thread
only loops over pairs belonging to its molecule.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from mlxmolkit.dg_extract import BatchedDGSystem

# fmt: off
_DG4D_ENERGY_GRAD_SOURCE = """
uint tid = thread_position_in_grid.x;
uint n_atoms_total = params_buf[0];
uint dim_val = params_buf[1];
uint n_mols = params_buf[2];

if (tid >= n_atoms_total * dim_val) {{ return; }}

uint global_atom = tid / dim_val;
uint coord = tid % dim_val;

// Binary-ish scan to find molecule
uint mol_id = 0;
for (uint m = 0; m < n_mols; m++) {{
    if (global_atom < (uint)atom_starts[m + 1]) {{
        mol_id = m;
        break;
    }}
}}

float grad_val = 0.0f;
float e_contrib = 0.0f;

// === Distance violation ===
uint d_start = (uint)dist_starts[mol_id];
uint d_end   = (uint)dist_starts[mol_id + 1];

for (uint p = d_start; p < d_end; p++) {{
    uint a = (uint)dist_i1[p];
    uint b = (uint)dist_i2[p];

    bool is_a = (a == global_atom);
    bool is_b = (b == global_atom);
    if (!is_a && !is_b) continue;

    float d2 = 0.0f;
    for (uint c = 0; c < dim_val; c++) {{
        float dc = pos[a * dim_val + c] - pos[b * dim_val + c];
        d2 += dc * dc;
    }}
    d2 = max(d2, 1e-12f);

    float lb2 = dist_lb[p];
    float ub2 = dist_ub[p];
    float w   = dist_w[p];

    float my_diff = pos[a * dim_val + coord] - pos[b * dim_val + coord];
    if (!is_a) my_diff = -my_diff;

    if (d2 > ub2) {{
        float ratio = d2 / ub2 - 1.0f;
        if (coord == 0u && is_a) e_contrib += w * ratio * ratio;
        grad_val += w * 4.0f * ratio / ub2 * my_diff;
    }} else if (d2 < lb2) {{
        float denom = lb2 + d2;
        float ratio = 2.0f * lb2 / denom - 1.0f;
        if (coord == 0u && is_a) e_contrib += w * ratio * ratio;
        grad_val += w * 2.0f * ratio * (-4.0f * lb2 / (denom * denom)) * my_diff;
    }}
}}

// === Chiral volume ===
uint ch_start = (uint)chiral_starts[mol_id];
uint ch_end   = (uint)chiral_starts[mol_id + 1];

for (uint ci = ch_start; ci < ch_end; ci++) {{
    uint i1 = (uint)ch_i1[ci];
    uint i2 = (uint)ch_i2[ci];
    uint i3 = (uint)ch_i3[ci];
    uint i4 = (uint)ch_i4[ci];

    if (global_atom != i1 && global_atom != i2 &&
        global_atom != i3 && global_atom != i4) continue;
    if (coord >= 3u) continue;

    float p1x = pos[i1*dim_val], p1y = pos[i1*dim_val+1], p1z = pos[i1*dim_val+2];
    float p2x = pos[i2*dim_val], p2y = pos[i2*dim_val+1], p2z = pos[i2*dim_val+2];
    float p3x = pos[i3*dim_val], p3y = pos[i3*dim_val+1], p3z = pos[i3*dim_val+2];
    float p4x = pos[i4*dim_val], p4y = pos[i4*dim_val+1], p4z = pos[i4*dim_val+2];

    float v1x = p1x-p4x, v1y = p1y-p4y, v1z = p1z-p4z;
    float v2x = p2x-p4x, v2y = p2y-p4y, v2z = p2z-p4z;
    float v3x = p3x-p4x, v3y = p3y-p4y, v3z = p3z-p4z;

    // cross = v2 × v3
    float cx = v2y*v3z - v2z*v3y;
    float cy = v2z*v3x - v2x*v3z;
    float cz = v2x*v3y - v2y*v3x;

    float vol = v1x*cx + v1y*cy + v1z*cz;

    float vol_lo = ch_lo[ci];
    float vol_hi = ch_hi[ci];

    float violation = 0.0f;
    if (vol < vol_lo) violation = vol - vol_lo;
    else if (vol > vol_hi) violation = vol - vol_hi;
    else continue;

    if (coord == 0u && global_atom == i1) e_contrib += violation * violation;

    // dE/dV = 2·violation.  dV/d(atom,coord) depends on which atom.
    float dV = 0.0f;
    if (global_atom == i1) {{
        if (coord == 0u) dV = cx;
        else if (coord == 1u) dV = cy;
        else dV = cz;
    }} else if (global_atom == i2) {{
        float c2x = v3y*v1z - v3z*v1y;
        float c2y = v3z*v1x - v3x*v1z;
        float c2z = v3x*v1y - v3y*v1x;
        if (coord == 0u) dV = c2x;
        else if (coord == 1u) dV = c2y;
        else dV = c2z;
    }} else if (global_atom == i3) {{
        float c3x = v1y*v2z - v1z*v2y;
        float c3y = v1z*v2x - v1x*v2z;
        float c3z = v1x*v2y - v1y*v2x;
        if (coord == 0u) dV = c3x;
        else if (coord == 1u) dV = c3y;
        else dV = c3z;
    }} else {{
        float c2x = v3y*v1z - v3z*v1y;
        float c2y = v3z*v1x - v3x*v1z;
        float c2z = v3x*v1y - v3y*v1x;
        float c3x = v1y*v2z - v1z*v2y;
        float c3y = v1z*v2x - v1x*v2z;
        float c3z = v1x*v2y - v1y*v2x;
        if (coord == 0u) dV = -(cx + c2x + c3x);
        else if (coord == 1u) dV = -(cy + c2y + c3y);
        else dV = -(cz + c2z + c3z);
    }}

    grad_val += 2.0f * violation * dV;
}}

// === Fourth dimension penalty ===
if (dim_val == 4u && coord == 3u) {{
    float x4 = pos[global_atom * dim_val + 3u];
    float w4 = fourth_w[0];
    e_contrib += w4 * x4 * x4;
    grad_val += 2.0f * w4 * x4;
}}

grad_out[tid] = grad_val;
if (coord == 0u) {{
    energy_parts[global_atom] = e_contrib;
}}
"""
# fmt: on

_dg4d_kernel = None


def _get_dg4d_kernel():
    global _dg4d_kernel
    if _dg4d_kernel is None:
        _dg4d_kernel = mx.fast.metal_kernel(
            name="dg4d_energy_grad",
            input_names=[
                "pos", "params_buf",
                "atom_starts",
                "dist_i1", "dist_i2", "dist_lb", "dist_ub", "dist_w",
                "dist_starts",
                "ch_i1", "ch_i2", "ch_i3", "ch_i4", "ch_lo", "ch_hi",
                "chiral_starts",
                "fourth_w",
            ],
            output_names=["grad_out", "energy_parts"],
            source=_DG4D_ENERGY_GRAD_SOURCE,
            ensure_row_contiguous=True,
        )
    return _dg4d_kernel


def make_dg4d_energy_grad(
    system: BatchedDGSystem,
    fourth_dim_weight: float = 10.0,
):
    """
    Build a batched 4D DG energy+gradient function backed by a single Metal kernel.

    Returns:
        fn(pos_flat) → (energy_parts, grad)
        where pos_flat is (n_atoms_total * dim,) float32,
        energy_parts is (n_atoms_total,) float32,
        grad is (n_atoms_total * dim,) float32.

        Also returns (atom_starts, n_atoms_total, n_mols, dim) for the caller.
    """
    dim = system.dim
    n_atoms_total = system.n_atoms_total
    n_mols = system.n_mols
    total_coords = n_atoms_total * dim

    params_buf = mx.array([n_atoms_total, dim, n_mols], dtype=mx.uint32)
    atom_starts_mx = mx.array(system.atom_starts, dtype=mx.int32)

    # Distance terms — use empty sentinel arrays if no terms
    dist_i1 = mx.array(system.dist_idx1) if len(system.dist_idx1) else mx.zeros((1,), dtype=mx.int32)
    dist_i2 = mx.array(system.dist_idx2) if len(system.dist_idx2) else mx.zeros((1,), dtype=mx.int32)
    dist_lb = mx.array(system.dist_lb2) if len(system.dist_lb2) else mx.zeros((1,), dtype=mx.float32)
    dist_ub = mx.array(system.dist_ub2) if len(system.dist_ub2) else mx.zeros((1,), dtype=mx.float32)
    dist_w  = mx.array(system.dist_weight) if len(system.dist_weight) else mx.zeros((1,), dtype=mx.float32)
    dist_starts_mx = mx.array(system.dist_term_starts, dtype=mx.int32)

    # Chiral terms
    ch_i1 = mx.array(system.chiral_idx1) if len(system.chiral_idx1) else mx.zeros((1,), dtype=mx.int32)
    ch_i2 = mx.array(system.chiral_idx2) if len(system.chiral_idx2) else mx.zeros((1,), dtype=mx.int32)
    ch_i3 = mx.array(system.chiral_idx3) if len(system.chiral_idx3) else mx.zeros((1,), dtype=mx.int32)
    ch_i4 = mx.array(system.chiral_idx4) if len(system.chiral_idx4) else mx.zeros((1,), dtype=mx.int32)
    ch_lo  = mx.array(system.chiral_vol_lower) if len(system.chiral_vol_lower) else mx.zeros((1,), dtype=mx.float32)
    ch_hi  = mx.array(system.chiral_vol_upper) if len(system.chiral_vol_upper) else mx.zeros((1,), dtype=mx.float32)
    chiral_starts_mx = mx.array(system.chiral_term_starts, dtype=mx.int32)

    fourth_w = mx.array([fourth_dim_weight], dtype=mx.float32)

    kernel = _get_dg4d_kernel()

    def energy_grad_fn(pos_flat: mx.array) -> tuple[mx.array, mx.array]:
        grad_out, energy_parts = kernel(
            inputs=[
                pos_flat, params_buf,
                atom_starts_mx,
                dist_i1, dist_i2, dist_lb, dist_ub, dist_w,
                dist_starts_mx,
                ch_i1, ch_i2, ch_i3, ch_i4, ch_lo, ch_hi,
                chiral_starts_mx,
                fourth_w,
            ],
            grid=(total_coords, 1, 1),
            threadgroup=(min(256, total_coords), 1, 1),
            output_shapes=[(total_coords,), (n_atoms_total,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        return energy_parts, grad_out

    return energy_grad_fn, system.atom_starts, n_atoms_total, n_mols, dim
