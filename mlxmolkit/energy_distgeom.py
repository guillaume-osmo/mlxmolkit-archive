"""
Distance geometry energy and gradient on Metal.

Simple spring model: E = Σ_{(i,j)} (d_ij - target_ij)²
where d_ij = ||x_i - x_j|| and target_ij is the target distance.

This is the simplest force field for 3D embedding from distance bounds,
matching nvMolKit's DG (Distance Geometry) mode.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

_DG_ENERGY_GRAD_SOURCE = """
uint tid = thread_position_in_grid.x;
uint n_atoms = n_atoms_buf[0];
uint n_pairs = n_pairs_buf[0];
uint dim_val = 3u;

if (tid >= n_atoms * dim_val) {{ return; }}

uint atom_i = tid / dim_val;
uint coord = tid % dim_val;

float grad_val = 0.0f;
float e_contrib = 0.0f;

for (uint p = 0; p < n_pairs; p++) {{
    uint a = (uint)pairs[p * 2];
    uint b = (uint)pairs[p * 2 + 1];

    uint my_atom = 0xFFFFFFFFu;
    uint other_atom = 0xFFFFFFFFu;
    float sign = 0.0f;

    if (a == atom_i) {{
        my_atom = a; other_atom = b; sign = 1.0f;
    }} else if (b == atom_i) {{
        my_atom = b; other_atom = a; sign = 1.0f;
    }}

    if (my_atom == 0xFFFFFFFFu) continue;

    float dx = pos[a * dim_val + 0] - pos[b * dim_val + 0];
    float dy = pos[a * dim_val + 1] - pos[b * dim_val + 1];
    float dz = pos[a * dim_val + 2] - pos[b * dim_val + 2];
    float dist = sqrt(dx * dx + dy * dy + dz * dz + 1e-12f);
    float target = targets[p];
    float diff = dist - target;

    float my_coord_diff;
    if (coord == 0u) my_coord_diff = dx;
    else if (coord == 1u) my_coord_diff = dy;
    else my_coord_diff = dz;

    if (a != atom_i) my_coord_diff = -my_coord_diff;

    grad_val += 2.0f * diff * my_coord_diff / dist;

    if (coord == 0u && a == atom_i) {{
        e_contrib += diff * diff;
    }}
}}

grad_out[tid] = grad_val;
if (coord == 0u) {{
    energy_parts[atom_i] = e_contrib;
}}
"""


_dg_kernel = None


def _get_dg_kernel():
    global _dg_kernel
    if _dg_kernel is None:
        _dg_kernel = mx.fast.metal_kernel(
            name="dg_energy_grad",
            input_names=["pos", "pairs", "targets", "n_atoms_buf", "n_pairs_buf"],
            output_names=["grad_out", "energy_parts"],
            source=_DG_ENERGY_GRAD_SOURCE,
            ensure_row_contiguous=True,
        )
    return _dg_kernel


def make_distgeom_energy_grad(
    pairs: np.ndarray,
    targets: np.ndarray,
    n_atoms: int,
):
    """
    Create an energy/gradient function for distance geometry optimization.

    Args:
        pairs: (n_pairs, 2) int32 array of atom index pairs.
        targets: (n_pairs,) float32 array of target distances.
        n_atoms: number of atoms.

    Returns:
        Callable(pos_flat) → (energy, grad), where pos_flat is (n_atoms*3,) float32.
    """
    pairs_mx = mx.array(pairs.astype(np.int32).flatten())
    targets_mx = mx.array(targets.astype(np.float32))
    n_pairs = len(targets)
    n_atoms_buf = mx.array([n_atoms], dtype=mx.uint32)
    n_pairs_buf = mx.array([n_pairs], dtype=mx.uint32)

    k = _get_dg_kernel()
    dim3 = n_atoms * 3

    def energy_grad_fn(pos_flat: mx.array) -> tuple[mx.array, mx.array]:
        grad_out, energy_parts = k(
            inputs=[pos_flat, pairs_mx, targets_mx, n_atoms_buf, n_pairs_buf],
            grid=(dim3, 1, 1),
            threadgroup=(min(256, dim3), 1, 1),
            output_shapes=[(dim3,), (n_atoms,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        energy = mx.sum(energy_parts)
        return energy, grad_out

    return energy_grad_fn


def make_rosenbrock_energy_grad():
    """
    Rosenbrock function for testing: f(x,y) = (1-x)² + 100(y-x²)².
    Minimum at (1, 1) with f=0.
    """
    def energy_grad_fn(pos: mx.array) -> tuple[mx.array, mx.array]:
        x_val = pos[0]
        y_val = pos[1]
        e = (1.0 - x_val) ** 2 + 100.0 * (y_val - x_val ** 2) ** 2
        dx = -2.0 * (1.0 - x_val) + 100.0 * 2.0 * (y_val - x_val ** 2) * (-2.0 * x_val)
        dy = 100.0 * 2.0 * (y_val - x_val ** 2)
        g = mx.array([dx, dy])
        return e, g

    return energy_grad_fn
