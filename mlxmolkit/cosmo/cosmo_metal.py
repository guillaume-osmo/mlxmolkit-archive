"""
Metal GPU kernel for COSMO matrix assembly.

Builds the A matrix and Φ potential vector for N molecules in parallel.
The solve (A·q = -Φ) stays on numpy since MLX linalg.solve is CPU-only.

For N=100 molecules × 500 segments each:
- A matrix build: 100 × 500² = 25M distance calculations → GPU
- Solve: 100 × 500³ = 12.5G flops → numpy (CPU)
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

_COSMO_MATRIX_SOURCE = """
// Build COSMO A matrix and Phi potential for one molecule
// Grid: (N_mols * max_seg * max_seg, 1, 1)
// Each thread computes one A[i,j] element

uint tid = thread_position_in_grid.x;
int n_mols = (int)config[0];
int max_seg = (int)config[1];
int max_atoms = (int)config[2];
int ms2 = max_seg * max_seg;

if (tid >= (uint)(n_mols * ms2)) return;

int mol = tid / ms2;
int rem = tid % ms2;
int i = rem / max_seg;
int j = rem % max_seg;

int n_seg = n_seg_arr[mol];
int n_at = n_atoms_arr[mol];

// Skip padding
if (i >= n_seg || j >= n_seg) {
    // Identity for padding (avoid singular matrix)
    A_out[tid] = (i == j) ? 1.0f : 0.0f;
    if (j == 0) Phi_out[mol * max_seg + i] = 0.0f;
    return;
}

int mol_seg_off = mol * max_seg * 3;
int mol_area_off = mol * max_seg;
int mol_atom_off = mol * max_atoms * 3;
int mol_charge_off = mol * max_atoms;

// A[i,j] = 1/|r_i - r_j| for i != j
// A[i,i] = 1.07 * sqrt(4*pi / area_i)
if (i == j) {
    float area_i = seg_area[mol_area_off + i];
    A_out[tid] = 1.07f * sqrt(4.0f * 3.14159265f / area_i);
} else {
    float dx = seg_pos[mol_seg_off + i*3 + 0] - seg_pos[mol_seg_off + j*3 + 0];
    float dy = seg_pos[mol_seg_off + i*3 + 1] - seg_pos[mol_seg_off + j*3 + 1];
    float dz = seg_pos[mol_seg_off + i*3 + 2] - seg_pos[mol_seg_off + j*3 + 2];
    float dist = sqrt(dx*dx + dy*dy + dz*dz);
    A_out[tid] = (dist > 1e-10f) ? (1.0f / dist) : 0.0f;
}

// Phi: each thread with j==0 computes Phi[i]
if (j == 0) {
    float phi = 0.0f;
    float ri_x = seg_pos[mol_seg_off + i*3 + 0];
    float ri_y = seg_pos[mol_seg_off + i*3 + 1];
    float ri_z = seg_pos[mol_seg_off + i*3 + 2];
    for (int a = 0; a < n_at; a++) {
        float dx = ri_x - atom_pos[mol_atom_off + a*3 + 0];
        float dy = ri_y - atom_pos[mol_atom_off + a*3 + 1];
        float dz = ri_z - atom_pos[mol_atom_off + a*3 + 2];
        float dist = sqrt(dx*dx + dy*dy + dz*dz);
        if (dist > 1e-10f) {
            phi += charges[mol_charge_off + a] / dist;
        }
    }
    Phi_out[mol * max_seg + i] = phi;
}
"""

_cosmo_kernel = None


def _get_cosmo_kernel():
    global _cosmo_kernel
    if _cosmo_kernel is None:
        if not hasattr(mx.fast, 'metal_kernel'):
            raise RuntimeError("MLX metal_kernel not available")
        _cosmo_kernel = mx.fast.metal_kernel(
            name="cosmo_matrix_build",
            input_names=["seg_pos", "seg_area", "atom_pos", "charges",
                         "n_seg_arr", "n_atoms_arr", "config"],
            output_names=["A_out", "Phi_out"],
            source=_COSMO_MATRIX_SOURCE,
        )
    return _cosmo_kernel


def build_cosmo_matrices_metal(
    seg_positions: list[np.ndarray],
    seg_areas: list[np.ndarray],
    atom_coords: list[np.ndarray],
    mulliken_charges: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build COSMO A matrices and Phi vectors on Metal GPU.

    Args:
        seg_positions: list of (n_seg_i, 3) arrays (in Bohr)
        seg_areas: list of (n_seg_i,) arrays (in Bohr²)
        atom_coords: list of (n_atoms_i, 3) arrays (in Bohr)
        mulliken_charges: list of (n_atoms_i,) arrays

    Returns:
        A_batch: (N, max_seg, max_seg) Coulomb matrices
        Phi_batch: (N, max_seg) potential vectors
    """
    N = len(seg_positions)
    max_seg = max(len(p) for p in seg_positions)
    max_atoms = max(len(c) for c in atom_coords)

    # Pad and flatten
    seg_pos_flat = np.zeros((N, max_seg, 3), dtype=np.float32)
    seg_area_flat = np.zeros((N, max_seg), dtype=np.float32)
    atom_pos_flat = np.zeros((N, max_atoms, 3), dtype=np.float32)
    charges_flat = np.zeros((N, max_atoms), dtype=np.float32)
    n_seg_arr = np.zeros(N, dtype=np.int32)
    n_atoms_arr = np.zeros(N, dtype=np.int32)

    for i in range(N):
        ns = len(seg_positions[i])
        na = len(atom_coords[i])
        seg_pos_flat[i, :ns] = seg_positions[i]
        seg_area_flat[i, :ns] = seg_areas[i]
        # Pad area with 1.0 to avoid sqrt(0) in diagonal
        seg_area_flat[i, ns:] = 1.0
        atom_pos_flat[i, :na] = atom_coords[i]
        charges_flat[i, :na] = mulliken_charges[i]
        n_seg_arr[i] = ns
        n_atoms_arr[i] = na

    config = np.array([N, max_seg, max_atoms], dtype=np.float32)
    n_elements = N * max_seg * max_seg

    kernel = _get_cosmo_kernel()
    outputs = kernel(
        inputs=[
            mx.array(seg_pos_flat.reshape(N, -1).flatten()),
            mx.array(seg_area_flat.flatten()),
            mx.array(atom_pos_flat.reshape(N, -1).flatten()),
            mx.array(charges_flat.flatten()),
            mx.array(n_seg_arr),
            mx.array(n_atoms_arr),
            mx.array(config),
        ],
        output_shapes=[(n_elements,), (N * max_seg,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_elements, 1, 1),
        threadgroup=(min(256, n_elements), 1, 1),
    )
    mx.eval(outputs[0], outputs[1])

    A_batch = np.array(outputs[0]).reshape(N, max_seg, max_seg).astype(np.float64)
    Phi_batch = np.array(outputs[1]).reshape(N, max_seg).astype(np.float64)

    return A_batch, Phi_batch


def cosmo_solve_metal(
    seg_positions: list[np.ndarray],
    seg_areas: list[np.ndarray],
    atom_coords: list[np.ndarray],
    mulliken_charges: list[np.ndarray],
    epsilon: float = 78.39,
) -> list[np.ndarray]:
    """Full COSMO solve: Metal GPU matrix build + numpy batch solve.

    Args:
        All inputs in Bohr / Bohr² units
        epsilon: dielectric constant

    Returns:
        list of (n_seg_i,) charge arrays
    """
    N = len(seg_positions)
    max_seg = max(len(p) for p in seg_positions)
    n_seg_list = [len(p) for p in seg_positions]

    # GPU: build A and Phi
    A_batch, Phi_batch = build_cosmo_matrices_metal(
        seg_positions, seg_areas, atom_coords, mulliken_charges
    )

    # CPU: batch solve A·q = -Phi
    q_batch = np.linalg.solve(A_batch, -Phi_batch[:, :, np.newaxis])[:, :, 0]

    # Dielectric scaling
    f_eps = (epsilon - 1.0) / (epsilon + 0.5)

    # Extract per-molecule charges
    charges = []
    for i in range(N):
        ns = n_seg_list[i]
        charges.append(f_eps * q_batch[i, :ns])

    return charges
