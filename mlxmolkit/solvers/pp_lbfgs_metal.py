"""
Fused PP-LBFGS Metal kernel for MMFF94 optimization.

Enhances the existing fused MMFF L-BFGS kernel with block-diagonal
Hessian preconditioning (PP-LBFGS, Klemsa & Řezáč 2013).

The key change: in the L-BFGS two-loop recursion, replace the
simple γI scaled-identity H₀ with a per-atom 3×3 block-diagonal
preconditioner computed from displaced gradients.

    OLD: r = γ * q                         (scaled identity)
    NEW: r[a] = H₀⁻¹[a] @ q[a] for each atom a   (block-diagonal solve)

The 3×3 blocks are pre-computed via 3 MMFF gradient evaluations
(one per xyz direction) before the kernel launch. Cost: negligible
compared to 200+ optimization iterations.

Usage:
    from mlxmolkit.solvers.pp_lbfgs_metal import build_preconditioner

    # Pre-compute block-diagonal Hessian from MMFF gradients
    precond = build_preconditioner(mol, positions, mmff_eg_fn)

    # Pass to MMFF optimizer (replaces γI in L-BFGS)
    result = mmff_optimize_pp_lbfgs(mol, precond=precond, ...)
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from rdkit import Chem


def build_preconditioner(
    mol: Chem.Mol,
    positions: np.ndarray,
    energy_grad_fn,
    fd_step: float = 0.005,
    reg: float = 1e-3,
) -> np.ndarray:
    """Build block-diagonal Hessian preconditioner from displaced MMFF gradients.

    For each xyz direction, displace ALL atoms simultaneously and compute
    the gradient difference → one column of each atom's 3×3 Hessian block.

    Total cost: 3 MMFF gradient evaluations (one per direction).

    Args:
        mol: RDKit molecule
        positions: (C, n_atoms, 3) or (n_atoms, 3) initial positions
        energy_grad_fn: callable(pos3d) → (energies, gradients) on Metal
        fd_step: finite difference step size (Angstrom)
        reg: Tikhonov regularization for each 3×3 block

    Returns:
        precond_blocks: (n_atoms * 9,) flattened 3×3 blocks per atom
            ready for Metal kernel buffer
    """
    n_atoms = mol.GetNumAtoms()

    if positions.ndim == 2:
        positions = positions[np.newaxis, :, :]  # (1, n_atoms, 3)

    C = positions.shape[0]
    pos_mx = mx.array(positions.astype(np.float32))

    # Reference gradient
    _, g0 = energy_grad_fn(pos_mx)
    mx.eval(g0)
    g0_np = np.array(g0).reshape(C, n_atoms, 3)  # (C, n_atoms, 3)

    # Blocks: (n_atoms, 3, 3) — averaged over conformers
    blocks = np.zeros((n_atoms, 3, 3), dtype=np.float32)

    for xyz in range(3):
        pos_disp = positions.copy()
        pos_disp[:, :, xyz] += fd_step  # displace all atoms in one direction
        pos_disp_mx = mx.array(pos_disp.astype(np.float32))

        _, g_disp = energy_grad_fn(pos_disp_mx)
        mx.eval(g_disp)
        g_disp_np = np.array(g_disp).reshape(C, n_atoms, 3)

        # Finite difference: dg/dx_xyz for each atom
        dg = (g_disp_np - g0_np) / fd_step  # (C, n_atoms, 3)

        # Average over conformers, store as column xyz of each 3×3 block
        blocks[:, :, xyz] = dg.mean(axis=0)  # (n_atoms, 3)

    # Symmetrize + ensure positive-definite via spectral shift
    for a in range(n_atoms):
        H = 0.5 * (blocks[a] + blocks[a].T)
        # Eigendecompose, shift negative eigenvalues to positive
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.maximum(eigvals, reg)  # floor at reg
        blocks[a] = (eigvecs * eigvals) @ eigvecs.T

    return blocks.ravel().astype(np.float32)  # (n_atoms * 9,)


# ─── Metal shader snippet for block-diagonal preconditioner ────────

PP_LBFGS_PRECOND_SNIPPET = """
        // ---- PP-LBFGS: Block-diagonal preconditioner instead of gamma*I ----
        // Solve r_atom = H0_inv_atom @ q_atom for each atom's 3x3 block
        // Using Cramer's rule (exact for 3x3, no iteration)
        if (hist_count > 0) {
            for (int a = (int)tid; a < n_atoms; a += (int)tg_size) {
                float q0 = my_q[a*3+0], q1 = my_q[a*3+1], q2 = my_q[a*3+2];

                // Read 3x3 block (row-major)
                const device float* H = &precond_blocks[a * 9];
                float h00=H[0], h01=H[1], h02=H[2];
                float h10=H[3], h11=H[4], h12=H[5];
                float h20=H[6], h21=H[7], h22=H[8];

                // Determinant
                float det = h00*(h11*h22 - h12*h21)
                          - h01*(h10*h22 - h12*h20)
                          + h02*(h10*h21 - h11*h20);
                float inv_det = 1.0f / max(abs(det), 1e-10f);
                if (det < 0.0f) inv_det = -inv_det;

                // Cofactor matrix (transposed = adjugate)
                float c00 = h11*h22 - h12*h21;
                float c01 = h02*h21 - h01*h22;
                float c02 = h01*h12 - h02*h11;
                float c10 = h12*h20 - h10*h22;
                float c11 = h00*h22 - h02*h20;
                float c12 = h02*h10 - h00*h12;
                float c20 = h10*h21 - h11*h20;
                float c21 = h01*h20 - h00*h21;
                float c22 = h00*h11 - h01*h10;

                // r = H^{-1} @ q = (1/det) * adj(H) @ q
                my_q[a*3+0] = inv_det * (c00*q0 + c01*q1 + c02*q2);
                my_q[a*3+1] = inv_det * (c10*q0 + c11*q1 + c12*q2);
                my_q[a*3+2] = inv_det * (c20*q0 + c21*q1 + c22*q2);
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
"""

GAMMA_SCALING_SNIPPET = """
        if (hist_count > 0) {
            int nw = (hist_idx - 1) % lbfgs_m; if (nw < 0) nw += lbfgs_m;
            float sy = parallel_dot(&my_S[nw*n_terms], &my_Y[nw*n_terms], n_terms, tid, tg_size, tg_reduce);
            float yy = parallel_dot(&my_Y[nw*n_terms], &my_Y[nw*n_terms], n_terms, tid, tg_size, tg_reduce);
            float gamma = sy / max(yy, 1e-30f);
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_q[i] *= gamma;
            threadgroup_barrier(mem_flags::mem_device);
        }
"""


def patch_lbfgs_source_for_pp(lbfgs_source: str) -> str:
    """Patch the MMFF L-BFGS Metal shader to use block-diagonal preconditioner.

    Replaces the gamma*I scaling in the two-loop recursion with
    per-atom 3×3 block-diagonal solve via Cramer's rule.

    The patched kernel expects an extra input buffer: precond_blocks (n_atoms*9,)
    """
    # Find the gamma scaling section and replace it
    # The pattern to find: "if (hist_count > 0) {" followed by gamma computation
    old_section = GAMMA_SCALING_SNIPPET.strip()
    new_section = PP_LBFGS_PRECOND_SNIPPET.strip()

    # Try exact match first
    if old_section in lbfgs_source:
        return lbfgs_source.replace(old_section, new_section)

    # Fallback: find by landmark
    landmark = "float gamma = sy / max(yy, 1e-30f);"
    if landmark in lbfgs_source:
        # Find the block containing gamma scaling (6 lines)
        lines = lbfgs_source.split('\n')
        start_idx = None
        for i, line in enumerate(lines):
            if landmark in line:
                # Go back to find "if (hist_count > 0)"
                for j in range(i, max(i-5, -1), -1):
                    if 'if (hist_count > 0)' in lines[j]:
                        start_idx = j
                        break
                break

        if start_idx is not None:
            # Find the closing brace
            end_idx = start_idx
            brace_count = 0
            for k in range(start_idx, min(start_idx + 10, len(lines))):
                brace_count += lines[k].count('{') - lines[k].count('}')
                if brace_count == 0 and k > start_idx:
                    end_idx = k
                    break

            # Replace
            new_lines = lines[:start_idx] + new_section.split('\n') + lines[end_idx + 1:]
            return '\n'.join(new_lines)

    # If we can't find the exact pattern, return original with a warning comment
    return "// WARNING: PP-LBFGS patch failed — using original gamma*I\n" + lbfgs_source
