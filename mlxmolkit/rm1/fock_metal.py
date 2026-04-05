"""
Metal kernel for Fock matrix construction in RM1.

One thread per (μ,ν) matrix element. Each thread computes:
  F[μ,ν] = H[μ,ν] + G_one_center[μ,ν] + G_two_center[μ,ν]

For batch processing: one threadgroup per molecule.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

_FOCK_HEADER = """
// RM1 one-center two-electron integrals (Slater-Condon)
// gss, gsp, gpp, gp2, hsp stored per atom in param_buf
// Layout: param_buf[atom_idx * 5 + {0:gss, 1:gsp, 2:gpp, 3:gp2, 4:hsp}]
"""

_FOCK_SOURCE = """
uint tid = thread_position_in_grid.x;
uint n_basis = (uint)config[0];
uint n_atoms = (uint)config[1];

if (tid >= n_basis * n_basis) return;

uint mu = tid / n_basis;
uint nu = tid % n_basis;

// Look up which atom owns mu and nu
int atom_mu = atom_map[mu];
int atom_nu = atom_map[nu];
int type_mu = type_map[mu];  // 0=s, 1=px, 2=py, 3=pz
int type_nu = type_map[nu];

float f = H_core[tid];  // start with core Hamiltonian

// === One-center contribution (same atom only) ===
if (atom_mu == atom_nu) {
    int a = atom_mu;
    float gss = params[a*5+0];
    float gsp = params[a*5+1];
    float gpp = params[a*5+2];
    float gp2 = params[a*5+3];
    float hsp = params[a*5+4];
    int a_start = atom_starts[a];
    int n_orb = atom_starts[a+1] - a_start;

    if (mu == nu) {
        // Diagonal
        float P_self = P[mu * n_basis + mu];

        if (n_orb == 1) {
            // H: only (ss|ss)
            f += P_self * gss * 0.5f;
        } else {
            int s = a_start;
            if (type_mu == 0) {
                // s orbital
                float Ppp = 0.0f;
                for (int k = 1; k < 4; k++) Ppp += P[(s+k)*n_basis+(s+k)];
                f += P_self * gss * 0.5f + Ppp * (gsp - 0.5f * hsp);
            } else {
                // p orbital
                float Pss = P[s*n_basis+s];
                float Ppp_total = 0.0f;
                for (int k = 1; k < 4; k++) Ppp_total += P[(s+k)*n_basis+(s+k)];
                float Ppp_self = P[mu*n_basis+mu];
                f += Pss * (gsp - 0.5f * hsp)
                   + Ppp_self * gpp * 0.5f
                   + (Ppp_total - Ppp_self) * (gp2 - 0.5f * (gpp - gp2));
            }
        }
    } else {
        // Off-diagonal, same atom
        if (n_orb > 1) {
            if ((type_mu == 0 && type_nu > 0) || (type_mu > 0 && type_nu == 0)) {
                // s-p exchange
                f += P[mu*n_basis+nu] * (2.0f * hsp - 0.5f * gsp);
            } else if (type_mu > 0 && type_nu > 0 && type_mu != type_nu) {
                // p-p' exchange
                f += P[mu*n_basis+nu] * (0.5f * (gpp - gp2) - 0.5f * gp2);
            }
        }
    }
}

// === Two-center Coulomb ===
// F[mu,mu] += Σ_B density_on_B * (ss|ss)_AB
if (mu == nu) {
    for (uint b = 0; b < n_atoms; b++) {
        if ((int)b == atom_mu) continue;
        float PB = 0.0f;
        int b_start = atom_starts[b];
        int b_end = atom_starts[b+1];
        for (int lb = b_start; lb < b_end; lb++) PB += P[lb*n_basis+lb];
        f += PB * ssss_integrals[atom_mu * n_atoms + b];
    }
}
// Two-center exchange (s-s only)
if (atom_mu != atom_nu && type_mu == 0 && type_nu == 0) {
    f -= 0.5f * P[mu*n_basis+nu] * ssss_integrals[atom_mu * n_atoms + atom_nu];
}

F_out[tid] = f;
"""

_fock_kernel = None


def _get_fock_kernel():
    global _fock_kernel
    if _fock_kernel is None:
        _fock_kernel = mx.fast.metal_kernel(
            name="rm1_fock_build",
            input_names=["H_core", "P", "params", "atom_map", "type_map",
                         "atom_starts", "ssss_integrals", "config"],
            output_names=["F_out"],
            header=_FOCK_HEADER,
            source=_FOCK_SOURCE,
        )
    return _fock_kernel


def build_fock_metal(
    H_core: np.ndarray,
    P: np.ndarray,
    atom_params: np.ndarray,
    atom_map: np.ndarray,
    type_map: np.ndarray,
    atom_starts: np.ndarray,
    ssss_integrals: np.ndarray,
    n_basis: int,
    n_atoms: int,
) -> np.ndarray:
    """Build Fock matrix on Metal GPU.

    Args:
        H_core: (n_basis²,) flat core Hamiltonian
        P: (n_basis²,) flat density matrix
        atom_params: (n_atoms, 5) [gss, gsp, gpp, gp2, hsp] per atom
        atom_map: (n_basis,) basis → atom index
        type_map: (n_basis,) basis → orbital type (0=s, 1=px, 2=py, 3=pz)
        atom_starts: (n_atoms+1,) CSR-style basis offsets
        ssss_integrals: (n_atoms, n_atoms) pairwise (ss|ss) integrals
        n_basis, n_atoms: sizes

    Returns:
        F: (n_basis²,) flat Fock matrix
    """
    config = np.array([n_basis, n_atoms], dtype=np.float32)
    n_elements = n_basis * n_basis

    kernel = _get_fock_kernel()
    outputs = kernel(
        inputs=[
            mx.array(H_core.flatten().astype(np.float32)),
            mx.array(P.flatten().astype(np.float32)),
            mx.array(atom_params.flatten().astype(np.float32)),
            mx.array(atom_map.astype(np.int32)),
            mx.array(type_map.astype(np.int32)),
            mx.array(atom_starts.astype(np.int32)),
            mx.array(ssss_integrals.flatten().astype(np.float32)),
            mx.array(config),
        ],
        output_shapes=[(n_elements,)],
        output_dtypes=[mx.float32],
        grid=(n_elements, 1, 1),
        threadgroup=(min(256, n_elements), 1, 1),
    )
    mx.eval(outputs[0])
    return np.array(outputs[0]).reshape(n_basis, n_basis)
