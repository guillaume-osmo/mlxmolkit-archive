"""
Metal kernel for batch Fock matrix construction in RM1.

Batch version: one thread per (mol, mu, nu) triple.
Grid: (N * MB * MB, 1, 1) where N = n_mols, MB = max_basis.

Computes F = H_core + G_one_center(P) + G_two_center(P, w)
using the full rotated w tensor (not just ss|ss).

One-center factors verified against PYSEQM fock.py _one_center.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx


_FOCK_BATCH_SOURCE = """
uint tid = thread_position_in_grid.x;
int n_mols = (int)config[0];
int MB = (int)config[1];      // max_basis (padded)
int MA = (int)config[2];      // max_atoms (padded)
int MB2 = MB * MB;

if (tid >= (uint)(n_mols * MB2)) return;

int mol = tid / MB2;
int rem = tid % MB2;
int mu = rem / MB;
int nu = rem % MB;

int n_bas = n_basis_arr[mol];
int n_at  = n_atoms_arr[mol];

// Skip padding region
if (mu >= n_bas || nu >= n_bas) {
    F_out[tid] = 0.0f;
    return;
}

// Base offsets for this molecule
int mol_MB2 = mol * MB2;
int mol_MA  = mol * (MA + 1);
int mol_MA5 = mol * MA * 5;
int mol_MB  = mol * MB;
int mol_W   = mol * MA * MA * 256;

// Atom info for mu and nu
int atom_mu = atom_map[mol_MB + mu];
int atom_nu = atom_map[mol_MB + nu];
int type_mu = type_map[mol_MB + mu];
int type_nu = type_map[mol_MB + nu];

float f = H_core[mol_MB2 + mu * MB + nu];

// =========================================================
// ONE-CENTER (same atom)
// =========================================================
if (atom_mu == atom_nu) {
    int a = atom_mu;
    float gss = atom_params[mol_MA5 + a*5 + 0];
    float gsp = atom_params[mol_MA5 + a*5 + 1];
    float gpp = atom_params[mol_MA5 + a*5 + 2];
    float gp2 = atom_params[mol_MA5 + a*5 + 3];
    float hsp = atom_params[mol_MA5 + a*5 + 4];
    int a_start = atom_starts[mol_MA + a];
    int n_orb = atom_starts[mol_MA + a + 1] - a_start;

    if (mu == nu) {
        // Diagonal
        float P_self = P[mol_MB2 + mu * MB + mu];
        if (n_orb == 1) {
            // H: only (ss|ss)
            f += P_self * gss * 0.5f;
        } else {
            int s = a_start;
            if (type_mu == 0) {
                // s orbital: F(s,s) = 0.5*Pss*gss + Pptot*(gsp - 0.5*hsp)
                float Ppp = 0.0f;
                for (int k = 1; k < 4; k++)
                    Ppp += P[mol_MB2 + (s+k)*MB + (s+k)];
                f += P_self * gss * 0.5f + Ppp * (gsp - 0.5f * hsp);
            } else {
                // p orbital: PYSEQM factors
                float Pss = P[mol_MB2 + s*MB + s];
                float Ppp_total = 0.0f;
                for (int k = 1; k < 4; k++)
                    Ppp_total += P[mol_MB2 + (s+k)*MB + (s+k)];
                float Ppp_self = P[mol_MB2 + mu*MB + mu];
                float pp_fac_d = 1.25f * gp2 - 0.25f * gpp;
                f += Pss * (gsp - 0.5f * hsp)
                   + Ppp_self * gpp * 0.5f
                   + (Ppp_total - Ppp_self) * pp_fac_d;
            }
        }
    } else {
        // Off-diagonal, same atom
        if (n_orb > 1) {
            float Pmn = P[mol_MB2 + mu*MB + nu];
            if ((type_mu == 0) != (type_nu == 0)) {
                // s-p: sp_fac_2 = 1.5*hsp - 0.5*gsp
                float sp_fac_2 = 1.5f * hsp - 0.5f * gsp;
                f += Pmn * sp_fac_2;
            } else if (type_mu > 0 && type_nu > 0 && type_mu != type_nu) {
                // p-p': pp_fac_off = 0.75*gpp - 1.25*gp2
                float pp_fac_off = 0.75f * gpp - 1.25f * gp2;
                f += Pmn * pp_fac_off;
            }
        }
    }
}

// =========================================================
// TWO-CENTER: Coulomb + Exchange using full w tensor
// =========================================================
// w tensor layout: w[mol, atom_i, atom_j, k*64 + l*16 + m*4 + n]
//   where k,l index on atom_i (0..3), m,n on atom_j (0..3)

int mu_off = mu - atom_starts[mol_MA + atom_mu];
int nu_off = nu - atom_starts[mol_MA + atom_nu];

if (atom_mu == atom_nu) {
    // Coulomb on atom A from all other atoms B:
    // F[mu_A, nu_A] += sum_{lam,sig on B} P[lam,sig] * w[A,B,mu_off,nu_off,lam_off,sig_off]
    for (int b = 0; b < n_at; b++) {
        if (b == atom_mu) continue;
        int b_start = atom_starts[mol_MA + b];
        int b_end = atom_starts[mol_MA + b + 1];
        int nB = b_end - b_start;
        // w index for pair (atom_mu, b)
        int w_base = mol_W + atom_mu * MA * 256 + b * 256;
        for (int ls = 0; ls < nB; ls++) {
            for (int ss = 0; ss < nB; ss++) {
                int lam = b_start + ls;
                int sig = b_start + ss;
                // w[mu_off, nu_off, ls, ss]
                int w_idx = mu_off * 64 + nu_off * 16 + ls * 4 + ss;
                f += P[mol_MB2 + lam * MB + sig] * w[w_base + w_idx];
            }
        }
    }
} else {
    // Exchange A-B:
    // F[mu_A, lam_B] -= 0.5 * sum_{nu_A, sig_B} P[nu_A, sig_B] * w[A,B,mu_off,nu_off,lam_off,sig_off]
    // Here mu is on atom_mu, nu is on atom_nu
    // So this is F[mu, nu] where atoms differ → exchange contribution
    int a = atom_mu;
    int b = atom_nu;
    int a_start = atom_starts[mol_MA + a];
    int a_end = atom_starts[mol_MA + a + 1];
    int nA = a_end - a_start;
    int b_start = atom_starts[mol_MA + b];
    int b_end = atom_starts[mol_MA + b + 1];
    int nB = b_end - b_start;

    int lam_off = nu_off;  // nu is on atom_nu = B, so lam_off = nu_off

    // F[mu_A, lam_B] -= 0.5 * sum_{nu_A on A, sig_B on B} P[nu_A, sig_B] * w[A,B,mu_off,nu_off_A,lam_off_B,sig_off_B]
    // But w is stored as w[A,B, kk,ll, mm,nn] where kk,ll are on A and mm,nn on B
    float exch = 0.0f;
    int w_base = mol_W + a * MA * 256 + b * 256;
    for (int nA_off = 0; nA_off < nA; nA_off++) {
        for (int sB_off = 0; sB_off < nB; sB_off++) {
            int nu_global = a_start + nA_off;
            int sig_global = b_start + sB_off;
            // w[A,B, mu_off, nA_off, lam_off, sB_off]
            int w_idx = mu_off * 64 + nA_off * 16 + lam_off * 4 + sB_off;
            exch += P[mol_MB2 + nu_global * MB + sig_global] * w[w_base + w_idx];
        }
    }
    f -= 0.5f * exch;
}

F_out[tid] = f;
"""

_fock_batch_kernel = None


def _get_fock_batch_kernel():
    global _fock_batch_kernel
    if _fock_batch_kernel is None:
        if not hasattr(mx.fast, 'metal_kernel'):
            raise RuntimeError(
                f"MLX {getattr(mx, '__version__', '?')} does not support metal_kernel. "
                f"Upgrade to MLX >= 0.16: pip install 'mlx>=0.16'"
            )
        _fock_batch_kernel = mx.fast.metal_kernel(
            name="rm1_fock_batch",
            input_names=["H_core", "P", "w", "atom_params",
                         "atom_map", "type_map", "atom_starts",
                         "n_atoms_arr", "n_basis_arr", "config"],
            output_names=["F_out"],
            source=_FOCK_BATCH_SOURCE,
        )
    return _fock_batch_kernel


class MetalFockContext:
    """Pre-allocated GPU buffers for batch Fock kernel.

    Upload static data ONCE, only update P each SCF iteration.
    """
    def __init__(self, batch):
        N = batch.n_mols
        MB = batch.max_basis
        MA = batch.max_atoms
        self.N = N
        self.MB = MB
        self.MA = MA
        self.n_elements = N * MB * MB

        # Static buffers — uploaded ONCE to GPU
        self._H_core = mx.array(batch.H_core.flatten().astype(np.float32))
        self._w = mx.array(batch.w.flatten().astype(np.float32))
        self._atom_params = mx.array(batch.atom_params.flatten().astype(np.float32))
        self._atom_map = mx.array(batch.atom_map.flatten().astype(np.int32))
        self._type_map = mx.array(batch.type_map.flatten().astype(np.int32))
        self._atom_starts = mx.array(batch.atom_starts.flatten().astype(np.int32))
        self._n_atoms_arr = mx.array(batch.n_atoms_arr.astype(np.int32))
        self._n_basis_arr = mx.array(batch.n_basis_arr.astype(np.int32))
        self._config = mx.array(np.array([N, MB, MA], dtype=np.float32))
        self._kernel = _get_fock_batch_kernel()

    def build_fock(self, P: np.ndarray) -> np.ndarray:
        """Build Fock from density P. Only P is transferred each call."""
        P_gpu = mx.array(P.flatten().astype(np.float32))
        outputs = self._kernel(
            inputs=[
                self._H_core, P_gpu, self._w,
                self._atom_params, self._atom_map, self._type_map,
                self._atom_starts, self._n_atoms_arr, self._n_basis_arr,
                self._config,
            ],
            output_shapes=[(self.n_elements,)],
            output_dtypes=[mx.float32],
            grid=(self.n_elements, 1, 1),
            threadgroup=(min(256, self.n_elements), 1, 1),
        )
        mx.eval(outputs[0])
        return np.array(outputs[0]).reshape(self.N, self.MB, self.MB).astype(np.float64)


def build_fock_batch_metal(batch) -> np.ndarray:
    """Build Fock matrices for all molecules in batch on Metal GPU.

    NOTE: For SCF loops, use MetalFockContext instead to avoid
    re-uploading static buffers every iteration.
    """
    ctx = MetalFockContext(batch)
    return ctx.build_fock(batch.P)


def build_fock_batch_cpu(batch) -> np.ndarray:
    """Build Fock matrices for all molecules on CPU (reference).

    Uses the same data structures as Metal for verification.
    """
    N = batch.n_mols
    MB = batch.max_basis
    MA = batch.max_atoms
    F_all = np.zeros((N, MB, MB), dtype=np.float64)

    for mol in range(N):
        n_bas = batch.n_basis_arr[mol]
        n_at = batch.n_atoms_arr[mol]
        H = batch.H_core[mol, :n_bas, :n_bas]
        P = batch.P[mol, :n_bas, :n_bas]
        F = H.copy()

        b2a = batch.atom_map[mol, :n_bas]
        btype = batch.type_map[mol, :n_bas]
        starts = batch.atom_starts[mol]

        # One-center
        for a in range(n_at):
            a_start = starts[a]
            a_end = starts[a + 1]
            n_orb = a_end - a_start
            gss, gsp, gpp, gp2, hsp = batch.atom_params[mol, a]

            if n_orb == 1:
                s = a_start
                F[s, s] += P[s, s] * gss * 0.5
            else:
                s = a_start
                Pss = P[s, s]
                Ppp = P[s+1, s+1] + P[s+2, s+2] + P[s+3, s+3]

                sp_fac_1 = gsp - 0.5 * hsp
                sp_fac_2 = 1.5 * hsp - 0.5 * gsp
                pp_fac_d = 1.25 * gp2 - 0.25 * gpp
                pp_fac_off = 0.75 * gpp - 1.25 * gp2

                F[s, s] += Pss * gss * 0.5 + Ppp * sp_fac_1

                for k in range(1, 4):
                    pk = s + k
                    F[pk, pk] += (Pss * sp_fac_1
                                  + P[pk, pk] * gpp * 0.5
                                  + (Ppp - P[pk, pk]) * pp_fac_d)

                for k in range(1, 4):
                    pk = s + k
                    F[s, pk] += P[s, pk] * sp_fac_2
                    F[pk, s] += P[pk, s] * sp_fac_2

                for k in range(1, 4):
                    for l in range(k + 1, 4):
                        pk, pl = s + k, s + l
                        F[pk, pl] += P[pk, pl] * pp_fac_off
                        F[pl, pk] += P[pl, pk] * pp_fac_off

        # Two-center: full w tensor
        for a in range(n_at):
            for b in range(a + 1, n_at):
                sA = starts[a]
                sB = starts[b]
                nA = starts[a + 1] - sA
                nB = starts[b + 1] - sB
                w = batch.w[mol, a, b].reshape(4, 4, 4, 4)

                for mu_a in range(nA):
                    for nu_a in range(nA):
                        mu = sA + mu_a
                        nu = sA + nu_a
                        for lam_b in range(nB):
                            for sig_b in range(nB):
                                lam = sB + lam_b
                                sig = sB + sig_b
                                wval = w[mu_a, nu_a, lam_b, sig_b]
                                # Coulomb A from B
                                F[mu, nu] += P[lam, sig] * wval
                                # Coulomb B from A
                                F[lam, sig] += P[mu, nu] * wval
                                # Exchange (both triangles)
                                F[mu, lam] -= 0.5 * P[nu, sig] * wval
                                F[lam, mu] -= 0.5 * P[sig, nu] * wval

        F_all[mol, :n_bas, :n_bas] = F

    return F_all
