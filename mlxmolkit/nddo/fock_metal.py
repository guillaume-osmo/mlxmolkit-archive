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

from .fock_d import fock_d_one_center
from .packing import index_matrix, packed_size, unpack


_FOCK_BATCH_SOURCE = """
uint tid = thread_position_in_grid.x;
int n_mols = (int)config[0];
int MB = (int)config[1];      // max_basis (padded)
int MA = (int)config[2];      // max_atoms (padded)
int MO = (int)config[3];      // widest per-atom basis: 4 for sp, 9 with d
int WSTRIDE = (int)config[4]; // per-molecule packed two-centre buffer length
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
int mol_W   = mol * WSTRIDE;
int mol_MA2 = mol * MA * MA;

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
            } else if (type_mu < 4) {
                // p orbital: PYSEQM factors. Guarded to type < 4 — on a
                // 9-orbital atom types 4-8 are d, and the sp formulas do not
                // apply to them; their one-centre terms come from the W
                // integrals below instead.
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
        // Same guard as the diagonal: any pair touching a d orbital is
        // handled by the W integrals, not by the sp factors.
        if (n_orb > 1 && type_mu < 4 && type_nu < 4) {
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
// ONE-CENTER d: the s-d, p-d and d-d cross terms
// =========================================================
// Additive on top of the sp formulas above, which already cover orbitals
// 0-3 of a 9-orbital atom. Each thread evaluates exactly the packed element
// it owns, so the sparse PM6 F-local map becomes a short CSR walk.
if (atom_mu == atom_nu) {
    int a = atom_mu;
    int a_start = atom_starts[mol_MA + a];
    int n_orb = atom_starts[mol_MA + a + 1] - a_start;
    if (n_orb == 9) {
        int mo = mu - a_start;
        int no = nu - a_start;
        int col = (mo >= no) ? (mo * (mo + 1) / 2 + no) : (no * (no + 1) / 2 + mo);
        int wbase = (mol * MA + a) * 243;
        float acc = 0.0f;
        for (int e = flocal_start[col]; e < flocal_start[col + 1]; e++) {
            int pi = flocal_p[e];
            int ii = tril_i[pi];
            int jj = tril_j[pi];
            float weight = (ii == jj) ? 1.0f : 2.0f;
            acc += atom_w[wbase + flocal_w[e]]
                 * P[mol_MB2 + (a_start + ii) * MB + (a_start + jj)] * weight;
        }
        f += acc;
    }
}

// =========================================================
// TWO-CENTER: Coulomb + Exchange using the packed w blocks
// =========================================================
// Packed layout: pair_offset[mol, i, j] gives the start of the (i, j) block
// inside this molecule's buffer. A block is (packed(nA), packed(nB)) row
// major, packed(n) = n(n+1)/2, and the pair index is the lower-triangle one.

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
        int w_base = mol_W + pair_offset[mol_MA2 + atom_mu * MA + b];
        int pB = nB * (nB + 1) / 2;
        int row = ((mu_off >= nu_off) ? (mu_off * (mu_off + 1) / 2 + nu_off)
                                      : (nu_off * (nu_off + 1) / 2 + mu_off)) * pB;
        for (int ls = 0; ls < nB; ls++) {
            for (int ss = 0; ss < nB; ss++) {
                int lam = b_start + ls;
                int sig = b_start + ss;
                int col = (ls >= ss) ? (ls * (ls + 1) / 2 + ss)
                                     : (ss * (ss + 1) / 2 + ls);
                f += P[mol_MB2 + lam * MB + sig] * w[w_base + row + col];
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
    int w_base = mol_W + pair_offset[mol_MA2 + a * MA + b];
    int pB = nB * (nB + 1) / 2;
    for (int nA_off = 0; nA_off < nA; nA_off++) {
        for (int sB_off = 0; sB_off < nB; sB_off++) {
            int nu_global = a_start + nA_off;
            int sig_global = b_start + sB_off;
            int row = ((mu_off >= nA_off) ? (mu_off * (mu_off + 1) / 2 + nA_off)
                                          : (nA_off * (nA_off + 1) / 2 + mu_off)) * pB;
            int col = (lam_off >= sB_off) ? (lam_off * (lam_off + 1) / 2 + sB_off)
                                          : (sB_off * (sB_off + 1) / 2 + lam_off);
            exch += P[mol_MB2 + nu_global * MB + sig_global] * w[w_base + row + col];
        }
    }
    f -= 0.5f * exch;
}

F_out[tid] = f;
"""

_fock_batch_kernel = None



def _flocal_csr():
    """Flatten PM6_FLOCAL_MAP into CSR arrays the kernel can walk.

    The map is a fixed sparse contraction: each of the 45 packed Fock elements
    on a d atom is a short sum over one-centre W integrals times packed density
    elements. Flattened once here so a thread can evaluate only its own element.
    """
    from .fock_d import PM6_FLOCAL_MAP, TRIL_I, TRIL_J

    start = np.zeros(46, dtype=np.int32)
    w_idx, p_idx = [], []
    for col, ws, ps in PM6_FLOCAL_MAP:
        start[col + 1] = len(ws)
        w_idx.extend(ws)
        p_idx.extend(ps)
    start = np.cumsum(start).astype(np.int32)
    return (start, np.array(w_idx, dtype=np.int32), np.array(p_idx, dtype=np.int32),
            TRIL_I.astype(np.int32), TRIL_J.astype(np.int32))


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
                         "n_atoms_arr", "n_basis_arr", "config",
                         "pair_offset", "atom_w",
                         "flocal_start", "flocal_w", "flocal_p",
                         "tril_i", "tril_j"],
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
        self._pair_offset = mx.array(batch.pair_offset.flatten().astype(np.int32))
        self._atom_w = mx.array(batch.atom_w.flatten().astype(np.float32))
        fs, fw, fp, ti, tj = _flocal_csr()
        self._flocal_start = mx.array(fs)
        self._flocal_w = mx.array(fw)
        self._flocal_p = mx.array(fp)
        self._tril_i = mx.array(ti)
        self._tril_j = mx.array(tj)
        self._config = mx.array(np.array(
            [N, MB, MA, batch.max_orb, batch.w.shape[1]], dtype=np.float32))
        self._kernel = _get_fock_batch_kernel()

    def build_fock(self, P: np.ndarray) -> np.ndarray:
        """Build Fock from density P. Only P is transferred each call.

        NumPy in / NumPy out. For all-MLX SCF loops use :meth:`build_fock_mlx`
        instead — it avoids the per-iteration GPU→CPU round-trip.
        """
        P_mx = mx.array(P.flatten().astype(np.float32))
        F_mx = self._kernel_call(P_mx)
        mx.eval(F_mx)
        return np.array(F_mx).reshape(self.N, self.MB, self.MB).astype(np.float64)

    def build_fock_mlx(self, P_mx: mx.array) -> mx.array:
        """Build Fock from density P (mx.array in, mx.array out, no host copy).

        ``P_mx`` may be ``(N, MB, MB)`` or pre-flattened ``(N*MB*MB,)``;
        returns ``(N, MB, MB)`` ``mx.array`` of dtype ``float32``.
        """
        if P_mx.ndim == 3:
            P_flat = mx.reshape(P_mx, (self.n_elements,))
        else:
            P_flat = P_mx
        if P_flat.dtype != mx.float32:
            P_flat = P_flat.astype(mx.float32)
        F_flat = self._kernel_call(P_flat)
        return mx.reshape(F_flat, (self.N, self.MB, self.MB))

    def _kernel_call(self, P_flat_mx: mx.array) -> mx.array:
        outputs = self._kernel(
            inputs=[
                self._H_core, P_flat_mx, self._w,
                self._atom_params, self._atom_map, self._type_map,
                self._atom_starts, self._n_atoms_arr, self._n_basis_arr,
                self._config,
                self._pair_offset, self._atom_w,
                self._flocal_start, self._flocal_w, self._flocal_p,
                self._tril_i, self._tril_j,
            ],
            output_shapes=[(self.n_elements,)],
            output_dtypes=[mx.float32],
            grid=(self.n_elements, 1, 1),
            threadgroup=(min(256, self.n_elements), 1, 1),
        )
        return outputs[0]


def build_fock_batch_metal(batch) -> np.ndarray:
    """Build Fock matrices for all molecules in batch on Metal GPU.

    NOTE: For SCF loops, use MetalFockContext instead to avoid
    re-uploading static buffers every iteration.
    """
    ctx = MetalFockContext(batch)
    return ctx.build_fock(batch.P)


def build_fock_batch_cpu_reference(batch) -> np.ndarray:
    """Build Fock matrices for all molecules on CPU, one scalar element at a time.

    The obvious implementation, kept as the oracle
    :func:`build_fock_batch_cpu` is checked against. It is deliberately dumb —
    five nested Python loops over (molecule, pair, mu, nu, lam, sig) — so that
    reading it is enough to believe it.

    Do not call this in anger. It runs 256 scalar iterations per sp pair per
    molecule per SCF iteration: 57 s for a 200-molecule PM6 batch, against
    2.5 s for simply looping `nddo_energy` over the same molecules.
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

        # One-centre d contribution. The sp formulas above already cover
        # orbitals 0-3 of a 9-orbital atom correctly; the W integrals add the
        # s-d, p-d and d-d cross terms on top rather than replacing anything.
        for a in range(n_at):
            if starts[a + 1] - starts[a] == 9:
                from .fock_d import fock_d_one_center
                F = fock_d_one_center(F, P, batch.atom_w[mol, a], starts[a])

        # Two-center: full w tensor
        for a in range(n_at):
            for b in range(a + 1, n_at):
                sA = starts[a]
                sB = starts[b]
                nA = starts[a + 1] - sA
                nB = starts[b + 1] - sB
                # Packed pair block -> dense (nA, nA, nB, nB) for the loop
                # below. The reference path keeps the dense contraction; only
                # the storage changed.
                off = int(batch.pair_offset[mol, a, b])
                pa, pb = packed_size(nA), packed_size(nB)
                w = unpack(batch.w[mol, off:off + pa * pb].reshape(pa, pb),
                           nA, nB)

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


def _fock_batch_plan(batch):
    """Everything a batched CPU Fock build needs that does not depend on P.

    The pair list, the orbital shapes, the unpacked w tensors and the flat
    scatter indices are all fixed for a batch's geometry, yet the reference
    path rederived them inside its innermost loop on every SCF iteration.

    Groups are keyed on (nA, nB) across the *whole batch*, not per molecule:
    a 200-molecule batch of ordinary organics has three or four distinct
    orbital shapes and tens of thousands of pairs, so one contraction per
    shape replaces one per pair.

    Returns:
        (one_centre, groups) where `groups` is a list of
        (w, rows_a, rows_b, flat) with `w` of shape (G, nA, nA, nB, nB) and
        `flat` the concatenated scatter indices into a flattened (N, MB, MB).
    """
    N, MB = batch.n_mols, batch.max_basis
    stride = MB * MB

    shapes: dict[tuple[int, int], list] = {}
    one_centre = []
    for mol in range(N):
        n_at = batch.n_atoms_arr[mol]
        starts = batch.atom_starts[mol]
        for a in range(n_at):
            if starts[a + 1] - starts[a] == 9:
                one_centre.append((mol, a, int(starts[a])))
            for b in range(a + 1, n_at):
                off = int(batch.pair_offset[mol, a, b])
                if off < 0:          # no block stored for this pair
                    continue
                nA = int(starts[a + 1] - starts[a])
                nB = int(starts[b + 1] - starts[b])
                shapes.setdefault((nA, nB), []).append(
                    (mol, int(starts[a]), int(starts[b]), off))

    groups = []
    for (nA, nB), entries in shapes.items():
        mols = np.fromiter((e[0] for e in entries), int, len(entries))
        sA = np.fromiter((e[1] for e in entries), int, len(entries))
        sB = np.fromiter((e[2] for e in entries), int, len(entries))
        offs = np.fromiter((e[3] for e in entries), int, len(entries))

        pa, pb = packed_size(nA), packed_size(nB)
        block = pa * pb
        # One gather of every packed block in the group, then one unpack.
        flat_w = batch.w[mols[:, None], offs[:, None] + np.arange(block)]
        ia = index_matrix(nA).ravel()
        ib = index_matrix(nB).ravel()
        w = flat_w.reshape(-1, pa, pb)[:, ia[:, None], ib[None, :]]
        w = w.reshape(-1, nA, nA, nB, nB)

        rows_a = sA[:, None] + np.arange(nA)
        rows_b = sB[:, None] + np.arange(nB)
        base = mols * stride
        flat = np.concatenate([
            (base[:, None, None] + r[:, :, None] * MB + c[:, None, :]).ravel()
            for r, c in ((rows_a, rows_a), (rows_b, rows_b),
                         (rows_a, rows_b), (rows_b, rows_a))])

        # The three density contractions as batched matrix products rather
        # than einsum. numpy's einsum does not reach BLAS for (G, 4, 4, 4, 4)
        # against (G, 4, 4) and runs its own loop; `matmul` on (G, m, m) by
        # (G, m, 1) dispatches to batched GEMM and is 3.9x faster at G = 60000,
        # agreeing to 1e-14.
        #
        # Two layouts of the same tensor: [ab, cd] serves the two Coulomb
        # terms, [ac, bd] the exchange one, which contracts b with d. Both are
        # geometry-only, so the reordered copy is made once here and reused by
        # every SCF iteration.
        w_ab_cd = w.reshape(-1, nA * nA, nB * nB)
        w_ac_bd = np.ascontiguousarray(
            w.transpose(0, 1, 3, 2, 4)).reshape(-1, nA * nB, nA * nB)
        groups.append((w_ab_cd, w_ac_bd, nA, nB, mols, rows_a, rows_b, flat))

    return one_centre, groups


def build_fock_batch_cpu(batch) -> np.ndarray:
    """Build Fock matrices for all molecules on CPU, batched by orbital shape.

    Same arithmetic as :func:`build_fock_batch_cpu_reference`, which pins it,
    but the two-centre contraction runs once per distinct orbital shape over
    the whole batch instead of once per (molecule, pair) with a 256-iteration
    Python loop inside.

    The plan is cached on the batch: it depends on the geometry, and an SCF
    holds the geometry fixed for ~20 iterations.
    """
    N, MB = batch.n_mols, batch.max_basis
    F_all = np.zeros((N, MB, MB), dtype=np.float64)

    plan = getattr(batch, '_fock_cpu_plan', None)
    if plan is None:
        plan = _fock_batch_plan(batch)
        batch._fock_cpu_plan = plan
    one_centre, groups = plan

    # One-centre: still per molecule, and cheap — n_atoms terms against
    # n_atoms**2 / 2 pairs, with no inner loop over orbital quartets.
    for mol in range(N):
        n_bas = batch.n_basis_arr[mol]
        n_at = batch.n_atoms_arr[mol]
        P = batch.P[mol, :n_bas, :n_bas]
        F = batch.H_core[mol, :n_bas, :n_bas].copy()
        starts = batch.atom_starts[mol]

        for a in range(n_at):
            s = int(starts[a])
            n_orb = int(starts[a + 1] - s)
            gss, gsp, gpp, gp2, hsp = batch.atom_params[mol, a]
            if n_orb == 1:
                F[s, s] += P[s, s] * gss * 0.5
                continue

            Pss = P[s, s]
            pk = slice(s + 1, s + 4)
            # The p-block diagonal, not the p-block: F[pk, pk] with pk a slice
            # would write all nine entries of the 3x3.
            pdiag = np.arange(s + 1, s + 4)
            diag = P[pdiag, pdiag]
            Ppp = float(diag.sum())
            sp_fac_1 = gsp - 0.5 * hsp
            sp_fac_2 = 1.5 * hsp - 0.5 * gsp
            pp_fac_d = 1.25 * gp2 - 0.25 * gpp
            pp_fac_off = 0.75 * gpp - 1.25 * gp2

            F[s, s] += Pss * gss * 0.5 + Ppp * sp_fac_1
            F[pdiag, pdiag] += (Pss * sp_fac_1 + diag * gpp * 0.5
                                + (Ppp - diag) * pp_fac_d)
            F[s, pk] += P[s, pk] * sp_fac_2
            F[pk, s] += P[pk, s] * sp_fac_2
            for k in range(1, 4):
                for l in range(k + 1, 4):
                    F[s + k, s + l] += P[s + k, s + l] * pp_fac_off
                    F[s + l, s + k] += P[s + l, s + k] * pp_fac_off

        F_all[mol, :n_bas, :n_bas] = F

    for mol, a, s in one_centre:
        n_bas = batch.n_basis_arr[mol]
        F_all[mol, :n_bas, :n_bas] = fock_d_one_center(
            F_all[mol, :n_bas, :n_bas], batch.P[mol, :n_bas, :n_bas],
            batch.atom_w[mol, a], s)

    # Two-centre, one contraction per orbital shape across the whole batch.
    if groups:
        flat_idx, flat_val = [], []
        for w_ab_cd, w_ac_bd, nA, nB, mols, rows_a, rows_b, flat in groups:
            g = mols.size
            P_AA = batch.P[mols[:, None, None], rows_a[:, :, None], rows_a[:, None, :]]
            P_BB = batch.P[mols[:, None, None], rows_b[:, :, None], rows_b[:, None, :]]
            P_AB = batch.P[mols[:, None, None], rows_a[:, :, None], rows_b[:, None, :]]

            t_aa = np.matmul(w_ab_cd,
                             P_BB.reshape(g, nB * nB, 1)).reshape(g, nA, nA)
            t_bb = np.matmul(P_AA.reshape(g, 1, nA * nA),
                             w_ab_cd).reshape(g, nB, nB)
            t_ab = -0.5 * np.matmul(
                w_ac_bd, P_AB.reshape(g, nA * nB, 1)).reshape(g, nA, nB)

            flat_idx.append(flat)
            flat_val.append(np.concatenate([
                t_aa.ravel(), t_bb.ravel(), t_ab.ravel(),
                np.swapaxes(t_ab, 1, 2).ravel()]))

        F_all += np.bincount(
            np.concatenate(flat_idx), weights=np.concatenate(flat_val),
            minlength=N * MB * MB).reshape(N, MB, MB)

    return F_all
