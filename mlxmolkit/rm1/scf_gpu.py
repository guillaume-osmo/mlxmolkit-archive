"""All-on-GPU PM6_D SCF via MLX + mlx-addons.

Strategy:
  - Pre-compute all per-atom and per-pair integrals on CPU (numpy + PYSEQM
    delegation), then materialize them as ``mx.array`` once before the
    SCF loop.
  - The iteration runs entirely on Metal:
        Fock build via mx.einsum (JIT-compiled via mx.compile)
        DIIS extrapolation via mlx_addons.solvers.pulay_diis
        Eigendecomposition via mlx_addons.linalg.batched_eigh
            (Metal Jacobi for N ≤ 32, CPU fallback for larger)
        Density update via matmul
  - For multiple molecules, the integrals stay per-molecule (different
    sparsity patterns make padding wasteful) but the SCF loop runs
    GPU-resident for each one, with a single ``mx.eval`` at the end.

Requires:
  - mlx (Metal): pip install mlx
  - mlx-addons:  pip install mlx-addons  (or local checkout on PYTHONPATH)
"""
from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
    from mlx_addons.linalg import batched_eigh
    from mlx_addons.solvers import pulay_diis, commutator_error
    _HAS_MLX = True
except ImportError:  # pragma: no cover
    _HAS_MLX = False
    mx = None  # type: ignore

from .methods import get_params
from .scf import _build_basis_info, _build_core_hamiltonian
from .scf_mlx import precompute_pair_cache, _precompute_one_center_W


# --------------------------------------------------------------------------
# Materialize cached integrals as MLX arrays (GPU-resident)
# --------------------------------------------------------------------------

def _mx(arr, dtype=None):
    if dtype is None:
        dtype = mx.float32  # Metal Jacobi requires float32
    return mx.array(np.asarray(arr, dtype=np.float32 if dtype == mx.float32 else np.float64),
                    dtype=dtype)


def _materialize_cache_mx(cache, starts, n_basis):
    """Stack per-pair integrals into a list of MLX arrays + index tables."""
    pairs_mx = []
    for (i, j), entry in cache.items():
        e_mx = {'nA': entry['nA'], 'nB': entry['nB'],
                'sA': starts[i], 'sB': starts[j]}
        e_mx['w_sp'] = _mx(entry['w_sp'])
        for k in ('w_yh_A', 'w_yh_B', 'w_yx_AB', 'w_yx_BA', 'w_yy'):
            if k in entry:
                e_mx[k] = _mx(entry[k])
        pairs_mx.append(e_mx)
    return pairs_mx


# --------------------------------------------------------------------------
# One-center G(P) on MLX
# --------------------------------------------------------------------------

def _one_center_g_mx(P, params_info, W_one_mx):
    """One-center G contribution. Walks atoms in Python but the work per
    atom is small (no integral re-computation; just contractions).
    """
    params = params_info['params']
    starts = params_info['atom_basis_start']
    G = mx.zeros_like(P)
    for i, p in enumerate(params):
        s = starts[i]
        Pss = P[s, s]
        if p.n_basis == 1:
            G = G.at[s, s].add(Pss * p.gss * 0.5)
            continue
        Ppx = P[s + 1, s + 1]; Ppy = P[s + 2, s + 2]; Ppz = P[s + 3, s + 3]
        Ppp_total = Ppx + Ppy + Ppz
        sp_fac_1 = p.gsp - 0.5 * p.hsp
        sp_fac_2 = 1.5 * p.hsp - 0.5 * p.gsp
        pp_fac_d = 1.25 * p.gp2 - 0.25 * p.gpp
        pp_fac_off = 0.75 * p.gpp - 1.25 * p.gp2
        G = G.at[s, s].add(Pss * p.gss * 0.5 + Ppp_total * sp_fac_1)
        for k in range(1, 4):
            pk = s + k
            G = G.at[pk, pk].add(
                Pss * sp_fac_1
                + P[pk, pk] * p.gpp * 0.5
                + (Ppp_total - P[pk, pk]) * pp_fac_d
            )
            G = G.at[s, pk].add(P[s, pk] * sp_fac_2)
            G = G.at[pk, s].add(P[pk, s] * sp_fac_2)
        for k in range(1, 4):
            for l in range(k + 1, 4):
                pk, pl = s + k, s + l
                G = G.at[pk, pl].add(P[pk, pl] * pp_fac_off)
                G = G.at[pl, pk].add(P[pl, pk] * pp_fac_off)
        if p.n_basis == 9 and W_one_mx[i] is not None:
            # d-orbital one-center via packed 243-element W
            G = _fock_d_one_center_mx(G, P, W_one_mx[i], s)
    return G


# Precompute index tables for fock_d_one_center on MLX
def _tril9():
    I, J = [], []
    for i in range(9):
        for j in range(i + 1):
            I.append(i); J.append(j)
    return np.array(I), np.array(J)


_TRIL9_I, _TRIL9_J = _tril9()
_WEIGHT_45 = np.where(_TRIL9_I == _TRIL9_J, 1.0, 2.0).astype(np.float32)
# PM6 FLOCAL_MAP from fock_d.py — same as numpy version
from .fock_d import PM6_FLOCAL_MAP


def _fock_d_one_center_mx(G_mx, P_mx, W_mx, s):
    """MLX port of fock_d_one_center — adds d-orbital one-center to G."""
    # Pack P 9x9 block into lower-triangle 45 elements (with weight 2 for off-diag)
    weights = mx.array(_WEIGHT_45, dtype=G_mx.dtype)
    i_idx = mx.array(_TRIL9_I, dtype=mx.int32)
    j_idx = mx.array(_TRIL9_J, dtype=mx.int32)
    # P_packed[k] = P[s+i, s+j] * weight[k]
    P_block = P_mx[s:s + 9, s:s + 9]
    P_packed = P_block[i_idx, j_idx] * weights  # (45,)

    F_packed = mx.zeros(45, dtype=G_mx.dtype)
    for col, w_idxs, p_idxs in PM6_FLOCAL_MAP:
        w_sel = W_mx[mx.array(w_idxs, dtype=mx.int32)]
        p_sel = P_packed[mx.array(p_idxs, dtype=mx.int32)]
        F_packed = F_packed.at[col].add((w_sel * p_sel).sum())

    # Unpack back to 9×9 symmetric
    for k in range(45):
        i, j = int(_TRIL9_I[k]), int(_TRIL9_J[k])
        v = F_packed[k]
        G_mx = G_mx.at[s + i, s + j].add(v)
        if i != j:
            G_mx = G_mx.at[s + j, s + i].add(v)
    return G_mx


# --------------------------------------------------------------------------
# Two-center G(P) on MLX (vectorized einsum per pair)
# --------------------------------------------------------------------------

def _two_center_g_mx(P, pairs_mx):
    """MLX two-center G contribution via einsum per pair."""
    G = mx.zeros_like(P)
    for entry in pairs_mx:
        sA, sB = entry['sA'], entry['sB']
        nA, nB = entry['nA'], entry['nB']
        nA_sp = min(nA, 4); nB_sp = min(nB, 4)
        # sp×sp×sp×sp portion (always present)
        w_sp = entry['w_sp'][:nA_sp, :nA_sp, :nB_sp, :nB_sp]
        P_AA_sp = P[sA:sA + nA_sp, sA:sA + nA_sp]
        P_BB_sp = P[sB:sB + nB_sp, sB:sB + nB_sp]
        P_AB_sp = P[sA:sA + nA_sp, sB:sB + nB_sp]
        J_AA = mx.einsum('abcd,cd->ab', w_sp, P_BB_sp)
        J_BB = mx.einsum('abcd,ab->cd', w_sp, P_AA_sp)
        K_sp = -0.5 * mx.einsum('abcd,bd->ac', w_sp, P_AB_sp)
        G = G.at[sA:sA + nA_sp, sA:sA + nA_sp].add(J_AA)
        G = G.at[sB:sB + nB_sp, sB:sB + nB_sp].add(J_BB)
        G = G.at[sA:sA + nA_sp, sB:sB + nB_sp].add(K_sp)
        G = G.at[sB:sB + nB_sp, sA:sA + nA_sp].add(mx.swapaxes(K_sp, 0, 1))

        # YH / YX / YY extensions follow the same skip-the-sp-block pattern
        if 'w_yh_A' in entry:
            W = entry['w_yh_A']  # (9, 9)
            P_BB_full = P[sB, sB]
            J = P_BB_full * W
            J_sub = P_BB_full * W[:4, :4]
            G = G.at[sA:sA + 9, sA:sA + 9].add(J)
            G = G.at[sA:sA + 4, sA:sA + 4].add(-J_sub)
            jc = (P[sA:sA + 9, sA:sA + 9] * W).sum()
            jc_sp = (P[sA:sA + 4, sA:sA + 4] * W[:4, :4]).sum()
            G = G.at[sB, sB].add(jc - jc_sp)
            ksum = (P[sA:sA + 9, sB] * W).sum(axis=1)
            ksum_sp = (P[sA:sA + 4, sB] * W[:4, :4]).sum(axis=1)
            G = G.at[sA:sA + 9, sB].add(-0.5 * ksum)
            G = G.at[sB, sA:sA + 9].add(-0.5 * ksum)
            G = G.at[sA:sA + 4, sB].add(0.5 * ksum_sp)
            G = G.at[sB, sA:sA + 4].add(0.5 * ksum_sp)
        elif 'w_yh_B' in entry:
            W = entry['w_yh_B']
            P_AA_full = P[sA, sA]
            J = P_AA_full * W
            J_sub = P_AA_full * W[:4, :4]
            G = G.at[sB:sB + 9, sB:sB + 9].add(J)
            G = G.at[sB:sB + 4, sB:sB + 4].add(-J_sub)
            jc = (P[sB:sB + 9, sB:sB + 9] * W).sum()
            jc_sp = (P[sB:sB + 4, sB:sB + 4] * W[:4, :4]).sum()
            G = G.at[sA, sA].add(jc - jc_sp)
            ksum = (P[sB:sB + 9, sA] * W).sum(axis=1)
            ksum_sp = (P[sB:sB + 4, sA] * W[:4, :4]).sum(axis=1)
            G = G.at[sB:sB + 9, sA].add(-0.5 * ksum)
            G = G.at[sA, sB:sB + 9].add(-0.5 * ksum)
            G = G.at[sB:sB + 4, sA].add(0.5 * ksum_sp)
            G = G.at[sA, sB:sB + 4].add(0.5 * ksum_sp)
        elif 'w_yx_AB' in entry:
            W = entry['w_yx_AB']  # (9, 9, 4, 4)
            P_AA = P[sA:sA + 9, sA:sA + 9]
            P_BB = P[sB:sB + 4, sB:sB + 4]
            P_AB = P[sA:sA + 9, sB:sB + 4]
            J_A = mx.einsum('abcd,cd->ab', W, P_BB)
            J_A_sp = mx.einsum('abcd,cd->ab', W[:4, :4, :, :], P_BB)
            G = G.at[sA:sA + 9, sA:sA + 9].add(J_A)
            G = G.at[sA:sA + 4, sA:sA + 4].add(-J_A_sp)
            J_B = mx.einsum('abcd,ab->cd', W, P_AA)
            J_B_sp = mx.einsum('abcd,ab->cd', W[:4, :4, :, :], P_AA[:4, :4])
            G = G.at[sB:sB + 4, sB:sB + 4].add(J_B - J_B_sp)
            K = -0.5 * mx.einsum('abcd,bd->ac', W, P_AB)
            K_sp = -0.5 * mx.einsum('abcd,bd->ac', W[:4, :4, :, :], P_AB[:4, :])
            G = G.at[sA:sA + 9, sB:sB + 4].add(K)
            G = G.at[sB:sB + 4, sA:sA + 9].add(mx.swapaxes(K, 0, 1))
            G = G.at[sA:sA + 4, sB:sB + 4].add(-K_sp)
            G = G.at[sB:sB + 4, sA:sA + 4].add(-mx.swapaxes(K_sp, 0, 1))
        elif 'w_yx_BA' in entry:
            W = entry['w_yx_BA']
            P_AA = P[sA:sA + 4, sA:sA + 4]
            P_BB = P[sB:sB + 9, sB:sB + 9]
            P_BA = P[sB:sB + 9, sA:sA + 4]
            J_B = mx.einsum('abcd,cd->ab', W, P_AA)
            J_B_sp = mx.einsum('abcd,cd->ab', W[:4, :4, :, :], P_AA)
            G = G.at[sB:sB + 9, sB:sB + 9].add(J_B)
            G = G.at[sB:sB + 4, sB:sB + 4].add(-J_B_sp)
            J_A = mx.einsum('abcd,ab->cd', W, P_BB)
            J_A_sp = mx.einsum('abcd,ab->cd', W[:4, :4, :, :], P_BB[:4, :4])
            G = G.at[sA:sA + 4, sA:sA + 4].add(J_A - J_A_sp)
            K = -0.5 * mx.einsum('abcd,bd->ac', W, P_BA)
            K_sp = -0.5 * mx.einsum('abcd,bd->ac', W[:4, :4, :, :], P_BA[:4, :])
            G = G.at[sB:sB + 9, sA:sA + 4].add(K)
            G = G.at[sA:sA + 4, sB:sB + 9].add(mx.swapaxes(K, 0, 1))
            G = G.at[sB:sB + 4, sA:sA + 4].add(-K_sp)
            G = G.at[sA:sA + 4, sB:sB + 4].add(-mx.swapaxes(K_sp, 0, 1))
        elif 'w_yy' in entry:
            W = entry['w_yy']  # (9, 9, 9, 9)
            P_AA = P[sA:sA + 9, sA:sA + 9]
            P_BB = P[sB:sB + 9, sB:sB + 9]
            P_AB = P[sA:sA + 9, sB:sB + 9]
            J_A = mx.einsum('abcd,cd->ab', W, P_BB)
            J_A_sp = mx.einsum('abcd,cd->ab', W[:4, :4, :4, :4], P_BB[:4, :4])
            G = G.at[sA:sA + 9, sA:sA + 9].add(J_A)
            G = G.at[sA:sA + 4, sA:sA + 4].add(-J_A_sp)
            J_B = mx.einsum('cdab,cd->ab', W, P_AA)
            J_B_sp = mx.einsum('cdab,cd->ab', W[:4, :4, :4, :4], P_AA[:4, :4])
            G = G.at[sB:sB + 9, sB:sB + 9].add(J_B)
            G = G.at[sB:sB + 4, sB:sB + 4].add(-J_B_sp)
            K = -0.5 * mx.einsum('abcd,bd->ac', W, P_AB)
            K_sp = -0.5 * mx.einsum('abcd,bd->ac',
                                    W[:4, :4, :4, :4], P_AB[:4, :4])
            G = G.at[sA:sA + 9, sB:sB + 9].add(K)
            G = G.at[sB:sB + 9, sA:sA + 9].add(mx.swapaxes(K, 0, 1))
            G = G.at[sA:sA + 4, sB:sB + 4].add(-K_sp)
            G = G.at[sB:sB + 4, sA:sA + 4].add(-mx.swapaxes(K_sp, 0, 1))
    return G


# --------------------------------------------------------------------------
# Main entry point — per-molecule GPU SCF
# --------------------------------------------------------------------------

def rm1_pm6d_native_gpu(atoms, coords, max_iter=200, conv_tol=1e-5,
                       use_jit=True, dtype='float32'):
    """Per-molecule native PM6_D SCF on Metal GPU.

    Pre-computes integrals once on CPU (numpy + PYSEQM delegation), then
    runs the SCF iteration entirely on Metal via mlx-addons.

    Args:
        atoms: list of atomic numbers
        coords: (N, 3) coordinates in Å
        max_iter: SCF iteration cap
        conv_tol: density convergence threshold (RMS)
        use_jit: wrap Fock build in mx.compile for graph fusion
        dtype: 'float32' (GPU Jacobi requires) or 'float64' (CPU fallback)

    Returns:
        dict with 'q', 'converged', 'n_iter', 'P'.
    """
    if not _HAS_MLX:
        raise ImportError("mlx + mlx-addons required for GPU SCF")
    dt = mx.float32 if dtype == 'float32' else mx.float64

    PARAMS = get_params('PM6_D')
    coords = np.asarray(coords, dtype=np.float64)
    info = _build_basis_info(atoms, PARAMS)
    n_basis = info['n_basis']; n_occ = info['n_occ']
    params = info['params']; starts = info['atom_basis_start']

    # Pre-compute integrals on CPU
    H_np = _build_core_hamiltonian(atoms, coords, info)
    W_one_np = [_precompute_one_center_W(p) for p in params]
    cache_np = precompute_pair_cache(atoms, coords, params)

    # Move to MLX (GPU)
    H = mx.array(H_np.astype(np.float32 if dtype == 'float32' else np.float64), dtype=dt)
    W_one_mx = [mx.array(w.astype(np.float32 if dtype == 'float32' else np.float64),
                         dtype=dt) if w is not None else None for w in W_one_np]
    pairs_mx = _materialize_cache_mx(cache_np, starts, n_basis)
    if dtype == 'float32':
        # cast pairs_mx in-place
        for e in pairs_mx:
            for k in list(e.keys()):
                if isinstance(e[k], mx.array):
                    e[k] = e[k].astype(mx.float32)

    # Initial density (PYSEQM diagonal init)
    P_init = np.zeros((n_basis, n_basis), dtype=np.float64)
    for i, p in enumerate(params):
        sA = starts[i]
        if p.n_basis == 1:
            P_init[sA, sA] = 1.0
        else:
            val = p.n_valence / 4.0
            for k in range(4):
                P_init[sA + k, sA + k] = val
    P = mx.array(P_init.astype(np.float32 if dtype == 'float32' else np.float64), dtype=dt)

    # Fock builder (optionally JIT-compiled)
    def build_F(P):
        G = _one_center_g_mx(P, info, W_one_mx)
        G = G + _two_center_g_mx(P, pairs_mx)
        F = H + G
        return 0.5 * (F + mx.swapaxes(F, 0, 1))

    if use_jit:
        build_F = mx.compile(build_F)

    # SCF
    has_d = any(p.n_basis > 4 for p in params)
    diis_F: list = []
    diis_E: list = []
    converged = False
    it = 0
    for it in range(max_iter):
        F = build_F(P)
        # DIIS
        if it >= 2:
            err = commutator_error(F, P)
            diis_F.append(F); diis_E.append(err)
            if len(diis_F) > 6:
                diis_F.pop(0); diis_E.pop(0)
            if len(diis_F) >= 2:
                F = pulay_diis(diis_F, diis_E, max_history=6)
        _, C = batched_eigh(F)
        C_occ = C[:, :n_occ]
        Pn = 2.0 * C_occ @ mx.swapaxes(C_occ, 0, 1)
        d = mx.sqrt(((Pn - P) ** 2).mean())
        d_val = float(d.item())
        if d_val < conv_tol:
            converged = True
            P = Pn
            break
        # Mixing
        if it < 3:
            mix = 0.3 if has_d else 0.5
        elif d_val > 0.1:
            mix = 0.05 if has_d else 0.4
        elif d_val > 0.01:
            mix = 0.5
        else:
            mix = 0.8
        P = mix * Pn + (1.0 - mix) * P
    P_np = np.asarray(P)
    q = np.array([
        p.n_valence - float(sum(P_np[starts[i] + k, starts[i] + k] for k in range(p.n_basis)))
        for i, p in enumerate(params)
    ])
    return {'q': q, 'converged': converged, 'n_iter': it + 1, 'P': P_np}
