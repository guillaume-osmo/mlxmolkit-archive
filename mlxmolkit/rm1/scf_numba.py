"""Numba-JIT'd Fock contractions for the inner SCF loop.

Profile data on this codebase:
  - sp×sp 4×4×4×4 contractions called many times per SCF iter (e.g. 190
    pair-iterations for a 20-atom molecule × ~25 SCF iters = 4750 calls).
    For tensors this small, np.einsum's per-call overhead dominates.
    Numba-JIT'd explicit loops are ~8× faster.
  - YY 9×9×9×9 contractions: numba and einsum are roughly tied (BLAS
    handles the larger inner reduction well).

This module replaces the sp×sp portion of ``_two_center_g_numpy`` with
``@numba.njit``-compiled kernels. The d-orbital extensions (YH/YX/YY)
stay einsum-based.

Caveat: integral precompute (PYSEQM delegation for YY/YX W tensors)
still dominates d-orbital workloads. Numba helps most for sp-only
molecules where the SCF loop is the bottleneck.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
    def njit(*a, **kw):
        # No-op decorator if numba missing
        if len(a) == 1 and callable(a[0]):
            return a[0]
        def deco(fn): return fn
        return deco

from .methods import get_params
from .scf import _build_basis_info, _build_core_hamiltonian
from .scf_mlx import precompute_pair_cache, _precompute_one_center_W
from .fock_d import fock_d_one_center


@njit(cache=True, fastmath=True)
def _sp_two_center_jit(F, P, w_sp, sA, sB, nA_sp, nB_sp):
    """Add sp×sp J(A from B) + J(B from A) + K cross-atom contributions."""
    # J on A
    for mu in range(nA_sp):
        for nu in range(nA_sp):
            s = 0.0
            for lam in range(nB_sp):
                for sig in range(nB_sp):
                    s += P[sB+lam, sB+sig] * w_sp[mu, nu, lam, sig]
            F[sA+mu, sA+nu] += s
    # J on B
    for lam in range(nB_sp):
        for sig in range(nB_sp):
            s = 0.0
            for mu in range(nA_sp):
                for nu in range(nA_sp):
                    s += P[sA+mu, sA+nu] * w_sp[mu, nu, lam, sig]
            F[sB+lam, sB+sig] += s
    # K cross-atom
    for mu in range(nA_sp):
        for lam in range(nB_sp):
            s = 0.0
            for nu in range(nA_sp):
                for sig in range(nB_sp):
                    s += P[sA+nu, sB+sig] * w_sp[mu, nu, lam, sig]
            K = -0.5 * s
            F[sA+mu, sB+lam] += K
            F[sB+lam, sA+mu] += K


@njit(cache=True, fastmath=True)
def _one_center_sp_jit(F, P, s, gss, gsp, gpp, gp2, hsp, has_p):
    """sp-only one-center G contribution (atoms with n_basis ∈ {1, 4})."""
    Pss = P[s, s]
    if not has_p:
        F[s, s] += Pss * gss * 0.5
        return
    Ppx = P[s+1, s+1]; Ppy = P[s+2, s+2]; Ppz = P[s+3, s+3]
    Ppp_total = Ppx + Ppy + Ppz
    sp_fac_1 = gsp - 0.5 * hsp
    sp_fac_2 = 1.5 * hsp - 0.5 * gsp
    pp_fac_d = 1.25 * gp2 - 0.25 * gpp
    pp_fac_off = 0.75 * gpp - 1.25 * gp2
    F[s, s] += Pss * gss * 0.5 + Ppp_total * sp_fac_1
    for k in range(1, 4):
        pk = s + k
        F[pk, pk] += (Pss * sp_fac_1 + P[pk, pk] * gpp * 0.5
                      + (Ppp_total - P[pk, pk]) * pp_fac_d)
        F[s, pk] += P[s, pk] * sp_fac_2
        F[pk, s] += P[pk, s] * sp_fac_2
    for k in range(1, 4):
        for l in range(k+1, 4):
            pk, pl = s + k, s + l
            F[pk, pl] += P[pk, pl] * pp_fac_off
            F[pl, pk] += P[pl, pk] * pp_fac_off


def _build_G_numba(P, info, W_one_np, cache):
    """G(P) build: numba kernels for sp parts + einsum for d-orbital parts."""
    params = info['params']
    starts = info['atom_basis_start']
    G = np.zeros_like(P)
    # One-center
    for i, p in enumerate(params):
        s = starts[i]
        if p.n_basis == 1:
            _one_center_sp_jit(G, P, s, p.gss, 0.0, 0.0, 0.0, 0.0, False)
        else:
            _one_center_sp_jit(G, P, s, p.gss, p.gsp, p.gpp, p.gp2, p.hsp, True)
        if p.n_basis == 9 and W_one_np[i] is not None:
            G = fock_d_one_center(G, P, W_one_np[i], s, n_basis=9)
    # Two-center
    for (i, j), entry in cache.items():
        sA, sB = starts[i], starts[j]
        nA, nB = entry['nA'], entry['nB']
        nA_sp = min(nA, 4); nB_sp = min(nB, 4)
        w_sp = entry['w_sp']
        # Numba kernel handles sp×sp
        _sp_two_center_jit(G, P, w_sp, sA, sB, nA_sp, nB_sp)
        # d-orbital extensions: einsum (currently optimal for 9-basis tensors)
        if 'w_yh_A' in entry:
            W = entry['w_yh_A']
            P_BB_full = P[sB, sB]
            J = P_BB_full * W
            J[:4, :4] -= P_BB_full * W[:4, :4]
            G[sA:sA+9, sA:sA+9] += J
            jc = (P[sA:sA+9, sA:sA+9] * W).sum()
            jc_sp = (P[sA:sA+4, sA:sA+4] * W[:4, :4]).sum()
            G[sB, sB] += jc - jc_sp
            ksum = (P[sA:sA+9, sB] * W).sum(axis=1)
            ksum_sp = (P[sA:sA+4, sB] * W[:4, :4]).sum(axis=1)
            G[sA:sA+9, sB] -= 0.5 * ksum
            G[sB, sA:sA+9] -= 0.5 * ksum
            G[sA:sA+4, sB] += 0.5 * ksum_sp
            G[sB, sA:sA+4] += 0.5 * ksum_sp
        elif 'w_yh_B' in entry:
            W = entry['w_yh_B']
            P_AA_full = P[sA, sA]
            J = P_AA_full * W
            J[:4, :4] -= P_AA_full * W[:4, :4]
            G[sB:sB+9, sB:sB+9] += J
            jc = (P[sB:sB+9, sB:sB+9] * W).sum()
            jc_sp = (P[sB:sB+4, sB:sB+4] * W[:4, :4]).sum()
            G[sA, sA] += jc - jc_sp
            ksum = (P[sB:sB+9, sA] * W).sum(axis=1)
            ksum_sp = (P[sB:sB+4, sA] * W[:4, :4]).sum(axis=1)
            G[sB:sB+9, sA] -= 0.5 * ksum
            G[sA, sB:sB+9] -= 0.5 * ksum
            G[sB:sB+4, sA] += 0.5 * ksum_sp
            G[sA, sB:sB+4] += 0.5 * ksum_sp
        elif 'w_yx_AB' in entry:
            W = entry['w_yx_AB']
            P_AA = P[sA:sA+9, sA:sA+9]
            P_BB = P[sB:sB+4, sB:sB+4]
            P_AB = P[sA:sA+9, sB:sB+4]
            # Only the (μ or ν in d) extensions; sp×sp already handled
            J_A = np.einsum('abcd,cd->ab', W, P_BB)
            J_A_sp = np.einsum('abcd,cd->ab', W[:4, :4, :, :], P_BB)
            G[sA:sA+9, sA:sA+9] += J_A
            G[sA:sA+4, sA:sA+4] -= J_A_sp
            J_B = np.einsum('abcd,ab->cd', W, P_AA)
            J_B_sp = np.einsum('abcd,ab->cd', W[:4, :4, :, :], P_AA[:4, :4])
            G[sB:sB+4, sB:sB+4] += J_B - J_B_sp
            K = -0.5 * np.einsum('abcd,bd->ac', W, P_AB)
            K_sp = -0.5 * np.einsum('abcd,bd->ac', W[:4, :4, :, :], P_AB[:4, :])
            G[sA:sA+9, sB:sB+4] += K
            G[sB:sB+4, sA:sA+9] += K.T
            G[sA:sA+4, sB:sB+4] -= K_sp
            G[sB:sB+4, sA:sA+4] -= K_sp.T
        elif 'w_yx_BA' in entry:
            W = entry['w_yx_BA']
            P_AA = P[sA:sA+4, sA:sA+4]
            P_BB = P[sB:sB+9, sB:sB+9]
            P_BA = P[sB:sB+9, sA:sA+4]
            J_B = np.einsum('abcd,cd->ab', W, P_AA)
            J_B_sp = np.einsum('abcd,cd->ab', W[:4, :4, :, :], P_AA)
            G[sB:sB+9, sB:sB+9] += J_B
            G[sB:sB+4, sB:sB+4] -= J_B_sp
            J_A = np.einsum('abcd,ab->cd', W, P_BB)
            J_A_sp = np.einsum('abcd,ab->cd', W[:4, :4, :, :], P_BB[:4, :4])
            G[sA:sA+4, sA:sA+4] += J_A - J_A_sp
            K = -0.5 * np.einsum('abcd,bd->ac', W, P_BA)
            K_sp = -0.5 * np.einsum('abcd,bd->ac', W[:4, :4, :, :], P_BA[:4, :])
            G[sB:sB+9, sA:sA+4] += K
            G[sA:sA+4, sB:sB+9] += K.T
            G[sB:sB+4, sA:sA+4] -= K_sp
            G[sA:sA+4, sB:sB+4] -= K_sp.T
        elif 'w_yy' in entry:
            W = entry['w_yy']
            P_AA = P[sA:sA+9, sA:sA+9]
            P_BB = P[sB:sB+9, sB:sB+9]
            P_AB = P[sA:sA+9, sB:sB+9]
            J_A = np.einsum('abcd,cd->ab', W, P_BB)
            J_A_sp = np.einsum('abcd,cd->ab', W[:4, :4, :4, :4], P_BB[:4, :4])
            G[sA:sA+9, sA:sA+9] += J_A
            G[sA:sA+4, sA:sA+4] -= J_A_sp
            J_B = np.einsum('cdab,cd->ab', W, P_AA)
            J_B_sp = np.einsum('cdab,cd->ab', W[:4, :4, :4, :4], P_AA[:4, :4])
            G[sB:sB+9, sB:sB+9] += J_B
            G[sB:sB+4, sB:sB+4] -= J_B_sp
            K = -0.5 * np.einsum('abcd,bd->ac', W, P_AB)
            K_sp = -0.5 * np.einsum('abcd,bd->ac',
                                    W[:4, :4, :4, :4], P_AB[:4, :4])
            G[sA:sA+9, sB:sB+9] += K
            G[sB:sB+9, sA:sA+9] += K.T
            G[sA:sA+4, sB:sB+4] -= K_sp
            G[sB:sB+4, sA:sA+4] -= K_sp.T
    return G


def rm1_pm6d_native_numba(atoms, coords, max_iter=200, conv_tol=1e-7):
    """PM6_D SCF with Numba-JIT'd sp Fock contractions."""
    PARAMS = get_params('PM6_D')
    coords = np.asarray(coords, dtype=np.float64)
    info = _build_basis_info(atoms, PARAMS)
    n_basis = info['n_basis']; n_occ = info['n_occ']
    params = info['params']; starts = info['atom_basis_start']
    H = _build_core_hamiltonian(atoms, coords, info)
    W_one = [_precompute_one_center_W(p) for p in params]
    cache = precompute_pair_cache(atoms, coords, params)

    P = np.zeros((n_basis, n_basis))
    for i, p in enumerate(params):
        sA = starts[i]
        if p.n_basis == 1:
            P[sA, sA] = 1.0
        else:
            v = p.n_valence / 4.0
            for k in range(4):
                P[sA + k, sA + k] = v

    has_d = any(p.n_basis > 4 for p in params)
    diis_max = 6; diis_F = []; diis_E = []
    converged = False
    it = 0
    for it in range(max_iter):
        F = H + _build_G_numba(P, info, W_one, cache)
        F = 0.5 * (F + F.T)
        if it >= 2:
            err = F @ P - P @ F
            diis_F.append(F.copy()); diis_E.append(err.copy())
            if len(diis_F) > diis_max:
                diis_F.pop(0); diis_E.pop(0)
            nd = len(diis_F)
            if nd >= 2:
                B = np.zeros((nd + 1, nd + 1))
                for ii in range(nd):
                    for jj in range(nd):
                        B[ii, jj] = float((diis_E[ii] * diis_E[jj]).sum())
                B[nd, :nd] = -1.0; B[:nd, nd] = -1.0
                rhs = np.zeros(nd + 1); rhs[nd] = -1.0
                try:
                    c = np.linalg.solve(B, rhs)
                    F = sum(c[k] * diis_F[k] for k in range(nd))
                except np.linalg.LinAlgError:
                    pass
        _, C = np.linalg.eigh(F)
        Pn = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        d = float(np.sqrt(((Pn - P) ** 2).mean()))
        if d < conv_tol:
            converged = True; P = Pn; break
        if it < 3: mix = 0.3 if has_d else 0.5
        elif d > 0.1: mix = 0.05 if has_d else 0.4
        elif d > 0.01: mix = 0.5
        else: mix = 0.8
        P = mix * Pn + (1 - mix) * P
    q = np.array([
        p.n_valence - float(sum(P[starts[i]+k, starts[i]+k] for k in range(p.n_basis)))
        for i, p in enumerate(params)
    ])
    return {'q': q, 'converged': converged, 'n_iter': it + 1}
