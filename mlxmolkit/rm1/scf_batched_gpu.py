"""Batched PM6_D SCF on Metal GPU.

Processes B molecules simultaneously by padding them to a common basis
size and running the SCF loop on stacked ``(B, N, N)`` MLX arrays. The
per-iteration kernels (Fock build via mx.einsum, batched_eigh, pulay_diis)
each run once per SCF iter instead of once per molecule per iter.

GPU sweet spot: batches of 32-128 same-size molecules. The mlx-addons
Jacobi eigh handles N ≤ 32 on GPU; for larger N it falls back to CPU
batched eigh (still much faster than per-molecule serial calls).

Limitations of this first cut:
  - All molecules in a batch must have the same basis size (caller groups
    them; the test harness here picks identical molecules for now).
  - Per-pair integrals (sp×sp 4^4, YH 9×9, YX 9×9×4×4, YY 9×9×9×9) are
    pre-computed per molecule on CPU then stacked. A future revision can
    group pairs across the batch by type for fused einsum.
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
    mx = None

from .methods import get_params
from .scf import _build_basis_info, _build_core_hamiltonian
from .scf_mlx import precompute_pair_cache, _precompute_one_center_W


def _build_full_G_matrix(P_np, info, W_one_np, cache_np):
    """Build the FULL G(P) matrix on CPU and return as numpy.

    Same algorithm as scf_mlx._one_center_g_numpy + _two_center_g_numpy,
    but returns a single numpy array. Used per-molecule before stacking
    for the batched GPU SCF (since per-pair integrals have variable
    shapes that don't pad cleanly).
    """
    from .scf_mlx import _one_center_g_numpy, _two_center_g_numpy
    G = _one_center_g_numpy(P_np, info, W_one_np)
    G = G + _two_center_g_numpy(P_np, info, cache_np)
    return G


def rm1_pm6d_native_batched_gpu(
    molecules,
    max_iter: int = 200,
    conv_tol: float = 1e-5,
    dtype: str = 'float32',
):
    """Batched native PM6_D SCF on Metal GPU.

    All molecules must have the same n_basis (caller's responsibility);
    raises if they don't.

    Args:
        molecules: list of (atoms, coords) tuples
        max_iter, conv_tol: SCF convergence parameters
        dtype: 'float32' (for GPU Jacobi eigh) or 'float64'

    Returns:
        list of dicts with 'q', 'converged', 'n_iter'
    """
    if not _HAS_MLX:
        raise ImportError("mlx + mlx-addons required for batched GPU SCF")
    dt = mx.float32 if dtype == 'float32' else mx.float64
    np_dt = np.float32 if dtype == 'float32' else np.float64

    PARAMS = get_params('PM6_D')
    B = len(molecules)
    if B == 0:
        return []

    # Pre-compute per-molecule CPU data
    pre = []
    for atoms, coords in molecules:
        coords = np.asarray(coords, dtype=np.float64)
        info = _build_basis_info(atoms, PARAMS)
        H = _build_core_hamiltonian(atoms, coords, info)
        W_one = [_precompute_one_center_W(p) for p in info['params']]
        cache = precompute_pair_cache(atoms, coords, info['params'])
        # Initial diagonal density
        P = np.zeros((info['n_basis'], info['n_basis']), dtype=np.float64)
        for i, p in enumerate(info['params']):
            sA = info['atom_basis_start'][i]
            if p.n_basis == 1:
                P[sA, sA] = 1.0
            else:
                v = p.n_valence / 4.0
                for k in range(4):
                    P[sA + k, sA + k] = v
        pre.append({'atoms': atoms, 'info': info, 'H': H,
                    'W_one': W_one, 'cache': cache, 'P': P})

    # Verify uniform basis
    Ns = [p['info']['n_basis'] for p in pre]
    if len(set(Ns)) > 1:
        raise ValueError(f"Heterogeneous batch n_basis={set(Ns)} — group by size first")
    N = Ns[0]
    n_occs = [p['info']['n_occ'] for p in pre]
    # All same n_basis but possibly different n_occ — that's fine

    # Stack into batched MLX arrays
    H_batch = mx.array(np.stack([p['H'] for p in pre]).astype(np_dt), dtype=dt)
    P_batch = mx.array(np.stack([p['P'] for p in pre]).astype(np_dt), dtype=dt)

    has_d = any(any(p.n_basis > 4 for p in pre_i['info']['params']) for pre_i in pre)
    diis_F = []
    diis_E = []
    converged = np.zeros(B, dtype=bool)
    n_iter_out = np.zeros(B, dtype=int)

    for it in range(max_iter):
        # Build G per molecule on CPU (per-pair einsums in numpy), stack
        P_np_batch = np.asarray(P_batch)
        G_batch_np = np.empty((B, N, N), dtype=np_dt)
        for b in range(B):
            if converged[b]:
                G_batch_np[b] = 0.0
                continue
            G_batch_np[b] = _build_full_G_matrix(
                P_np_batch[b].astype(np.float64), pre[b]['info'],
                pre[b]['W_one'], pre[b]['cache']
            ).astype(np_dt)
        G_batch = mx.array(G_batch_np, dtype=dt)
        F = H_batch + G_batch
        F = 0.5 * (F + mx.swapaxes(F, -1, -2))

        # DIIS (batched)
        if it >= 2:
            err = commutator_error(F, P_batch)
            diis_F.append(F); diis_E.append(err)
            if len(diis_F) > 6:
                diis_F.pop(0); diis_E.pop(0)
            if len(diis_F) >= 2:
                F = pulay_diis(diis_F, diis_E, max_history=6)

        # Batched eigh on (B, N, N)
        _, C = batched_eigh(F)
        # Build density per molecule (n_occ varies) — do on CPU after eval
        C_np = np.asarray(C)
        P_new_np = np.empty_like(C_np)
        for b in range(B):
            nocc = n_occs[b]
            P_new_np[b] = 2.0 * C_np[b, :, :nocc] @ C_np[b, :, :nocc].T
        P_new = mx.array(P_new_np.astype(np_dt), dtype=dt)
        d = mx.sqrt(((P_new - P_batch) ** 2).mean(axis=(1, 2)))
        d_np = np.asarray(d)

        for b in range(B):
            if not converged[b] and d_np[b] < conv_tol:
                converged[b] = True
                n_iter_out[b] = it + 1

        if np.all(converged):
            P_batch = P_new
            break

        # Per-molecule adaptive mixing
        mixes = np.empty(B, dtype=np_dt)
        for b in range(B):
            d_b = d_np[b]
            if it < 3:
                m = 0.3 if has_d else 0.5
            elif d_b > 0.1:
                m = 0.05 if has_d else 0.4
            elif d_b > 0.01:
                m = 0.5
            else:
                m = 0.8
            mixes[b] = m
        mix_arr = mx.array(mixes.reshape(B, 1, 1), dtype=dt)
        P_batch = mix_arr * P_new + (1.0 - mix_arr) * P_batch

    # Mark unconverged
    for b in range(B):
        if not converged[b]:
            n_iter_out[b] = max_iter

    # Mulliken charges per molecule
    P_final_np = np.asarray(P_batch)
    results = []
    for b in range(B):
        info = pre[b]['info']; atoms = pre[b]['atoms']
        params = info['params']; starts = info['atom_basis_start']
        P_b = P_final_np[b]
        q = np.array([
            p.n_valence - float(sum(P_b[starts[i] + k, starts[i] + k]
                                    for k in range(p.n_basis)))
            for i, p in enumerate(params)
        ])
        results.append({
            'q': q, 'converged': bool(converged[b]),
            'n_iter': int(n_iter_out[b]),
        })
    return results
