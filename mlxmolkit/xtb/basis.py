# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""STO-NG AO basis assembly for GFN0-xTB and the full overlap matrix.

Phase A scope: pure-numpy single-mol overlap with STO-3G primitives for
s and p shells (d shells are silently skipped for now — they only appear
on heavy elements outside the CHNO subset). Single-mol because the
overlap matrix is computed once per energy evaluation in GFN0 (no SCF
iteration → no hot-path concern); the boundary cast to ``mx.array``
happens in the orchestrator.

Caveats / deferred:
- xtb actually uses **mixed STO-NG** (different N per shell per element);
  we use STO-3G everywhere for simplicity. Expected error vs xtb's S:
  small for valence shells (s/p of CHNO use 3-4 Gaussians), larger for
  contracted core orbitals.
- d-shells of S/P/Cl... and transition metals are **not yet
  implemented**. Phase A0 supports H, C, N, O, F (no d).
- Pure (spherical) p is identical to Cartesian p, so we stay in
  Cartesian throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params_gfn0 import GFN0_PARAMS, GFN0ElementParams, GFN0Shell
from .sto_ng import (
    get_sto_ng,
    gfn0_n_gauss,
    primitive_norm_d,
    primitive_norm_p,
    primitive_norm_s,
)


_ANG_TO_BOHR = 1.8897259886


@dataclass
class BasisFunction:
    """One Cartesian basis function in the molecular AO basis.

    The ``is_valence`` flag matches xtb's ``valenceShell`` flag and
    determines which branch of ``h0scal`` (val-val, val-aux, aux-aux)
    builds the off-diagonal Hcore element with this BF.

    For d-shells, the Cartesian basis has 6 functions per shell
    (xx, yy, zz, xy, xz, yz). They are emitted with the canonical
    ``primitive_norm_d`` normalization and grouped via
    ``shell_id``; the 6→5 CAO→SAO reduction is applied later by
    :func:`cao_to_sao_transform` (mirror of xtb's ``dtrf2``).
    """
    atom_idx: int
    l_total: int
    l_xyz: tuple[int, int, int]
    center: np.ndarray
    alphas: np.ndarray
    coeffs: np.ndarray
    is_valence: bool = True
    shell_id: int = -1     # opaque tag; same id for the 6 BFs of one d-shell


def _contraction_norm(
    alphas: np.ndarray,
    raw_coeffs: np.ndarray,
    l_total: int,
) -> float:
    """Compute 1/sqrt(self-overlap) for a contracted Cartesian Gaussian.

    Works for any number of primitives ``len(alphas) == len(raw_coeffs)``.
    For l_total = 2, the self-overlap is computed for one of the d
    components with angular factor ``1/(2p)²`` (e.g. xx · xx = 3/(2p)²,
    xy · xy = 1/(2p)²); we use the average xtb-style ``primitive_norm_d``
    convention so the resulting d primitive is the bare Cartesian
    function before the CAO→SAO transform.
    """
    n = len(alphas)
    ov = 0.0
    for i in range(n):
        for j in range(n):
            p = alphas[i] + alphas[j]
            base = (np.pi / p) ** 1.5
            if l_total == 0:
                ang = 1.0
                norm_i = primitive_norm_s(alphas[i])
                norm_j = primitive_norm_s(alphas[j])
            elif l_total == 1:
                ang = 1.0 / (2.0 * p)
                norm_i = primitive_norm_p(alphas[i])
                norm_j = primitive_norm_p(alphas[j])
            elif l_total == 2:
                # Self-overlap of an xx-type Cartesian d primitive:
                # the angular factor for xx · xx = 3/(2p)². Combined with
                # primitive_norm_d's 1/sqrt(3), this normalizes the
                # Cartesian-d shell as a *group* (each of the 3 axial
                # components xx/yy/zz has angular = 3/(2p)²; the off-axis
                # xy/xz/yz have angular = 1/(2p)²; the dtrf2 transform
                # then mixes them into 5 pure-d functions). Using xx
                # here matches xtb's convention.
                ang = 3.0 / (2.0 * p) ** 2
                norm_i = primitive_norm_d(alphas[i])
                norm_j = primitive_norm_d(alphas[j])
            else:
                raise NotImplementedError(f"l_total={l_total} not supported")
            ov += raw_coeffs[i] * raw_coeffs[j] * float(norm_i) * float(norm_j) * base * ang
    return 1.0 / float(np.sqrt(ov))


def build_basis(
    atoms: list[int],
    coords_ang: np.ndarray,
    params_dict=None,
    n_gauss_fn=None,
) -> list[BasisFunction]:
    """Build the AO basis for one molecule (Cartesian, STO-NG, s+p only).

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` coordinates in Angstrom.
        params_dict: element-params dict; entries must have a ``shells``
            tuple whose elements expose ``.l``, ``.n``, ``.zeta``.
            Defaults to :data:`mlxmolkit.xtb.params_gfn0.GFN0_PARAMS`.
        n_gauss_fn: callable ``(Z, l, n, is_valence) -> int`` returning
            the STO-NG order to use. Defaults to GFN0's rule
            (:func:`mlxmolkit.xtb.sto_ng.gfn0_n_gauss`).

    Returns:
        A list of :class:`BasisFunction`, ordered (atom-major, then shell,
        then Cartesian component).

    Raises:
        NotImplementedError: if any shell has ``l > 1`` (d/f shells).
        KeyError: if any ``Z`` is not in ``params_dict``.
    """
    if params_dict is None:
        params_dict = GFN0_PARAMS
    if n_gauss_fn is None:
        n_gauss_fn = gfn0_n_gauss
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR

    out: list[BasisFunction] = []
    next_shell_id = 0
    for atom_idx, Z in enumerate(atoms):
        p = params_dict[int(Z)]
        center = coords_bohr[atom_idx]
        # Track first-shell-of-each-l on this atom to support Gram-Schmidt
        # orthogonalization of aux s shells against the valence shell.
        # xtb's basis has same-atom <val|aux> = 0 (verified by reading off
        # MO coefs from xtb-python on H2). Without this orthogonalization
        # the overlap matrix is near-singular and the generalized eigh
        # produces spurious -24 Ha eigenvalues.
        first_l_alphas: dict[int, np.ndarray] = {}
        first_l_coeffs: dict[int, np.ndarray] = {}
        seen_l: set[int] = set()
        for shell in p.shells:
            if shell.l > 2:
                # f shells deferred — none of GFN1/GFN2 require them
                continue
            is_valence = shell.l not in seen_l
            seen_l.add(shell.l)
            # Mixed STO-NG per xtb's setGFN0NumberOfPrimitives:
            # H/He valence s = STO-3G, aux s = STO-2G; everything else
            # we have vendored = STO-3G (heavier-element STO-4G+ TBD).
            n_gauss = n_gauss_fn(int(Z), shell.l, shell.n, is_valence)
            sto = get_sto_ng(shell.n, shell.l, n_gauss)
            zeta_sq = shell.zeta * shell.zeta
            alphas = np.array(sto.alphas, dtype=np.float64) * zeta_sq
            raw_coeffs = np.array(sto.coeffs, dtype=np.float64)
            N_contraction = _contraction_norm(alphas, raw_coeffs, shell.l)
            if shell.l == 0:
                prim_norms = primitive_norm_s(alphas)
            elif shell.l == 1:
                prim_norms = primitive_norm_p(alphas)
            else:   # l == 2
                prim_norms = primitive_norm_d(alphas)
            full_coeffs = raw_coeffs * prim_norms * N_contraction

            # Gram-Schmidt: if this is the aux shell, subtract its same-atom
            # projection onto the valence shell, then renormalize. Only s
            # shells in this Phase A0 (no aux p), so we only handle l == 0.
            if (not is_valence) and shell.l == 0:
                v_alphas = first_l_alphas[shell.l]
                v_coeffs = first_l_coeffs[shell.l]
                # <val|aux> with both shells fully normalized & contracted.
                s_va = 0.0
                for i in range(len(v_alphas)):
                    for j in range(len(alphas)):
                        p_ij = v_alphas[i] + alphas[j]
                        s_va += v_coeffs[i] * full_coeffs[j] * (np.pi / p_ij) ** 1.5
                # 2s_orth = (2s - s · 1s) / sqrt(1 - s²); store as union.
                denom = float(np.sqrt(max(1.0 - s_va * s_va, 1e-12)))
                aux_alphas = np.concatenate([alphas, v_alphas])
                aux_coeffs = np.concatenate([
                    full_coeffs / denom,
                    -s_va * v_coeffs / denom,
                ])
                alphas = aux_alphas
                full_coeffs = aux_coeffs
            elif is_valence:
                first_l_alphas[shell.l] = alphas.copy()
                first_l_coeffs[shell.l] = full_coeffs.copy()

            shell_id = next_shell_id
            next_shell_id += 1
            if shell.l == 0:
                out.append(BasisFunction(
                    atom_idx=atom_idx, l_total=0, l_xyz=(0, 0, 0),
                    center=center, alphas=alphas, coeffs=full_coeffs,
                    is_valence=is_valence, shell_id=shell_id,
                ))
            elif shell.l == 1:  # l == 1: p_x, p_y, p_z
                for axis in range(3):
                    l_xyz = [0, 0, 0]
                    l_xyz[axis] = 1
                    out.append(BasisFunction(
                        atom_idx=atom_idx, l_total=1, l_xyz=tuple(l_xyz),
                        center=center, alphas=alphas, coeffs=full_coeffs,
                        is_valence=is_valence, shell_id=shell_id,
                    ))
            else:  # l == 2: 6 Cartesian d functions (xx, yy, zz, xy, xz, yz).
                # Order matches xtb's CAO convention used by dtrf2.
                _D_LXYZ = (
                    (2, 0, 0), (0, 2, 0), (0, 0, 2),  # xx, yy, zz
                    (1, 1, 0), (1, 0, 1), (0, 1, 1),  # xy, xz, yz
                )
                for l_xyz in _D_LXYZ:
                    out.append(BasisFunction(
                        atom_idx=atom_idx, l_total=2, l_xyz=l_xyz,
                        center=center, alphas=alphas, coeffs=full_coeffs,
                        is_valence=is_valence, shell_id=shell_id,
                    ))
    return out


def _primitive_overlap(
    alpha_a: float, A: np.ndarray, l_xyz_a: tuple[int, int, int],
    alpha_b: float, B: np.ndarray, l_xyz_b: tuple[int, int, int],
) -> float:
    """Cartesian-Gaussian primitive overlap for s/p shells (Obara–Saika
    base + first-level recurrence).

    Returns the unnormalized integral
    ``∫ g_a(r) g_b(r) dr`` for two primitives. The angular factors handle
    s and p; higher angular momentum is rejected as not-yet-supported.
    """
    p = alpha_a + alpha_b
    mu = alpha_a * alpha_b / p
    P = (alpha_a * A + alpha_b * B) / p
    R2 = float(np.sum((A - B) ** 2))
    base = (np.pi / p) ** 1.5 * float(np.exp(-mu * R2))
    # Multiply per-axis angular factors. Obara-Saika recursion extended
    # through (la, lb) ∈ {0, 1, 2} per axis (sufficient for s/p/d):
    #
    #   S(0,0) = 1
    #   S(1,0) = PA
    #   S(0,1) = PB
    #   S(1,1) = PA·PB + 1/(2p)
    #   S(2,0) = PA² + 1/(2p)
    #   S(0,2) = PB² + 1/(2p)
    #   S(2,1) = PA²·PB + PA/(2p)·(1 + δ)  → use OS up-recurrence
    #   ...
    #
    # We implement OS up-recurrence via the canonical ``S(la+1, lb) =
    # PA · S(la, lb) + (1/2p) · (la · S(la-1, lb) + lb · S(la, lb-1))``.
    out = base
    inv2p = 1.0 / (2.0 * p)
    for axis in range(3):
        la = l_xyz_a[axis]
        lb = l_xyz_b[axis]
        PA = float(P[axis] - A[axis])
        PB = float(P[axis] - B[axis])
        # Build the (la_max+1) × (lb_max+1) Obara-Saika table.
        n = la + 1
        m = lb + 1
        S = np.zeros((n, m), dtype=np.float64)
        S[0, 0] = 1.0
        # Fill column j=0 by raising la.
        for i in range(1, n):
            term = PA * S[i - 1, 0]
            if i >= 2:
                term += inv2p * (i - 1) * S[i - 2, 0]
            S[i, 0] = term
        # Fill rows j>=1 by raising lb.
        for j in range(1, m):
            for i in range(n):
                term = PB * S[i, j - 1]
                if j >= 2:
                    term += inv2p * (j - 1) * S[i, j - 2]
                if i >= 1:
                    term += inv2p * i * S[i - 1, j - 1]
                S[i, j] = term
        out *= float(S[la, lb])
    return out


def cao_to_sao_transform(basis: list[BasisFunction]) -> np.ndarray:
    """Build the (n_sao, n_cao) CAO→SAO transform matrix.

    Mirrors xtb's ``dtrf2`` (intgrad.f90:144-231): for each d-shell,
    the 6 Cartesian d functions (xx, yy, zz, xy, xz, yz) are mapped to
    5 pure-d functions

        d_{x²-y²} = ½√3 · (xx − yy)
        d_{z²}    = ½ · (xx + yy) − 1 · zz
        d_{xy}    = xy
        d_{xz}    = xz
        d_{yz}    = yz

    The 6th component (½√(1/5) · (xx + yy + zz), pure-s by symmetry)
    is dropped — it would otherwise add a spurious basis function with
    almost-100% overlap with the same-atom valence s shell.

    For non-d basis functions, T is the identity.
    """
    n_cao = len(basis)
    # Group BFs by (shell_id) for d-shells, otherwise each is its own row.
    rows: list[np.ndarray] = []
    skip = set()
    for mu in range(n_cao):
        if mu in skip:
            continue
        bm = basis[mu]
        if bm.l_total < 2:
            row = np.zeros(n_cao, dtype=np.float64)
            row[mu] = 1.0
            rows.append(row)
        else:
            # Find all 6 BFs of this d-shell.
            sid = bm.shell_id
            assert sid >= 0, "d-shell BFs must have shell_id set"
            d_idx: list[int] = []
            for nu in range(mu, n_cao):
                if basis[nu].shell_id == sid:
                    d_idx.append(nu)
            assert len(d_idx) == 6, (
                f"d-shell {sid}: found {len(d_idx)} BFs, expected 6"
            )
            for k in d_idx:
                skip.add(k)
            # CAO order: xx, yy, zz, xy, xz, yz  (matches build_basis).
            xx, yy, zz, xy, xz, yz = d_idx
            # 5 pure-d rows, in xtb's dtrf2 output order:
            #   2 → d_{x²-y²}   ½√3 · (xx − yy)
            #   3 → d_{z²}      ½ · (xx + yy) − zz
            #   4 → d_{xy}      √3 · xy
            #   5 → d_{xz}      √3 · xz
            #   6 → d_{yz}      √3 · yz
            #
            # The √3 factor on d_{xy/xz/yz} is necessary to give those
            # SAO functions unit self-overlap — the bare CAO xy/xz/yz
            # primitives have self-overlap 1/3 (because primitive_norm_d
            # is calibrated for axial xx-style components). Without
            # this factor the SAO d basis is non-orthonormal in a way
            # that mismatches xtb's convention (verified by recovering
            # S from tblite's MO coefs on Br₂: tblite has S_diag = 1
            # uniformly across the d sub-block).
            sqrt3 = float(np.sqrt(3.0))
            r1 = np.zeros(n_cao, dtype=np.float64)
            r1[xx] = 0.5 * sqrt3
            r1[yy] = -0.5 * sqrt3
            rows.append(r1)
            r2 = np.zeros(n_cao, dtype=np.float64)
            r2[xx] = 0.5
            r2[yy] = 0.5
            r2[zz] = -1.0
            rows.append(r2)
            r3 = np.zeros(n_cao, dtype=np.float64)
            r3[xy] = sqrt3
            rows.append(r3)
            r4 = np.zeros(n_cao, dtype=np.float64)
            r4[xz] = sqrt3
            rows.append(r4)
            r5 = np.zeros(n_cao, dtype=np.float64)
            r5[yz] = sqrt3
            rows.append(r5)
    return np.asarray(rows, dtype=np.float64)


def sao_basis_metadata(
    basis: list[BasisFunction],
) -> tuple[list[BasisFunction], np.ndarray]:
    """Build the (length-n_sao) BasisFunction list paired with each SAO row.

    For non-d shells, each SAO BF is the same as the underlying CAO BF.
    For d-shells, the 5 SAO d functions all share the same atom/shell
    metadata (atom_idx, l_total=2, alphas, coeffs, shell_id, is_valence)
    of the original Cartesian d primitives — only ``l_xyz`` is dropped
    (set to ``(0, 0, 0)`` since pure-d is no longer single-axial).
    Use this when downstream code (Hcore, Mulliken) needs per-BF
    metadata.

    Returns:
        ``(sao_basis, T)`` where ``sao_basis`` has length n_sao and
        ``T`` is the (n_sao, n_cao) transform.
    """
    T = cao_to_sao_transform(basis)
    n_cao = len(basis)
    out: list[BasisFunction] = []
    skip = set()
    for mu in range(n_cao):
        if mu in skip:
            continue
        bm = basis[mu]
        if bm.l_total < 2:
            out.append(bm)
        else:
            sid = bm.shell_id
            d_idx = [k for k in range(mu, n_cao) if basis[k].shell_id == sid]
            for k in d_idx:
                skip.add(k)
            # Emit 5 SAO d functions sharing the underlying contraction.
            for _ in range(5):
                out.append(BasisFunction(
                    atom_idx=bm.atom_idx, l_total=2, l_xyz=(0, 0, 0),
                    center=bm.center, alphas=bm.alphas, coeffs=bm.coeffs,
                    is_valence=bm.is_valence, shell_id=sid,
                ))
    assert len(out) == T.shape[0]
    return out, T


def overlap_matrix(basis: list[BasisFunction]) -> np.ndarray:
    """Compute the full ``(n_basis, n_basis)`` AO overlap matrix.

    Args:
        basis: list of basis functions from :func:`build_basis`.

    Returns:
        ``S_μν`` as a numpy ``float64`` array. Diagonal is 1 (within
        numerical accuracy) since basis functions are normalized.
    """
    n = len(basis)
    S = np.zeros((n, n), dtype=np.float64)
    for mu in range(n):
        bm = basis[mu]
        for nu in range(mu, n):
            bn = basis[nu]
            s_munu = 0.0
            for i in range(len(bm.alphas)):
                for j in range(len(bn.alphas)):
                    s_munu += bm.coeffs[i] * bn.coeffs[j] * _primitive_overlap(
                        bm.alphas[i], bm.center, bm.l_xyz,
                        bn.alphas[j], bn.center, bn.l_xyz,
                    )
            S[mu, nu] = s_munu
            S[nu, mu] = s_munu
    return S


def basis_summary(basis: list[BasisFunction]) -> dict:
    """Useful per-mol metadata for downstream consumers (Hcore, density)."""
    return {
        "n_basis": len(basis),
        "atom_idx": np.array([b.atom_idx for b in basis], dtype=np.int32),
        "l_total": np.array([b.l_total for b in basis], dtype=np.int32),
    }
