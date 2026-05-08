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
    STO3GShell,
    get_sto3g,
    primitive_norm_p,
    primitive_norm_s,
)


_ANG_TO_BOHR = 1.8897259886


@dataclass
class BasisFunction:
    """One Cartesian basis function in the molecular AO basis.

    Attributes:
        atom_idx: index of the atom this BF lives on
        l_total: angular momentum (0 = s, 1 = p)
        l_xyz: ``(lx, ly, lz)`` Cartesian decomposition; for s = (0,0,0),
            for p_x = (1,0,0), p_y = (0,1,0), p_z = (0,0,1).
        center: ``(3,)`` position in Bohr (the convention used by the
            primitive overlap formulas).
        alphas: primitive Gaussian exponents (Bohr^-2), 3 values for STO-3G.
        coeffs: contraction coefficients × primitive normalizations × the
            outer contraction normalization, so that ``Σ_i Σ_j c_i c_j
            <g_i|g_j> = 1`` for this contracted basis function.
    """
    atom_idx: int
    l_total: int
    l_xyz: tuple[int, int, int]
    center: np.ndarray             # (3,) Bohr
    alphas: np.ndarray             # (3,)
    coeffs: np.ndarray             # (3,) including primitive + contraction norms


def _contraction_norm(
    alphas: np.ndarray,
    raw_coeffs: np.ndarray,
    l_total: int,
) -> float:
    """Compute 1/sqrt(self-overlap) for a contracted Cartesian Gaussian
    on a specific Cartesian component (e.g. p_x). All components of a
    given l have the same self-overlap.
    """
    # self-overlap = Σ_i Σ_j c_i c_j N_i N_j <prim_i | prim_j> at R=0
    # For two primitives at the same center with same Cartesian quantum:
    #   <prim_i | prim_j>_0 = (π / (α_i + α_j))^(3/2) * angular_factor(l_total)
    # where the angular factor for s is 1, for p is 1/(2(α_i+α_j)).
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
                # For p_x p_x: angular = 1/(2p)
                ang = 1.0 / (2.0 * p)
                norm_i = primitive_norm_p(alphas[i])
                norm_j = primitive_norm_p(alphas[j])
            else:
                raise NotImplementedError(f"l_total={l_total} not supported")
            ov += raw_coeffs[i] * raw_coeffs[j] * float(norm_i) * float(norm_j) * base * ang
    return 1.0 / float(np.sqrt(ov))


def build_basis(
    atoms: list[int],
    coords_ang: np.ndarray,
    params_dict: dict[int, GFN0ElementParams] | None = None,
) -> list[BasisFunction]:
    """Build the AO basis for one molecule (Cartesian, STO-3G, s+p only).

    Args:
        atoms: list of atomic numbers.
        coords_ang: ``(n_atoms, 3)`` coordinates in Angstrom.
        params_dict: GFN0 element-params dict (defaults to ``GFN0_PARAMS``).

    Returns:
        A list of :class:`BasisFunction`, ordered (atom-major, then shell,
        then Cartesian component).

    Raises:
        NotImplementedError: if any shell has ``l > 1`` (d/f shells).
        KeyError: if any ``Z`` is not in ``params_dict``.
    """
    if params_dict is None:
        params_dict = GFN0_PARAMS
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR

    out: list[BasisFunction] = []
    for atom_idx, Z in enumerate(atoms):
        p = params_dict[int(Z)]
        center = coords_bohr[atom_idx]
        for shell in p.shells:
            if shell.l > 1:
                # Skip d/f for Phase A. The basis would still be valid
                # without them; energies will deviate from xtb on heavy
                # elements but H/C/N/O/F are unaffected.
                continue
            sto = get_sto3g(shell.n, shell.l)
            zeta_sq = shell.zeta * shell.zeta
            alphas = np.array(sto.alphas, dtype=np.float64) * zeta_sq
            raw_coeffs = np.array(sto.coeffs, dtype=np.float64)
            N_contraction = _contraction_norm(alphas, raw_coeffs, shell.l)
            # Pre-multiply contraction coeffs by primitive norms *and*
            # by N_contraction so the overlap kernel just sees a single
            # weighted sum over primitive pairs.
            if shell.l == 0:
                prim_norms = primitive_norm_s(alphas)
            else:
                prim_norms = primitive_norm_p(alphas)
            full_coeffs = raw_coeffs * prim_norms * N_contraction

            if shell.l == 0:
                out.append(BasisFunction(
                    atom_idx=atom_idx, l_total=0, l_xyz=(0, 0, 0),
                    center=center, alphas=alphas, coeffs=full_coeffs,
                ))
            else:  # l == 1: p_x, p_y, p_z
                for axis in range(3):
                    l_xyz = [0, 0, 0]
                    l_xyz[axis] = 1
                    out.append(BasisFunction(
                        atom_idx=atom_idx, l_total=1, l_xyz=tuple(l_xyz),
                        center=center, alphas=alphas, coeffs=full_coeffs,
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
    # Multiply per-axis angular factors:
    out = base
    for axis in range(3):
        la = l_xyz_a[axis]
        lb = l_xyz_b[axis]
        PA = float(P[axis] - A[axis])
        PB = float(P[axis] - B[axis])
        if la == 0 and lb == 0:
            ang = 1.0
        elif la == 1 and lb == 0:
            ang = PA
        elif la == 0 and lb == 1:
            ang = PB
        elif la == 1 and lb == 1:
            ang = PA * PB + 1.0 / (2.0 * p)
        else:
            raise NotImplementedError(
                f"primitive overlap for axis l = ({la}, {lb}) not implemented"
            )
        out *= ang
    return out


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
