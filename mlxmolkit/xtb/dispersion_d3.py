# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""GFN1's D3 dispersion correction with Becke-Johnson damping.

Verbatim port of the two-body D3-BJ E_disp pieces in
``xtb/src/disp/dftd3.f90``:

    E = -½ Σ_{i≠j}  C6_ij · ( s6 / (R²³ + R0⁶)
                              + s8 · 3 r4r2_i · r4r2_j / (R²⁴ + R0⁸) )

with the Becke-Johnson radius ``R0 = a1·sqrt(3·r4r2_i·r4r2_j) + a2``.
The C6 coefficients are CN-interpolated from a per-element reference
table via Gaussian weighting (``weight_references`` in dftd3.f90:52-113).

GFN1-xTB damping parameters (from gfn1.f90 disp_dftd3 init):
    s6 = 1.0  s8 = 2.4  a1 = 0.63  a2 = 5.0  s10 = 0  wf = 4.0

The s10 term is zero for GFN1, so we omit it. ATM (Axilrod-Teller-Muto)
three-body is also off for GFN1 (xtb's TGFN1 only enables the two-body
disp).

Data tables are vendored from xtb's dftd3_parameters.f90 +
r4r2_expectation_values.f90; see ``params/d3_data.npz`` (parsed by
``tools/parse_dftd3_params.py``, run once at install time).
"""

from __future__ import annotations

import numpy as np
import os

from .params_gfn1 import GFN1_GLOBALS

_DATA_PATH = os.path.join(os.path.dirname(__file__), "params", "d3_data.npz")
_D3 = np.load(_DATA_PATH)
_NUMBER_OF_REFERENCES = _D3["number_of_references"]   # (94,)
_REFERENCE_CN = _D3["reference_cn"]                   # (94, 5)
_C6AB = _D3["c6ab"]                                    # (5, 5, 4465)
_R4_OVER_R2 = _D3["r4_over_r2"]                       # (118,)

_ANG_TO_BOHR = 1.8897259886
_HARTREE_PER_EV = 1.0 / 27.211386245988

# Pre-compute sqrt(0.5 · r4_over_r2 · sqrt(Z)) per xtb's
# ``sqrt_z_r4_over_r2`` (mctc/r4r2_expectation_values.f90:56-57).
_Z_ARR = np.arange(1, len(_R4_OVER_R2) + 1, dtype=np.float64)
_R4R2 = np.sqrt(0.5 * _R4_OVER_R2 * np.sqrt(_Z_ARR))   # (118,)

GFN1_D3_PARAMS = {"s6": 1.0, "s8": 2.4, "a1": 0.63, "a2": 5.0, "wf": 4.0}


def _pair_index(zi: int, zj: int) -> tuple[int, bool]:
    """Map (Zi, Zj) to ``ic`` (0-based) packed-pair index, with a flag
    indicating whether the C6 lookup must swap (iref, jref) — mirrors
    ``get_c6`` (dftd3_parameters.f90:1051-1061).

    Returns ``(ic, swap)`` where ``c6ab[iref, jref, ic]`` is correct
    for the canonical ordering, but if ``swap=True`` the caller must
    use ``c6ab[jref, iref, ic]`` (i.e., transpose the lookup).
    """
    if zi > zj:
        ic = zj + zi * (zi - 1) // 2     # 1-based; (atj, ati)
        swap = False
    else:
        ic = zi + zj * (zj - 1) // 2     # 1-based; (ati, atj)
        swap = True
    return ic - 1, swap


def _weight_references(cn: np.ndarray, atoms: np.ndarray, wf: float = 4.0) -> np.ndarray:
    """Gaussian weights w[i, k] of each reference system k for atom i.

    Mirrors ``weight_references`` (dftd3.f90:52-113). The Gaussian is
    ``exp(-wf · (cn - refcn)²)`` normalized by the sum over all valid
    refs of the same atom. Atoms with all-zero refs (e.g. He) get
    zero weights.
    """
    n_at = len(atoms)
    w = np.zeros((n_at, 5), dtype=np.float64)
    for i, Z in enumerate(atoms):
        nref_i = int(_NUMBER_OF_REFERENCES[Z - 1])
        if nref_i == 0:
            continue
        rcn = _REFERENCE_CN[Z - 1, :nref_i]
        gw = np.exp(-wf * (cn[i] - rcn) ** 2)
        norm = float(np.sum(gw))
        if norm < 1e-80:
            # Pin the weight to the highest-CN reference (xtb fallback).
            k_max = int(np.argmax(rcn))
            w[i, k_max] = 1.0
        else:
            w[i, :nref_i] = gw / norm
    return w


def _atomic_c6(atoms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-pair C6 from CN-interpolated references.

    ``c6[i, j] = Σ_{kl} w[i, k] · w[j, l] · refc6(k, l, ic_ij)``
    """
    n_at = len(atoms)
    c6 = np.zeros((n_at, n_at), dtype=np.float64)
    for i in range(n_at):
        zi = int(atoms[i])
        for j in range(i + 1):
            zj = int(atoms[j])
            ic, swap = _pair_index(zi, zj)
            block = _C6AB[:, :, ic]
            if swap:
                # Need to transpose: c6 is stored for the canonical ic;
                # if our (zi, zj) order doesn't match, transpose lookup.
                block = block.T
            # Σ_kl w_i[k] · w_j[l] · block[k, l]
            c6_ij = float(weights[i] @ block @ weights[j])
            c6[i, j] = c6_ij
            c6[j, i] = c6_ij
    return c6


def d3bj_dispersion_energy(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    cn: np.ndarray | None = None,
    *,
    s6: float = 1.0,
    s8: float = 2.4,
    a1: float = 0.63,
    a2: float = 5.0,
    wf: float = 4.0,
) -> float:
    """D3(BJ) two-body dispersion energy in Hartree.

    Args:
        atoms: list of atomic numbers (length n_atoms).
        coords_ang: (n_atoms, 3) Angstrom coordinates.
        cn: (n_atoms,) erf-CN array; if None, recompute via
            :func:`mlxmolkit.xtb.cn.coordination_number_erf`. Pass
            yours when you already have it (e.g. from the hcore build).
        s6, s8, a1, a2, wf: D3-BJ damping parameters
            (defaults are the GFN1-xTB values).

    Returns:
        Total dispersion energy in Hartree (negative = attractive).
    """
    atoms_arr = np.asarray(atoms, dtype=np.int64)
    coords = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    n_at = len(atoms_arr)
    if n_at < 2:
        return 0.0

    if cn is None:
        from .cn import coordination_number_erf
        import mlx.core as mx
        cn_mx = coordination_number_erf(
            mx.array(np.asarray(coords_ang, dtype=np.float32)),
            mx.array(np.asarray(atoms_arr, dtype=np.int32)),
        )
        mx.eval(cn_mx)
        cn = np.asarray(cn_mx).astype(np.float64)
    cn = np.asarray(cn, dtype=np.float64)

    weights = _weight_references(cn, atoms_arr, wf=wf)
    c6 = _atomic_c6(atoms_arr, weights)

    # r4r2 per atom (1-based Z → 0-based index).
    r4r2_per = _R4R2[atoms_arr - 1]    # (n_at,)

    # Pairwise sum. Note xtb's loop double-counts by symmetry and pre-
    # multiplies by 0.5; we use the unique-pair convention.
    e = 0.0
    for i in range(n_at):
        for j in range(i + 1, n_at):
            r4r2ij = 3.0 * r4r2_per[i] * r4r2_per[j]
            r0 = a1 * np.sqrt(r4r2ij) + a2
            R = np.linalg.norm(coords[i] - coords[j])
            R2 = R * R
            t6 = 1.0 / (R2 ** 3 + r0 ** 6)
            t8 = 1.0 / (R2 ** 4 + r0 ** 8)
            disp = s6 * t6 + s8 * r4r2ij * t8
            e += -float(c6[i, j]) * disp
    return e


def d3bj_dispersion_gfn1(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    cn: np.ndarray | None = None,
) -> float:
    """GFN1-flavored D3-BJ wrapper using the published GFN1 damping
    parameters (s6=1, s8=2.4, a1=0.63, a2=5.0, wf=4)."""
    p = GFN1_D3_PARAMS
    return d3bj_dispersion_energy(atoms, coords_ang, cn, **p)
