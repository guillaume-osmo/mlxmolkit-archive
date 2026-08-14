# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT
#
# D3-style covalent radii vendored from grimme-lab/xtb's
# `src/param/covalentradd3.f90` (LGPL-3.0). The xtb code applies
# `* aatoau * 4/3`; here we keep the raw Pyykkö values in Angstrom and
# apply the 4/3 scale at use time, working in Angstrom throughout to
# match RDKit's coordinate convention.

"""Coordination-number primitives for xTB.

GFN0 uses two CN variants in its energy expression:

- **erf-CN** (this module): plain error-function counting,
  ``CN_A = Σ_{B≠A} ½ (1 + erf(−k · (R_AB − R_AB^cov) / R_AB^cov))``,
  with ``k = 7.5`` and ``R_AB^cov = (R_A^Pyykkö + R_B^Pyykkö) · 4/3``.
  Used by ``H_μμ`` (CN-shift on the diagonal) and EEQ (RHS coupling).
- **cov-CN** (deferred to Phase A5 — D4 dispersion): same kernel but
  with an additional electronegativity-difference weighting per pair.

Reference: ``grimme-lab/xtb/src/disp/coordinationnumber.F90`` (LGPL-3.0),
``erfCount`` at lines 483-498. Matches Caldeweyher 2019 (J. Chem. Phys.
150:154122) for the shape and constant.
"""

from __future__ import annotations

import mlx.core as mx


# ---------------------------------------------------------------------------
# Pyykkö-style covalent radii in Angstrom, indexed 1..118.
# Index 0 is a SENTINEL (0.0) for padded atoms (Z=0 in batched arrays);
# this lets `mx.take(_COV_RAD, atoms)` map padding to zero radius cleanly.
# Vendored from grimme-lab/xtb/src/param/covalentradd3.f90 (LGPL-3.0).
# Multiply by 4/3 at use time to get the "extended" R^cov used in CN.
# ---------------------------------------------------------------------------
_COV_RAD_PYYKKO = (
    0.0,                                                    # 0  — sentinel
    0.32, 0.46,                                             # H, He
    1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67,         # Li-Ne
    1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96,         # Na-Ar
    1.76, 1.54,                                             # K, Ca
    1.33, 1.22, 1.21, 1.10, 1.07,                           # Sc-
    1.04, 1.00, 0.99, 1.01, 1.09,                           # -Zn
    1.12, 1.09, 1.15, 1.10, 1.14, 1.17,                     # Ga-Kr
    1.89, 1.67,                                             # Rb, Sr
    1.47, 1.39, 1.32, 1.24, 1.15,                           # Y-
    1.13, 1.13, 1.08, 1.15, 1.23,                           # -Cd
    1.28, 1.26, 1.26, 1.23, 1.32, 1.31,                     # In-Xe
    2.09, 1.76,                                             # Cs, Ba
    1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51,               # La-Eu
    1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53,               # Gd-Yb
    1.46, 1.37, 1.31, 1.23, 1.18,                           # Lu-
    1.16, 1.11, 1.12, 1.13, 1.32,                           # -Hg
    1.30, 1.30, 1.36, 1.31, 1.38, 1.42,                     # Tl-Rn
    2.01, 1.81,                                             # Fr, Ra
    1.67, 1.58, 1.52, 1.53, 1.54, 1.55, 1.49,               # Ac-Am
    1.49, 1.51, 1.51, 1.48, 1.50, 1.56, 1.58,               # Cm-No
    1.45, 1.41, 1.34, 1.29, 1.27,                           # Lr-
    1.21, 1.16, 1.15, 1.09, 1.22,                           # -Cn
    1.36, 1.43, 1.46, 1.58, 1.48, 1.57,                     # Nh-Og
)
assert len(_COV_RAD_PYYKKO) == 119  # 0 sentinel + 118 elements

# Module-level mx.array: built once on first import.
_COV_RAD_TABLE: mx.array = mx.array(_COV_RAD_PYYKKO, dtype=mx.float32)
_COV_SCALE = 4.0 / 3.0   # the "extended" scaling used in xtb's CN

KCN_ERF = 7.5            # default counting-function steepness
MAX_CN = 8.0             # saturation per atom (cutCoordinationNumber)


def coordination_number_erf(
    coords: mx.array,
    atoms: mx.array,
    n_atoms_arr: mx.array | None = None,
    *,
    k: float = KCN_ERF,
    max_cn: float = MAX_CN,
) -> mx.array:
    """Erf-counting coordination numbers — batched, MLX-native.

    ``CN_A = Σ_{B≠A} ½ (1 + erf(−k · (R_AB − R_AB^cov) / R_AB^cov))``,
    with ``R_AB^cov = (R_A + R_B) · 4/3`` (Pyykkö raw radii, scaled).

    Args:
        coords: ``(B, max_atoms, 3)`` Angstrom positions. Padded atoms
            should sit at any value (their contribution is masked out
            via ``n_atoms_arr``). Single-mol input ``(max_atoms, 3)``
            is auto-promoted to B=1 and squeezed back on return.
        atoms: ``(B, max_atoms)`` ``int32``/``int64`` atomic numbers.
            Padding is encoded as ``Z = 0`` (the sentinel slot in the
            covalent-radius table); padded slots map to ``R = 0`` and
            are zeroed by the validity mask anyway.
        n_atoms_arr: optional ``(B,)`` ``int`` per-mol atom count. If
            ``None``, all leading atoms are assumed valid (single-mol
            or fully-filled batch).
        k: counting-function steepness. Default ``7.5`` matches
            xtb's ``cnType%erf``.
        max_cn: per-atom saturation cap (``cutCoordinationNumber`` in
            xtb). Default ``8.0``.

    Returns:
        ``(B, max_atoms)`` (or ``(max_atoms,)`` if input was single-mol)
        coordination number per atom; padded slots are zero.
    """
    was_2d = coords.ndim == 2
    if was_2d:
        coords = coords[None]
        atoms = atoms[None]
    B, n_max, _ = coords.shape

    # Pairwise distance matrix: (B, n_max, n_max)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    R = mx.sqrt(mx.sum(diff * diff, axis=-1) + mx.array(1e-30, dtype=coords.dtype))

    # Covalent-radius lookup: (B, n_max) → (B, n_max, n_max) pair sum scaled by 4/3.
    R_atom = mx.take(_COV_RAD_TABLE, atoms, axis=0).astype(coords.dtype)
    R_cov_pair = (R_atom[:, :, None] + R_atom[:, None, :]) * mx.array(
        _COV_SCALE, dtype=coords.dtype
    )

    # Erf counting: 0.5 * (1 + erf(-k * (R - R_cov) / R_cov))
    eps = mx.array(1e-30, dtype=coords.dtype)
    arg = -k * (R - R_cov_pair) / mx.maximum(R_cov_pair, eps)
    f_pair = 0.5 * (1.0 + mx.erf(arg))                      # (B, n_max, n_max)

    # Zero diagonal (A=A self-counting).
    eye = mx.eye(n_max, dtype=coords.dtype)
    f_pair = f_pair * (1.0 - eye[None, :, :])

    # Validity mask: pair (i, j) contributes only if BOTH i and j are real
    # atoms in their molecule. Also handles the case where atoms[..., i] = 0
    # (padding); R_cov_pair is then small and would produce spurious counting.
    if n_atoms_arr is not None:
        arange_n = mx.arange(n_max, dtype=mx.int32)
        valid = (arange_n[None, :] < n_atoms_arr[:, None]).astype(coords.dtype)
        pair_valid = valid[:, :, None] * valid[:, None, :]
        f_pair = f_pair * pair_valid

    CN = mx.sum(f_pair, axis=-1)                            # (B, n_max)
    CN = mx.minimum(CN, mx.array(max_cn, dtype=CN.dtype))

    if was_2d:
        CN = CN[0]
    return CN
