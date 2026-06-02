# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""MCTC packed van-der-Waals pair radii recovered from the g-xTB binary."""

from __future__ import annotations

from functools import lru_cache
import os

import numpy as np


MAX_Z = 103
_DATA_PATH = os.path.join(os.path.dirname(__file__), "params", "mctc_vdwrad.npz")


@lru_cache(maxsize=1)
def load_mctc_vdwrad_packed() -> np.ndarray:
    """Load packed pair radii used by ``mctc_data_vdwrad``.

    The packed index mirrors ``get_vdw_rad_pair_num`` in the binary:

    ``idx = min(Za, Zb) + max(Za, Zb) * (max(Za, Zb) - 1) / 2 - 1``.

    Values are pair radii in Bohr. The diagonal is twice the atomic vdW radius.
    """

    with np.load(_DATA_PATH, allow_pickle=False) as data:
        packed = np.asarray(data["vdwrad_pair_packed"], dtype=np.float64)
    expected = MAX_Z * (MAX_Z + 1) // 2
    if packed.shape != (expected,):
        raise ValueError(f"expected {expected} packed vdW radii, got shape {packed.shape}")
    return packed


def mctc_vdw_pair_radius_bohr(za: int, zb: int) -> float:
    """Return the MCTC pair vdW radius in Bohr for two atomic numbers."""

    ia = int(za)
    ib = int(zb)
    if not (1 <= ia <= MAX_Z and 1 <= ib <= MAX_Z):
        raise ValueError(f"MCTC vdW radii cover Z=1..{MAX_Z}; got {za}, {zb}")
    lo = min(ia, ib)
    hi = max(ia, ib)
    idx = lo + hi * (hi - 1) // 2 - 1
    return float(load_mctc_vdwrad_packed()[idx])


def mctc_vdw_pair_matrix_bohr(atomic_numbers: np.ndarray | list[int]) -> np.ndarray:
    """Return the full symmetric MCTC pair vdW radius matrix in Bohr."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    if atoms.ndim != 1:
        raise ValueError("atomic_numbers must be a 1D array")
    if np.any(atoms < 1) or np.any(atoms > MAX_Z):
        raise ValueError(f"MCTC vdW radii cover Z=1..{MAX_Z}")

    packed = load_mctc_vdwrad_packed()
    hi = np.maximum(atoms[:, None], atoms[None, :])
    lo = np.minimum(atoms[:, None], atoms[None, :])
    idx = lo + hi * (hi - 1) // 2 - 1
    return np.asarray(packed[idx], dtype=np.float64)
