# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Experimental fast GFN2 wrappers.

This sidecar keeps :mod:`scf_gfn2` untouched and swaps selected
reference functions with vectorized equivalents for benchmarking.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import scf_gfn2 as _ref
from .aes_fast import (
    mmompop_vectorized,
    mulliken_shell_charges_vectorized,
    setvsdq_vectorized,
)
from .multipole_integrals_cpp import CPP_AVAILABLE, mmompop_cpp, multipole_matrices_cpp


def gfn2_energy_fast(
    atoms: list[int],
    coords_ang: np.ndarray,
    **kwargs: Any,
) -> dict:
    """GFN2 energy with vectorized AES multipole populations."""

    old_mmompop = _ref.mmompop
    old_setvsdq = _ref.setvsdq
    old_mulliken_shell_charges = _ref._mulliken_shell_charges
    old_multipole_matrices = _ref.multipole_matrices
    _ref.mmompop = mmompop_cpp if CPP_AVAILABLE else mmompop_vectorized
    _ref.setvsdq = setvsdq_vectorized
    _ref._mulliken_shell_charges = mulliken_shell_charges_vectorized
    if CPP_AVAILABLE:
        _ref.multipole_matrices = multipole_matrices_cpp
    try:
        return _ref.gfn2_energy(atoms, coords_ang, **kwargs)
    finally:
        _ref.mmompop = old_mmompop
        _ref.setvsdq = old_setvsdq
        _ref._mulliken_shell_charges = old_mulliken_shell_charges
        _ref.multipole_matrices = old_multipole_matrices
