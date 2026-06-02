# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Small MCTC coordination-number kernels recovered from the g-xTB binary."""

from __future__ import annotations

import math

import numpy as np


def erf_coordination_number(
    atomic_numbers: np.ndarray | list[int],
    coords: np.ndarray,
    rcov_by_z: np.ndarray,
    *,
    k: float,
    power: float = 1.0,
    cutoff: float = 25.0,
) -> np.ndarray:
    """Return the ``mctc_ncoord`` ERF coordination number.

    The pair formula is recovered from ``mctc_ncoord_erf::ncoord_count``:

    ``0.5 * (1 + erf(-k * (r - r0) / r0**power))``

    where ``r0 = rcov[ZA] + rcov[ZB]``. Coordinates and radii must be in the
    same units.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    xyz = np.asarray(coords, dtype=np.float64)
    rcov = np.asarray(rcov_by_z, dtype=np.float64)
    if xyz.shape != (atoms.size, 3):
        raise ValueError("coords must have shape (nat, 3)")
    if np.any(atoms < 1) or np.any(atoms > rcov.size):
        raise ValueError("atomic_numbers are outside the supplied rcov table")

    atom_rcov = rcov[atoms - 1]
    cn = np.zeros(atoms.size, dtype=np.float64)
    for i in range(atoms.size):
        for j in range(i + 1, atoms.size):
            rij = float(np.linalg.norm(xyz[i] - xyz[j]))
            if rij < 1.0e-12 or rij > cutoff:
                continue
            r0 = float(atom_rcov[i] + atom_rcov[j])
            denom = max(r0**power, 1.0e-12)
            fij = 0.5 * (1.0 + math.erf(-k * (rij - r0) / denom))
            cn[i] += fij
            cn[j] += fij
    return cn
