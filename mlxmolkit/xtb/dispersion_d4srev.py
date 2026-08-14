# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""g-xTB D4Srev dispatch scaffold.

The release binary ships a dedicated ``dftd4_model_d4srev`` model and g-xTB
reference-charge tables.  Those tables are visible in the Mach-O symbol table,
but there is no Python binding for that model in the installed ``dftd4``
package.  This module keeps the native g-xTB driver wired and explicit: by
default it uses the existing D4 backend as a numerical fallback, while callers
can disable it or replace it when the D4Srev table extractor lands.
"""

from __future__ import annotations

import numpy as np

from .dispersion_d4 import d4_dispersion_gfn2


def d4srev_dispersion_gxtb(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    enabled: bool = True,
) -> tuple[float, str]:
    """Return ``(energy_hartree, backend_label)`` for the current g-xTB path."""

    if not enabled:
        return 0.0, "disabled"
    return float(d4_dispersion_gfn2(atomic_numbers, coords_ang)), "gfn2-d4-fallback"
