"""GFN-xTB tight-binding methods for mlxmolkit.

The headline production entry point is :func:`gfn2_alpb_water_optimize`,
the in-process replacement for ``xtb --opt --alpb water`` used by the
openCOSMO-RS conformer pipeline. It runs ANCopt on a tblite-backed
GFN2-xTB + ALPB(water) gradient.

For lower-level access, the family of single-point methods is exposed
directly: GFN0 (non-SCF), GFN1 (SCC on shell charges), GFN2 (SCC +
anisotropic electrostatics). Analytical gradients are available for
GFN1 (FD-floor accuracy) and partially for GFN2 (band+SCC analytical;
AES via FD on E_aes; ~6e-3 Ha/Å residual on small organics — for
production opt use the tblite-backed pipeline).

Future hook for g-xTB (Grimme 2026): when the ``thfroitzheim/save_tblite``
module goes public, the calc factory in :mod:`solvation_alpb` will
swap ``method='GFN2-xTB'`` → ``'g-xtb'`` with no API changes.
"""

# --- Production API ---
from .solvation_alpb import (  # noqa: F401
    alpb_water_correction,
    gfn2_alpb_water_optimize,
    gfn2_energy_alpb_water,
)
from .solvation_alpb_native import (  # noqa: F401
    alpb_water_correction_native,
    gfn2_alpb_water_native_singlepoint,
)

# --- Single-point methods ---
from .energy import gfn0_energy  # noqa: F401
from .scf_gfn1 import gfn1_energy  # noqa: F401
from .scf_gfn2 import gfn2_energy  # noqa: F401

# --- Gradients ---
from .gradient_gfn0 import gfn0_gradient  # noqa: F401
from .gradient_gfn1 import gfn1_gradient, gfn1_gradient_analytical  # noqa: F401
from .gradient_gfn2 import gfn2_gradient, gfn2_gradient_analytical  # noqa: F401

# --- Optimizer ---
from .optimizer import ancopt  # noqa: F401

__all__ = [
    "gfn2_alpb_water_optimize",
    "gfn2_energy_alpb_water",
    "alpb_water_correction",
    "alpb_water_correction_native",
    "gfn2_alpb_water_native_singlepoint",
    "gfn0_energy",
    "gfn1_energy",
    "gfn2_energy",
    "gfn0_gradient",
    "gfn1_gradient",
    "gfn1_gradient_analytical",
    "gfn2_gradient",
    "gfn2_gradient_analytical",
    "ancopt",
]
