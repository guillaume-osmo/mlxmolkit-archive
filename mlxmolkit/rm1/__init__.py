"""mlxmolkit.rm1 — PM6_D / RM1 semi-empirical SCF on Apple Silicon.

Public API — only the entry points covered by the test suite.
Lower-level internals are reachable via submodule imports but are not
part of the stable surface.

Tested entry points
-------------------

SCF
    nddo_energy(atoms, coords, method='RM1') -> dict
        Single-molecule SCF. Returns electronic + nuclear + heat-of-formation
        energies plus the converged density. Methods: 'RM1', 'AM1', 'PM3',
        'PM6_SP' (sp-only), 'PM6_D' (full d-orbital, requires the vendored
        NumPy PYSEQM port — no external dep).
        Tests: tests/test_rm1_scf.py, tests/test_pm6_d_native.py

    nddo_energy_batch(atoms_list, coords_list, method='RM1') -> list[dict]
        Batched version. Tests: tests/test_rm1_scf.py

PM6-D3H4 corrections (post-SCF)
    pm6_d3h4_correction(atoms, coords) -> float
        Sum of Grimme D3 dispersion + Rezáč-Hobza H4 H-bond
        + HH-repulsion (eV). Tests: tests/test_pm6_d3h4.py

    d3_energy(atoms, coords) -> float
    h4_energy(atoms, coords) -> float
    hh_repulsion(atoms, coords) -> float
        Individual components.

Parameters
    METHOD_PARAMS : dict[str, dict[int, ElementParams]]
        Method-name -> Z -> parameters lookup.
    get_params(method) -> dict[int, ElementParams]
    ElementParams
        Dataclass holding one element's NDDO parameters.

Bit-exact reference primitives (vendored from PYSEQM, BSD-3, LANL)
    These match PYSEQM to machine precision; the regression suite in
    tests/test_pyseqm_port.py asserts this on every commit.

    from mlxmolkit.rm1._pyseqm_port import (
        diatom_overlap_matrixD,     # qn=1..6 diatomic overlap (incl. d)
        two_elec_two_center_int,    # full TETCI per pair
        qn_int, qnD_int,            # periodic-table tables
    )
"""

# --- public API ---
from .scf import nddo_energy, nddo_energy_batch
from .gradient import nddo_gradient, nddo_optimize, nddo_optimize_batch
from .pm6_d3h4 import (
    pm6_d3h4_correction,
    d3_energy,
    h4_energy,
    hh_repulsion,
)
from .methods import METHOD_PARAMS, get_params
from .params import ElementParams, ANG_TO_BOHR, principal_qn, RM1_PARAMS

__all__ = [
    # SCF
    "nddo_energy",
    "nddo_energy_batch",
    # Gradient + geometry optimization
    "nddo_gradient",
    "nddo_optimize",
    "nddo_optimize_batch",
    # PM6-D3H4 corrections
    "pm6_d3h4_correction",
    "d3_energy",
    "h4_energy",
    "hh_repulsion",
    # Parameters
    "METHOD_PARAMS",
    "get_params",
    "ElementParams",
    "RM1_PARAMS",
    # Constants
    "ANG_TO_BOHR",
    "principal_qn",
]
