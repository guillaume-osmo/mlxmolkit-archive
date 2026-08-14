# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Clean-room, binary-guided g-xTB reconstruction pieces.

This module is intentionally narrower than a full g-xTB calculator.  It turns
the recovered repulsion constants and native pair-loop microkernel into a
callable component so we can run molecule-scale numerical probes while the
Hamiltonian/SCC pieces are still being reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gxtb_cpp import (
    GXTBRepulsionState,
    repulsion_energy_gradient_asm,
    repulsion_state,
)
from .mctc_vdwrad import mctc_vdw_pair_matrix_bohr
from .mctc_ncoord import erf_coordination_number
from .params_gxtb import GXTB_PARAMS, GXTB_REPULSION_LITERAL_BY_ADDR


ANG_TO_BOHR = 1.8897259886


@dataclass(frozen=True)
class GXTBReconstructedRepulsionConstants:
    """Scalar constants mapped from the recovered repulsion constructor calls."""

    stored_scalar_1p5: float
    exp_power_1: float
    exp_power_2: float
    exp2_scale: float
    exp2_weight: float
    cubic_coeff: float
    quartic_coeff: float
    light_pair_coeff: float
    heavy_pair_coeff: float


@dataclass(frozen=True)
class GXTBReconstructedRepulsion:
    """Result of the current reconstructed g-xTB repulsion block."""

    energy: float
    gradient: np.ndarray
    matvec: np.ndarray
    state: GXTBRepulsionState
    cn: np.ndarray
    pair_rvdw: np.ndarray
    pair_roffset: np.ndarray
    linear_coeff: np.ndarray
    quadratic_coeff: np.ndarray
    constants: GXTBReconstructedRepulsionConstants
    metadata: dict[str, object]


def repulsion_constants_from_binary() -> GXTBReconstructedRepulsionConstants:
    """Return the current scalar mapping from the disassembled g-xTB binary.

    The mapping below is the direct working hypothesis from ``add_repulsion``
    and ``new_repulsion_gxtb``:

    * 1.5 is stored in the repulsion object at offset 0x170.
    * 2.068 and 2.0 are the two exponential powers.
    * 0.73 and 0.0046511298 scale/weight the second exponential.
    * 0.0110955395 and 0.0116077951 are the global rvdw/R cubic/quartic
      coefficients loaded in the pair matrix loop.
    * 0.0120981314 and 0.0085442527 are the light/heavy atom-class constants
      selected by ``Z < 3`` in the constructor loop.
    * The two recovered average objects are geometric (ID 1, for rvdw scale)
      and arithmetic (ID 0, for pair coefficient matrices).
    """

    lit = GXTB_REPULSION_LITERAL_BY_ADDR
    return GXTBReconstructedRepulsionConstants(
        stored_scalar_1p5=lit[0x73B268],
        exp_power_1=lit[0x73B270],
        exp_power_2=lit[0x73B278],
        exp2_scale=lit[0x73B280],
        exp2_weight=lit[0x73B288],
        cubic_coeff=lit[0x73B298],
        quartic_coeff=lit[0x73B290],
        light_pair_coeff=lit[0x73B2A0],
        heavy_pair_coeff=lit[0x73B2A8],
    )


def _arithmetic_pair_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return 0.5 * (arr[:, None] + arr[None, :])


def _geometric_pair_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("geometric average requires non-negative values")
    return np.sqrt(arr[:, None] * arr[None, :])


def _coefficient_matrices(
    atomic_numbers: np.ndarray,
    constants: GXTBReconstructedRepulsionConstants,
) -> tuple[np.ndarray, np.ndarray]:
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    atom_k1 = np.asarray(GXTB_PARAMS["pa_rep_k1"][atoms - 1], dtype=np.float64)
    atom_coeff = np.where(
        atoms < 3,
        constants.light_pair_coeff,
        constants.heavy_pair_coeff,
    ).astype(np.float64)
    linear = _arithmetic_pair_average(atom_k1)
    quadratic = _arithmetic_pair_average(atom_coeff)
    np.fill_diagonal(linear, 0.0)
    np.fill_diagonal(quadratic, 0.0)
    return linear, quadratic


def _vdw_pair_matrix_bohr(
    atomic_numbers: np.ndarray,
    constants: GXTBReconstructedRepulsionConstants,
) -> np.ndarray:
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    scale = np.asarray(GXTB_PARAMS["pa_rvdw_scale"][atoms - 1], dtype=np.float64)
    pair_scale = _geometric_pair_average(scale)
    pair = mctc_vdw_pair_matrix_bohr(atoms) * pair_scale
    np.fill_diagonal(pair, 0.0)
    return pair


def _pair_roffset_matrix(atomic_numbers: np.ndarray) -> np.ndarray:
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    atom_roffset = np.asarray(GXTB_PARAMS["pa_rep_roffset"][atoms - 1], dtype=np.float64)
    pair = 0.5 * (atom_roffset[:, None] + atom_roffset[None, :])
    np.fill_diagonal(pair, 0.0)
    return pair


def _gxtb_erf_coordination_number(
    atomic_numbers: np.ndarray,
    coords_ang: np.ndarray,
    *,
    k: float = 2.068,
    power: float = 1.0,
    cutoff: float = 25.0,
) -> np.ndarray:
    """Binary-guided g-xTB ``mctc_ncoord`` ERF coordination number.

    ``new_gxtb_calculator`` passes count type ID 3 to ``new_ncoord``. The
    dispatcher maps ID 3 to ``new_erf_ncoord`` and passes the literal at
    ``0x73b270`` as the ERF steepness. The recovered pair formula is:

    ``0.5 * (1 + erf(-k * (r - r0) / r0**power))``.
    """

    return erf_coordination_number(
        atomic_numbers,
        coords_ang,
        GXTB_PARAMS["pa_cn_rcov"],
        k=k,
        power=power,
        cutoff=cutoff,
    )


def gxtb_reconstructed_repulsion(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
    *,
    descriptor: np.ndarray | None = None,
    cn: np.ndarray | None = None,
    cutoff_bohr: float = 25.0,
) -> GXTBReconstructedRepulsion:
    """Compute the current reconstructed g-xTB repulsion energy/gradient.

    Coordinates are supplied in Angstrom. The native pair loop runs in Bohr and
    returns dE/dBohr, which is converted to Hartree/Angstrom for the public
    gradient field.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    if coords.shape != (atoms.size, 3):
        raise ValueError("coords_ang must have shape (nat, 3)")

    constants = repulsion_constants_from_binary()
    cn_arr = _gxtb_erf_coordination_number(atoms, coords) if cn is None else np.asarray(cn, dtype=np.float64)
    if descriptor is None:
        descriptor_arr = np.zeros(atoms.size, dtype=np.float64)
    else:
        descriptor_arr = np.asarray(descriptor, dtype=np.float64)
    if cn_arr.shape != atoms.shape:
        raise ValueError("cn must have one value per atom")
    if descriptor_arr.shape != atoms.shape:
        raise ValueError("descriptor must have one value per atom")

    pair_roffset = _pair_roffset_matrix(atoms)
    state = repulsion_state(
        atoms,
        descriptor=descriptor_arr,
        cn=cn_arr,
        pair_roffset=pair_roffset,
    )
    pair_rvdw = _vdw_pair_matrix_bohr(atoms, constants)
    linear_coeff, quadratic_coeff = _coefficient_matrices(atoms, constants)

    energy, gradient_bohr, matvec = repulsion_energy_gradient_asm(
        coords * ANG_TO_BOHR,
        state.scaled_zeff,
        state.alpha,
        pair_rvdw,
        pair_roffset,
        linear_coeff,
        quadratic_coeff,
        constants.cubic_coeff,
        constants.quartic_coeff,
        constants.exp_power_1,
        constants.exp_power_2,
        constants.exp2_scale,
        constants.exp2_weight,
        cutoff=cutoff_bohr,
    )

    return GXTBReconstructedRepulsion(
        energy=float(energy),
        gradient=np.asarray(gradient_bohr, dtype=np.float64) * ANG_TO_BOHR,
        matvec=np.asarray(matvec, dtype=np.float64),
        state=state,
        cn=cn_arr,
        pair_rvdw=pair_rvdw,
        pair_roffset=pair_roffset,
        linear_coeff=linear_coeff,
        quadratic_coeff=quadratic_coeff,
        constants=constants,
        metadata={
            "component": "gxtb_repulsion",
            "reconstruction": "binary-guided",
            "complete_gxtb": False,
            "missing_for_full_energy": (
                "EEQ_BC",
                "q-vSZP basis",
                "overlap/H0",
                "shell-charge SCC",
                "exchange",
                "anisotropic H0",
                "p-ACP",
                "D4Srev",
            ),
            "pair_builder_status": "candidate; pair-loop kernel/constants, average IDs, exact MCTC vdW pair table, and ERF CN type/k recovered; H0/SCC still absent",
        },
    )
