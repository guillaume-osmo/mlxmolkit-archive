# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Optional native float64 CPU kernels for clean-room g-xTB reconstruction.

Important: this module is not yet a complete C++ g-xTB calculator. The Python
reconstruction now has EEQ_BC, q-vSZP, H0, SCC, and a finite-difference driver
in :mod:`mlxmolkit.xtb.scf_gxtb`; this C++ facade still exposes only recovered
microkernels and status for the native rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params_gxtb import GXTB_PARAMS

try:
    from . import _gxtb_cpp
except Exception:  # pragma: no cover - depends on local extension build
    _gxtb_cpp = None


CPP_AVAILABLE = _gxtb_cpp is not None


@dataclass(frozen=True)
class GXTBCppBlock:
    """One recoverable block in the clean-room g-xTB native implementation."""

    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class GXTBSymbolKernel:
    """Native wrapper tied to one observed g-xTB binary symbol."""

    symbol: str
    wrapper: str
    status: str
    evidence: str


@dataclass(frozen=True)
class GXTBRepulsionState:
    """Atom-resolved repulsion state after the observed update/scaling steps."""

    atomic_numbers: np.ndarray
    descriptor: np.ndarray
    cn: np.ndarray
    scaled_zeff: np.ndarray
    alpha: np.ndarray
    dalpha_dcn: np.ndarray
    atom_roffset: np.ndarray
    pair_roffset: np.ndarray


IMPLEMENTATION_STATUS: tuple[GXTBCppBlock, ...] = (
    GXTBCppBlock(
        "binary parameter tables",
        "implemented",
        "103-element g-xTB arrays vendored in params/gxtb_binary_params.npz",
    ),
    GXTBCppBlock(
        "repulsion scalar helpers",
        "implemented",
        "scaled Zeff, CN-scaled alpha, descriptor potential, pair value and d/dr",
    ),
    GXTBCppBlock(
        "repulsion pair matrix/gradient microkernel",
        "implemented",
        "native float64 CPU matrix, gradient, and matvec contraction kernels, including rvdw/R assembly form",
    ),
    GXTBCppBlock(
        "MCTC vdW pair table",
        "implemented",
        "packed 103-element pair-radius table and triangular indexing recovered from mctc_data_vdwrad",
    ),
    GXTBCppBlock(
        "MCTC ERF coordination number",
        "implemented",
        "new_gxtb_calculator uses ncoord type 3 with k=2.068 and pa_cn_rcov custom radii",
    ),
    GXTBCppBlock(
        "EEQ_BC 2025 parameter tables",
        "implemented",
        "nine 103-element EEQ_BC arrays recovered from multicharge_param_eeqbc2025",
    ),
    GXTBCppBlock(
        "multipole damping microkernels",
        "implemented",
        "observed g-xTB damping-pair, damping derivative, and mrad indexing helpers",
    ),
    GXTBCppBlock(
        "EEQ_BC basis-charge model",
        "implemented",
        "pure-numpy xvec/coulomb solve is wired in eeqbc.py; derivatives still separate",
    ),
    GXTBCppBlock(
        "q-vSZP basis and overlap",
        "implemented",
        "binary q-vSZP tables extracted and active pa_nshell basis/overlap built in gxtb_basis.py",
    ),
    GXTBCppBlock(
        "g-xTB H0 build",
        "partial",
        "CN-shifted diagonal/off-diagonal H0 scaffold is wired; anisotropic H0 terms remain",
    ),
    GXTBCppBlock(
        "shell-charge SCC Fock",
        "partial",
        "shell-charge Coulomb and optional high-order table hooks exist; exact multipole Coulomb still pending",
    ),
    GXTBCppBlock(
        "one-center exchange",
        "partial",
        "recovered tables are exposed; exact INDO-like Fock assembly is not enabled by default",
    ),
    GXTBCppBlock(
        "p-ACP and D4Srev assembly",
        "partial",
        "p-ACP proxy and D4Srev fallback are isolated; exact projector/reference model still pending",
    ),
    GXTBCppBlock(
        "SCF driver and analytic gradient assembly",
        "partial",
        "Python SCF and central-difference gradient exist in scf_gxtb.py; native C++ and analytic dE/dR pending",
    ),
)


DISASSEMBLED_KERNELS: tuple[GXTBSymbolKernel, ...] = (
    GXTBSymbolKernel(
        "tblite_repulsion_gxtb::get_scaled_zeff",
        "scaled_zeff / repulsion_scaled_zeff",
        "implemented",
        "ARM64 fmsub/fmul loop: zeff[Z] * (1 - scale[Z] * descriptor[i])",
    ),
    GXTBSymbolKernel(
        "tblite_repulsion_gxtb::update",
        "cn_scaled_parameter / repulsion_cn_scaled_parameter",
        "implemented",
        "ARM64 fsqrt/fmadd/fdiv loop: base[Z] * (1 + slope[Z] * (sqrt(cn+eps2)-eps))",
    ),
    GXTBSymbolKernel(
        "tblite_repulsion_gxtb::get_energy",
        "repulsion_energy_from_matvec",
        "implemented",
        "calls get_scaled_zeff and DSYMV, then contracts scaled_zeff dot matvec",
    ),
    GXTBSymbolKernel(
        "tblite_repulsion_gxtb::get_potential",
        "repulsion_descriptor_potential",
        "implemented",
        "same DSYMV path; descriptor derivative is -zeff[Z] * scale[Z] * matvec",
    ),
    GXTBSymbolKernel(
        "tblite_repulsion_gxtb::{matrix,derivs} pair loop",
        "repulsion_pair_matrix_asm / repulsion_energy_gradient_asm",
        "partial",
        "two-exponential rvdw/R inverse-polynomial microkernel is native; MCTC vdW, average IDs, and ERF CN type/k are mapped",
    ),
    GXTBSymbolKernel(
        "tblite_coulomb_multipole_gxtb::get_damping_pair",
        "multipole_damping_pair / multipole_damping_pair_derivs",
        "implemented",
        "four erf damping channels and d/d(a-b) derivative",
    ),
)


def _require_cpp() -> None:
    if _gxtb_cpp is None:
        raise ImportError(
            "mlxmolkit.xtb._gxtb_cpp is not built. Run "
            "`python setup.py build_ext --inplace` in the osmo env."
        )


def implementation_status() -> tuple[GXTBCppBlock, ...]:
    """Return the native g-xTB reconstruction status block-by-block."""

    return IMPLEMENTATION_STATUS


def disassembled_kernel_status() -> tuple[GXTBSymbolKernel, ...]:
    """Return the binary symbols already translated into callable wrappers."""

    return DISASSEMBLED_KERNELS


def full_calculator_available() -> bool:
    """Return whether this module can compute standalone g-xTB energies."""

    return all(block.status == "implemented" for block in IMPLEMENTATION_STATUS)


def missing_calculator_blocks() -> tuple[GXTBCppBlock, ...]:
    """Return blocks still needed for a complete native g-xTB calculator."""

    return tuple(block for block in IMPLEMENTATION_STATUS if block.status != "implemented")


def gxtb_energy_gradient(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder for the eventual native g-xTB energy/gradient calculator.

    The explicit guard is intentional: callers should use the executable oracle
    for g-xTB until every block in :func:`missing_calculator_blocks` is closed.
    """

    del args, kwargs
    missing = ", ".join(block.name for block in missing_calculator_blocks())
    raise NotImplementedError(
        "native C++ g-xTB is not complete yet. Implemented pieces are only "
        "microkernels/parameter accessors; missing blocks: "
        f"{missing}"
    )


def scaled_zeff(
    atomic_numbers: np.ndarray,
    zeff_by_z: np.ndarray,
    scale_by_z: np.ndarray,
    descriptor: np.ndarray,
) -> np.ndarray:
    """Compute ``zeff[Z] * (1 - scale[Z] * descriptor)`` in C++."""

    _require_cpp()
    return _gxtb_cpp.scaled_zeff(
        np.asarray(atomic_numbers, dtype=np.intp),
        np.asarray(zeff_by_z, dtype=np.float64),
        np.asarray(scale_by_z, dtype=np.float64),
        np.asarray(descriptor, dtype=np.float64),
    )


def repulsion_scaled_zeff(
    atomic_numbers: np.ndarray,
    descriptor: np.ndarray,
    scale: str = "pa_rep_q",
) -> np.ndarray:
    """Convenience wrapper using the binary-extracted g-xTB repulsion tables."""

    return scaled_zeff(
        atomic_numbers,
        GXTB_PARAMS["pa_rep_zeff"],
        GXTB_PARAMS[scale],
        descriptor,
    )


def cn_scaled_parameter(
    atomic_numbers: np.ndarray,
    base_by_z: np.ndarray,
    slope_by_z: np.ndarray,
    cn: np.ndarray,
    eps2: float = 1.0e-12,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the observed g-xTB CN scaling and its CN derivative."""

    _require_cpp()
    return _gxtb_cpp.cn_scaled_parameter(
        np.asarray(atomic_numbers, dtype=np.intp),
        np.asarray(base_by_z, dtype=np.float64),
        np.asarray(slope_by_z, dtype=np.float64),
        np.asarray(cn, dtype=np.float64),
        float(eps2),
        float(eps),
    )


def repulsion_cn_scaled_parameter(
    atomic_numbers: np.ndarray,
    cn: np.ndarray,
    base: str = "pa_rep_alpha",
    slope: str = "pa_rep_cn",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper using binary-extracted g-xTB repulsion tables."""

    return cn_scaled_parameter(
        atomic_numbers,
        GXTB_PARAMS[base],
        GXTB_PARAMS[slope],
        cn,
    )


def repulsion_atom_roffset(atomic_numbers: np.ndarray) -> np.ndarray:
    """Return binary-extracted atom roffset values for the repulsion block."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    return np.asarray(GXTB_PARAMS["pa_rep_roffset"][atoms - 1], dtype=np.float64)


def repulsion_pair_roffset_arithmetic(atomic_numbers: np.ndarray) -> np.ndarray:
    """Build a symmetric candidate pair roffset from atom roffset parameters.

    The binary exposes atom-level ``pa_rep_roffset``. The exact pair-combiner is
    part of the denser repulsion pair-loop still being mapped, so this function
    is intentionally named as the arithmetic candidate rather than the final
    production g-xTB combiner.
    """

    atom_roffset = repulsion_atom_roffset(atomic_numbers)
    return 0.5 * (atom_roffset[:, None] + atom_roffset[None, :])


def repulsion_state(
    atomic_numbers: np.ndarray,
    descriptor: np.ndarray | None = None,
    cn: np.ndarray | None = None,
    pair_roffset: np.ndarray | None = None,
) -> GXTBRepulsionState:
    """Apply the disassembled repulsion scaling/update formulas to one system."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    if descriptor is None:
        descriptor_arr = np.zeros(atoms.shape, dtype=np.float64)
    else:
        descriptor_arr = np.asarray(descriptor, dtype=np.float64)
    if cn is None:
        cn_arr = np.zeros(atoms.shape, dtype=np.float64)
    else:
        cn_arr = np.asarray(cn, dtype=np.float64)
    if descriptor_arr.shape != atoms.shape:
        raise ValueError("descriptor must have one value per atom")
    if cn_arr.shape != atoms.shape:
        raise ValueError("cn must have one value per atom")

    scaled = repulsion_scaled_zeff(atoms, descriptor_arr)
    alpha, dalpha_dcn = repulsion_cn_scaled_parameter(atoms, cn_arr)
    atom_roffset = repulsion_atom_roffset(atoms)
    if pair_roffset is None:
        pair_roffset_arr = repulsion_pair_roffset_arithmetic(atoms)
    else:
        pair_roffset_arr = np.asarray(pair_roffset, dtype=np.float64)
    if pair_roffset_arr.shape != (atoms.size, atoms.size):
        raise ValueError("pair_roffset must be an (nat, nat) matrix")

    return GXTBRepulsionState(
        atomic_numbers=atoms,
        descriptor=descriptor_arr,
        cn=cn_arr,
        scaled_zeff=scaled,
        alpha=alpha,
        dalpha_dcn=dalpha_dcn,
        atom_roffset=atom_roffset,
        pair_roffset=pair_roffset_arr,
    )


def repulsion_energy_from_matvec(scaled: np.ndarray, matvec: np.ndarray) -> float:
    """Contract the cached repulsion matrix-vector product with scaled Zeff."""

    _require_cpp()
    return float(
        _gxtb_cpp.repulsion_energy_from_matvec(
            np.asarray(scaled, dtype=np.float64),
            np.asarray(matvec, dtype=np.float64),
        )
    )


def repulsion_descriptor_potential(
    atomic_numbers: np.ndarray,
    base_by_z: np.ndarray,
    scale_by_z: np.ndarray,
    matvec: np.ndarray,
) -> np.ndarray:
    """Compute ``-base[Z] * scale[Z] * matvec`` from the repulsion potential path."""

    _require_cpp()
    return _gxtb_cpp.repulsion_descriptor_potential(
        np.asarray(atomic_numbers, dtype=np.intp),
        np.asarray(base_by_z, dtype=np.float64),
        np.asarray(scale_by_z, dtype=np.float64),
        np.asarray(matvec, dtype=np.float64),
    )


def repulsion_q_potential(
    atomic_numbers: np.ndarray,
    matvec: np.ndarray,
    base: str = "pa_rep_zeff",
    scale: str = "pa_rep_q",
) -> np.ndarray:
    """Convenience descriptor-potential wrapper for the default repulsion-q path."""

    return repulsion_descriptor_potential(
        atomic_numbers,
        GXTB_PARAMS[base],
        GXTB_PARAMS[scale],
        matvec,
    )


def repulsion_pair_value(
    r: float,
    alpha_a: float,
    alpha_b: float,
    roffset: float,
    coeffs: np.ndarray,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
) -> float:
    """Compute the two-exponential inverse-polynomial repulsion pair form."""

    _require_cpp()
    return float(
        _gxtb_cpp.repulsion_pair_value(
            float(r),
            float(alpha_a),
            float(alpha_b),
            float(roffset),
            np.asarray(coeffs, dtype=np.float64),
            float(exp_power_1),
            float(exp_power_2),
            float(exp2_scale),
            float(exp2_weight),
        )
    )


def repulsion_pair_value_deriv(
    r: float,
    alpha_a: float,
    alpha_b: float,
    roffset: float,
    coeffs: np.ndarray,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
) -> tuple[float, float]:
    """Return pair value and derivative with respect to pair distance ``r``."""

    _require_cpp()
    value, deriv = _gxtb_cpp.repulsion_pair_value_deriv(
        float(r),
        float(alpha_a),
        float(alpha_b),
        float(roffset),
        np.asarray(coeffs, dtype=np.float64),
        float(exp_power_1),
        float(exp_power_2),
        float(exp2_scale),
        float(exp2_weight),
    )
    return float(value), float(deriv)


def repulsion_pair_value_asm(
    r: float,
    alpha_a: float,
    alpha_b: float,
    pair_rvdw: float,
    roffset: float,
    linear_coeff: float,
    quadratic_coeff: float,
    cubic_coeff: float,
    quartic_coeff: float,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
) -> float:
    """Compute the repulsion pair form seen in the g-xTB pair-loop assembly."""

    _require_cpp()
    return float(
        _gxtb_cpp.repulsion_pair_value_asm(
            float(r),
            float(alpha_a),
            float(alpha_b),
            float(pair_rvdw),
            float(roffset),
            float(linear_coeff),
            float(quadratic_coeff),
            float(cubic_coeff),
            float(quartic_coeff),
            float(exp_power_1),
            float(exp_power_2),
            float(exp2_scale),
            float(exp2_weight),
        )
    )


def repulsion_pair_value_asm_deriv(
    r: float,
    alpha_a: float,
    alpha_b: float,
    pair_rvdw: float,
    roffset: float,
    linear_coeff: float,
    quadratic_coeff: float,
    cubic_coeff: float,
    quartic_coeff: float,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
) -> tuple[float, float]:
    """Return the assembly-derived pair value and d/dR derivative."""

    _require_cpp()
    value, deriv = _gxtb_cpp.repulsion_pair_value_asm_deriv(
        float(r),
        float(alpha_a),
        float(alpha_b),
        float(pair_rvdw),
        float(roffset),
        float(linear_coeff),
        float(quadratic_coeff),
        float(cubic_coeff),
        float(quartic_coeff),
        float(exp_power_1),
        float(exp_power_2),
        float(exp2_scale),
        float(exp2_weight),
    )
    return float(value), float(deriv)


def repulsion_pair_matrix(
    coords: np.ndarray,
    alpha: np.ndarray,
    pair_roffset: np.ndarray,
    coeffs: np.ndarray,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
    cutoff: float = 25.0,
) -> np.ndarray:
    """Build the symmetric pair matrix for the recovered repulsion pair form."""

    _require_cpp()
    return _gxtb_cpp.repulsion_pair_matrix(
        np.asarray(coords, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(pair_roffset, dtype=np.float64),
        np.asarray(coeffs, dtype=np.float64),
        float(exp_power_1),
        float(exp_power_2),
        float(exp2_scale),
        float(exp2_weight),
        float(cutoff),
    )


def repulsion_pair_matrix_asm(
    coords: np.ndarray,
    alpha: np.ndarray,
    pair_rvdw: np.ndarray,
    pair_roffset: np.ndarray,
    linear_coeff: np.ndarray,
    quadratic_coeff: np.ndarray,
    cubic_coeff: float,
    quartic_coeff: float,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
    cutoff: float = 25.0,
) -> np.ndarray:
    """Build the symmetric pair matrix for the assembly-derived pair form."""

    _require_cpp()
    return _gxtb_cpp.repulsion_pair_matrix_asm(
        np.asarray(coords, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(pair_rvdw, dtype=np.float64),
        np.asarray(pair_roffset, dtype=np.float64),
        np.asarray(linear_coeff, dtype=np.float64),
        np.asarray(quadratic_coeff, dtype=np.float64),
        float(cubic_coeff),
        float(quartic_coeff),
        float(exp_power_1),
        float(exp_power_2),
        float(exp2_scale),
        float(exp2_weight),
        float(cutoff),
    )


def repulsion_energy_gradient(
    coords: np.ndarray,
    scaled: np.ndarray,
    alpha: np.ndarray,
    pair_roffset: np.ndarray,
    coeffs: np.ndarray,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
    cutoff: float = 25.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute energy, Cartesian gradient, and cached matvec for the pair form."""

    _require_cpp()
    energy, gradient, matvec = _gxtb_cpp.repulsion_energy_gradient(
        np.asarray(coords, dtype=np.float64),
        np.asarray(scaled, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(pair_roffset, dtype=np.float64),
        np.asarray(coeffs, dtype=np.float64),
        float(exp_power_1),
        float(exp_power_2),
        float(exp2_scale),
        float(exp2_weight),
        float(cutoff),
    )
    return float(energy), gradient, matvec


def repulsion_energy_gradient_asm(
    coords: np.ndarray,
    scaled: np.ndarray,
    alpha: np.ndarray,
    pair_rvdw: np.ndarray,
    pair_roffset: np.ndarray,
    linear_coeff: np.ndarray,
    quadratic_coeff: np.ndarray,
    cubic_coeff: float,
    quartic_coeff: float,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
    cutoff: float = 25.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute energy, gradient, and matvec for the assembly-derived pair form."""

    _require_cpp()
    energy, gradient, matvec = _gxtb_cpp.repulsion_energy_gradient_asm(
        np.asarray(coords, dtype=np.float64),
        np.asarray(scaled, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(pair_rvdw, dtype=np.float64),
        np.asarray(pair_roffset, dtype=np.float64),
        np.asarray(linear_coeff, dtype=np.float64),
        np.asarray(quadratic_coeff, dtype=np.float64),
        float(cubic_coeff),
        float(quartic_coeff),
        float(exp_power_1),
        float(exp_power_2),
        float(exp2_scale),
        float(exp2_weight),
        float(cutoff),
    )
    return float(energy), gradient, matvec


def repulsion_energy_gradient_parameterized(
    atomic_numbers: np.ndarray,
    coords: np.ndarray,
    descriptor: np.ndarray | None,
    cn: np.ndarray | None,
    coeffs: np.ndarray,
    exp_power_1: float,
    exp_power_2: float,
    exp2_scale: float,
    exp2_weight: float,
    pair_roffset: np.ndarray | None = None,
    cutoff: float = 25.0,
) -> tuple[float, np.ndarray, np.ndarray, GXTBRepulsionState]:
    """Run the recovered repulsion state plus native pair energy/gradient.

    This is not yet a complete g-xTB repulsion component because the final
    binary pair-combiner/constants are still being mapped. It is the direct
    callable bridge from observed atom updates to the native pair microkernel.
    """

    state = repulsion_state(
        atomic_numbers,
        descriptor=descriptor,
        cn=cn,
        pair_roffset=pair_roffset,
    )
    energy, gradient, matvec = repulsion_energy_gradient(
        coords,
        state.scaled_zeff,
        state.alpha,
        state.pair_roffset,
        coeffs,
        exp_power_1,
        exp_power_2,
        exp2_scale,
        exp2_weight,
        cutoff=cutoff,
    )
    return energy, gradient, matvec, state


def multipole_damping_pair(
    a: float,
    b: float,
    amplitudes: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Compute the four observed g-xTB multipole damping-pair values."""

    _require_cpp()
    return _gxtb_cpp.multipole_damping_pair(
        float(a),
        float(b),
        np.asarray(amplitudes, dtype=np.float64),
        np.asarray(betas, dtype=np.float64),
    )


def multipole_mrad_pair(table: np.ndarray, i: int, j: int) -> float:
    """Fetch one multipole-radius pair table entry with the observed indexing."""

    _require_cpp()
    return float(_gxtb_cpp.multipole_mrad_pair(np.asarray(table, dtype=np.float64), int(i), int(j)))


def multipole_damping_pair_derivs(
    a: float,
    b: float,
    amplitudes: np.ndarray,
    betas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return damping values and derivatives with respect to ``a - b``."""

    _require_cpp()
    return _gxtb_cpp.multipole_damping_pair_derivs(
        float(a),
        float(b),
        np.asarray(amplitudes, dtype=np.float64),
        np.asarray(betas, dtype=np.float64),
    )
