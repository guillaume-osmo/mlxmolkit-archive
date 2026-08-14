# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""EEQ_BC 2025 charge model used by g-xTB basis setup.

This is the non-periodic molecular path from multicharge's EEQ_BC model:

* ERF coordination number with ``kcn=2.0`` and ``norm_exp=0.75``.
* Electronegativity-weighted ERF CN for the local charge ``qloc``.
* Maxwell capacitance matrix ``C`` from the bond-capacitor pair function.
* RHS ``x = C @ (-chi + kcnchi*CN + kqchi*qloc, total_charge)``.
* Augmented Coulomb matrix solve ``A @ (q, lambda) = x``.

The public API accepts Angstrom coordinates. The EEQ_BC 2025 tables recovered
from the g-xTB binary are used directly, matching the molecular branch of the
source model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .eeqbc2025_params import EEQBC2025_PARAMS
from .mctc_ncoord import erf_coordination_number
from .mctc_vdwrad import mctc_vdw_pair_matrix_bohr
from .params_gfn2 import _GFN2_PAULING_EN


ANG_PER_BOHR = 0.529177210903
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
DEFAULT_KCN_RAD = 0.30
DEFAULT_KBC = 0.75
DEFAULT_CN_EXP = 2.0
DEFAULT_NORM_EXP = 0.75
DEFAULT_CUTOFF = 25.0


@dataclass(frozen=True)
class EEQBCResult:
    """Result bundle for the molecular EEQ_BC solve."""

    charges: np.ndarray
    energy: float
    xvec: np.ndarray
    xtmp: np.ndarray
    amat: np.ndarray
    cmat: np.ndarray
    cn: np.ndarray
    qloc: np.ndarray
    radii: np.ndarray


def _pauling_en_normalized(atomic_numbers: np.ndarray) -> np.ndarray:
    """Return EEQ_BC Pauling EN values normalized to fluorine."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    en = np.empty(atoms.size, dtype=np.float64)
    for i, z in enumerate(atoms):
        if 1 <= z < len(_GFN2_PAULING_EN):
            val = float(_GFN2_PAULING_EN[z])
        elif 87 <= z <= 103:
            # Mirrors multicharge_param::new_eeqbc2025_model actinide fixes.
            val = 1.30
            if z == 87:
                val = 0.80
            elif z == 89:
                val = 1.00
            elif z in (90, 91, 92, 95):
                val = 1.10
            elif z in (93, 94, 97, 103):
                val = 1.20
        else:
            val = 1.50
        en[i] = val / 3.98
    return en


def eeqbc_coordination_number(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
) -> np.ndarray:
    """EEQ_BC ERF coordination number used in ``xvec`` and charge widths."""

    return erf_coordination_number(
        atomic_numbers,
        coords_ang,
        EEQBC2025_PARAMS["cov_radii"],
        k=DEFAULT_CN_EXP,
        power=DEFAULT_NORM_EXP,
        cutoff=DEFAULT_CUTOFF,
    )


def eeqbc_local_charge(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
    *,
    total_charge: float = 0.0,
) -> np.ndarray:
    """Electronegativity-weighted ERF CN plus uniform total-charge shift."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    rcov = EEQBC2025_PARAMS["cov_radii"][atoms - 1]
    en = _pauling_en_normalized(atoms)
    qloc = np.zeros(atoms.size, dtype=np.float64)
    for i in range(atoms.size):
        for j in range(i):
            rij = float(np.linalg.norm(coords[i] - coords[j]))
            if rij < 1.0e-12 or rij > DEFAULT_CUTOFF:
                continue
            r0 = float(rcov[i] + rcov[j])
            count = 0.5 * (
                1.0
                + math.erf(-DEFAULT_CN_EXP * (rij - r0) / max(r0**DEFAULT_NORM_EXP, 1.0e-12))
            )
            den = en[j] - en[i]
            qloc[i] += den * count
            qloc[j] -= den * count
    qloc += float(total_charge) / float(atoms.size)
    return qloc


def eeqbc_pair_rvdw_matrix_ang(atomic_numbers: np.ndarray | list[int]) -> np.ndarray:
    """Return the EEQ_BC bond-capacitor vdW pair matrix in Angstrom."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    scale = EEQBC2025_PARAMS["rvdw_scale"][atoms - 1]
    pair_scale = 0.5 * (scale[:, None] + scale[None, :])
    return mctc_vdw_pair_matrix_bohr(atoms) * ANG_PER_BOHR * pair_scale


def eeqbc_capacitance_pair(
    r_ang: float,
    rvdw_ang: float,
    cap_i: float,
    cap_j: float,
    *,
    kbc: float = DEFAULT_KBC,
) -> float:
    """Bond-capacitor pair capacitance."""

    if r_ang < 1.0e-12:
        return 0.0
    arg = -kbc * (r_ang - rvdw_ang) / max(rvdw_ang, 1.0e-12)
    return math.sqrt(cap_i * cap_j) * 0.5 * (1.0 + math.erf(arg))


def eeqbc_capacitance_matrix(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
) -> np.ndarray:
    """Return the augmented Maxwell capacitance matrix ``C``."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    n = atoms.size
    if coords.shape != (n, 3):
        raise ValueError("coords_ang must have shape (nat, 3)")
    caps = EEQBC2025_PARAMS["cap"][atoms - 1]
    rvdw = eeqbc_pair_rvdw_matrix_ang(atoms)
    cmat = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n):
        for j in range(i):
            rij = float(np.linalg.norm(coords[j] - coords[i]))
            cij = eeqbc_capacitance_pair(rij, rvdw[i, j], caps[i], caps[j])
            cmat[j, i] = -cij
            cmat[i, j] = -cij
            cmat[i, i] += cij
            cmat[j, j] += cij
    cmat[n, n] = 1.0
    return cmat


def eeqbc_effective_radii(
    atomic_numbers: np.ndarray | list[int],
    cn: np.ndarray,
) -> np.ndarray:
    """CN-dependent Gaussian charge widths."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    base = EEQBC2025_PARAMS["rad"][atoms - 1]
    avg = EEQBC2025_PARAMS["avg_cn"][atoms - 1]
    norm = np.maximum(avg, 1.0e-12) ** DEFAULT_NORM_EXP
    return base * (1.0 - DEFAULT_KCN_RAD * np.asarray(cn, dtype=np.float64) / norm)


def eeqbc_xvec(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
    *,
    total_charge: float = 0.0,
    cn: np.ndarray | None = None,
    qloc: np.ndarray | None = None,
    cmat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xvec, xtmp)`` for the EEQ_BC linear system."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    if cn is None:
        cn = eeqbc_coordination_number(atoms, coords_ang)
    if qloc is None:
        qloc = eeqbc_local_charge(atoms, coords_ang, total_charge=total_charge)
    if cmat is None:
        cmat = eeqbc_capacitance_matrix(atoms, coords_ang)

    chi = EEQBC2025_PARAMS["chi"][atoms - 1]
    kcnchi = EEQBC2025_PARAMS["kcnchi"][atoms - 1]
    kqchi = EEQBC2025_PARAMS["kqchi"][atoms - 1]
    xtmp = np.empty(atoms.size + 1, dtype=np.float64)
    xtmp[:-1] = -chi + kcnchi * np.asarray(cn, dtype=np.float64) + kqchi * np.asarray(qloc, dtype=np.float64)
    xtmp[-1] = float(total_charge)
    return cmat @ xtmp, xtmp


def eeqbc_coulomb_matrix(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
    *,
    cn: np.ndarray | None = None,
    cmat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A, radii)`` for the augmented EEQ_BC solve."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    n = atoms.size
    if cn is None:
        cn = eeqbc_coordination_number(atoms, coords)
    if cmat is None:
        cmat = eeqbc_capacitance_matrix(atoms, coords)

    radii = eeqbc_effective_radii(atoms, cn)
    eta = EEQBC2025_PARAMS["eta"][atoms - 1]
    amat = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n):
        ri = radii[i]
        for j in range(i):
            rj = radii[j]
            rij = float(np.linalg.norm(coords[j] - coords[i]))
            gam2 = 1.0 / max(ri * ri + rj * rj, 1.0e-24)
            tmp = math.erf(math.sqrt(rij * rij * gam2)) / max(rij, 1.0e-12) * cmat[j, i]
            amat[j, i] = tmp
            amat[i, j] = tmp
        amat[i, i] += (eta[i] + SQRT_2_OVER_PI / max(ri, 1.0e-12)) * cmat[i, i] + 1.0

    amat[n, : n + 1] = 1.0
    amat[: n + 1, n] = 1.0
    amat[n, n] = 0.0
    return amat, radii


def eeqbc_charges(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
    *,
    total_charge: float = 0.0,
) -> np.ndarray:
    """Return EEQ_BC atomic charges for a non-periodic molecule."""

    return eeqbc_solve(atomic_numbers, coords_ang, total_charge=total_charge).charges


def eeqbc_solve(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
    *,
    total_charge: float = 0.0,
) -> EEQBCResult:
    """Solve the molecular EEQ_BC linear system."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    cn = eeqbc_coordination_number(atoms, coords)
    qloc = eeqbc_local_charge(atoms, coords, total_charge=total_charge)
    cmat = eeqbc_capacitance_matrix(atoms, coords)
    amat, radii = eeqbc_coulomb_matrix(atoms, coords, cn=cn, cmat=cmat)
    xvec, xtmp = eeqbc_xvec(atoms, coords, total_charge=total_charge, cn=cn, qloc=qloc, cmat=cmat)
    sol = np.linalg.solve(amat, xvec)
    q = sol[:-1]
    jmat = amat[:-1, :-1]
    energy = float(np.sum(q * (0.5 * (jmat @ q) - xvec[:-1])))
    return EEQBCResult(
        charges=q,
        energy=energy,
        xvec=xvec,
        xtmp=xtmp,
        amat=amat,
        cmat=cmat,
        cn=cn,
        qloc=qloc,
        radii=radii,
    )
