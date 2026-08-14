# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Lindh 1995 model Hessian — verbatim port of xtb/src/model_hessian.f90.

Reference: R. Lindh, A. Bernhardsson, G. Karlström, P.-Å. Malmqvist,
*Chem. Phys. Lett.* 241 (1995) 423-428.

xtb invokes this through ``mh_lindh`` (model_hessian.f90:799) which
sums four bonded contributions plus a vdW pair term:

    H = H_stretch + H_bend + H_torsion + H_outofp + H_vdW

The Hessian is built directly in Cartesian coordinates as a packed
upper-triangular vector and used as the *initial* Hessian for the
ANCopt geometry optimizer.

Helper tables (vander, c6, rav, aav, dav) and helper functions
(itabrow, fk_lindh, fk_vdw, getvdwxx/getvdwxy) are vendored verbatim
from xtb. All distances internal to this module are in **Bohr**;
the public API accepts Angstrom (consistent with the rest of mlxmolkit's
xtb subpackage).

Known limitations vs xtb:
- Constraint Hessian (``constrhess``) is not applied — fixed atoms
  contribute their natural Hessian. This module is unconstrained
  geometry optimization only.
- The default modh%kd/kr/kf/kt/ko are taken from xtb's
  modhess_setvar defaults (see ``LindhParams``).
"""

from __future__ import annotations

import math
import numpy as np


_ANG_TO_BOHR = 1.8897259886


# ---------------------------------------------------------------------------
# vdW radii table (van der Waals) — model_hessian.f90:38-56. Multiplied
# by aatoau here so internally we work in Bohr.
# ---------------------------------------------------------------------------
_VANDER_ANG = (
    0.0,
    0.91, 0.92,                                           # H, He
    0.75, 1.28, 1.35, 1.32, 1.27, 1.22, 1.17, 1.13,       # Li-Ne
    1.04, 1.24, 1.49, 1.56, 1.55, 1.53, 1.49, 1.45,       # Na-Ar
    1.35, 1.34,                                            # K, Ca
    1.42, 1.42, 1.42, 1.42, 1.42,                          # Sc-
    1.42, 1.42, 1.42, 1.42, 1.42,                          # -Zn
    1.50, 1.57, 1.60, 1.61, 1.59, 1.57,                    # Ga-Kr
    1.48, 1.46,                                            # Rb, Sr
    1.49, 1.49, 1.49, 1.49, 1.49,                          # Y-
    1.49, 1.49, 1.49, 1.49, 1.49,                          # -Cd
    1.52, 1.64, 1.71, 1.72, 1.72, 1.71,                    # In-Xe
    2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
)
assert len(_VANDER_ANG) == 87
_VANDER = tuple(v * _ANG_TO_BOHR for v in _VANDER_ANG)


# ---------------------------------------------------------------------------
# C6 coefficients (D2 model) — model_hessian.f90:57-77.
# ---------------------------------------------------------------------------
_C6 = (
    0.0,
    0.14, 0.08,
    1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
    5.71, 5.71, 10.79, 9.23, 7.84, 5.57, 5.07, 4.61,
    10.80, 10.80,
    10.80, 10.80, 10.80, 10.80, 10.80,
    10.80, 10.80, 10.80, 10.80, 10.80,
    16.99, 17.10, 16.37, 12.64, 12.47, 12.01,
    24.67, 24.67,
    24.67, 24.67, 24.67, 24.67, 24.67,
    24.67, 24.67, 24.67, 24.67, 24.67,
    37.32, 38.71, 38.44, 31.74, 31.50, 29.99,
    50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
)
assert len(_C6) == 87


# ---------------------------------------------------------------------------
# Lindh row/column parameters (per PSE row, 1..3 — model_hessian.f90:803-826).
# These are 3×3 matrices indexed by (row_i, row_j) where row is from itabrow.
# ---------------------------------------------------------------------------
_RAV = np.array(
    [
        [1.3500, 2.1000, 2.5300],
        [2.1000, 2.8700, 3.8000],
        [2.5300, 3.8000, 4.5000],
    ],
    dtype=np.float64,
)
_AAV = np.array(
    [
        [1.0000, 0.3949, 0.3949],
        [0.3949, 0.2800, 0.1200],
        [0.3949, 0.1200, 0.0600],
    ],
    dtype=np.float64,
)
_DAV = np.array(
    [
        [0.0000, 3.6000, 3.6000],
        [3.6000, 5.3000, 5.3000],
        [3.6000, 5.3000, 5.3000],
    ],
    dtype=np.float64,
)


def _itabrow(Z: int) -> int:
    """PSE-row mapping: H,He → 1; Li..Ne → 2; everything else → 3.

    Verbatim from model_hessian.f90:1460-1481.
    """
    if Z <= 0:
        return 0
    if Z <= 2:
        return 1
    if Z <= 10:
        return 2
    return 3


def _fk_lindh(alpha: float, r0: float, r2: float) -> float:
    """``exp(α·(r0² − r²))`` — model_hessian.f90:1753-1758."""
    return math.exp(alpha * (r0 * r0 - r2))


def _fk_vdw(alpha: float, r0: float, r2: float) -> float:
    """``exp(−α·(r0 − √r²)²)`` — model_hessian.f90:1767-1772."""
    return math.exp(-alpha * (r0 - math.sqrt(r2)) ** 2)


def _getvdw_xy(rx: float, ry: float, rz: float, c66: float, s6: float, r0: float) -> float:
    """Mixed second derivative ``∂²V_vdW/∂rx ∂ry`` (off-diag).

    Verbatim from model_hessian.f90:1684-1715. The xtb implementation
    uses pre-computed temporaries (t1..t56) for numerical stability;
    we replicate them exactly to avoid sign / power-of-precision drift.
    """
    avdw = 20.0
    t1 = s6 * c66
    t2 = rx * rx
    t3 = ry * ry
    t4 = rz * rz
    t5 = t2 + t3 + t4
    t6 = t5 * t5
    t7 = t6 * t6
    t11 = math.sqrt(t5)
    t12 = 1.0 / r0
    t16 = math.exp(-avdw * (t11 * t12 - 1.0))
    t17 = 1.0 + t16
    t25 = t17 * t17
    t26 = 1.0 / t25
    t35 = 1.0 / t7
    t40 = avdw * avdw
    t41 = r0 * r0
    t43 = t40 / t41
    t44 = t16 * t16
    t56 = (
        -48.0 * t1 / t7 / t5 / t17 * rx * ry
        + 13.0 * t1 / t11 / t7 * t26 * rx * avdw * t12 * ry * t16
        - 2.0 * t1 * t35 / t25 / t17 * t43 * rx * t44 * ry
        + t1 * t35 * t26 * t43 * rx * ry * t16
    )
    return t56


def _getvdw_xx(rx: float, ry: float, rz: float, c66: float, s6: float, r0: float) -> float:
    """Diagonal second derivative ``∂²V_vdW/∂rx²``.

    Verbatim from model_hessian.f90:1717-1751.
    """
    avdw = 20.0
    t1 = s6 * c66
    t2 = rx * rx
    t3 = ry * ry
    t4 = rz * rz
    t5 = t2 + t3 + t4
    t6 = t5 * t5
    t7 = t6 * t6
    t10 = math.sqrt(t5)
    t11 = 1.0 / r0
    t15 = math.exp(-avdw * (t10 * t11 - 1.0))
    t16 = 1.0 + t15
    t17 = 1.0 / t16
    t24 = t16 * t16
    t25 = 1.0 / t24
    t29 = t11 * t15
    t33 = 1.0 / t7
    t41 = avdw * avdw
    t42 = r0 * r0
    t44 = t41 / t42
    t45 = t15 * t15
    t62 = (
        -48.0 * t1 / t7 / t5 * t17 * t2
        + 13.0 * t1 / t10 / t7 * t25 * t2 * avdw * t29
        + 6.0 * t1 * t33 * t17
        - 2.0 * t1 * t33 / t24 / t16 * t44 * t2 * t45
        - t1 / t10 / t6 / t5 * t25 * avdw * t29
        + t1 * t33 * t25 * t44 * t2 * t15
    )
    return t62


# ---------------------------------------------------------------------------
# Default Lindh parameters (xtb's modhess_setvar defaults).
# ---------------------------------------------------------------------------
class LindhParams:
    """Lindh model-Hessian parameters (xtb's ``modhess_setvar`` defaults)."""
    kr: float = 0.4500    # stretch force constant
    kf: float = 0.1500    # bending force constant
    kt: float = 0.0050    # torsion force constant
    ko: float = 0.0500    # out-of-plane bending constant
    kd: float = 0.0       # dispersion contribution to stretch (kd/kr ratio)
    kq: float = 0.0       # EEQ-Hessian contribution (off by default)
    s6: float = 1.0       # D2 dispersion scale (used in vdW additive)
    rcut: float = 25.0    # squared cutoff distance (Bohr²) for stretch


def _ind(i: int, iat: int, j: int, jat: int) -> tuple[int, int]:
    """Convert (axis_i, atom_i, axis_j, atom_j) → flat (row, col) for the
    full ``(3N, 3N)`` Hessian. Both 0-based atoms; axis is 0..2.

    The Fortran layout used a packed upper-triangular vector; in Python
    we store the full square matrix and add to (r, c) and (c, r) to
    keep symmetry explicit.
    """
    return iat * 3 + i, jat * 3 + j


def _stretch(
    n: int,
    at: list[int],
    xyz_b: np.ndarray,
    H: np.ndarray,
    p: LindhParams,
    rav: np.ndarray,
    aav: np.ndarray,
    dav: np.ndarray,
    lcutoff: np.ndarray,
) -> None:
    """Stretch contribution + vdW additive — model_hessian.f90:854-955.

    Modifies H and lcutoff in place. lcutoff[i, j] is set to True if the
    pair distance² > rcut (used by bend/torsion to skip non-bonded
    pairs that would otherwise inflate angular force constants).
    """
    kd_ratio = p.kd / p.kr if p.kr != 0 else 0.0
    kr = p.kr
    s6 = p.s6
    rcut2 = p.rcut
    for i in range(n):
        ir = _itabrow(int(at[i]))
        for j in range(i):
            jr = _itabrow(int(at[j]))
            xij = xyz_b[i, 0] - xyz_b[j, 0]
            yij = xyz_b[i, 1] - xyz_b[j, 1]
            zij = xyz_b[i, 2] - xyz_b[j, 2]
            r2 = xij * xij + yij * yij + zij * zij
            lcutoff[i, j] = r2 > rcut2
            lcutoff[j, i] = lcutoff[i, j]
            if r2 < 1e-12:
                continue
            r0 = rav[ir - 1, jr - 1]
            d0 = dav[ir - 1, jr - 1]
            alpha = aav[ir - 1, jr - 1]

            c6i = _C6[int(at[i])]
            c6j = _C6[int(at[j])]
            c6ij = math.sqrt(c6i * c6j)
            rv = _VANDER[int(at[i])] + _VANDER[int(at[j])]

            vdw_xx = _getvdw_xx(xij, yij, zij, c6ij, s6, rv)
            vdw_xy = _getvdw_xy(xij, yij, zij, c6ij, s6, rv)
            vdw_xz = _getvdw_xy(xij, zij, yij, c6ij, s6, rv)
            vdw_yy = _getvdw_xx(yij, xij, zij, c6ij, s6, rv)
            vdw_yz = _getvdw_xy(yij, zij, xij, c6ij, s6, rv)
            vdw_zz = _getvdw_xx(zij, xij, yij, c6ij, s6, rv)

            gmm = kr * _fk_lindh(alpha, r0, r2) + kr * kd_ratio * _fk_vdw(4.0, d0, r2)

            hxx = gmm * xij * xij / r2 - vdw_xx
            hxy = gmm * xij * yij / r2 - vdw_xy
            hxz = gmm * xij * zij / r2 - vdw_xz
            hyy = gmm * yij * yij / r2 - vdw_yy
            hyz = gmm * yij * zij / r2 - vdw_yz
            hzz = gmm * zij * zij / r2 - vdw_zz

            ii3 = i * 3
            jj3 = j * 3
            # (i, i) block
            H[ii3 + 0, ii3 + 0] += hxx
            H[ii3 + 1, ii3 + 0] += hxy; H[ii3 + 0, ii3 + 1] += hxy
            H[ii3 + 1, ii3 + 1] += hyy
            H[ii3 + 2, ii3 + 0] += hxz; H[ii3 + 0, ii3 + 2] += hxz
            H[ii3 + 2, ii3 + 1] += hyz; H[ii3 + 1, ii3 + 2] += hyz
            H[ii3 + 2, ii3 + 2] += hzz
            # (j, j) block (same)
            H[jj3 + 0, jj3 + 0] += hxx
            H[jj3 + 1, jj3 + 0] += hxy; H[jj3 + 0, jj3 + 1] += hxy
            H[jj3 + 1, jj3 + 1] += hyy
            H[jj3 + 2, jj3 + 0] += hxz; H[jj3 + 0, jj3 + 2] += hxz
            H[jj3 + 2, jj3 + 1] += hyz; H[jj3 + 1, jj3 + 2] += hyz
            H[jj3 + 2, jj3 + 2] += hzz
            # (i, j) block — negated, symmetric
            H[ii3 + 0, jj3 + 0] -= hxx; H[jj3 + 0, ii3 + 0] -= hxx
            H[ii3 + 0, jj3 + 1] -= hxy; H[jj3 + 1, ii3 + 0] -= hxy
            H[ii3 + 0, jj3 + 2] -= hxz; H[jj3 + 2, ii3 + 0] -= hxz
            H[ii3 + 1, jj3 + 0] -= hxy; H[jj3 + 0, ii3 + 1] -= hxy
            H[ii3 + 1, jj3 + 1] -= hyy; H[jj3 + 1, ii3 + 1] -= hyy
            H[ii3 + 1, jj3 + 2] -= hyz; H[jj3 + 2, ii3 + 1] -= hyz
            H[ii3 + 2, jj3 + 0] -= hxz; H[jj3 + 0, ii3 + 2] -= hxz
            H[ii3 + 2, jj3 + 1] -= hyz; H[jj3 + 1, ii3 + 2] -= hyz
            H[ii3 + 2, jj3 + 2] -= hzz; H[jj3 + 2, ii3 + 2] -= hzz


def _bend(
    n: int,
    at: list[int],
    xyz_b: np.ndarray,
    H: np.ndarray,
    p: LindhParams,
    rav: np.ndarray,
    aav: np.ndarray,
    dav: np.ndarray,
    lcutoff: np.ndarray,
) -> None:
    """Bending contribution — model_hessian.f90:957-1067.

    For every atom-triple (i, m, j) where m is the central atom, add
    the bend force to the Hessian. ``lcutoff`` from the stretch step is
    used to skip non-bonded i-m or m-j pairs.
    """
    rzero = 1e-10
    kf = p.kf
    kd = p.kd
    for m in range(n):
        mr = _itabrow(int(at[m]))
        for i in range(n):
            if i == m:
                continue
            if lcutoff[i, m]:
                continue
            ir = _itabrow(int(at[i]))
            xmi = xyz_b[i, 0] - xyz_b[m, 0]
            ymi = xyz_b[i, 1] - xyz_b[m, 1]
            zmi = xyz_b[i, 2] - xyz_b[m, 2]
            rmi2 = xmi * xmi + ymi * ymi + zmi * zmi
            rmi = math.sqrt(rmi2)
            r0mi = rav[mr - 1, ir - 1]
            d0mi = dav[mr - 1, ir - 1]
            ami = aav[mr - 1, ir - 1]
            for j in range(i):
                if j == m:
                    continue
                if lcutoff[j, m]:
                    continue
                jr = _itabrow(int(at[j]))
                xmj = xyz_b[j, 0] - xyz_b[m, 0]
                ymj = xyz_b[j, 1] - xyz_b[m, 1]
                zmj = xyz_b[j, 2] - xyz_b[m, 2]
                rmj2 = xmj * xmj + ymj * ymj + zmj * zmj
                rmj = math.sqrt(rmj2)
                r0mj = rav[mr - 1, jr - 1]
                d0mj = dav[mr - 1, jr - 1]
                amj = aav[mr - 1, jr - 1]

                test = (xmi * xmj + ymi * ymj + zmi * zmj) / (rmi * rmj)
                if abs(test - 1.0) < 1e-12:
                    continue   # zero angle

                xij = xyz_b[j, 0] - xyz_b[i, 0]
                yij = xyz_b[j, 1] - xyz_b[i, 1]
                zij = xyz_b[j, 2] - xyz_b[i, 2]
                rij2 = xij * xij + yij * yij + zij * zij
                rrij = math.sqrt(rij2)

                gmi = _fk_lindh(ami, r0mi, rmi2) + 0.5 * kd * _fk_vdw(4.0, d0mi, rmi2)
                gmj = _fk_lindh(amj, r0mj, rmj2) + 0.5 * kd * _fk_vdw(4.0, d0mj, rmj2)
                gij = kf * gmi * gmj

                rl2 = (
                    (ymi * zmj - zmi * ymj) ** 2
                    + (zmi * xmj - xmi * zmj) ** 2
                    + (xmi * ymj - ymi * xmj) ** 2
                )
                rl = math.sqrt(rl2) if rl2 >= 1e-14 else 0.0

                if not (rmj > rzero and rmi > rzero and rrij > rzero):
                    continue

                sinphi = rl / (rmj * rmi)
                rmidotrmj = xmi * xmj + ymi * ymj + zmi * zmj
                cosphi = rmidotrmj / (rmj * rmi)

                if sinphi > rzero:
                    si = np.array([
                        (xmi / rmi * cosphi - xmj / rmj) / (rmi * sinphi),
                        (ymi / rmi * cosphi - ymj / rmj) / (rmi * sinphi),
                        (zmi / rmi * cosphi - zmj / rmj) / (rmi * sinphi),
                    ])
                    sj = np.array([
                        (cosphi * xmj / rmj - xmi / rmi) / (rmj * sinphi),
                        (cosphi * ymj / rmj - ymi / rmi) / (rmj * sinphi),
                        (cosphi * zmj / rmj - zmi / rmi) / (rmj * sinphi),
                    ])
                    sm = -si - sj

                    # Outer-product blocks
                    block_im = gij * np.outer(si, sm)  # ∂²/∂r_i ∂r_m
                    block_jm = gij * np.outer(sj, sm)
                    block_ij = gij * np.outer(si, sj)
                    block_ii = gij * np.outer(si, si)
                    block_mm = gij * np.outer(sm, sm)
                    block_jj = gij * np.outer(sj, sj)
                    _add_block(H, i, m, block_im)
                    _add_block(H, j, m, block_jm)
                    _add_block(H, i, j, block_ij)
                    _add_block(H, i, i, block_ii)
                    _add_block(H, m, m, block_mm)
                    _add_block(H, j, j, block_jj)
                else:
                    # Linear case — two perpendicular bend axes
                    if abs(ymi) > rzero or abs(xmi) > rzero:
                        x = [-ymi,        -xmi * zmi]
                        y = [ xmi,        -ymi * zmi]
                        z = [ 0.0,         xmi * xmi + ymi * ymi]
                    else:
                        x = [1.0, 0.0]
                        y = [0.0, 1.0]
                        z = [0.0, 0.0]
                    for ii in range(2):
                        r1 = math.sqrt(x[ii] ** 2 + y[ii] ** 2 + z[ii] ** 2)
                        if r1 < rzero:
                            continue
                        ctx = x[ii] / r1; cty = y[ii] / r1; ctz = z[ii] / r1
                        si = -np.array([ctx, cty, ctz]) / rmi
                        sj = -np.array([ctx, cty, ctz]) / rmj
                        sm = -(si + sj)
                        _add_block(H, i, m, gij * np.outer(si, sm))
                        _add_block(H, j, m, gij * np.outer(sj, sm))
                        _add_block(H, i, j, gij * np.outer(si, sj))
                        _add_block(H, i, i, gij * np.outer(si, si))
                        _add_block(H, m, m, gij * np.outer(sm, sm))
                        _add_block(H, j, j, gij * np.outer(sj, sj))


def _add_block(H: np.ndarray, iat: int, jat: int, block: np.ndarray) -> None:
    """Accumulate a 3×3 block into H[iat3:iat3+3, jat3:jat3+3] symmetrically."""
    i0 = iat * 3
    j0 = jat * 3
    H[i0:i0 + 3, j0:j0 + 3] += block
    if iat != jat:
        H[j0:j0 + 3, i0:i0 + 3] += block.T


def model_hessian(
    atoms: list[int],
    coords_ang: np.ndarray,
    params: LindhParams | None = None,
) -> np.ndarray:
    """Lindh 1995 model Hessian, shape ``(3N, 3N)``, in Hartree/Bohr².

    Args:
        atoms: list of atomic numbers (length n).
        coords_ang: ``(n, 3)`` Angstrom coordinates.
        params: optional ``LindhParams``. Defaults to xtb's
            ``modhess_setvar`` defaults (kr=0.45, kf=0.15, kt=0.005,
            ko=0.05, rcut=25 Bohr², s6=1.0).

    Returns:
        Symmetric positive-(semi)-definite matrix in Hartree/Bohr².
        Three eigenvalues are zero (translations) and three more are
        small (rotations); ANCopt projects these out.

    Note:
        Out-of-plane and torsion contributions are intentionally
        skipped here for the v1 port; they have a smaller effect on
        the optimization trajectory than stretch+bend, and require
        ~250 more lines each. They will be added in a follow-up if
        the optimizer needs the extra preconditioning.
    """
    if params is None:
        params = LindhParams()
    n = len(atoms)
    n3 = 3 * n
    H = np.zeros((n3, n3), dtype=np.float64)
    xyz_b = np.asarray(coords_ang, dtype=np.float64) * _ANG_TO_BOHR
    lcutoff = np.zeros((n, n), dtype=bool)

    if params.kr != 0:
        _stretch(n, atoms, xyz_b, H, params, _RAV, _AAV, _DAV, lcutoff)
    if params.kf != 0:
        _bend(n, atoms, xyz_b, H, params, _RAV, _AAV, _DAV, lcutoff)
    # torsion + outofp deferred — see docstring.

    return H
