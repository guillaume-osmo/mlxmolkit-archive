"""d-orbital rotation matrices for two-center integral rotation.

Port of PYSEQM ``GenerateRotationMatrix`` (lines 5-214) and
``RotateCore`` (lines 312-355) from RotationMatrixD.py
(BSD-3-Clause, github.com/lanl/PYSEQM).

Builds the (15, 45) rotation matrix that transforms 9-orbital
lower-triangle packed integrals from local (bond) frame to molecular
frame, including all sp-d cross terms via spherical-harmonic
multiplication tables.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

INDX = [0, 1, 3, 6, 10, 15, 21, 28, 36]
PT5SQ3 = 0.8660254037841
PT5 = 0.5


def generate_rotation_matrix(xij: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build (n_pairs, 15, 45) rotation matrix for 9-orbital integrals.

    Parameters
    ----------
    xij : (n_pairs, 3) float
        Unit vector from atom i to atom j.

    Returns
    -------
    matrix : (n_pairs, 15, 45)
        Pre-combined rotation matrix elements for local-frame →
        molecular-frame transformation of two-center integrals
        packed in 9-orbital lower-triangle form.
    """
    xij = np.asarray(xij)
    if xij.ndim == 1:
        xij = xij[None, :]
    dtype = xij.dtype
    xij_neg = -xij

    n_pairs = xij_neg.shape[0]
    xy = np.linalg.norm(xij_neg[:, :2], axis=1)

    tmp = np.where(
        xij_neg[:, 2] < 0.0, -1.0, np.where(xij_neg[:, 2] > 0.0, 1.0, 0.0)
    ).astype(dtype)

    cond_xy = xy >= 1.0e-10
    CA = tmp.copy()
    CA[cond_xy] = xij_neg[cond_xy, 0] / xy[cond_xy]

    CB = np.where(cond_xy, xij_neg[:, 2], tmp)
    SA = np.zeros_like(xy)
    SA[cond_xy] = xij_neg[cond_xy, 1] / xy[cond_xy]
    SB = np.where(cond_xy, xy, 0.0)

    C2A = 2.0 * CA * CA - 1.0
    C2B = 2.0 * CB * CB - 1.0
    S2A = 2.0 * SA * CA
    S2B = 2.0 * SB * CB

    P = np.zeros((n_pairs, 3, 3), dtype=dtype)
    P[:, 0, 0] = CA * SB
    P[:, 1, 0] = CA * CB
    P[:, 2, 0] = -SA
    P[:, 0, 1] = SA * SB
    P[:, 1, 1] = SA * CB
    P[:, 2, 1] = CA
    P[:, 0, 2] = CB
    P[:, 1, 2] = -SB

    D = np.zeros((n_pairs, 5, 5), dtype=dtype)
    D[:, 0, 0] = PT5SQ3 * C2A * SB * SB
    D[:, 1, 0] = PT5 * C2A * S2B
    D[:, 2, 0] = -S2A * SB
    D[:, 3, 0] = C2A * (CB * CB + PT5 * SB * SB)
    D[:, 4, 0] = -S2A * CB
    D[:, 0, 1] = PT5SQ3 * CA * S2B
    D[:, 1, 1] = CA * C2B
    D[:, 2, 1] = -SA * CB
    D[:, 3, 1] = -PT5 * CA * S2B
    D[:, 4, 1] = SA * SB
    D[:, 0, 2] = CB * CB - PT5 * SB * SB
    D[:, 1, 2] = -PT5SQ3 * S2B
    D[:, 3, 2] = PT5SQ3 * SB * SB
    D[:, 0, 3] = PT5SQ3 * SA * S2B
    D[:, 1, 3] = SA * C2B
    D[:, 2, 3] = CA * CB
    D[:, 3, 3] = -PT5 * SA * S2B
    D[:, 4, 3] = -CA * SB
    D[:, 0, 4] = PT5SQ3 * S2A * SB * SB
    D[:, 1, 4] = PT5 * S2A * S2B
    D[:, 2, 4] = C2A * SB
    D[:, 3, 4] = S2A * (CB * CB + PT5 * SB * SB)
    D[:, 4, 4] = C2A * CB

    matrix = np.zeros((n_pairs, 15, 45), dtype=dtype)

    # S-S
    matrix[:, 0, 0] = 1.0

    # P-S
    for K in range(3):
        KL = INDX[K + 1] + 1 - 1
        matrix[:, 0, KL] = P[:, K, 0]
        matrix[:, 1, KL] = P[:, K, 1]
        matrix[:, 2, KL] = P[:, K, 2]

    # P-P diagonal
    for K in range(3):
        KL = INDX[K + 1] + K + 1
        matrix[:, 0, KL] = P[:, K, 0] * P[:, K, 0]
        matrix[:, 1, KL] = P[:, K, 0] * P[:, K, 1]
        matrix[:, 2, KL] = P[:, K, 1] * P[:, K, 1]
        matrix[:, 3, KL] = P[:, K, 0] * P[:, K, 2]
        matrix[:, 4, KL] = P[:, K, 1] * P[:, K, 2]
        matrix[:, 5, KL] = P[:, K, 2] * P[:, K, 2]

    # P-P off-diagonal
    for K in range(1, 3):
        for L in range(K):
            KL = INDX[K + 1] + L + 1
            matrix[:, 0, KL] = P[:, K, 0] * P[:, L, 0] * 2.0
            matrix[:, 1, KL] = P[:, K, 0] * P[:, L, 1] + P[:, K, 1] * P[:, L, 0]
            matrix[:, 2, KL] = P[:, K, 1] * P[:, L, 1] * 2.0
            matrix[:, 3, KL] = P[:, K, 0] * P[:, L, 2] + P[:, K, 2] * P[:, L, 0]
            matrix[:, 4, KL] = P[:, K, 1] * P[:, L, 2] + P[:, K, 2] * P[:, L, 1]
            matrix[:, 5, KL] = P[:, K, 2] * P[:, L, 2] * 2.0

    # D-S
    for K in range(5):
        KL = INDX[K + 3 + 1]
        matrix[:, 0, KL] = D[:, K, 0]
        matrix[:, 1, KL] = D[:, K, 1]
        matrix[:, 2, KL] = D[:, K, 2]
        matrix[:, 3, KL] = D[:, K, 3]
        matrix[:, 4, KL] = D[:, K, 4]

    # D-P
    for K in range(5):
        for L in range(3):
            KL = INDX[K + 4] + L + 1
            matrix[:, 0, KL] = D[:, K, 0] * P[:, L, 0]
            matrix[:, 1, KL] = D[:, K, 0] * P[:, L, 1]
            matrix[:, 2, KL] = D[:, K, 0] * P[:, L, 2]
            matrix[:, 3, KL] = D[:, K, 1] * P[:, L, 0]
            matrix[:, 4, KL] = D[:, K, 1] * P[:, L, 1]
            matrix[:, 5, KL] = D[:, K, 1] * P[:, L, 2]
            matrix[:, 6, KL] = D[:, K, 2] * P[:, L, 0]
            matrix[:, 7, KL] = D[:, K, 2] * P[:, L, 1]
            matrix[:, 8, KL] = D[:, K, 2] * P[:, L, 2]
            matrix[:, 9, KL] = D[:, K, 3] * P[:, L, 0]
            matrix[:, 10, KL] = D[:, K, 3] * P[:, L, 1]
            matrix[:, 11, KL] = D[:, K, 3] * P[:, L, 2]
            matrix[:, 12, KL] = D[:, K, 4] * P[:, L, 0]
            matrix[:, 13, KL] = D[:, K, 4] * P[:, L, 1]
            matrix[:, 14, KL] = D[:, K, 4] * P[:, L, 2]

    # D-D diagonal
    for K in range(5):
        KL = INDX[K + 4] + K + 4
        matrix[:, 0, KL] = D[:, K, 0] * D[:, K, 0]
        matrix[:, 1, KL] = D[:, K, 0] * D[:, K, 1]
        matrix[:, 2, KL] = D[:, K, 1] * D[:, K, 1]
        matrix[:, 3, KL] = D[:, K, 0] * D[:, K, 2]
        matrix[:, 4, KL] = D[:, K, 1] * D[:, K, 2]
        matrix[:, 5, KL] = D[:, K, 2] * D[:, K, 2]
        matrix[:, 6, KL] = D[:, K, 0] * D[:, K, 3]
        matrix[:, 7, KL] = D[:, K, 1] * D[:, K, 3]
        matrix[:, 8, KL] = D[:, K, 2] * D[:, K, 3]
        matrix[:, 9, KL] = D[:, K, 3] * D[:, K, 3]
        matrix[:, 10, KL] = D[:, K, 0] * D[:, K, 4]
        matrix[:, 11, KL] = D[:, K, 1] * D[:, K, 4]
        matrix[:, 12, KL] = D[:, K, 2] * D[:, K, 4]
        matrix[:, 13, KL] = D[:, K, 3] * D[:, K, 4]
        matrix[:, 14, KL] = D[:, K, 4] * D[:, K, 4]

    # D-D off-diagonal
    for K in range(5):
        for L in range(K):
            KL = INDX[K + 4] + L + 4
            matrix[:, 0, KL] = D[:, K, 0] * D[:, L, 0] * 2.0
            matrix[:, 1, KL] = D[:, K, 0] * D[:, L, 1] + D[:, K, 1] * D[:, L, 0]
            matrix[:, 2, KL] = D[:, K, 1] * D[:, L, 1] * 2.0
            matrix[:, 3, KL] = D[:, K, 0] * D[:, L, 2] + D[:, K, 2] * D[:, L, 0]
            matrix[:, 4, KL] = D[:, K, 1] * D[:, L, 2] + D[:, K, 2] * D[:, L, 1]
            matrix[:, 5, KL] = D[:, K, 2] * D[:, L, 2] * 2.0
            matrix[:, 6, KL] = D[:, K, 0] * D[:, L, 3] + D[:, K, 3] * D[:, L, 0]
            matrix[:, 7, KL] = D[:, K, 1] * D[:, L, 3] + D[:, K, 3] * D[:, L, 1]
            matrix[:, 8, KL] = D[:, K, 2] * D[:, L, 3] + D[:, K, 3] * D[:, L, 2]
            matrix[:, 9, KL] = D[:, K, 3] * D[:, L, 3] * 2.0
            matrix[:, 10, KL] = D[:, K, 0] * D[:, L, 4] + D[:, K, 4] * D[:, L, 0]
            matrix[:, 11, KL] = D[:, K, 1] * D[:, L, 4] + D[:, K, 4] * D[:, L, 1]
            matrix[:, 12, KL] = D[:, K, 2] * D[:, L, 4] + D[:, K, 4] * D[:, L, 2]
            matrix[:, 13, KL] = D[:, K, 3] * D[:, L, 4] + D[:, K, 4] * D[:, L, 3]
            matrix[:, 14, KL] = D[:, K, 4] * D[:, L, 4] * 2.0

    return matrix


def rotate_core(core: NDArray[np.float64], matrix: NDArray[np.float64], index: int) -> NDArray[np.float64]:
    """Rotate the core-electron attraction vector from local to molecular frame.

    Parameters
    ----------
    core : (n_pairs, 45) float
        Local-frame coreYHLocal[..., 1:] (the 45 slots after the s-self term).
    matrix : (n_pairs, 15, 45) float
        Output of :func:`generate_rotation_matrix`.
    index : int
        1 = sp-only, 2 = include P-S/P-P rotation, 3 = include d-orbitals.
        For YH (PM6 with d-orbital atom paired with H), use 3.

    Returns
    -------
    rot_core : (n_pairs, 45) float
        Molecular-frame core matrix elements.
    """
    rot_core = np.zeros_like(core)
    if core.shape[0] == 0:
        return rot_core
    rot_core[:, 0] = core[:, 0]
    pp = [2, 4, 5, 7, 8, 9]
    if index > 1:
        # PS
        rot_core[:, 1] = core[:, 6] * matrix[:, 0, 1]
        rot_core[:, 3] = core[:, 6] * matrix[:, 1, 1]
        rot_core[:, 6] = core[:, 6] * matrix[:, 2, 1]
        # PP
        for I in range(6):
            rot_core[:, pp[I]] = (
                core[:, 9] * matrix[:, I, 2]
                + core[:, 2] * (matrix[:, I, 5] + matrix[:, I, 9])
            )
    if index > 2:
        # DD and DP
        dp = [11, 12, 13, 16, 17, 18, 22, 23, 24, 29, 30, 31, 37, 38, 39]
        dd = [14, 19, 20, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 44]
        for I in range(15):
            rot_core[:, dp[I]] = (
                core[:, 24] * matrix[:, I, 11]
                + core[:, 16] * (matrix[:, I, 17] + matrix[:, I, 24])
            )
            rot_core[:, dd[I]] = (
                core[:, 27] * matrix[:, I, 14]
                + core[:, 20] * (matrix[:, I, 20] + matrix[:, I, 27])
                + core[:, 14] * (matrix[:, I, 35] + matrix[:, I, 44])
            )
        # DS
        ds = [10, 15, 21, 28, 36]
        for I in range(5):
            rot_core[:, ds[I]] = core[:, 21] * matrix[:, I, 10]
    return rot_core
