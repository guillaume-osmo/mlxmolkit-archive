"""Quaternion-based bond-axis rotation for TETCI two-center integrals.

Port of PYSEQM ``rotate_with_quaternion`` (BSD-3-Clause,
github.com/lanl/PYSEQM, seqm/seqm_functions/two_elec_two_center_int.py
lines 1576-1703).

Given a unit vector ``v`` (bond axis), returns the 3×3 rotation matrix
that aligns ``v`` to the x-axis (1, 0, 0). Used as a building block by
the TETCI ``rotate()`` function that computes two-electron two-center
integrals in the local (bond) frame.

This is the first leaf-function port in the native MLX TETCI roadmap.
It has no internal PYSEQM dependencies — pure quaternion math.

NumPy implementation (CPU). MLX (Apple GPU) version planned once the
SCF loop is also ported to mlx.core arrays end-to-end; for now a NumPy
reference unblocks the porting of the larger ``rotate()`` function.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotate_with_quaternion(
    v: NDArray[np.float64],
    calculate_gradient: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotation matrices aligning a batch of unit vectors to the x-axis.

    Parameters
    ----------
    v : (n, 3) array
        Unit vectors (bond axes) — must be normalised; each row is one
        atom-pair direction.
    calculate_gradient : bool
        If True, additionally return the (n, 3, 3, 3) Jacobian
        ``dR/dv`` needed for analytic gradients.

    Returns
    -------
    rot : (n, 3, 3) array
        Rotation matrices such that ``rot @ v.T`` gives ``[1,0,0]``
        for each row.
    dRdv : (n, 3, 3, 3) array, only if ``calculate_gradient=True``
        ``dRdv[i, k, j, l] = ∂R_{jl}/∂v_k`` for vector ``i``.

    Notes
    -----
    Antipodal handling: when ``v`` ≈ ``-x_hat`` (i.e. ``1 + v_x ≈ 0``),
    the standard quaternion construction is degenerate. PYSEQM picks a
    180° flip about the z-axis in this case (quaternion ``(0,0,1,0)``).
    We use the same convention. Epsilon: 1e-7 for float64, 5e-4 for
    float32 (matches PYSEQM).
    """
    v = np.asarray(v)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"v must have shape (n, 3); got {v.shape}")
    n = v.shape[0]
    dtype = v.dtype

    # Quaternion components:
    #   u = v × x_hat  =  [0, v_z, -v_y]
    #   w = 1 + v_x   (scalar part)
    u = np.zeros_like(v)
    u[:, 1] = v[:, 2]
    u[:, 2] = -v[:, 1]
    w_ = 1.0 + v[:, 0]

    q_raw = np.concatenate((u, w_[:, None]), axis=-1)  # (n, 4) = (u_x, u_y, u_z, w)

    eps = 1.0e-7 if dtype == np.float64 else 5.0e-4
    mask = np.abs(w_) < eps  # (n,)
    # Antipodal fallback: 180° flip about z-axis → q = (0, 0, 1, 0)
    fallback = np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype)
    q_raw[mask] = fallback

    # Normalise
    N = np.linalg.norm(q_raw, axis=-1, keepdims=True)
    q = q_raw / N
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Build the 3×3 rotation matrix; qx is always zero by construction so
    # we omit terms involving qx (matches PYSEQM line 1616-1626).
    rot = np.empty((n, 3, 3), dtype=dtype)
    rot[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    rot[:, 0, 1] = -2 * (qz * qw)
    rot[:, 0, 2] = 2 * (qy * qw)
    rot[:, 1, 0] = 2 * (qz * qw)
    rot[:, 1, 1] = 1 - 2 * (qz * qz)
    rot[:, 1, 2] = 2 * (qy * qz)
    rot[:, 2, 0] = -2 * (qy * qw)
    rot[:, 2, 1] = 2 * (qy * qz)
    rot[:, 2, 2] = 1 - 2 * (qy * qy)

    if not calculate_gradient:
        return rot

    # ∂q_raw/∂v: constant entries (zeroed on the antipodal mask).
    #   q_raw = (0, v_z, -v_y, 1+v_x)
    #   ∂(u_y)/∂v_z = +1, ∂(u_z)/∂v_y = -1, ∂(w)/∂v_x = +1
    dq_raw_dv = np.zeros((n, 4, 3), dtype=dtype)
    dq_raw_dv[:, 1, 2] = 1.0
    dq_raw_dv[:, 2, 1] = -1.0
    dq_raw_dv[:, 3, 0] = 1.0
    # Zero on antipodal rows
    dq_raw_dv = dq_raw_dv * (~mask)[:, None, None]

    # dN/dv_j = (1/N) Σ_i q_raw_i · ∂q_raw_i/∂v_j  → shape (n, 3)
    dN_dv = (q_raw[:, :, None] * dq_raw_dv).sum(axis=1) / N

    # Quotient rule: ∂(q_raw_i / N)/∂v_j  → shape (n, 4, 3)
    dq_dv = (
        dq_raw_dv * N[:, :, None]
        - q_raw[:, :, None] * dN_dv[:, None, :]
    ) / (N[:, :, None] ** 2)

    # dr_dq[i, j, k, l] = ∂R_{jk}/∂q_l for vector i, layout matches PYSEQM
    dr_dq = np.zeros((n, 3, 3, 4), dtype=dtype)
    dr_dq[:, 0, 0, 1] = -4 * qy
    dr_dq[:, 0, 0, 2] = -4 * qz
    dr_dq[:, 0, 1, 0] = 2 * qy
    dr_dq[:, 0, 1, 2] = -2 * qw
    dr_dq[:, 0, 1, 3] = -2 * qz
    dr_dq[:, 0, 2, 0] = 2 * qz
    dr_dq[:, 0, 2, 1] = 2 * qw
    dr_dq[:, 0, 2, 3] = 2 * qy
    dr_dq[:, 1, 0, 0] = 2 * qy
    dr_dq[:, 1, 0, 2] = 2 * qw
    dr_dq[:, 1, 0, 3] = 2 * qz
    dr_dq[:, 1, 1, 2] = -4 * qz
    dr_dq[:, 1, 2, 0] = -2 * qw
    dr_dq[:, 1, 2, 1] = 2 * qz
    dr_dq[:, 1, 2, 2] = 2 * qy
    dr_dq[:, 2, 0, 0] = 2 * qz
    dr_dq[:, 2, 0, 1] = -2 * qw
    dr_dq[:, 2, 0, 3] = -2 * qy
    dr_dq[:, 2, 1, 0] = 2 * qw
    dr_dq[:, 2, 1, 1] = 2 * qz
    dr_dq[:, 2, 1, 2] = 2 * qy
    dr_dq[:, 2, 2, 1] = -4 * qy

    # Chain rule: dRdv[i, k, j, l] = Σ_d dr_dq[i, j, l, d] · dq_dv[i, d, k]
    # Note: PYSEQM uses einsum 'nijd,ndk->nkij'
    dRdv = np.einsum("nijd,ndk->nkij", dr_dq, dq_dv)

    return rot, dRdv
