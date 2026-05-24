"""MLX port of rotate_with_quaternion — runs on Apple GPU.

Drop-in replacement for :func:`tetci_quaternion.rotate_with_quaternion`
using ``mlx.core`` arrays. The math is identical; only the tensor
library changes.

Proof-of-concept for the full MLX migration: this is the smallest
self-contained piece and validates that the NumPy → MLX translation
is mechanical.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray


def rotate_with_quaternion_mlx(
    v: NDArray[np.float64] | "mx.array",
    calculate_gradient: bool = False,
):
    """MLX version of rotate_with_quaternion. Runs on default device (Metal GPU).

    Parameters
    ----------
    v : array-like of shape (n, 3)
        Unit vectors. Either NumPy ndarray or mlx.core.array.
    calculate_gradient : bool
        Returns ``(rot, dRdv)`` if True.

    Returns
    -------
    rot : mx.array shape (n, 3, 3)
        Rotation matrices on the Metal GPU.
    """
    if not isinstance(v, mx.array):
        v = mx.array(np.asarray(v))
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"v must have shape (n, 3); got {v.shape}")
    n = v.shape[0]
    dtype = v.dtype

    # u = v × x_hat = [0, v_z, -v_y]
    u = mx.zeros_like(v)
    u[:, 1] = v[:, 2]
    u[:, 2] = -v[:, 1]
    w_ = 1.0 + v[:, 0]

    q_raw = mx.concatenate([u, w_[:, None]], axis=-1)  # (n, 4)

    eps = 1.0e-7 if dtype == mx.float64 else 5.0e-4
    mask = mx.abs(w_) < eps  # (n,)

    # Antipodal fallback: q = (0, 0, 1, 0)
    fallback = mx.array([0.0, 0.0, 1.0, 0.0], dtype=dtype)
    # MLX boolean indexing assign
    if mask.any().item():
        idx = mx.array(np.where(np.asarray(mask))[0])
        q_raw[idx] = fallback

    N = mx.linalg.norm(q_raw, axis=-1, keepdims=True)
    q = q_raw / N
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 3×3 rotation matrix
    rot = mx.zeros((n, 3, 3), dtype=dtype)
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

    # Gradient computation: same structure as numpy version
    dq_raw_dv = mx.zeros((n, 4, 3), dtype=dtype)
    dq_raw_dv[:, 1, 2] = 1.0
    dq_raw_dv[:, 2, 1] = -1.0
    dq_raw_dv[:, 3, 0] = 1.0
    dq_raw_dv = dq_raw_dv * (~mask)[:, None, None]

    dN_dv = (q_raw[:, :, None] * dq_raw_dv).sum(axis=1) / N
    dq_dv = (
        dq_raw_dv * N[:, :, None]
        - q_raw[:, :, None] * dN_dv[:, None, :]
    ) / (N[:, :, None] ** 2)

    dr_dq = mx.zeros((n, 3, 3, 4), dtype=dtype)
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

    dRdv = mx.einsum("nijd,ndk->nkij", dr_dq, dq_dv)
    return rot, dRdv
