# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Experimental MLX-accelerated GFN2 SCF wrappers.

The stable implementation remains :mod:`mlxmolkit.xtb.scf_gfn2`. This
module swaps selected hotspots to MLX kernels in a scoped way so we can
benchmark speedups without changing the reference path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from . import scf_gfn2 as _ref
from .aes_mlx import mmompop_mlx


@dataclass
class _MmompopMlxAdapter:
    """NumPy-compatible wrapper around :func:`mmompop_mlx`.

    ``S``, ``dpint``, ``qpint``, ``aoat`` and ``coords_bohr`` are fixed
    during one SCF call, so they are cached as MLX arrays by object id.
    ``P`` changes every iteration and is uploaded each call.
    """

    dtype: mx.Dtype = mx.float32
    _cache: dict[tuple[int, int, int, int, int], tuple[mx.array, ...]] = field(
        default_factory=dict
    )

    def __call__(
        self,
        P: np.ndarray,
        S: np.ndarray,
        dpint: np.ndarray,
        qpint: np.ndarray,
        aoat: np.ndarray,
        coords_bohr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (id(S), id(dpint), id(qpint), id(aoat), id(coords_bohr))
        cached = self._cache.get(key)
        if cached is None:
            cached = (
                mx.array(S, dtype=self.dtype),
                mx.array(dpint, dtype=self.dtype),
                mx.array(qpint, dtype=self.dtype),
                mx.array(aoat, dtype=mx.int32),
                mx.array(coords_bohr, dtype=self.dtype),
            )
            self._cache[key] = cached
        S_mx, dpint_mx, qpint_mx, aoat_mx, coords_mx = cached
        P_mx = mx.array(P, dtype=self.dtype)
        dipm, qp = mmompop_mlx(P_mx, S_mx, dpint_mx, qpint_mx, aoat_mx, coords_mx)
        mx.eval(dipm, qp)
        return np.asarray(dipm), np.asarray(qp)


def gfn2_energy_mlx(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    allow_float32: bool = False,
    **kwargs: Any,
) -> dict:
    """GFN2 energy with experimental MLX AES population kernel.

    This is a drop-in sidecar for :func:`scf_gfn2.gfn2_energy`. It
    currently accelerates only ``mmompop``; all other behavior remains
    delegated to the reference implementation.

    MLX does not support ``float64`` arrays on the GPU. The reference
    GFN2 path is double precision, so this experimental GPU sidecar must
    be opted into explicitly with ``allow_float32=True``.
    """

    if not allow_float32:
        raise ValueError(
            "gfn2_energy_mlx uses MLX GPU float32 kernels. MLX float64 is "
            "CPU-only, so pass allow_float32=True only for exploratory "
            "speed/precision experiments. Use gfn2_energy_fast for exact "
            "float64 CPU acceleration."
        )

    old_mmompop = _ref.mmompop
    _ref.mmompop = _MmompopMlxAdapter(dtype=mx.float32)
    try:
        return _ref.gfn2_energy(atoms, coords_ang, **kwargs)
    finally:
        _ref.mmompop = old_mmompop
