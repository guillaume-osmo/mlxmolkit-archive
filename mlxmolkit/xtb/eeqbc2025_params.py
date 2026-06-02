# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""EEQ_BC 2025 element parameters recovered from the public g-xTB binary."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Mapping

import numpy as np


MAX_Z = 103
PARAMETER_NAMES = (
    "rvdw_scale",
    "rad",
    "kqchi",
    "kcnchi",
    "eta",
    "cov_radii",
    "chi",
    "cap",
    "avg_cn",
)
_DATA_PATH = os.path.join(os.path.dirname(__file__), "params", "eeqbc2025_params.npz")


@dataclass(frozen=True)
class EEQBC2025ElementParams:
    """Element-resolved EEQ_BC 2025 parameters."""

    Z: int
    rvdw_scale: float
    rad: float
    kqchi: float
    kcnchi: float
    eta: float
    cov_radii: float
    chi: float
    cap: float
    avg_cn: float


class EEQBC2025ParameterSet:
    """Typed access to binary-extracted EEQ_BC 2025 tables."""

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = dict(arrays)
        missing = sorted(set(PARAMETER_NAMES) - set(self.arrays))
        if missing:
            raise ValueError(f"missing EEQ_BC parameter arrays: {missing}")
        for name in PARAMETER_NAMES:
            arr = np.asarray(self.arrays[name], dtype=np.float64)
            if arr.shape != (MAX_Z,):
                raise ValueError(f"expected {name} shape {(MAX_Z,)}, got {arr.shape}")
            self.arrays[name] = arr

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "EEQBC2025ParameterSet":
        if path is None:
            path = _DATA_PATH
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in PARAMETER_NAMES}
        return cls(arrays)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def atom_value(self, name: str, Z: int) -> float:
        self._validate_z(Z)
        if name not in PARAMETER_NAMES:
            raise KeyError(name)
        return float(self.arrays[name][int(Z) - 1])

    def element(self, Z: int) -> EEQBC2025ElementParams:
        self._validate_z(Z)
        idx = int(Z) - 1
        return EEQBC2025ElementParams(
            Z=int(Z),
            rvdw_scale=float(self.arrays["rvdw_scale"][idx]),
            rad=float(self.arrays["rad"][idx]),
            kqchi=float(self.arrays["kqchi"][idx]),
            kcnchi=float(self.arrays["kcnchi"][idx]),
            eta=float(self.arrays["eta"][idx]),
            cov_radii=float(self.arrays["cov_radii"][idx]),
            chi=float(self.arrays["chi"][idx]),
            cap=float(self.arrays["cap"][idx]),
            avg_cn=float(self.arrays["avg_cn"][idx]),
        )

    @staticmethod
    def _validate_z(Z: int) -> None:
        if not 1 <= int(Z) <= MAX_Z:
            raise ValueError(f"EEQ_BC 2025 parameters cover Z=1..{MAX_Z}; got {Z}")


@lru_cache(maxsize=1)
def load_eeqbc2025_params() -> EEQBC2025ParameterSet:
    """Load the vendored binary-extracted EEQ_BC 2025 parameter bundle."""

    return EEQBC2025ParameterSet.load()


EEQBC2025_PARAMS = load_eeqbc2025_params()
