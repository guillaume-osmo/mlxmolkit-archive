# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Typed access to the binary-extracted q-vSZP basis tables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from typing import Mapping

import numpy as np


MAX_Z = 103
MAX_SHELLS = 4
MAX_PRIM = 12
SHELL_LABELS = ("s", "p", "d", "f")

_DATA_PATH = os.path.join(os.path.dirname(__file__), "params", "qvszp_binary_params.npz")


@dataclass(frozen=True)
class QVSZPShell:
    index: int
    label: str
    l: int
    n: int
    n_prim: int
    exponents: np.ndarray
    coefficients: np.ndarray
    coefficients_env: np.ndarray


@dataclass(frozen=True)
class QVSZPElement:
    Z: int
    cov_radius: float
    n_shell: int
    k0: float
    k1: float
    k2: float
    k3: float
    shells: tuple[QVSZPShell, ...]


class QVSZPParameterSet:
    """Binary q-vSZP basis constants in element/shell/primitive layout."""

    def __init__(self, arrays: Mapping[str, np.ndarray], meta: tuple[dict, ...] = ()) -> None:
        self.arrays = dict(arrays)
        self.meta = meta

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "QVSZPParameterSet":
        if path is None:
            path = _DATA_PATH
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files if name != "__meta_json__"}
            meta: tuple[dict, ...] = ()
            if "__meta_json__" in data.files:
                meta = tuple(json.loads(str(data["__meta_json__"])))
        return cls(arrays, meta)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def element(self, Z: int) -> QVSZPElement:
        self._validate_z(Z)
        idx = int(Z) - 1
        n_shell = int(self.arrays["nshell"][idx])
        shells = tuple(self.shell(Z, ish) for ish in range(n_shell))
        return QVSZPElement(
            Z=int(Z),
            cov_radius=float(self.arrays["cov_radii"][idx]),
            n_shell=n_shell,
            k0=float(self.arrays["p_k0"][idx]),
            k1=float(self.arrays["p_k1"][idx]),
            k2=float(self.arrays["p_k2"][idx]),
            k3=float(self.arrays["p_k3"][idx]),
            shells=shells,
        )

    def shell(self, Z: int, ish: int) -> QVSZPShell:
        self._validate_z(Z)
        if not 0 <= int(ish) < MAX_SHELLS:
            raise ValueError(f"q-vSZP shell index must be 0..{MAX_SHELLS - 1}; got {ish}")
        idx = int(Z) - 1
        ish = int(ish)
        n_prim = int(self.arrays["n_prim"][idx, ish])
        l = int(self.arrays["ang_shell"][idx, ish])
        if n_prim < 0 or n_prim > MAX_PRIM:
            raise ValueError(f"invalid q-vSZP primitive count for Z={Z}, shell={ish}: {n_prim}")
        return QVSZPShell(
            index=ish,
            label=SHELL_LABELS[l] if 0 <= l < len(SHELL_LABELS) else f"l={l}",
            l=l,
            n=int(self.arrays["principal_quantum_number"][idx, ish]),
            n_prim=n_prim,
            exponents=np.asarray(self.arrays["exponents"][idx, ish, :n_prim], dtype=np.float64),
            coefficients=np.asarray(self.arrays["coefficients"][idx, ish, :n_prim], dtype=np.float64),
            coefficients_env=np.asarray(
                self.arrays["coefficients_env"][idx, ish, :n_prim],
                dtype=np.float64,
            ),
        )

    @staticmethod
    def _validate_z(Z: int) -> None:
        if not 1 <= int(Z) <= MAX_Z:
            raise ValueError(f"q-vSZP parameters cover Z=1..{MAX_Z}; got {Z}")


@lru_cache(maxsize=1)
def load_qvszp_params() -> QVSZPParameterSet:
    return QVSZPParameterSet.load()


QVSZP_PARAMS = load_qvszp_params()
