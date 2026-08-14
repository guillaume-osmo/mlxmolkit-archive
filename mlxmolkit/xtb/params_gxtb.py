# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Clean-room g-xTB parameter accessors from the release binary.

The data in ``params/gxtb_binary_params.npz`` is extracted from the public
g-xTB release ``libxtb.dylib`` symbol table/constant section by
``tools/gxtb_disasm_pseudocpp.py``. This module intentionally does not claim
to be a source port of save_tblite; it is a typed, testable view of the
constants we can observe from the binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from typing import Mapping

import numpy as np


SHELL_LABELS = ("s", "p", "d", "f")
SHELL_ANGULAR = (0, 1, 2, 3)
MAX_Z = 103

GXTB_REPULSION_LITERAL_BY_ADDR = {
    0x73B268: 1.5,
    0x73B270: 2.068,
    0x73B278: 2.0,
    0x73B280: 0.73,
    0x73B288: 0.0046511298,
    0x73B290: 0.011607795128002491,
    0x73B298: 0.011095539524126988,
    0x73B2A0: 0.012098131381864387,
    0x73B2A8: 0.008544252691968662,
}

GXTB_REPULSION_LITERAL_SEQUENCE = tuple(GXTB_REPULSION_LITERAL_BY_ADDR.values())

_DATA_PATH = os.path.join(os.path.dirname(__file__), "params", "gxtb_binary_params.npz")


@dataclass(frozen=True)
class GXTBShellParams:
    """Per-shell g-xTB parameters for one element."""

    index: int
    label: str
    l: int
    reference_occ: float
    h0_selfenergy: float
    h0_selfenergy_cn: float
    h0_qvszp_exp_scal: float
    fock_shell_hubbard: float
    fock_avg_exp: float
    tb2_shell_hubbard: float
    tb1_zeffsh: float
    tb1_ipea: float
    acp_level: float
    acp_exp: float


@dataclass(frozen=True)
class GXTBElementParams:
    """Element-resolved g-xTB parameter view."""

    Z: int
    n_shell: int
    n_acp: int
    shells: tuple[GXTBShellParams, ...]
    rep_zeff: float
    rep_roffset: float
    rep_q: float
    rep_k1: float
    rep_cn: float
    rep_alpha: float
    rvdw_scale: float
    increment: float
    hubbard_parameter: float
    h0_shpoly2: float
    h0_qvszp_k0_scal: float
    h0_qvszp_k2_scal: float
    h0_qvszp_k3_scal: float
    h0_dip_scale: float
    fock_cscale: float
    fock_crad: float
    cn_rcov: float
    cn_average: float
    aes_dip_scale: float
    wll_scale: float
    tb3_hubbard_derivs: float
    tb2_hubbard_cn: float
    tb1_ipea_cn: float
    l_acp: tuple[int, ...]

    @property
    def reference_occ(self) -> tuple[float, ...]:
        return tuple(shell.reference_occ for shell in self.shells)


class GXTBParameterSet:
    """Typed access to extracted g-xTB release constants."""

    def __init__(self, arrays: Mapping[str, np.ndarray], meta: tuple[dict, ...] = ()) -> None:
        self.arrays = dict(arrays)
        self.meta = meta

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "GXTBParameterSet":
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

    def atom_value(self, name: str, Z: int) -> float:
        self._validate_z(Z)
        return float(self.arrays[name][Z - 1])

    def shell_values(self, name: str, Z: int) -> np.ndarray:
        self._validate_z(Z)
        n_shell = int(self.arrays["pa_nshell"][Z - 1])
        return np.asarray(self.arrays[name][Z - 1, :n_shell], dtype=np.float64)

    def element(self, Z: int) -> GXTBElementParams:
        self._validate_z(Z)
        idx = Z - 1
        n_shell = int(self.arrays["pa_nshell"][idx])
        n_acp = int(self.arrays["pa_nacp"][idx])
        shells = tuple(self._shell(Z, ish) for ish in range(n_shell))
        return GXTBElementParams(
            Z=Z,
            n_shell=n_shell,
            n_acp=n_acp,
            shells=shells,
            rep_zeff=float(self.arrays["pa_rep_zeff"][idx]),
            rep_roffset=float(self.arrays["pa_rep_roffset"][idx]),
            rep_q=float(self.arrays["pa_rep_q"][idx]),
            rep_k1=float(self.arrays["pa_rep_k1"][idx]),
            rep_cn=float(self.arrays["pa_rep_cn"][idx]),
            rep_alpha=float(self.arrays["pa_rep_alpha"][idx]),
            rvdw_scale=float(self.arrays["pa_rvdw_scale"][idx]),
            increment=float(self.arrays["pa_increment"][idx]),
            hubbard_parameter=float(self.arrays["pa_hubbard_parameter"][idx]),
            h0_shpoly2=float(self.arrays["pa_h0_shpoly2"][idx]),
            h0_qvszp_k0_scal=float(self.arrays["pa_h0_qvszp_k0_scal"][idx]),
            h0_qvszp_k2_scal=float(self.arrays["pa_h0_qvszp_k2_scal"][idx]),
            h0_qvszp_k3_scal=float(self.arrays["pa_h0_qvszp_k3_scal"][idx]),
            h0_dip_scale=float(self.arrays["pa_h0_dip_scale"][idx]),
            fock_cscale=float(self.arrays["pa_fock_cscale"][idx]),
            fock_crad=float(self.arrays["pa_fock_crad"][idx]),
            cn_rcov=float(self.arrays["pa_cn_rcov"][idx]),
            cn_average=float(self.arrays["pa_cn_average"][idx]),
            aes_dip_scale=float(self.arrays["pa_aes_dip_scale"][idx]),
            wll_scale=float(self.arrays["pa_wll_scale"][idx]),
            tb3_hubbard_derivs=float(self.arrays["pa_tb3_hubbard_derivs"][idx]),
            tb2_hubbard_cn=float(self.arrays["pa_tb2_hubbard_cn"][idx]),
            tb1_ipea_cn=float(self.arrays["pa_tb1_ipea_cn"][idx]),
            l_acp=tuple(int(v) for v in self.arrays["pa_l_acp"][idx, :n_acp]),
        )

    def reference_population(self, Z: int) -> np.ndarray:
        """Return active shell reference occupations for element ``Z``."""

        return self.shell_values("ps_reference_occ", Z)

    def shell_labels(self, Z: int) -> tuple[str, ...]:
        self._validate_z(Z)
        n_shell = int(self.arrays["pa_nshell"][Z - 1])
        return SHELL_LABELS[:n_shell]

    def _shell(self, Z: int, ish: int) -> GXTBShellParams:
        idx = Z - 1
        return GXTBShellParams(
            index=ish,
            label=SHELL_LABELS[ish],
            l=SHELL_ANGULAR[ish],
            reference_occ=float(self.arrays["ps_reference_occ"][idx, ish]),
            h0_selfenergy=float(self.arrays["ps_h0_selfenergy"][idx, ish]),
            h0_selfenergy_cn=float(self.arrays["ps_h0_selfenergy_cn"][idx, ish]),
            h0_qvszp_exp_scal=float(self.arrays["ps_h0_qvszp_exp_scal"][idx, ish]),
            fock_shell_hubbard=float(self.arrays["ps_fock_shell_hubbard"][idx, ish]),
            fock_avg_exp=float(self.arrays["ps_fock_avg_exp"][idx, ish]),
            tb2_shell_hubbard=float(self.arrays["ps_tb2_shell_hubbard"][idx, ish]),
            tb1_zeffsh=float(self.arrays["ps_tb1_zeffsh"][idx, ish]),
            tb1_ipea=float(self.arrays["ps_tb1_ipea"][idx, ish]),
            acp_level=float(self.arrays["ps_acp_level"][idx, ish]),
            acp_exp=float(self.arrays["ps_acp_exp"][idx, ish]),
        )

    @staticmethod
    def _validate_z(Z: int) -> None:
        if not 1 <= int(Z) <= MAX_Z:
            raise ValueError(f"g-xTB parameters cover Z=1..{MAX_Z}; got {Z}")


@lru_cache(maxsize=1)
def load_gxtb_params() -> GXTBParameterSet:
    """Load the vendored binary-extracted g-xTB parameter bundle."""

    return GXTBParameterSet.load()


GXTB_PARAMS = load_gxtb_params()
