# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT
#
# Loads GFN0-xTB element parameters from the vendored
# `param_gfn0-xtb.txt` (LGPL-3.0, grimme-lab/xtb), patches in EN
# defaults from the hardcoded electronegativity array in
# `xtb/src/xtb/gfn0.f90:251-269`, and exposes them as typed
# dataclasses keyed by atomic number.

"""Typed GFN0-xTB element parameters.

Module-level constants:

- :data:`GFN0_GLOBALS` — :class:`mlxmolkit.xtb.params_parser.GFN0Globals`
  with the 20 global floats parsed from the file's ``$globpar`` block.
- :data:`GFN0_PARAMS` — ``dict[int, GFN0ElementParams]`` keyed by atomic
  number (Z = 1..86, full coverage).

Anything not in the file (notably ``EN`` for many heavy elements) is
filled from xtb's source-code defaults at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .params_parser import GFN0Globals, GFN0RawElement, parse_gfn0_param_file


# ---------------------------------------------------------------------------
# Vendored fallback tables from grimme-lab/xtb/src/xtb/gfn0.f90:251-269
# (LGPL-3.0). Pauling-like electronegativity, Z = 1..86. Used to fill
# the EN field when the per-element block in the .txt file omits it.
# ---------------------------------------------------------------------------
_GFN0_EN_DEFAULTS: tuple[float, ...] = (
    1.92, 3.00, 0.98, 1.57, 2.04,
    2.48, 2.97, 3.44, 3.50, 3.50,
    0.93, 1.31, 1.61, 1.90, 2.19,
    2.58, 3.16, 3.50, 1.45, 1.80,
    1.73, 1.54, 1.63, 1.66, 1.55,
    1.83, 1.88, 1.91, 1.90, 1.65,
    1.81, 2.01, 2.18, 2.55, 2.96,
    3.00, 1.50, 1.50, 1.55, 1.33,
    1.60, 2.16, 1.90, 2.20, 2.28,
    2.20, 1.93, 1.69, 1.78, 1.96,
    2.05, 2.10, 2.66, 2.60, 1.50,
    1.60, 1.50, 1.50, 1.50, 1.50,
    1.50, 1.50, 1.50, 1.50, 1.50,
    1.50, 1.50, 1.50, 1.50, 1.50,
    1.50, 1.30, 1.50, 2.36, 1.90,
    2.20, 2.20, 2.28, 2.54, 2.00,
    1.62, 2.33, 2.02, 2.00, 2.20,
    2.20,
)
assert len(_GFN0_EN_DEFAULTS) == 86, "EN defaults table must cover Z=1..86"


# ---------------------------------------------------------------------------
# Element symbols (Z=1..86), used for human-readable error messages.
# ---------------------------------------------------------------------------
_ELEMENT_SYMBOLS: tuple[str, ...] = (
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
)
assert len(_ELEMENT_SYMBOLS) == 86


@dataclass(frozen=True)
class GFN0Shell:
    """One AO shell of a GFN0 element.

    Attributes:
        l: 0=s, 1=p, 2=d, 3=f
        n: principal quantum number
        h: shell self-energy (eV) — ``H_μμ`` baseline
        zeta: Slater orbital exponent (Bohr⁻¹)
        k_poly: shell-poly coefficient for ``Π(R)`` distance scaling
        k_cn: CN-shift coefficient for ``H_μμ`` diagonal correction
        k_q1: linear q-shift coefficient for ``H_μμ``
    """
    l: int
    n: int
    h: float
    zeta: float
    k_poly: float
    k_cn: float
    k_q1: float


@dataclass(frozen=True)
class GFN0ElementParams:
    """All GFN0-xTB per-element parameters.

    Attributes:
        Z: atomic number
        symbol: element symbol (e.g. ``"H"``, ``"C"``)
        shells: tuple of :class:`GFN0Shell`, ordered as in the param file
            (this is also the basis-function ordering)
        en: Pauling-like electronegativity for the EN-modulated K_AB
            in ``H_μν`` (NOT the EEQ χ — that's :attr:`eeq_chi`)
        k_q2: quadratic q-shift coefficient for ``H_μμ`` (atom-resolved)
        rep_alpha: repulsion exponent ``α_A`` in the classical pairwise
            term ``E_rep``
        rep_zeff: effective nuclear charge ``Z_A^eff`` in ``E_rep``
        eeq_chi: EEQ electronegativity ``χ_A`` (different from
            :attr:`en` above; this enters the EEQ Lagrangian)
        eeq_eta: EEQ chemical hardness ``η_A``
        eeq_kappa: EEQ CN-coupling ``κ_A``
        eeq_alpha: EEQ Gaussian charge width ``α_A``

    Notes:
        - ``atomic_radius`` (covalent radius for the shell-poly ``Π(R)``)
          is NOT a per-element field here; it lives in a separate
          ``xtb_param_atomicrad`` table and is loaded alongside.
        - ``e_atom`` (atomic reference energy for atomization energy) is
          also a separate table; deferred until the orchestrator phase.
    """
    Z: int
    symbol: str
    shells: tuple[GFN0Shell, ...]
    en: float
    k_q2: float
    rep_alpha: float
    rep_zeff: float
    eeq_chi: float
    eeq_eta: float
    eeq_kappa: float
    eeq_alpha: float


def _shells_from_raw(raw: GFN0RawElement) -> tuple[GFN0Shell, ...]:
    """Zip the per-shell arrays (lev, exp, POLY*, KCN*, KQ*) into shells."""
    polys = [raw.POLYS, raw.POLYP, raw.POLYD]
    kcns = [raw.KCNS, raw.KCNP, raw.KCND]
    kqs = [raw.KQS, raw.KQP, raw.KQD]
    out: list[GFN0Shell] = []
    for shell_idx, (n, l) in enumerate(raw.shells):
        out.append(GFN0Shell(
            l=l,
            n=n,
            h=raw.lev[shell_idx],
            zeta=raw.exp[shell_idx],
            k_poly=float(polys[l] if polys[l] is not None else 0.0),
            # KCNS/P/D in the txt file are 10× the runtime values used
            # in xtb (cf. gfn0.f90:361 hardcoded `kCN` table vs the
            # `param_gfn0-xtb.txt` `KCNS=0.7116904` line for H, where
            # the runtime value is 0.07116904). Apply the 0.1 scale here.
            k_cn=float(kcns[l] if kcns[l] is not None else 0.0) * 0.1,
            k_q1=float(kqs[l] if kqs[l] is not None else 0.0),
        ))
    return tuple(out)


def _build_element_params(raw: GFN0RawElement) -> GFN0ElementParams:
    en = raw.EN if raw.EN is not None else _GFN0_EN_DEFAULTS[raw.Z - 1]
    return GFN0ElementParams(
        Z=raw.Z,
        symbol=_ELEMENT_SYMBOLS[raw.Z - 1],
        shells=_shells_from_raw(raw),
        en=float(en),
        k_q2=raw.KQAT2,
        rep_alpha=raw.REPA,
        rep_zeff=raw.REPB,
        eeq_chi=raw.XI,
        eeq_eta=raw.GAM,
        eeq_kappa=raw.KAPPA,
        eeq_alpha=raw.ALPG,
    )


# ---------------------------------------------------------------------------
# Module-load: parse the vendored file once and freeze.
# ---------------------------------------------------------------------------
_PARAM_FILE = Path(__file__).parent / "params" / "gfn0" / "param_gfn0-xtb.txt"

GFN0_GLOBALS: GFN0Globals
GFN0_PARAMS: dict[int, GFN0ElementParams]

GFN0_GLOBALS, _raw_elements = parse_gfn0_param_file(_PARAM_FILE)
GFN0_PARAMS = {Z: _build_element_params(raw) for Z, raw in _raw_elements.items()}
del _raw_elements
