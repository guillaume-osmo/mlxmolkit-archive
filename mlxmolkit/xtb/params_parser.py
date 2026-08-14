# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT
#
# Parses the custom flat-text parameter file shipped by grimme-lab/xtb
# (LGPL-3.0). The file `param_gfn0-xtb.txt` lives at
# `mlxmolkit/xtb/params/gfn0/`, vendored verbatim from
# https://github.com/grimme-lab/xtb (`main` branch). License of the
# vendored data: LGPL-3.0. We only parse it; the dataclasses below are
# our own MIT-licensed code.

"""Parse `grimme-lab/xtb`'s `param_gfn0-xtb.txt` flat-text format.

File structure:

    $info
    level <int>
    name <str>
    doi <str>
    $globpar
    <key>  <float>     # one per line, e.g. "ks  2.0"
    ...
    $end
    $pairpar
    <key>  <float>     # GFN0 has none
    $end
    $Z= <int> <date>
     <key>= <float> <float> ...    # leading space; values can be 1+ floats
     ...
    $end
    ...
    $Z= <int> ...
    ...

Line conventions:
- `$globpar`/`$Z= n` blocks terminate at `$end`.
- Per-element fields use leading space + key + `=` + space-separated floats.
- Shell-resolved fields (lev=, exp=, POLYS/P/D=, KCNS/P/D=, KQS/P/Q=) appear
  only for shells the element has (read `ao=` to know which).
- Some fields (EN= for non-H elements) are absent from the file and live
  hardcoded in xtb's Fortran source (`gfn0.f90:251-269`); the loader
  patches them in from a vendored Python table at use sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_AO_SHELL_RE = re.compile(r"(\d)([spdf])")


def parse_ao_string(ao: str) -> list[tuple[int, int]]:
    """Parse a shell-list string like '1s2s', '2s2p', '3d4s4p' into
    [(n, l), ...] with l in {0, 1, 2, 3} for s/p/d/f.
    """
    matches = _AO_SHELL_RE.findall(ao.strip())
    if not matches:
        raise ValueError(f"Cannot parse ao= string: {ao!r}")
    l_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    return [(int(n), l_map[ang]) for n, ang in matches]


@dataclass
class GFN0RawElement:
    """Raw (untyped) per-element parameter block as parsed from the
    file. All values are either floats (scalar) or lists of floats
    (shell-resolved). `None` means "field not present in this block"
    — caller must handle defaults.
    """
    Z: int
    ao: str  # raw shell-list string, e.g. "2s2p"
    shells: list[tuple[int, int]]  # parsed (n, l) pairs
    lev: list[float]               # per shell
    exp: list[float]               # per shell
    EN: Optional[float]            # scalar; absent for most heavy atoms
    GAM: float                     # EEQ hardness (η)
    KQAT2: float                   # quadratic q-shift (atom-resolved)
    KCNS: float                    # CN-shift, s-shell
    KCNP: Optional[float]          # CN-shift, p-shell (None if no p)
    KCND: Optional[float]          # CN-shift, d-shell (None if no d)
    REPA: float                    # repulsion exponent α
    REPB: float                    # repulsion effective charge Z_eff
    POLYS: float                   # shell-poly coefficient, s
    POLYP: Optional[float]
    POLYD: Optional[float]
    KQS: float                     # linear q-shift, s-shell
    KQP: Optional[float]
    KQD: Optional[float]
    XI: float                      # EEQ electronegativity χ
    KAPPA: float                   # EEQ CN-coupling κ
    ALPG: float                    # EEQ Gaussian charge width α

    raw: dict = field(default_factory=dict, repr=False)  # any unknown fields


@dataclass
class GFN0Globals:
    """20-element global parameter block from `$globpar`."""
    ks: float
    kp: float
    kd: float
    kdiff: float
    ens: float
    enp: float
    end: float
    enscale4: float
    ipeashift: float
    srbshift: float
    srbpre: float
    srbexp: float
    srbken: float
    a1: float       # D4 BJ damping
    a2: float
    s8: float
    s9: float       # GFN0: 0.0 (no ATM)
    kexp: float
    kexplight: float
    renscale: float


def _parse_kv_line(line: str) -> tuple[str, list[float]]:
    """Parse a leading-space ' key= v1 v2 ...' line into (key, [floats])."""
    s = line.strip()
    eq = s.index("=")
    key = s[:eq].strip()
    rest = s[eq + 1:].strip()
    return key, [float(x) for x in rest.split()]


def _parse_globpar(lines: list[str]) -> GFN0Globals:
    """Parse the $globpar block (20 simple key float pairs)."""
    g: dict = {}
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 2:
            raise ValueError(f"Malformed $globpar line: {line!r}")
        g[parts[0]] = float(parts[1])
    # Map the file's 'end' (a Python keyword) to attribute 'end' explicitly.
    return GFN0Globals(**{k: g[k] for k in (
        "ks", "kp", "kd", "kdiff", "ens", "enp", "end", "enscale4",
        "ipeashift", "srbshift", "srbpre", "srbexp", "srbken",
        "a1", "a2", "s8", "s9", "kexp", "kexplight", "renscale",
    )})


def _parse_element_block(lines: list[str], Z: int) -> GFN0RawElement:
    """Parse the body of a single `$Z= n ... $end` block."""
    raw: dict[str, object] = {}
    for line in lines:
        if "=" not in line:
            continue
        # `ao=` is the only string-valued field; everything else is float(s).
        s = line.strip()
        eq = s.index("=")
        key = s[:eq].strip()
        if key == "ao":
            raw[key] = s[eq + 1:].strip()
            continue
        values = [float(x) for x in s[eq + 1:].strip().split()]
        raw[key] = values if len(values) > 1 else values[0]

    shells = parse_ao_string(raw["ao"])  # type: ignore[arg-type]

    def _list(key: str) -> list[float]:
        v = raw.get(key)
        if v is None:
            return []
        return v if isinstance(v, list) else [float(v)]

    def _opt(key: str) -> Optional[float]:
        v = raw.get(key)
        if v is None:
            return None
        return float(v) if not isinstance(v, list) else (v[0] if v else None)

    return GFN0RawElement(
        Z=Z,
        ao=raw["ao"],           # type: ignore[arg-type]
        shells=shells,
        lev=_list("lev"),
        exp=_list("exp"),
        EN=_opt("EN"),
        GAM=float(raw["GAM"]),
        KQAT2=float(raw["KQAT2"]),
        KCNS=float(raw["KCNS"]),
        KCNP=_opt("KCNP"),
        KCND=_opt("KCND"),
        REPA=float(raw["REPA"]),
        REPB=float(raw["REPB"]),
        POLYS=float(raw["POLYS"]),
        POLYP=_opt("POLYP"),
        POLYD=_opt("POLYD"),
        KQS=float(raw["KQS"]),
        KQP=_opt("KQP"),
        KQD=_opt("KQD"),
        XI=float(raw["XI"]),
        KAPPA=float(raw["KAPPA"]),
        ALPG=float(raw["ALPG"]),
        raw={k: v for k, v in raw.items() if k not in {
            "ao", "lev", "exp", "EN", "GAM", "KQAT2",
            "KCNS", "KCNP", "KCND", "REPA", "REPB",
            "POLYS", "POLYP", "POLYD", "KQS", "KQP", "KQD",
            "XI", "KAPPA", "ALPG",
        }},
    )


def parse_gfn0_param_file(
    path: str | Path,
) -> tuple[GFN0Globals, dict[int, GFN0RawElement]]:
    """Parse `param_gfn0-xtb.txt` end-to-end.

    Returns:
        (globals, per_element) — globals as :class:`GFN0Globals`,
        per_element as a dict keyed by atomic number.
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    sections: dict[str, list[str]] = {}
    elements: dict[int, list[str]] = {}

    cur_section: Optional[str] = None
    cur_buf: list[str] = []
    cur_z: Optional[int] = None

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("$end"):
            if cur_section is not None:
                sections[cur_section] = list(cur_buf)
                cur_buf.clear()
                cur_section = None
            elif cur_z is not None:
                elements[cur_z] = list(cur_buf)
                cur_buf.clear()
                cur_z = None
            continue
        if stripped.startswith("$Z="):
            # Header: "$Z= 1 Fri Nov 30 ..."
            tail = stripped[3:].strip()
            cur_z = int(tail.split()[0])
            cur_buf = []
            continue
        if stripped.startswith("$"):
            cur_section = stripped.split()[0][1:]  # drop the leading $
            cur_buf = []
            continue
        if cur_section is not None or cur_z is not None:
            cur_buf.append(line)

    if "globpar" not in sections:
        raise ValueError("Missing $globpar block in parameter file")

    globals_ = _parse_globpar(sections["globpar"])
    per_element: dict[int, GFN0RawElement] = {}
    for Z, body in elements.items():
        per_element[Z] = _parse_element_block(body, Z)
    return globals_, per_element
