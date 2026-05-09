"""Extract ALPB(water) parameters from xtb's param_alpb_water.fh
to a NumPy ``.npz`` for the pure-MLX port.

Type layout (xtb/src/solv/model.f90:94-106):
    epsv     — solvent dielectric
    smass    — solvent molecular mass
    rhos     — solvent density
    c1       — Born radius scaling
    rprobe   — solvent probe radius
    gshift   — global free-energy shift (kcal/mol)
    soset    — offset
    alpha    — ALPB α parameter
    gamscale(94) — per-element surface-tension scaling
    sx(94)   — per-element vdW-radius scaling
    tmp(94)  — per-element extra parameter (HB strength)

Each gfn-method block (gfn1, gfn2, gfn0_*, ...) is one record. We
extract the gfn2_alpb_water variant (used by GFN2-xTB ALPB(water)).
"""
from __future__ import annotations

import os
import re

import numpy as np


PARAM_FILE = "/private/tmp/xtb-src/include/param_alpb_water.fh"
OUT_PATH = "/Users/guillaume-osmo/Github/mlxmolkit/mlxmolkit/xtb/params/alpb_water.npz"

_FLOAT = re.compile(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?")


def parse_block(text: str, name: str) -> dict:
    """Parse a single ``gbsa_parameter`` instantiation by name."""
    pat = re.compile(
        rf"{name}\s*=\s*gbsa_parameter\s*\(\s*&?\s*(.*?)\s*\)\s*$",
        re.DOTALL | re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        raise KeyError(f"block {name!r} not found")
    body = m.group(1)
    # Strip Fortran continuations + comments
    body = re.sub(r"!.*?$", "", body, flags=re.MULTILINE)
    body = body.replace("&", " ").replace("\n", " ").replace("_wp", "")
    # The body is: 8 scalars, then 3 arrays (94 floats each), separated by `,`
    floats = [
        float(s.replace("d", "e").replace("D", "E"))
        for s in _FLOAT.findall(body)
    ]
    n_expected = 8 + 3 * 94
    if len(floats) != n_expected:
        raise ValueError(
            f"{name}: expected {n_expected} floats, got {len(floats)}"
        )
    return {
        "epsv": floats[0],
        "smass": floats[1],
        "rhos": floats[2],
        "c1": floats[3],
        "rprobe": floats[4],
        "gshift": floats[5],
        "soset": floats[6],
        "alpha": floats[7],
        "gamscale": np.array(floats[8 : 8 + 94], dtype=np.float64),
        "sx": np.array(floats[8 + 94 : 8 + 188], dtype=np.float64),
        "tmp": np.array(floats[8 + 188 : 8 + 282], dtype=np.float64),
    }


def main() -> None:
    with open(PARAM_FILE) as f:
        text = f.read()

    print("Parsing gfn2_alpb_water…")
    p = parse_block(text, "gfn2_alpb_water")
    for k, v in p.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:<10}  shape={v.shape}  range=[{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  {k:<10}  {v}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, **p)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
