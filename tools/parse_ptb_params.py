"""Parse PTB Fortran source (xtb_ptb_param + xtb_ptb_vdzp) to vendor
parameters as a NumPy ``.npz`` for the pure-MLX port.

PTB (Mueller, J. Chem. Phys. 158, 124111, 2023) is the immediate
predecessor of g-xTB and is the only public source code today that
shares the structural skeleton (vDZP basis, two-step SCF, +U,
Pauli-XC) with v2 g-xTB. Vendoring PTB gets us the full infrastructure
ahead of the eventual g-xTB source release.

Inputs:
    /tmp/gxtb-src/src/ptb/param.F90   — element-resolved parameters (3957 lines)
    /tmp/gxtb-src/src/ptb/vdzp.F90    — vDZP basis exponents/coeffs (1636 lines)

Output:
    mlxmolkit/xtb/params/ptb_data.npz  — keys mirror Fortran names
"""
from __future__ import annotations

import os
import re

import numpy as np


PARAM_F90 = "/tmp/gxtb-src/src/ptb/param.F90"
VDZP_F90 = "/tmp/gxtb-src/src/ptb/vdzp.F90"
OUT_PATH = "/Users/guillaume-osmo/Github/mlxmolkit/mlxmolkit/xtb/params/ptb_data.npz"

MAX_ELEM = 86
MAX_SHELL = 7
MAX_PRIM = 5
MAX_ANGMOM_PLUS1 = 3   # max_angmom = 2 (s,p,d) → +1 = 3 indices

_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?")
_INT_RE = re.compile(r"-?\d+")


def _strip(body: str) -> str:
    """Strip Fortran continuations (`&`), comments (`! ... \\n`), and newlines."""
    body = re.sub(r"!.*?$", "", body, flags=re.MULTILINE)
    body = body.replace("&", " ").replace("\n", " ").replace("_wp", "")
    return body


def parse_1d_real(text: str, name: str, n: int = MAX_ELEM) -> np.ndarray:
    """``real(wp), parameter :: NAME(...) = [ ...floats... ]``."""
    pat = re.compile(
        rf"real\(wp\),\s*parameter\s*::\s*{name}\s*\([^)]*\)\s*=\s*\[(.*?)\](?!\s*,)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise KeyError(f"1D real array {name} not found")
    body = _strip(m.group(1))
    floats = _FLOAT_RE.findall(body)
    vals = [float(f.replace("d", "e").replace("D", "E")) for f in floats]
    if len(vals) != n:
        raise ValueError(f"{name}: expected {n} reals, got {len(vals)}")
    return np.asarray(vals, dtype=np.float64)


def parse_1d_int(text: str, name: str, n: int = MAX_ELEM) -> np.ndarray:
    """``integer, parameter :: NAME(...) = [ ...ints... ]``."""
    pat = re.compile(
        rf"integer,\s*parameter\s*::\s*{name}\s*\([^)]*\)\s*=\s*\[(.*?)\](?!\s*,)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise KeyError(f"1D int array {name} not found")
    body = _strip(m.group(1))
    ints = _INT_RE.findall(body)
    vals = [int(s) for s in ints]
    if len(vals) != n:
        raise ValueError(f"{name}: expected {n} ints, got {len(vals)}")
    return np.asarray(vals, dtype=np.int64)


def parse_2d_real_reshape(
    text: str, name: str, dim1: int, dim2: int = MAX_ELEM,
) -> np.ndarray:
    """``real(wp), parameter :: NAME(D1, D2) = reshape([...], shape(NAME))``.

    Returns ``(D2, D1)`` — i.e., (n_elem, n_shell) row-major Python
    convention. Fortran is column-major: flat list is
    ``[v(1,1), v(2,1), ..., v(D1,1), v(1,2), ...]``, so we use
    ``order='F'`` then transpose.
    """
    pat = re.compile(
        rf"real\(wp\),\s*parameter\s*::\s*{name}\s*\([^)]*\)\s*="
        rf"\s*reshape\(\s*\[(.*?)\]\s*,\s*shape\({name}\)\s*\)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise KeyError(f"2D real reshape {name} not found")
    body = _strip(m.group(1))
    floats = _FLOAT_RE.findall(body)
    vals = [float(f.replace("d", "e").replace("D", "E")) for f in floats]
    if len(vals) != dim1 * dim2:
        raise ValueError(
            f"{name}: expected {dim1*dim2} reals (={dim1}*{dim2}), "
            f"got {len(vals)}"
        )
    flat = np.asarray(vals, dtype=np.float64)
    arr_F = flat.reshape((dim1, dim2), order="F")  # (D1, D2) Fortran-order
    return arr_F.T  # (D2, D1) — (n_elem, n_shell or n_angmom+1)


def parse_2d_int_reshape(
    text: str, name: str, dim1: int, dim2: int = MAX_ELEM,
) -> np.ndarray:
    pat = re.compile(
        rf"integer,\s*parameter\s*::\s*{name}\s*\([^)]*\)\s*="
        rf"\s*reshape\(\s*\[(.*?)\]\s*,\s*shape\({name}\)\s*\)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise KeyError(f"2D int reshape {name} not found")
    body = _strip(m.group(1))
    ints = _INT_RE.findall(body)
    vals = [int(s) for s in ints]
    if len(vals) != dim1 * dim2:
        raise ValueError(f"{name}: expected {dim1*dim2}, got {len(vals)}")
    flat = np.asarray(vals, dtype=np.int64)
    arr_F = flat.reshape((dim1, dim2), order="F")
    return arr_F.T


def parse_globals(text: str) -> dict:
    """``ptbGlobals = TPTBParameter(kpol=..., kpolres=..., ...)``."""
    pat = re.compile(
        r"ptbGlobals\s*=\s*TPTBParameter\(\s*(.*?)\)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise KeyError("ptbGlobals struct not found")
    body = _strip(m.group(1))
    out = {}
    for kv in re.finditer(r"(\w+)\s*=\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)", body):
        out[kv.group(1)] = float(kv.group(2).replace("d", "e").replace("D", "E"))
    return out


def parse_vdzp_exponents(text: str, kind: str = "exponents") -> np.ndarray:
    """``exponents(:, :, Z) = reshape([...], (/max_prim, max_shell/))`` for
    Z = 1..86 inside ``setCGTOexponents``. Returns ``(86, max_shell, max_prim)``.

    Each Z block holds ``max_prim * max_shell = 35`` floats, column-major
    in Fortran (max_prim varies fastest). For the (Z, ish) block of length
    ``max_prim`` we want: ``out[Z-1, ish, :max_prim]`` filled with that
    block.
    """
    out = np.zeros((MAX_ELEM, MAX_SHELL, MAX_PRIM), dtype=np.float64)
    pat = re.compile(
        rf"{kind}\(:,\s*:,\s*(\d+)\)\s*=\s*reshape\(\s*\[(.*?)\]\s*,",
        re.DOTALL,
    )
    seen = set()
    for m in pat.finditer(text):
        Z = int(m.group(1))
        body = _strip(m.group(2))
        floats = _FLOAT_RE.findall(body)
        vals = [
            float(f.replace("d", "e").replace("D", "E")) for f in floats
        ]
        if len(vals) != MAX_PRIM * MAX_SHELL:
            raise ValueError(
                f"{kind}(Z={Z}): expected {MAX_PRIM*MAX_SHELL}, got {len(vals)}"
            )
        flat = np.asarray(vals, dtype=np.float64)
        # Fortran column-major: arr_F[prim, shell] reshape order='F'
        arr_F = flat.reshape((MAX_PRIM, MAX_SHELL), order="F")  # (prim, shell)
        out[Z - 1] = arr_F.T  # (shell, prim)
        seen.add(Z)
    missing = set(range(1, MAX_ELEM + 1)) - seen
    print(f"  {kind}: {len(seen)} elements parsed, "
          f"{len(missing)} missing ({sorted(missing)[:8]}…)" if missing
          else f"  {kind}: all {len(seen)} elements parsed")
    return out


def main() -> None:
    with open(PARAM_F90) as f:
        param_src = f.read()
    with open(VDZP_F90) as f:
        vdzp_src = f.read()

    print("Parsing PTB globals…")
    globs = parse_globals(param_src)
    print(f"  found {len(globs)} fields: {sorted(globs)}")

    print("\nParsing 1D real arrays (n_elem = 86)…")
    real_1d = [
        "kr", "kocod", "kcnstar", "kshift", "rf", "kxc1", "kits0",
        "kecpepsilon", "alpeeq", "chieeq", "cnfeeq", "gameeq",
        "kto", "cud", "cu1", "cu2",
        # Repulsion + +U residual scaling (last block in param.F90)
        "ar", "arcn", "avcn", "kares", "kueffres", "cvesres",
    ]
    one_d_data = {}
    for name in real_1d:
        try:
            one_d_data[name] = parse_1d_real(param_src, name)
            print(f"  {name:<14} ✓ shape={one_d_data[name].shape}")
        except (KeyError, ValueError) as e:
            print(f"  {name:<14} ✗ {e}")

    print("\nParsing 2D real reshape arrays…")
    # (max_angmom+1, max_elem):
    angmom_arrs = ["kla", "ksla"]
    # (max_shell, max_elem):
    shell_arrs = ["hla", "klh", "kxc2l", "kalphah0l", "klalphaxc",
                  "keta1", "keta2", "cueffl"]
    two_d_data = {}
    for name in angmom_arrs:
        try:
            two_d_data[name] = parse_2d_real_reshape(
                param_src, name, MAX_ANGMOM_PLUS1
            )
            print(f"  {name:<14} ✓ shape={two_d_data[name].shape}")
        except (KeyError, ValueError) as e:
            print(f"  {name:<14} ✗ {e}")
    for name in shell_arrs:
        try:
            two_d_data[name] = parse_2d_real_reshape(param_src, name, MAX_SHELL)
            print(f"  {name:<14} ✓ shape={two_d_data[name].shape}")
        except (KeyError, ValueError) as e:
            print(f"  {name:<14} ✗ {e}")

    print("\nParsing 1D int arrays (vDZP basis layout)…")
    int_arrs = {}
    for name in ["nshell"]:
        int_arrs[name] = parse_1d_int(vdzp_src, name)
        print(f"  {name:<14} ✓ shape={int_arrs[name].shape}, "
              f"max={int_arrs[name].max()}")
    # core basis n_shell from param.F90
    int_arrs["cbas_nshell"] = parse_1d_int(param_src, "cbas_nshell")
    print(f"  cbas_nshell    ✓ shape={int_arrs['cbas_nshell'].shape}")

    print("\nParsing 2D int reshape arrays (vDZP basis layout)…")
    for name in ["ang_shell", "n_prim"]:
        int_arrs[name] = parse_2d_int_reshape(vdzp_src, name, MAX_SHELL)
        print(f"  {name:<14} ✓ shape={int_arrs[name].shape}")

    print("\nParsing val_el (vDZP valence electron count)…")
    int_arrs["val_el"] = parse_1d_real(vdzp_src, "val_el")
    print(f"  val_el         ✓ shape={int_arrs['val_el'].shape}")

    print("\nParsing vDZP exponents + coefficients (per-element reshape)…")
    vdzp_exp = parse_vdzp_exponents(vdzp_src, "exponents")
    vdzp_coeff = parse_vdzp_exponents(vdzp_src, "coefficients")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    payload = dict(one_d_data)
    payload.update(two_d_data)
    payload.update(int_arrs)
    payload["vdzp_exponents"] = vdzp_exp
    payload["vdzp_coefficients"] = vdzp_coeff
    payload["globals_keys"] = np.array(sorted(globs.keys()))
    payload["globals_values"] = np.array(
        [globs[k] for k in sorted(globs.keys())], dtype=np.float64,
    )
    np.savez(OUT_PATH, **payload)

    print(f"\nWrote {OUT_PATH}")
    print(f"  {len(payload)} arrays packed.")
    nz = int(np.count_nonzero(np.any(vdzp_exp != 0, axis=(1, 2))))
    print(f"  vDZP exponents non-zero for {nz} elements.")


if __name__ == "__main__":
    main()
