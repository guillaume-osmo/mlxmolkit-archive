#!/usr/bin/env python3
"""Extract q-vSZP basis tables from the public g-xTB release binary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np


MAX_Z = 103
MAX_SHELLS = 4
MAX_PRIM = 12

SYMBOLS = {
    "cov_radii": "___tblite_basis_qvszp_MOD_qvszp_cov_radii",
    "principal_quantum_number": "___tblite_basis_qvszp_MOD_principal_quantum_number",
    "p_k3": "___tblite_basis_qvszp_MOD_p_k3",
    "p_k2": "___tblite_basis_qvszp_MOD_p_k2",
    "p_k1": "___tblite_basis_qvszp_MOD_p_k1",
    "p_k0": "___tblite_basis_qvszp_MOD_p_k0",
    "nshell": "___tblite_basis_qvszp_MOD_nshell",
    "n_prim": "___tblite_basis_qvszp_MOD_n_prim",
    "ang_shell": "___tblite_basis_qvszp_MOD_ang_shell",
    "exponents": "___tblite_basis_qvszp_MOD_exponents",
    "coefficients_env": "___tblite_basis_qvszp_MOD_coefficients_env",
    "coefficients": "___tblite_basis_qvszp_MOD_coefficients",
}


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _symbol_addresses(lib: Path) -> dict[str, int]:
    out = _run(["nm", "-nm", str(lib)])
    found: dict[str, int] = {}
    patterns = {
        name: re.compile(rf"^([0-9a-fA-F]+)\s+\([^)]+\)\s+.*\s+{re.escape(symbol)}$")
        for name, symbol in SYMBOLS.items()
    }
    for line in out.splitlines():
        for name, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                found[name] = int(match.group(1), 16)
    missing = sorted(set(SYMBOLS) - set(found))
    if missing:
        raise ValueError(f"could not find q-vSZP symbols in {lib}: {missing}")
    return found


def _sections(lib: Path) -> list[dict[str, int | str]]:
    out = _run(["otool", "-l", str(lib)])
    sections: list[dict[str, int | str]] = []
    current: dict[str, int | str] | None = None
    for raw in out.splitlines():
        line = raw.strip()
        if line.startswith("sectname "):
            if current and {"addr", "size", "offset", "sectname"} <= current.keys():
                sections.append(current)
            current = {"sectname": line.split()[1]}
        elif current is not None and line.startswith("addr "):
            current["addr"] = int(line.split()[1], 16)
        elif current is not None and line.startswith("size "):
            current["size"] = int(line.split()[1], 16)
        elif current is not None and line.startswith("offset "):
            current["offset"] = int(line.split()[1])
        elif current is not None and line.startswith("flags "):
            if {"addr", "size", "offset", "sectname"} <= current.keys():
                sections.append(current)
            current = None
    return sections


def _file_offset(sections: list[dict[str, int | str]], addr: int, nbytes: int) -> int:
    for section in sections:
        start = int(section["addr"])
        end = start + int(section["size"])
        if start <= addr and addr + nbytes <= end:
            return int(section["offset"]) + (addr - start)
    raise ValueError(f"address 0x{addr:x}+{nbytes} is outside known sections")


def _read_array(
    blob: bytes,
    sections: list[dict[str, int | str]],
    addr: int,
    dtype: str,
    shape: tuple[int, ...],
    *,
    order: str = "C",
) -> np.ndarray:
    dt = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
    offset = _file_offset(sections, addr, nbytes)
    return np.frombuffer(blob[offset : offset + nbytes], dtype=dt).copy().reshape(shape, order=order)


def extract_qvszp_params(lib: Path) -> dict[str, np.ndarray]:
    """Return q-vSZP arrays in Python layout ``[Z, shell, primitive]``."""

    addresses = _symbol_addresses(lib)
    sections = _sections(lib)
    blob = lib.read_bytes()
    arrays: dict[str, np.ndarray] = {}

    arrays["cov_radii"] = _read_array(
        blob, sections, addresses["cov_radii"], "<f8", (MAX_Z,)
    ).astype(np.float64)
    for name in ("p_k0", "p_k1", "p_k2", "p_k3"):
        arrays[name] = _read_array(blob, sections, addresses[name], "<f8", (MAX_Z,)).astype(
            np.float64
        )
    arrays["nshell"] = _read_array(
        blob, sections, addresses["nshell"], "<i4", (MAX_Z,)
    ).astype(np.int32)
    for name in ("principal_quantum_number", "n_prim", "ang_shell"):
        arrays[name] = _read_array(
            blob, sections, addresses[name], "<i4", (MAX_Z, MAX_SHELLS)
        ).astype(np.int32)

    for name in ("exponents", "coefficients_env", "coefficients"):
        raw = _read_array(
            blob,
            sections,
            addresses[name],
            "<f8",
            (MAX_Z, MAX_SHELLS, MAX_PRIM),
        )
        arrays[name] = raw.astype(np.float64)

    meta = [
        {
            "name": name,
            "symbol": SYMBOLS[name],
            "address": f"0x{addresses[name]:08x}",
        }
        for name in SYMBOLS
    ]
    arrays["__meta_json__"] = np.array(json.dumps(meta, indent=2))
    arrays["max_z"] = np.array(MAX_Z, dtype=np.int32)
    arrays["max_shells"] = np.array(MAX_SHELLS, dtype=np.int32)
    arrays["max_prim"] = np.array(MAX_PRIM, dtype=np.int32)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, default=Path("/tmp/gxtb-v2-macos/lib/libxtb.dylib"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mlxmolkit/xtb/params/qvszp_binary_params.npz"),
    )
    args = parser.parse_args()

    arrays = extract_qvszp_params(args.lib)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(
        f"wrote {args.out} with q-vSZP tables "
        f"({MAX_Z} elements, {MAX_SHELLS} shells, {MAX_PRIM} primitives)"
    )


if __name__ == "__main__":
    main()
