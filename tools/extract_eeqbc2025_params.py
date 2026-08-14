#!/usr/bin/env python3
"""Extract EEQ_BC 2025 element parameter tables from the public g-xTB binary."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np


MAX_Z = 103
SYMBOLS = {
    "rvdw_scale": "___multicharge_param_eeqbc2025_MOD_eeqbc_rvdw_scale",
    "rad": "___multicharge_param_eeqbc2025_MOD_eeqbc_rad",
    "kqchi": "___multicharge_param_eeqbc2025_MOD_eeqbc_kqchi",
    "kcnchi": "___multicharge_param_eeqbc2025_MOD_eeqbc_kcnchi",
    "eta": "___multicharge_param_eeqbc2025_MOD_eeqbc_eta",
    "cov_radii": "___multicharge_param_eeqbc2025_MOD_eeqbc_cov_radii",
    "chi": "___multicharge_param_eeqbc2025_MOD_eeqbc_chi",
    "cap": "___multicharge_param_eeqbc2025_MOD_eeqbc_cap",
    "avg_cn": "___multicharge_param_eeqbc2025_MOD_eeqbc_avg_cn",
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
        raise ValueError(f"could not find EEQ_BC symbols in {lib}: {missing}")
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


def _file_offset(lib: Path, addr: int, nbytes: int) -> int:
    for section in _sections(lib):
        start = int(section["addr"])
        end = start + int(section["size"])
        if start <= addr and addr + nbytes <= end:
            return int(section["offset"]) + (addr - start)
    raise ValueError(f"address 0x{addr:x}+{nbytes} is outside known sections")


def extract_eeqbc2025_params(lib: Path) -> dict[str, np.ndarray]:
    addresses = _symbol_addresses(lib)
    raw = lib.read_bytes()
    nbytes = MAX_Z * np.dtype("<f8").itemsize
    arrays: dict[str, np.ndarray] = {}
    for name, addr in addresses.items():
        offset = _file_offset(lib, addr, nbytes)
        arrays[name] = np.frombuffer(raw[offset : offset + nbytes], dtype="<f8").astype(
            np.float64,
            copy=True,
        )
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, default=Path("/tmp/gxtb-v2-macos/lib/libxtb.dylib"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mlxmolkit/xtb/params/eeqbc2025_params.npz"),
    )
    args = parser.parse_args()

    arrays = extract_eeqbc2025_params(args.lib)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **arrays,
        max_z=np.array(MAX_Z, dtype=np.int32),
        source_symbols=np.array([SYMBOLS[name] for name in arrays], dtype="U96"),
        source_names=np.array(list(arrays), dtype="U32"),
    )
    print(f"wrote {args.out} with {len(arrays)} EEQ_BC 2025 arrays of length {MAX_Z}")


if __name__ == "__main__":
    main()
