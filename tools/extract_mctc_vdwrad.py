#!/usr/bin/env python3
"""Extract the packed MCTC vdW pair-radius table from the public g-xTB binary."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np


SYMBOL = "___mctc_data_vdwrad_MOD_vdwrad"
MAX_Z = 103
N_PACKED = MAX_Z * (MAX_Z + 1) // 2


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _symbol_address(lib: Path, symbol: str) -> int:
    out = _run(["nm", "-nm", str(lib)])
    pattern = re.compile(rf"^([0-9a-fA-F]+)\s+\([^)]+\)\s+.*\s+{re.escape(symbol)}$")
    for line in out.splitlines():
        match = pattern.match(line)
        if match:
            return int(match.group(1), 16)
    raise ValueError(f"could not find symbol {symbol} in {lib}")


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


def extract_vdwrad_packed(lib: Path) -> np.ndarray:
    addr = _symbol_address(lib, SYMBOL)
    nbytes = N_PACKED * np.dtype("<f8").itemsize
    offset = _file_offset(lib, addr, nbytes)
    data = lib.read_bytes()[offset : offset + nbytes]
    return np.frombuffer(data, dtype="<f8").astype(np.float64, copy=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, default=Path("/tmp/gxtb-v2-macos/lib/libxtb.dylib"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mlxmolkit/xtb/params/mctc_vdwrad.npz"),
    )
    args = parser.parse_args()

    packed = extract_vdwrad_packed(args.lib)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        vdwrad_pair_packed=packed,
        max_z=np.array(MAX_Z, dtype=np.int32),
        source_symbol=np.array(SYMBOL),
    )
    print(f"wrote {args.out} with {packed.size} packed pair radii")


if __name__ == "__main__":
    main()
