#!/usr/bin/env python3
"""Run the current clean-room reconstructed g-xTB pieces on test molecules."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gxtb_oracle_probe import MOLECULES, molecule, run_xtb
from mlxmolkit.xtb.gxtb_reconstructed import gxtb_reconstructed_repulsion


def _fmt_vec(vec: np.ndarray) -> str:
    return "[" + " ".join(f"{x: .6f}" for x in vec) + "]"


def run_one(
    name: str,
    *,
    use_oracle_charges: bool,
    xtb: Path,
    acc: float,
) -> None:
    atoms, coords = molecule(name)
    descriptor = None
    oracle_energy = None
    if use_oracle_charges:
        oracle_energy, oracle_atoms, _ = run_xtb(atoms, coords, xtb=xtb, acc=acc, charge=0, uhf=0)
        descriptor = np.array([atom.charge for atom in oracle_atoms], dtype=np.float64)

    result = gxtb_reconstructed_repulsion(atoms, coords, descriptor=descriptor)
    grad = result.gradient

    print(f"molecule: {name}")
    print(f"atoms:    {len(atoms)}")
    print(f"block:    reconstructed g-xTB repulsion only")
    print(f"energy:   {result.energy: .12f} Ha")
    print(f"max|grad|:{np.max(np.abs(grad)): .6e} Ha/Ang")
    print(f"sum grad: {_fmt_vec(np.sum(grad, axis=0))} Ha/Ang")
    print(f"CN range: {np.min(result.cn):.6f} .. {np.max(result.cn):.6f}")
    print(f"alpha:    {np.min(result.state.alpha):.6f} .. {np.max(result.state.alpha):.6f}")
    print(f"Zeff*:    {np.min(result.state.scaled_zeff):.6f} .. {np.max(result.state.scaled_zeff):.6f}")
    if oracle_energy is not None:
        print(f"oracle g-xTB total energy: {oracle_energy: .12f} Ha")
        print(f"repulsion minus total:     {result.energy - oracle_energy: .12f} Ha")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", choices=[*sorted(MOLECULES), "all"], default="all")
    parser.add_argument("--oracle-charges", action="store_true")
    parser.add_argument("--xtb", type=Path, default=Path("/tmp/gxtb-v2-macos/bin/xtb"))
    parser.add_argument("--acc", type=float, default=0.1)
    args = parser.parse_args()

    names = sorted(MOLECULES) if args.molecule == "all" else [args.molecule]
    for name in names:
        run_one(name, use_oracle_charges=args.oracle_charges, xtb=args.xtb, acc=args.acc)


if __name__ == "__main__":
    main()
