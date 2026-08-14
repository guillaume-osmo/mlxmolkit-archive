#!/usr/bin/env python3
"""Probe the public g-xTB binary and compare printed shell populations.

This is a clean-room oracle tool. It only consumes observable CLI output from
``xtb --gxtb`` and the binary-extracted parameter arrays exposed by
``mlxmolkit.xtb.params_gxtb``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlxmolkit.xtb.params_gxtb import GXTB_PARAMS


MOLECULES = {
    "water": (
        [8, 1, 1],
        np.array(
            [
                [0.0, 0.0, 0.117790],
                [0.0, 0.755453, -0.471160],
                [0.0, -0.755453, -0.471160],
            ],
            dtype=np.float64,
        ),
    ),
    # Vanillin, RDKit/MMFF geometry is generated if RDKit is available.
    "vanillin": "COc1cc(C=O)ccc1O",
    # Hedione / methyl dihydrojasmonate.
    "hedione": "CCCCCC1C(CCC1=O)CC(=O)OC",
}


@dataclass(frozen=True)
class OracleAtom:
    index: int
    Z: int
    symbol: str
    charge: float
    shell_pop: tuple[float, ...]


def rdkit_coords(smiles: str, seed: int = 42) -> tuple[list[int], np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"RDKit embedding failed: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    conf = mol.GetConformer()
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coords = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=np.float64,
    )
    return atoms, coords


def molecule(name: str) -> tuple[list[int], np.ndarray]:
    item = MOLECULES[name]
    if isinstance(item, tuple):
        return item
    return rdkit_coords(item)


def write_xyz(path: Path, atoms: list[int], coords: np.ndarray) -> None:
    symbols = {
        1: "H",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        35: "Br",
        53: "I",
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\nprobe\n")
        for Z, xyz in zip(atoms, coords):
            f.write(f"{symbols.get(int(Z), str(Z))} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}\n")


def parse_total_energy(log: str) -> float:
    match = re.search(r"total energy\s+([-+0-9.Ee]+)\s+Eh", log)
    if not match:
        raise ValueError("could not parse total energy")
    return float(match.group(1))


def parse_shell_populations(log: str) -> list[OracleAtom]:
    lines = log.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "Atomic charges and shell populations" in line:
            start = i
            break
    if start is None:
        raise ValueError("could not find shell population block")

    atoms: list[OracleAtom] = []
    row = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]+)\s+([-+0-9.Ee]+)\s+(.+)$")
    in_rows = False
    for line in lines[start + 1 :]:
        if line.strip().startswith("#"):
            in_rows = True
            continue
        if not in_rows:
            continue
        if line.strip().startswith("---"):
            if atoms:
                break
            continue
        m = row.match(line)
        if not m:
            continue
        shell_pop = tuple(float(x) for x in m.group(5).split())
        atoms.append(
            OracleAtom(
                index=int(m.group(1)),
                Z=int(m.group(2)),
                symbol=m.group(3),
                charge=float(m.group(4)),
                shell_pop=shell_pop,
            )
        )
    if not atoms:
        raise ValueError("shell population block was present but empty")
    return atoms


def run_xtb(
    atoms: list[int],
    coords: np.ndarray,
    xtb: Path,
    acc: float,
    charge: int,
    uhf: int,
) -> tuple[float, list[OracleAtom], str]:
    with tempfile.TemporaryDirectory(prefix="gxtb-oracle-") as tmp:
        cwd = Path(tmp)
        xyz = cwd / "mol.xyz"
        write_xyz(xyz, atoms, coords)
        env = os.environ.copy()
        libdir = str(xtb.parent.parent / "lib")
        bindir = str(xtb.parent)
        env["DYLD_LIBRARY_PATH"] = f"{libdir}:{bindir}:{env.get('DYLD_LIBRARY_PATH', '')}"
        cmd = [
            str(xtb),
            str(xyz),
            "--gxtb",
            "--grad",
            "--acc",
            str(acc),
            "--chrg",
            str(charge),
            "--uhf",
            str(uhf),
        ]
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)
        log = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"xtb failed with code {proc.returncode}\n{log[-4000:]}")
        return parse_total_energy(log), parse_shell_populations(log), log


def print_comparison(energy: float, oracle_atoms: list[OracleAtom]) -> None:
    print(f"g-xTB total energy: {energy:.12f} Ha")
    print("")
    print("idx Z  shell labels  reference_occ -> oracle_pop       charge")
    for atom in oracle_atoms:
        labels = GXTB_PARAMS.shell_labels(atom.Z)
        ref = GXTB_PARAMS.reference_population(atom.Z)
        pop = np.array(atom.shell_pop, dtype=np.float64)
        ref_s = " ".join(f"{x:7.3f}" for x in ref)
        pop_s = " ".join(f"{x:7.3f}" for x in pop)
        lab_s = ",".join(labels)
        print(f"{atom.index:3d} {atom.Z:2d} {lab_s:>8s}  [{ref_s}] -> [{pop_s}]  {atom.charge:+.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xtb", type=Path, default=Path("/tmp/gxtb-v2-macos/bin/xtb"))
    parser.add_argument("--molecule", choices=sorted(MOLECULES), default="water")
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--uhf", type=int, default=0)
    args = parser.parse_args()

    atoms, coords = molecule(args.molecule)
    energy, oracle_atoms, _ = run_xtb(atoms, coords, args.xtb, args.acc, args.charge, args.uhf)
    print(f"molecule: {args.molecule}")
    print(f"atoms:    {len(atoms)}")
    print_comparison(energy, oracle_atoms)


if __name__ == "__main__":
    main()
