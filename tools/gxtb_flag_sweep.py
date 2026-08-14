#!/usr/bin/env python3
"""g-xTB term-by-term flag sweep against the official xtb --gxtb binary.

For a small CHNOFS molecule set, this evaluates the official xtb g-xTB total
energy once, then runs the local scf_gxtb.gxtb_energy with every flag
combination of interest (baseline, +offsite, +two-body 3rd, +both, ...).
The point is to identify which "approximate/disabled" reconstructed terms
actually close the residual against the binary.

Run from the repo root with:

    PYTHONPATH=.:/Users/guillaume-osmo/Github/mlx-addons/src \\
    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \\
        tools/gxtb_flag_sweep.py --n 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_ADDONS_SRC = Path.home() / "Github" / "mlx-addons" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if MLX_ADDONS_SRC.exists() and str(MLX_ADDONS_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_ADDONS_SRC))

from tools.gxtb_100_molecule_benchmark import (  # noqa: E402
    KCAL_PER_HA,
    ALLOWED_Z,
    MoleculeCase,
    embed_smiles,
    parse_total_energy,
    smiles_passes_filter,
)


COMBOS = [
    # (label, kwargs to merge into gxtb_energy)
    ("base", {}),
    ("+offsite", {"use_first_order_offsite": True}),
    ("+twobody3", {"use_twobody_third_order": True}),
    ("+third3", {"use_third_order": True}),
    ("+fourth4", {"use_fourth_order": True}),
    ("+exchange", {"use_exchange": True}),
    ("+offsite+twobody3", {"use_first_order_offsite": True, "use_twobody_third_order": True}),
    (
        "all_on",
        {
            "use_first_order_offsite": True,
            "use_twobody_third_order": True,
            "use_third_order": True,
            "use_fourth_order": True,
            "use_exchange": True,
        },
    ),
]


def run_xtb_binary_gxtb(case: MoleculeCase, *, xtb: Path, acc: float) -> float:
    """One xtb --gxtb call per molecule; returns total energy in Ha."""

    with tempfile.TemporaryDirectory(prefix="gxtb-fs-") as tmp:
        cwd = Path(tmp)
        xyz = cwd / "mol.xyz"
        with xyz.open("w") as f:
            f.write(f"{len(case.atoms)}\n\n")
            for z, xyz_row in zip(case.atoms, case.coords):
                from rdkit.Chem import GetPeriodicTable

                sym = GetPeriodicTable().GetElementSymbol(int(z))
                f.write(f"{sym}  {xyz_row[0]:.10f}  {xyz_row[1]:.10f}  {xyz_row[2]:.10f}\n")
        env = os.environ.copy()
        libdir = str(xtb.parent.parent / "lib")
        bindir = str(xtb.parent)
        env["DYLD_LIBRARY_PATH"] = f"{libdir}:{bindir}:{env.get('DYLD_LIBRARY_PATH', '')}"
        cmd = [str(xtb), str(xyz.name), "--gxtb", "--acc", str(acc), "--chrg", "0", "--uhf", "0"]
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)
        log = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"xtb --gxtb failed: {log[-2000:]}")
        return parse_total_energy(log)


def run_local_combo(case: MoleculeCase, *, conv_tol: float, max_iter: int, extra: dict) -> tuple[float, float]:
    """Local scf_gxtb.gxtb_energy with the given flag overrides."""

    from mlxmolkit.xtb.scf_gxtb import gxtb_energy

    kwargs = {
        "charge": 0,
        "conv_tol": conv_tol,
        "max_iter": max_iter,
        "use_d4srev": True,
        "use_pacp": True,
        "use_first_order": True,
        "use_first_order_offsite": False,
        "use_mfx_exchange": True,
        "use_twobody_third_order": False,
        "use_acp_hamiltonian": True,
    }
    kwargs.update(extra)
    t0 = time.perf_counter()
    res = gxtb_energy(case.atoms, case.coords, **kwargs)
    wall = time.perf_counter() - t0
    return float(res["energy_plus_increment_hartree"]), wall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "benchmark_1000_smiles.csv")
    parser.add_argument("--xtb", type=Path, default=Path("/tmp/gxtb-v2-macos/bin/xtb"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--conv-tol", type=float, default=1e-7)
    parser.add_argument("--max-iter", type=int, default=240)
    args = parser.parse_args()

    import csv

    cases: list[MoleculeCase] = []
    with args.input.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            smi = row["smiles"].strip()
            if not smi:
                continue
            if not smiles_passes_filter(smi, max_atoms=args.max_atoms, allow_zwitterions=False):
                continue
            try:
                atoms, coords = embed_smiles(smi, seed=args.seed + idx)
            except Exception:
                continue
            cases.append(MoleculeCase(name=f"m{idx:04d}", smiles=smi, atoms=atoms, coords=coords))
            if len(cases) >= args.n:
                break

    print(f"Prepared {len(cases)} molecules; CHNOFS only; max_atoms={args.max_atoms}\n")

    # Collect oracle energies first
    e_oracle: list[float] = []
    print("Running xtb --gxtb oracle...")
    for c in cases:
        try:
            e = run_xtb_binary_gxtb(c, xtb=args.xtb, acc=args.acc)
            e_oracle.append(e)
            print(f"  {c.name:<8} nat={len(c.atoms):>2}  E_oracle = {e:+.8f} Ha  {c.smiles}")
        except Exception as exc:
            e_oracle.append(float("nan"))
            print(f"  {c.name:<8} FAILED: {exc}")

    # For each combo, evaluate locally; compare to oracle
    deltas_per_combo: dict[str, list[float]] = {label: [] for label, _ in COMBOS}
    walls_per_combo: dict[str, list[float]] = {label: [] for label, _ in COMBOS}
    print("\nRunning local scf_gxtb per molecule for each combo...")
    for ci, c in enumerate(cases):
        if not np.isfinite(e_oracle[ci]):
            continue
        print(f"\n  {c.name} nat={len(c.atoms)}  E_oracle={e_oracle[ci]:+.6f} Ha")
        for label, extra in COMBOS:
            try:
                e_loc, wall = run_local_combo(c, conv_tol=args.conv_tol, max_iter=args.max_iter, extra=extra)
                dE = e_loc - e_oracle[ci]
                deltas_per_combo[label].append(dE)
                walls_per_combo[label].append(wall)
                print(f"     {label:<22}  E={e_loc:+.6f}  ΔE={dE:+.4f} Ha ({dE*KCAL_PER_HA:+.1f} kcal/mol)  {wall:.2f}s")
            except Exception as exc:
                print(f"     {label:<22}  FAILED: {exc}")

    print("\n" + "=" * 88)
    print(f"{'combo':<22}  {'mean|Δ|':>10}  {'median|Δ|':>10}  {'max|Δ|':>10}  {'mean Δ':>10}  {'wall':>8}")
    print("-" * 88)
    for label, _ in COMBOS:
        deltas = np.asarray(deltas_per_combo[label], dtype=np.float64)
        walls = np.asarray(walls_per_combo[label], dtype=np.float64)
        if deltas.size == 0:
            continue
        absd = np.abs(deltas)
        print(
            f"{label:<22}  {absd.mean():>10.4f}  {np.median(absd):>10.4f}  "
            f"{absd.max():>10.4f}  {deltas.mean():>+10.4f}  {walls.mean():>7.2f}s"
        )
    print("(units: Ha)")


if __name__ == "__main__":
    main()
