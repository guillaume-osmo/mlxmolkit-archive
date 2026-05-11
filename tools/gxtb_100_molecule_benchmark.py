#!/usr/bin/env python3
"""Run a 100-molecule g-xTB energy-deviation sweep.

The harness builds deterministic RDKit 3D geometries from the local benchmark
SMILES file, evaluates the public ``xtb`` executable as an oracle, and compares
it with the reconstructed local g-xTB implementation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_ADDONS_SRC = Path.home() / "Github" / "mlx-addons" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if MLX_ADDONS_SRC.exists() and str(MLX_ADDONS_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_ADDONS_SRC))

from tools.gxtb_oracle_probe import MOLECULES, write_xyz  # noqa: E402


KCAL_PER_HA = 627.5094740631
ALLOWED_Z = {1, 6, 7, 8, 9, 16}
EXEC_METHOD_ARGS = {
    "gxtb": ("--gxtb",),
    "gfn2": ("--gfn", "2"),
}


@dataclass(frozen=True)
class MoleculeCase:
    name: str
    smiles: str
    atoms: list[int]
    coords: np.ndarray


def parse_total_energy(log: str) -> float:
    match = re.search(r"total energy\s+([-+0-9.Ee]+)\s+Eh", log)
    if not match:
        raise ValueError("could not parse total energy from xtb output")
    return float(match.group(1))


def embed_smiles(smiles: str, *, seed: int) -> tuple[list[int], np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol0)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"RDKit embedding failed: {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=300)

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


def smiles_passes_filter(smiles: str, *, max_atoms: int, allow_zwitterions: bool) -> bool:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return False
    if Chem.GetFormalCharge(mol) != 0:
        return False
    if not allow_zwitterions and any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return False
    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() > max_atoms:
        return False
    return all(atom.GetAtomicNum() in ALLOWED_Z for atom in mol_h.GetAtoms())


def iter_cases(
    input_csv: Path,
    *,
    n_success: int,
    max_atoms: int,
    seed: int,
    include_anchors: bool,
    allow_zwitterions: bool,
) -> list[MoleculeCase]:
    seen_smiles: set[str] = set()
    cases: list[MoleculeCase] = []

    if include_anchors:
        for name, item in MOLECULES.items():
            if isinstance(item, tuple):
                atoms, coords = item
                smiles = name
            else:
                smiles = item
                atoms, coords = embed_smiles(smiles, seed=seed)
                seen_smiles.add(smiles)
            cases.append(MoleculeCase(name=name, smiles=smiles, atoms=list(atoms), coords=np.asarray(coords)))

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            smiles = row["smiles"].strip()
            if not smiles or smiles in seen_smiles:
                continue
            if not smiles_passes_filter(smiles, max_atoms=max_atoms, allow_zwitterions=allow_zwitterions):
                continue
            try:
                atoms, coords = embed_smiles(smiles, seed=seed + idx)
            except Exception:
                continue
            seen_smiles.add(smiles)
            name = row.get("id") or row.get("name") or f"bm{idx:04d}"
            cases.append(MoleculeCase(name=name, smiles=smiles, atoms=atoms, coords=coords))
            if len(cases) >= max(n_success * 2, n_success + 25):
                break
    return cases


def run_xtb_energy(
    case: MoleculeCase,
    *,
    xtb: Path,
    method: str,
    acc: float,
    charge: int,
    uhf: int,
) -> tuple[float, float]:
    with tempfile.TemporaryDirectory(prefix=f"gxtb100-{method}-") as tmp:
        cwd = Path(tmp)
        xyz = cwd / "mol.xyz"
        write_xyz(xyz, case.atoms, case.coords)
        env = os.environ.copy()
        libdir = str(xtb.parent.parent / "lib")
        bindir = str(xtb.parent)
        env["DYLD_LIBRARY_PATH"] = f"{libdir}:{bindir}:{env.get('DYLD_LIBRARY_PATH', '')}"
        cmd = [
            str(xtb),
            str(xyz.name),
            *EXEC_METHOD_ARGS[method],
            "--acc",
            str(acc),
            "--chrg",
            str(charge),
            "--uhf",
            str(uhf),
        ]
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)
        wall = time.perf_counter() - start
        log = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"xtb {method} failed with code {proc.returncode}\n{log[-3000:]}")
        return parse_total_energy(log), wall


def run_local_gxtb(
    case: MoleculeCase,
    *,
    charge: int,
    conv_tol: float,
    max_iter: int,
    local_d4srev: bool,
    local_pacp_proxy: bool,
    use_first_order_offsite: bool = False,
    use_twobody_third_order: bool = False,
    use_third_order: bool = False,
    use_fourth_order: bool = False,
    use_exchange: bool = False,
) -> tuple[dict[str, object], float]:
    from mlxmolkit.xtb.scf_gxtb import gxtb_energy

    start = time.perf_counter()
    res = gxtb_energy(
        case.atoms,
        case.coords,
        charge=charge,
        conv_tol=conv_tol,
        max_iter=max_iter,
        use_d4srev=local_d4srev,
        use_pacp=local_pacp_proxy,
        use_first_order=True,
        use_first_order_offsite=use_first_order_offsite,
        use_mfx_exchange=True,
        use_third_order=use_third_order,
        use_twobody_third_order=use_twobody_third_order,
        use_fourth_order=use_fourth_order,
        use_exchange=use_exchange,
        use_acp_hamiltonian=True,
    )
    return res, time.perf_counter() - start


def run_local_gfn2_fast(
    case: MoleculeCase,
    *,
    charge: int,
    conv_tol: float,
    max_iter: int,
) -> tuple[dict[str, object], float]:
    from mlxmolkit.xtb.scf_gfn2_fast import gfn2_energy_fast

    start = time.perf_counter()
    res = gfn2_energy_fast(
        case.atoms,
        case.coords,
        charge=charge,
        conv_tol=conv_tol,
        max_iter=max_iter,
    )
    return res, time.perf_counter() - start


def _blank_row(case: MoleculeCase, attempt: int) -> dict[str, object]:
    formula = "".join(str(z) for z in case.atoms)
    return {
        "attempt": attempt,
        "name": case.name,
        "smiles": case.smiles,
        "natoms": len(case.atoms),
        "atom_z_compact": formula,
        "status": "ok",
        "error": "",
    }


def _add_delta(row: dict[str, object], prefix: str, local_e: float, ref_e: float) -> None:
    de = local_e - ref_e
    row[f"{prefix}_delta_ha"] = de
    row[f"{prefix}_delta_kcal_mol"] = de * KCAL_PER_HA
    row[f"{prefix}_abs_delta_ha"] = abs(de)
    row[f"{prefix}_abs_delta_kcal_mol"] = abs(de) * KCAL_PER_HA


def summarize(rows: list[dict[str, object]], delta_key: str, time_keys: list[str]) -> None:
    ok = [row for row in rows if row.get("status") == "ok" and row.get(delta_key) not in ("", None)]
    if not ok:
        print(f"No successful rows for {delta_key}")
        return
    deltas = np.array([float(row[delta_key]) for row in ok], dtype=np.float64)
    abs_deltas = np.abs(deltas)
    print(f"\nSummary for {delta_key} ({len(ok)} successful comparisons)")
    print(f"  mean signed: {np.mean(deltas):+.6e} Ha ({np.mean(deltas) * KCAL_PER_HA:+.3f} kcal/mol)")
    print(f"  mean abs:    {np.mean(abs_deltas):.6e} Ha ({np.mean(abs_deltas) * KCAL_PER_HA:.3f} kcal/mol)")
    print(f"  median abs:  {np.median(abs_deltas):.6e} Ha ({np.median(abs_deltas) * KCAL_PER_HA:.3f} kcal/mol)")
    print(f"  RMSE:        {math.sqrt(float(np.mean(deltas * deltas))):.6e} Ha")
    print(f"  p90 abs:     {np.percentile(abs_deltas, 90):.6e} Ha")
    print(f"  max abs:     {np.max(abs_deltas):.6e} Ha")
    for key in time_keys:
        vals = [float(row[key]) for row in ok if row.get(key) not in ("", None)]
        if vals:
            print(f"  median {key}: {median(vals):.4f} s")
    print("  top |delta|:")
    for row in sorted(ok, key=lambda r: abs(float(r[delta_key])), reverse=True)[:10]:
        print(
            f"    {row['attempt']:>3} {row['name']:<12} nat={row['natoms']:>2} "
            f"delta={float(row[delta_key]):+.6e} Ha  smiles={row['smiles']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "benchmark_1000_smiles.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "gxtb_100_molecule_deviation.csv")
    parser.add_argument("--xtb", type=Path, default=Path("/tmp/gxtb-v2-macos/bin/xtb"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--uhf", type=int, default=0)
    parser.add_argument("--conv-tol", type=float, default=1e-7)
    parser.add_argument("--max-iter", type=int, default=240)
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument(
        "--allow-zwitterions",
        action="store_true",
        help="allow formal charge-separated molecules whose total charge is zero",
    )
    parser.add_argument("--no-local-d4srev", action="store_true")
    parser.add_argument("--local-pacp-proxy", action="store_true")
    parser.add_argument("--include-gfn2-exec", action="store_true")
    parser.add_argument("--include-local-gfn2-fast", action="store_true")
    parser.add_argument("--local-first-order-offsite", action="store_true",
                        help="enable the offsite (xvec) first-order TB term in scf_gxtb")
    parser.add_argument("--local-twobody-third-order", action="store_true",
                        help="enable the two-body third-order TB term in scf_gxtb")
    parser.add_argument("--local-third-order", action="store_true",
                        help="enable the diagonal onsite third-order term in scf_gxtb")
    parser.add_argument("--local-fourth-order", action="store_true",
                        help="enable the diagonal onsite fourth-order term in scf_gxtb")
    parser.add_argument("--local-exchange", action="store_true",
                        help="enable the diagonal shell-exchange proxy in scf_gxtb")
    args = parser.parse_args()

    cases = iter_cases(
        args.input,
        n_success=args.n,
        max_atoms=args.max_atoms,
        seed=args.seed,
        include_anchors=not args.no_anchors,
        allow_zwitterions=args.allow_zwitterions,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    success = 0
    print(f"Prepared {len(cases)} candidate geometries; target successful comparisons: {args.n}", flush=True)
    print(f"Writing CSV to {args.out}", flush=True)

    for attempt, case in enumerate(cases, start=1):
        if success >= args.n:
            break
        row = _blank_row(case, attempt)
        try:
            e_xtb_gxtb, t_xtb_gxtb = run_xtb_energy(
                case,
                xtb=args.xtb,
                method="gxtb",
                acc=args.acc,
                charge=args.charge,
                uhf=args.uhf,
            )
            row["xtb_gxtb_energy_ha"] = e_xtb_gxtb
            row["xtb_gxtb_wall_s"] = t_xtb_gxtb

            local_gxtb, t_local_gxtb = run_local_gxtb(
                case,
                charge=args.charge,
                conv_tol=args.conv_tol,
                max_iter=args.max_iter,
                local_d4srev=not args.no_local_d4srev,
                local_pacp_proxy=args.local_pacp_proxy,
                use_first_order_offsite=args.local_first_order_offsite,
                use_twobody_third_order=args.local_twobody_third_order,
                use_third_order=args.local_third_order,
                use_fourth_order=args.local_fourth_order,
                use_exchange=args.local_exchange,
            )
            row["local_gxtb_energy_ha"] = float(local_gxtb["energy_hartree"])
            row["local_gxtb_energy_plus_increment_ha"] = float(local_gxtb["energy_plus_increment_hartree"])
            row["local_gxtb_wall_s"] = t_local_gxtb
            row["local_gxtb_converged"] = bool(local_gxtb["converged"])
            row["local_gxtb_n_iter"] = int(local_gxtb["n_iter"])
            row["local_gxtb_n_basis"] = int(local_gxtb["n_basis"])
            row["local_gxtb_h0_ha"] = float(local_gxtb["h0_hartree"])
            row["local_gxtb_first_order_ha"] = float(local_gxtb["first_order_hartree"])
            row["local_gxtb_coulomb_ha"] = float(local_gxtb["coulomb_hartree"])
            row["local_gxtb_mfx_ha"] = float(local_gxtb["mfx_exchange_hartree"])
            row["local_gxtb_acp_hamiltonian_ha"] = float(local_gxtb["acp_hamiltonian_hartree"])
            row["local_gxtb_repulsion_ha"] = float(local_gxtb["repulsion_hartree"])
            row["local_gxtb_dispersion_ha"] = float(local_gxtb["dispersion_hartree"])
            row["local_gxtb_raw_increment_ha"] = float(local_gxtb["raw_increment_hartree"])
            row["local_gxtb_halide_increment_correction_ha"] = float(local_gxtb["halide_increment_correction_hartree"])
            row["local_gxtb_increment_ha"] = float(local_gxtb["increment_hartree"])
            _add_delta(row, "local_gxtb_vs_xtb_gxtb", float(row["local_gxtb_energy_plus_increment_ha"]), e_xtb_gxtb)

            if args.include_gfn2_exec:
                e_xtb_gfn2, t_xtb_gfn2 = run_xtb_energy(
                    case,
                    xtb=args.xtb,
                    method="gfn2",
                    acc=args.acc,
                    charge=args.charge,
                    uhf=args.uhf,
                )
                row["xtb_gfn2_energy_ha"] = e_xtb_gfn2
                row["xtb_gfn2_wall_s"] = t_xtb_gfn2
                _add_delta(row, "xtb_gxtb_vs_xtb_gfn2", e_xtb_gxtb, e_xtb_gfn2)

            if args.include_local_gfn2_fast:
                local_gfn2, t_local_gfn2 = run_local_gfn2_fast(
                    case,
                    charge=args.charge,
                    conv_tol=args.conv_tol,
                    max_iter=args.max_iter,
                )
                row["local_gfn2_fast_energy_ha"] = float(local_gfn2["energy_hartree"])
                row["local_gfn2_fast_wall_s"] = t_local_gfn2
                row["local_gfn2_fast_converged"] = bool(local_gfn2["converged"])
                row["local_gfn2_fast_n_iter"] = int(local_gfn2["n_iter"])
                if args.include_gfn2_exec:
                    _add_delta(row, "local_gfn2_fast_vs_xtb_gfn2", float(row["local_gfn2_fast_energy_ha"]), float(row["xtb_gfn2_energy_ha"]))

            success += 1
            print(
                f"[{success:3d}/{args.n}] {case.name:<12} nat={len(case.atoms):>2} "
                f"delta(local-gxtb)={float(row['local_gxtb_vs_xtb_gxtb_delta_ha']):+.4e} Ha",
                flush=True,
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc).replace("\n", " | ")[:3000]
            print(f"[fail] {case.name:<12} nat={len(case.atoms):>2} {row['error'][:220]}", flush=True)
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} attempted rows with {success} successful comparisons to {args.out}")
    summarize(
        rows,
        "local_gxtb_vs_xtb_gxtb_delta_ha",
        ["xtb_gxtb_wall_s", "local_gxtb_wall_s"],
    )
    if args.include_gfn2_exec:
        summarize(rows, "xtb_gxtb_vs_xtb_gfn2_delta_ha", ["xtb_gxtb_wall_s", "xtb_gfn2_wall_s"])
    if args.include_local_gfn2_fast and args.include_gfn2_exec:
        summarize(rows, "local_gfn2_fast_vs_xtb_gfn2_delta_ha", ["xtb_gfn2_wall_s", "local_gfn2_fast_wall_s"])


if __name__ == "__main__":
    main()
