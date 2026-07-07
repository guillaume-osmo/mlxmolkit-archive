#!/usr/bin/env python
"""Prepare CHEESE-style ESP/RESP charge-training labels.

Default label path:

    RDKit 3D molecule -> AM1-BCC charges -> Connolly ESP -> ESP/RESP fits

The result is one compact NPZ with variable-size molecular graph inputs and two
per-atom targets, ``q_esp`` and ``q_resp``. This is intended for training two
parallel geometric charge transformers, matching the public CHEESE description.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem

from tools.conformer_source import embed_molecule_3d

from mlxmolkit.am1bcc import am1_bcc_charges_from_rdkit_mol
from mlxmolkit.charge_model import bond_matrix_from_rdkit_mol
from mlxmolkit.charge_training_dataset import (
    CheeseChargeTrainingRecord,
    esp_resp_labels_from_reference_charges,
    failed_charge_training_record,
    write_cheese_charge_training_npz,
)
from mlxmolkit.esp_resp import pm6_esp_resp_charge_labels


def iter_input_molecules(
    path: Path,
    *,
    input_format: str = "auto",
    smiles_col: str = "SMILES",
    id_col: str | None = None,
) -> Iterable[tuple[str, str, Chem.Mol]]:
    """Yield ``(identifier, smiles, mol)`` from SMI/TXT/CSV/SDF input."""

    fmt = _infer_format(path, input_format)
    if fmt in {"smi", "txt"}:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as file:
            for row_index, line in enumerate(file):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                smiles = parts[0]
                identifier = parts[1] if len(parts) > 1 else str(row_index)
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    yield identifier, smiles, mol
        return

    if fmt == "csv":
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", newline="") as file:
            reader = csv.DictReader(file)
            if smiles_col not in (reader.fieldnames or []):
                raise ValueError(f"CSV has no SMILES column {smiles_col!r}")
            for row_index, row in enumerate(reader):
                smiles = row[smiles_col].strip()
                if not smiles:
                    continue
                identifier = row.get(id_col, "") if id_col else ""
                identifier = identifier or row.get("ID", "") or row.get("id", "") or str(row_index)
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    yield identifier, smiles, mol
        return

    if fmt == "sdf":
        handle = gzip.open(path, "rb") if str(path).endswith(".gz") else path.open("rb")
        try:
            supplier = Chem.ForwardSDMolSupplier(handle, removeHs=False)
            for row_index, mol in enumerate(supplier):
                if mol is None:
                    continue
                identifier = mol.GetProp("_Name") if mol.HasProp("_Name") else str(row_index)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                yield identifier, smiles, mol
        finally:
            handle.close()
        return

    raise ValueError(f"unsupported input format: {fmt}")


def prepare_molecule_3d(
    mol: Chem.Mol,
    *,
    add_hs: bool = True,
    optimize: bool = True,
    random_seed: int = 0xC0FFEE,
    max_iters: int = 200,
) -> Chem.Mol:
    """Return a conformer-bearing RDKit molecule suitable for charge labels.

    Thin wrapper over the shared CHEESE conformer source of truth so the charge
    cache, teacher ensembles, and LIT-PCBA eval all embed the same ETKDG version.
    """

    return embed_molecule_3d(
        mol,
        add_hs=add_hs,
        optimize=optimize,
        seed=random_seed,
        max_iters=max_iters,
    )


def charge_training_record_from_mol(
    identifier: str,
    smiles: str,
    mol: Chem.Mol,
    *,
    label_source: str = "am1bcc",
    am1_method: str = "AM1",
    allow_partial_bcc: bool = False,
    require_scf_convergence: bool = False,
    point_density: float = 1.0,
    shell_factors: Sequence[float] = (1.4, 1.6, 1.8, 2.0),
    restraint_a: float = 5.0e-4,
    restraint_b: float = 0.1,
    restrain_hydrogens: bool = False,
    resp_max_iter: int = 100,
    resp_conv_tol: float = 1.0e-6,
) -> CheeseChargeTrainingRecord:
    """Compute one dual-label record from a prepared RDKit molecule."""

    atoms = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    bonds = bond_matrix_from_rdkit_mol(mol)
    total_charge = float(Chem.GetFormalCharge(mol))

    try:
        if label_source == "am1bcc":
            am1bcc = am1_bcc_charges_from_rdkit_mol(
                mol,
                total_charge=total_charge,
                am1_method=am1_method,
                add_hs=False,
                validate_bcc_coverage=not allow_partial_bcc,
                require_scf_convergence=require_scf_convergence,
            )
            q_ref = np.asarray(am1bcc.charges, dtype=np.float32)
            esp, resp = esp_resp_labels_from_reference_charges(
                atoms,
                coords,
                q_ref,
                total_charge=total_charge,
                shell_factors=shell_factors,
                point_density=point_density,
                restraint_a=restraint_a,
                restraint_b=restraint_b,
                restrain_hydrogens=restrain_hydrogens,
                max_iter=resp_max_iter,
                conv_tol=resp_conv_tol,
            )
            metadata = {
                "am1_method": am1_method,
                "reference_charge_model": "AM1-BCC",
                **am1bcc.metadata,
            }
        elif label_source == "pm6-proxy":
            esp, resp = pm6_esp_resp_charge_labels(
                atoms,
                coords,
                total_charge=total_charge,
                method=am1_method,
                shell_factors=shell_factors,
                point_density=point_density,
            )
            q_ref = np.asarray(esp.charges, dtype=np.float32)
            metadata = {
                "semiempirical_method": am1_method,
                "reference_charge_model": f"{am1_method} ESP proxy",
            }
        else:
            raise ValueError(f"unsupported label source: {label_source}")

        return CheeseChargeTrainingRecord(
            identifier=str(identifier),
            smiles=str(smiles),
            atomic_numbers=atoms,
            coords=coords,
            bond_matrix=bonds,
            total_charge=total_charge,
            q_reference=q_ref,
            q_esp=np.asarray(esp.charges, dtype=np.float32),
            q_resp=np.asarray(resp.charges, dtype=np.float32),
            esp_rmse=float(esp.rmse),
            resp_rmse=float(resp.rmse),
            resp_converged=bool(resp.converged),
            n_grid=int(resp.n_grid),
            ok=True,
            metadata=metadata,
        )
    except Exception as exc:
        return failed_charge_training_record(
            identifier,
            smiles,
            atoms,
            coords,
            bonds,
            total_charge,
            exc,
            metadata={"label_source": label_source, "am1_method": am1_method},
        )


def write_manifest(records: Sequence[CheeseChargeTrainingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "id",
                "smiles",
                "n_atoms",
                "total_charge",
                "ok",
                "n_grid",
                "esp_rmse",
                "resp_rmse",
                "resp_converged",
                "error",
            ],
        )
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "index": index,
                    "id": record.identifier,
                    "smiles": record.smiles,
                    "n_atoms": int(record.atomic_numbers.shape[0]),
                    "total_charge": record.total_charge,
                    "ok": record.ok,
                    "n_grid": record.n_grid,
                    "esp_rmse": record.esp_rmse,
                    "resp_rmse": record.resp_rmse,
                    "resp_converged": record.resp_converged,
                    "error": record.error,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/cheese_charge_training.npz"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--input-format", choices=["auto", "smi", "txt", "csv", "sdf"], default="auto")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--label-source", choices=["am1bcc", "pm6-proxy"], default="am1bcc")
    parser.add_argument("--am1-method", choices=["AM1", "RM1", "PM3", "PM6", "PM6_SP", "PM6_D"], default="AM1")
    parser.add_argument("--max-mols", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--no-add-hs", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0xC0FFEE)
    parser.add_argument("--point-density", type=float, default=1.0)
    parser.add_argument("--shell-factors", default="1.4,1.6,1.8,2.0")
    parser.add_argument("--resp-a", type=float, default=5.0e-4)
    parser.add_argument("--resp-b", type=float, default=0.1)
    parser.add_argument("--resp-max-iter", type=int, default=100)
    parser.add_argument("--resp-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--restrain-hydrogens", action="store_true")
    parser.add_argument("--allow-partial-bcc", action="store_true")
    parser.add_argument("--require-scf-convergence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shell_factors = tuple(float(x) for x in args.shell_factors.split(",") if x.strip())
    records: list[CheeseChargeTrainingRecord] = []

    for source_index, (identifier, smiles, mol) in enumerate(
        iter_input_molecules(
            args.input,
            input_format=args.input_format,
            smiles_col=args.smiles_col,
            id_col=args.id_col,
        )
    ):
        if source_index < args.start:
            continue
        if args.max_mols is not None and len(records) >= args.max_mols:
            break
        try:
            prepared = prepare_molecule_3d(
                mol,
                add_hs=not args.no_add_hs,
                optimize=not args.no_optimize,
                random_seed=args.random_seed + source_index,
            )
            record = charge_training_record_from_mol(
                identifier,
                Chem.MolToSmiles(prepared, isomericSmiles=True),
                prepared,
                label_source=args.label_source,
                am1_method=args.am1_method,
                allow_partial_bcc=args.allow_partial_bcc,
                require_scf_convergence=args.require_scf_convergence,
                point_density=args.point_density,
                shell_factors=shell_factors,
                restraint_a=args.resp_a,
                restraint_b=args.resp_b,
                restrain_hydrogens=args.restrain_hydrogens,
                resp_max_iter=args.resp_max_iter,
                resp_conv_tol=args.resp_conv_tol,
            )
        except Exception as exc:
            fallback = Chem.MolFromSmiles(smiles)
            if fallback is None:
                continue
            fallback = Chem.AddHs(fallback)
            atoms = [atom.GetAtomicNum() for atom in fallback.GetAtoms()]
            coords = np.zeros((len(atoms), 3), dtype=np.float32)
            bonds = bond_matrix_from_rdkit_mol(fallback)
            record = failed_charge_training_record(
                identifier,
                smiles,
                atoms,
                coords,
                bonds,
                float(Chem.GetFormalCharge(fallback)),
                exc,
            )
        records.append(record)
        status = "ok" if record.ok else "failed"
        print(f"{len(records):6d} {status:6s} id={identifier} atoms={record.atomic_numbers.shape[0]}", flush=True)

    summary = write_cheese_charge_training_npz(
        records,
        args.out,
        label_source=args.label_source,
        metadata={
            "am1_method": args.am1_method,
            "point_density": args.point_density,
            "shell_factors": shell_factors,
            "resp_a": args.resp_a,
            "resp_b": args.resp_b,
            "resp_max_iter": args.resp_max_iter,
            "resp_conv_tol": args.resp_conv_tol,
            "restrain_hydrogens": args.restrain_hydrogens,
        },
    )
    manifest = args.manifest or args.out.with_suffix(".manifest.csv")
    write_manifest(records, manifest)
    print(json.dumps({**summary, "manifest": str(manifest)}, indent=2, sort_keys=True))


def _infer_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    suffixes = [suffix.lower() for suffix in path.suffixes]
    suffixes = [suffix for suffix in suffixes if suffix != ".gz"]
    if not suffixes:
        return "smi"
    suffix = suffixes[-1]
    if suffix == ".sdf":
        return "sdf"
    if suffix == ".csv":
        return "csv"
    if suffix in {".smi", ".smiles"}:
        return "smi"
    if suffix == ".txt":
        return "txt"
    raise ValueError(f"could not infer input format from {path}")


if __name__ == "__main__":
    main()
