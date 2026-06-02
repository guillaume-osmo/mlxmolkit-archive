#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mlxmolkit.dipole_features import export_dipole_atom_feature_dataset


DEMO_ROWS = [
    {"name": "water", "smiles": "O", "dipole_debye": "1.8546"},
    {"name": "methanol", "smiles": "CO", "dipole_debye": "1.70"},
    {"name": "ethanol", "smiles": "CCO", "dipole_debye": "1.69"},
    {"name": "methyl_iodide", "smiles": "CI", "dipole_debye": "1.64"},
]


def _write_demo_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "smiles", "dipole_debye"])
        writer.writeheader()
        writer.writerows(DEMO_ROWS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export Chemprop-style atom feature NPZ files for dipole moment data. "
            "The input CSV must contain a SMILES column; target columns are preserved."
        )
    )
    parser.add_argument("--input", type=Path, help="input CSV with a SMILES column")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for exported artifacts")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name")
    parser.add_argument(
        "--target-cols",
        default="",
        help="comma-separated target columns to record in metadata, e.g. dipole_debye",
    )
    parser.add_argument("--seed", type=int, default=42, help="RDKit ETKDG seed")
    parser.add_argument("--max-rows", type=int, default=None, help="optional row limit")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print progress every N input rows",
    )
    parser.add_argument(
        "--explicit-h",
        action="store_true",
        help=(
            "export explicit-H atom rows for custom GNNs. Leave off for standard "
            "Chemprop SMILES graphs, where implicit hydrogens are folded onto heavy atoms."
        ),
    )
    parser.add_argument("--strict", action="store_true", help="raise on the first failed molecule")
    parser.add_argument(
        "--no-graph-tensors",
        action="store_true",
        help="skip graph_tensors.npz; useful for full Chemprop atom-feature exports",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="write and export a four-molecule demo set: water, methanol, ethanol, methyl iodide",
    )
    args = parser.parse_args()

    input_csv = args.input
    if args.demo:
        input_csv = args.out_dir / "dipole_demo_input.csv"
        _write_demo_csv(input_csv)
    if input_csv is None:
        parser.error("--input is required unless --demo is used")

    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    metadata = export_dipole_atom_feature_dataset(
        input_csv,
        args.out_dir,
        smiles_col=args.smiles_col,
        target_cols=target_cols,
        seed=args.seed,
        include_h=args.explicit_h,
        max_rows=args.max_rows,
        strict=args.strict,
        write_graph_tensors=not args.no_graph_tensors,
        progress_every=args.progress_every,
    )
    print(f"Exported {metadata['n_exported']} molecules to {args.out_dir}")
    print(f"  CSV:              {args.out_dir / 'dipole_chemprop.csv'}")
    print(f"  atom features:    {args.out_dir / 'atom_features.npz'}")
    print(f"  molecule features:{args.out_dir / 'molecule_features.npz'}")
    if not args.no_graph_tensors:
        print(f"  graph tensors:    {args.out_dir / 'graph_tensors.npz'}")
    print(f"  metadata:         {args.out_dir / 'metadata.json'}")
    if metadata["n_failed"]:
        print(f"  failures:         {metadata['n_failed']}")


if __name__ == "__main__":
    main()
