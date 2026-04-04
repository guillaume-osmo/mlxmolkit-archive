#!/usr/bin/env python
"""
3D Conformer Generation Example — mlxmolkit

Generates 3D conformers for drug-like molecules on Apple Silicon GPU.
Full pipeline: DG (4D) → ETK (3D torsion) → MMFF94 optimization.

Usage:
    python examples/conf3d_example.py
    python examples/conf3d_example.py --n-mols 100 --n-confs 20 --mmff
    python examples/conf3d_example.py --variant srETKDGv3 --mmff --mmff-variant MMFF94s
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="3D conformer generation with mlxmolkit")
    parser.add_argument("--n-mols", type=int, default=20, help="Number of molecules (default: 20)")
    parser.add_argument("--n-confs", type=int, default=10, help="Conformers per molecule (default: 10)")
    parser.add_argument("--batch-size", type=int, default=500, help="Max conformers per GPU batch (default: 500)")
    parser.add_argument("--variant", type=str, default="ETKDGv2",
                        choices=["DG", "KDG", "ETDG", "ETKDG", "ETKDGv2", "ETKDGv3", "srETKDGv3"],
                        help="ETKDG variant (default: ETKDGv2)")
    parser.add_argument("--mmff", action="store_true", help="Run MMFF94 optimization after DG+ETK")
    parser.add_argument("--mmff-variant", type=str, default="MMFF94", choices=["MMFF94", "MMFF94s"],
                        help="MMFF force field variant (default: MMFF94)")
    parser.add_argument("--mmff-lbfgs", action="store_true", help="Use L-BFGS for MMFF (less memory, slower)")
    parser.add_argument("--smiles", type=str, nargs="+", help="Custom SMILES (overrides --n-mols)")
    args = parser.parse_args()

    # ---- Build SMILES list ----
    if args.smiles:
        smiles_list = args.smiles
    else:
        # Drug-like molecules of varying complexity
        drug_smiles = [
            "c1ccccc1",                             # benzene
            "CC(=O)O",                              # acetic acid
            "CCO",                                  # ethanol
            "c1ccncc1",                             # pyridine
            "CC(=O)Oc1ccccc1C(=O)O",               # aspirin
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",          # ibuprofen
            "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",     # pyrene
            "OC(=O)c1ccccc1O",                      # salicylic acid
            "c1ccc(cc1)C(=O)Nc2ccccc2",             # benzanilide
            "CC(=O)Nc1ccc(O)cc1",                   # acetaminophen
            "C1CCCCC1",                             # cyclohexane
            "c1ccc(cc1)O",                          # phenol
            "CC(C)O",                               # isopropanol
            "c1ccoc1",                              # furan
            "c1ccsc1",                              # thiophene
            "CC#N",                                 # acetonitrile
            "c1ccc(cc1)N",                          # aniline
            "CC(=O)N",                              # acetamide
            "CCOC(=O)C",                            # ethyl acetate
            "c1ccc(cc1)F",                          # fluorobenzene
            "CC(=O)OC",                             # methyl acetate
            "C1CCNCC1",                             # piperidine
            "CCC(=O)O",                             # propionic acid
            "CCOCC",                                # diethyl ether
            "c1ccc(cc1)Cl",                         # chlorobenzene
            "CC=O",                                 # acetaldehyde
            "CCCC",                                 # butane
            "CC(C)C",                               # isobutane
            "c1ccc(cc1)C(=O)O",                     # benzoic acid
            "Oc1ccc(cc1)O",                         # hydroquinone
        ]
        smiles_list = drug_smiles[:args.n_mols]
        if args.n_mols > len(drug_smiles):
            # Repeat to fill
            smiles_list = (drug_smiles * ((args.n_mols // len(drug_smiles)) + 1))[:args.n_mols]

    N = len(smiles_list)
    k = args.n_confs
    C = N * k

    print(f"mlxmolkit 3D Conformer Generation")
    print(f"=" * 50)
    print(f"Molecules:        {N}")
    print(f"Conformers/mol:   {k}")
    print(f"Total conformers: {C}")
    print(f"Variant:          {args.variant}")
    print(f"MMFF:             {'Yes (' + args.mmff_variant + ')' if args.mmff else 'No'}")
    print(f"Batch size:       {args.batch_size}")
    print()

    # ---- Import and run ----
    sys.path.insert(0, ".")
    from mlxmolkit.conformer_pipeline_v2 import generate_conformers_nk

    t0 = time.time()
    result = generate_conformers_nk(
        smiles_list=smiles_list,
        n_confs_per_mol=k,
        variant=args.variant,
        run_mmff=args.mmff,
        mmff_variant=args.mmff_variant,
        mmff_use_lbfgs=args.mmff_lbfgs,
        max_confs_per_batch=args.batch_size,
    )
    t_total = time.time() - t0

    # ---- Results ----
    total_conv = sum(sum(m.converged) for m in result.molecules)
    print(f"Results")
    print(f"-" * 50)
    print(f"Time:             {t_total:.2f}s")
    print(f"Throughput:       {C / t_total:.0f} conformers/s")
    print(f"Convergence:      {total_conv}/{C} ({total_conv / C * 100:.1f}%)")
    print(f"Batches:          {result.n_batches}")
    print()

    # Per-molecule summary
    print(f"{'Mol':>4s}  {'Atoms':>5s}  {'Conv':>6s}  {'E_mean':>10s}  SMILES")
    print(f"{'-' * 4}  {'-' * 5}  {'-' * 6}  {'-' * 10}  {'-' * 30}")
    for i, mol in enumerate(result.molecules):
        conv = sum(mol.converged)
        e_mean = np.mean(mol.energies) if mol.energies else 0
        smi = smiles_list[i][:30]
        print(f"{i:4d}  {mol.n_atoms:5d}  {conv:3d}/{len(mol.positions_3d):<2d}  {e_mean:10.2f}  {smi}")

    # Verify 3D coordinates
    n_valid = 0
    for mol in result.molecules:
        for pos in mol.positions_3d:
            if pos.shape[1] == 3 and np.max(np.abs(pos)) > 0.01:
                n_valid += 1
    print()
    print(f"Valid 3D conformers: {n_valid}/{C}")


if __name__ == "__main__":
    main()
