"""Benchmark: multi-molecule parallel MMFF optimisation on Metal.

Compares three strategies:
  1. RDKit native (sequential per molecule, multithreaded per conformer)
  2. Metal batch per molecule (one molecule at a time, batch conformers)
  3. Metal multi-molecule (ALL molecules + conformers in parallel)
"""
import os
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from mlxmolkit.mmff_metal_optimizer import (
    mmff_optimize_metal_batch,
    mmff_optimize_metal_fused_multi_mol,
    mmff_optimize_metal_multi_mol,
)

SMILES_10 = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "OC(=O)CC1=CC=CC=C1NC1=C(Cl)C=CC=C1Cl",  # Diclofenac
    "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen
    "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",  # Testosterone
    "OC(=O)C1=CC=CC=C1O",  # Salicylic acid
    "COC1=CC(=CC(=C1O)OC)C=O",  # Syringaldehyde
    "CC(=O)OCC=C",  # Allyl acetate
    "OC1=CC=CC2=CC=CC=C12",  # 1-Naphthol
]

SMILES_50 = SMILES_10 + [
    "CC(=O)OC1=CC=CC=C1",  # Phenyl acetate
    "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
    "CC(O)C1=CC=CC=C1",  # 1-Phenylethan-1-ol
    "OC1=CC=C(C=C1)C(O)=O",  # 4-Hydroxybenzoic
    "CC1=CC=C(C=C1)NC(=O)C",  # 4-Methylacetanilide
    "COC1=CC=C(C=C1)C(C)C(=O)O",  # 4-Methoxy-ibuprofen analog
    "CC1=CC(=CC=C1)C(=O)O",  # 3-Toluic acid
    "OC1=CC=CC=C1C=O",  # Salicylaldehyde
    "CC(=O)C1=CC=CC=C1",  # Acetophenone
    "C1CCC(CC1)C(=O)O",  # Cyclohexanecarboxylic
    "C1=CC=C(C=C1)CC(=O)O",  # Phenylacetic acid
    "CC1=CC=CC=C1O",  # 2-Methylphenol
    "OC(=O)C1CCCCC1",  # Cyclohexanecarboxylic
    "CC(C)(C)C1=CC=C(C=C1)O",  # 4-tert-Butylphenol
    "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene
    "CC1=CC2=CC=CC=C2C=C1",  # 2-Methylnaphthalene
    "OC1=CC=C2C=CC=CC2=C1",  # 2-Naphthol
    "CC(=O)NC1=CC=CC=C1",  # Acetanilide
    "COC1=CC=CC=C1O",  # Guaiacol
    "CC1=CC=CC=C1",  # Toluene
    "C1CCCCC1O",  # Cyclohexanol
    "CC(C)C1=CC=CC=C1",  # Isopropylbenzene
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",  # Citric acid
    "C1=CC=C(C=C1)N1C=CC=C1",  # 1-Phenylpyrrole
    "CC(=O)OC1CC2CCC1C2",  # Bornyl acetate
    "CC(C)=CCCC(C)=CCO",  # Geraniol
    "CC(C)=CCCC(C)=CC=O",  # Citral
    "CC1=CCC(CC1)C(C)=C",  # Limonene
    "C1CC2CCCCC2CC1",  # Decalin
    "OC(=O)C1CC1",  # Cyclopropanecarboxylic
    "CC1(C)CC1C(=O)O",  # Chrysanthemic acid
    "OC1=CC=C(C=C1)CC1=CC=C(O)C=C1",  # BPA
    "CC=CC(=O)O",  # Crotonic acid
    "C=CC(=O)O",  # Acrylic acid
    "CC(=O)OCC",  # Ethyl acetate
    "CCOC(=O)C1=CC=CC=C1",  # Ethyl benzoate
    "CC(C)CC(=O)O",  # Isovaleric acid
    "CCCCCC(=O)O",  # Hexanoic acid
    "CCCCCCCC(=O)O",  # Octanoic acid
    "OC(=O)C=CC1=CC=CC=C1",  # Cinnamic acid
]


def prepare_mols(smiles_list: list[str], n_conf: int) -> list[Chem.Mol]:
    """Embed conformers for each SMILES."""
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    mols = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        m = Chem.AddHs(m)
        cids = AllChem.EmbedMultipleConfs(m, numConfs=n_conf, params=params)
        if len(cids) == 0:
            continue
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(m)
        if props is None:
            continue
        mols.append(m)
    return mols


def bench_rdkit(mols: list[Chem.Mol], n_threads: int = 1):
    """RDKit native MMFF optimisation (sequential per molecule)."""
    copies = [Chem.RWMol(m) for m in mols]
    t0 = time.perf_counter()
    all_e = []
    for m in copies:
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            m, maxIters=200, numThreads=n_threads,
        )
        all_e.extend(e for (_, e) in results)
    dt = time.perf_counter() - t0
    return dt, np.array(all_e)


def bench_metal_per_mol(mols: list[Chem.Mol]):
    """Metal kernel: one mmff_optimize_metal_batch per molecule (sequential)."""
    copies = [Chem.RWMol(m) for m in mols]
    t0 = time.perf_counter()
    all_e = []
    for m in copies:
        r = mmff_optimize_metal_batch(m, max_iters=200)
        all_e.extend(r.energies.tolist())
    dt = time.perf_counter() - t0
    return dt, np.array(all_e)


def bench_metal_multi_mol(mols: list[Chem.Mol]):
    """Metal kernel: all molecules in parallel via mega-kernel + Python L-BFGS."""
    copies = [Chem.RWMol(m) for m in mols]
    specs = [(m, None) for m in copies]
    t0 = time.perf_counter()
    result = mmff_optimize_metal_multi_mol(specs, max_iters=200)
    dt = time.perf_counter() - t0
    all_e = []
    for mr in result.mol_results:
        all_e.extend(mr.energies.tolist())
    return dt, np.array(all_e)


def bench_metal_fused(
    mols: list[Chem.Mol], max_iters: int = 500, grad_lr: float = 0.005,
):
    """Metal: entire optimization in one kernel launch (zero Python overhead)."""
    copies = [Chem.RWMol(m) for m in mols]
    specs = [(m, None) for m in copies]
    t0 = time.perf_counter()
    result = mmff_optimize_metal_fused_multi_mol(
        specs, max_iters=max_iters, grad_lr=grad_lr, max_step=0.3,
    )
    dt = time.perf_counter() - t0
    all_e = []
    for mr in result.mol_results:
        all_e.extend(mr.energies.tolist())
    return dt, np.array(all_e)


def print_table(rows, ref_time):
    hdr = (
        f"{'Method':<35} {'Total(s)':>9} {'ms/conf':>9}"
        f" {'mean|dE|':>9} {'max|dE|':>9} {'speedup':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, dt, total_confs, e_arr, ref_e in rows:
        ms = dt / total_confs * 1000
        speedup = ref_time / dt if dt > 0 else float("inf")
        if ref_e is not None:
            diffs = np.abs(e_arr - ref_e)
            mdiff = np.mean(diffs)
            xdiff = np.max(diffs)
            print(
                f"{label:<35} {dt:9.2f} {ms:9.1f}"
                f" {mdiff:9.3f} {xdiff:9.3f} {speedup:8.2f}x"
            )
        else:
            print(
                f"{label:<35} {dt:9.2f} {ms:9.1f}"
                f" {'(ref)':>9} {'(ref)':>9} {'1.00x':>9}"
            )


def run_bench(smiles_list, n_conf, label):
    print(f"\n{'=' * 75}")
    print(f"  {label}: {len(smiles_list)} SMILES x {n_conf} conformers")
    print(f"{'=' * 75}")

    mols = prepare_mols(smiles_list, n_conf)
    total_confs = sum(m.GetNumConformers() for m in mols)
    atoms = [m.GetNumAtoms() for m in mols]
    print(
        f"  Prepared {len(mols)} molecules, {total_confs} conformers"
        f" (atoms: {min(atoms)}-{max(atoms)}, mean {np.mean(atoms):.0f})"
    )
    print()

    n_cpu = os.cpu_count() or 4

    # RDKit 1-thread baseline
    t_rdk1, e_rdk1 = bench_rdkit(mols, n_threads=1)
    # RDKit multi-thread
    t_rdkN, e_rdkN = bench_rdkit(mols, n_threads=n_cpu)
    # Metal mega-kernel + Python L-BFGS (warm-up + real)
    _, _ = bench_metal_multi_mol(mols)
    t_mm, e_mm = bench_metal_multi_mol(mols)
    # Metal FUSED BB optimizer (warm-up + real)
    _, _ = bench_metal_fused(mols, max_iters=200, grad_lr=0.01)
    t_f200, e_f200 = bench_metal_fused(mols, max_iters=200, grad_lr=0.01)
    _, _ = bench_metal_fused(mols, max_iters=500, grad_lr=0.005)
    t_f500, e_f500 = bench_metal_fused(mols, max_iters=500, grad_lr=0.005)

    rows = [
        ("RDKit native (1 thread)", t_rdk1, total_confs, e_rdk1, None),
        (f"RDKit native ({n_cpu} threads)", t_rdkN, total_confs, e_rdkN, e_rdk1),
        ("Mega-kernel + Python L-BFGS", t_mm, total_confs, e_mm, e_rdk1),
        ("FUSED BB 200it (1 launch)", t_f200, total_confs, e_f200, e_rdk1),
        ("FUSED BB 500it (1 launch)", t_f500, total_confs, e_f500, e_rdk1),
    ]
    print_table(rows, t_rdk1)
    return rows


def main():
    print("Multi-molecule parallel MMFF Metal benchmark")
    print(f"CPUs: {os.cpu_count()}\n")

    run_bench(SMILES_10, 10, "10 mols x 10 confs")
    run_bench(SMILES_10, 50, "10 mols x 50 confs")
    run_bench(SMILES_10, 100, "10 mols x 100 confs")
    run_bench(SMILES_50, 10, "50 mols x 10 confs")
    run_bench(SMILES_50, 50, "50 mols x 50 confs")


if __name__ == "__main__":
    main()
