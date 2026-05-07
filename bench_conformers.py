#!/usr/bin/env python3
"""
Head-to-head benchmark: Metal ETKDG vs RDKit CPU ETKDG.

Compares:
  1. RDKit EmbedMultipleConfs (CPU ETKDG)
  2. mlxmolkit generate_conformers (Metal GPU ETKDG)

Metrics: wall time, conformers/sec, success rate.
"""
import time
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from mlxmolkit import generate_conformers

BENCHMARK_MOLECULES = [
    ("ethanol", "CCO"),
    ("benzene", "c1ccccc1"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine", "Cn1c(=O)c2c(ncn2C)n(c1=O)C"),
    ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("diazepam", "CN1C(=O)CN=C(c2ccccc21)c3ccc(Cl)cc3"),
    ("celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),
]

N_CONFS = 10


def bench_rdkit(smiles_list, n_confs, n_warmup=1, n_repeat=3):
    """Benchmark RDKit EmbedMultipleConfs."""
    # Warmup
    for _ in range(n_warmup):
        for _, smi in smiles_list[:2]:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=42,
                                       useExpTorsionAnglePrefs=True,
                                       useBasicKnowledge=True)

    results = {}
    total_time = 0.0
    total_confs = 0

    for name, smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)

        times = []
        n_success = 0
        for rep in range(n_repeat):
            t0 = time.perf_counter()
            cids = AllChem.EmbedMultipleConfs(
                mol, numConfs=n_confs, randomSeed=42 + rep,
                useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)
            n_success = len(cids)

        avg_t = np.mean(times)
        total_time += avg_t
        total_confs += n_success
        results[name] = {
            "time": avg_t,
            "success": n_success,
            "confs_per_sec": n_success / avg_t if avg_t > 0 else 0,
        }

    return results, total_time, total_confs


def bench_metal(smiles_list, n_confs, n_warmup=1, n_repeat=3):
    """Benchmark Metal ETKDG pipeline."""
    smi_list = [smi for _, smi in smiles_list]

    # Warmup
    for _ in range(n_warmup):
        generate_conformers(smi_list[:2], n_confs=2, max_attempts=1)

    results = {}
    total_time = 0.0
    total_confs = 0

    for name, smi in smiles_list:
        times = []
        n_success = 0
        for rep in range(n_repeat):
            t0 = time.perf_counter()
            res = generate_conformers([smi], n_confs=n_confs,
                                       max_attempts=3, seed=42 + rep)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            n_success = res.results[0].n_generated

        avg_t = np.mean(times)
        total_time += avg_t
        total_confs += n_success
        results[name] = {
            "time": avg_t,
            "success": n_success,
            "confs_per_sec": n_success / avg_t if avg_t > 0 else 0,
        }

    return results, total_time, total_confs


def bench_metal_batched(smiles_list, n_confs, n_warmup=1, n_repeat=3):
    """Benchmark Metal ETKDG pipeline — all molecules batched together."""
    smi_list = [smi for _, smi in smiles_list]

    # Warmup
    for _ in range(n_warmup):
        generate_conformers(smi_list[:2], n_confs=2, max_attempts=1)

    times = []
    n_success_total = 0
    for rep in range(n_repeat):
        t0 = time.perf_counter()
        res = generate_conformers(smi_list, n_confs=n_confs,
                                   max_attempts=3, seed=42 + rep)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        n_success_total = res.total_conformers

    avg_t = np.mean(times)
    return avg_t, n_success_total


def main():
    print("=" * 70)
    print("Head-to-head: Metal ETKDG vs RDKit CPU ETKDG")
    print(f"Molecules: {len(BENCHMARK_MOLECULES)}, Conformers/mol: {N_CONFS}")
    print("=" * 70)

    print("\n--- RDKit CPU ETKDG ---")
    rdkit_results, rdkit_total, rdkit_confs = bench_rdkit(
        BENCHMARK_MOLECULES, N_CONFS, n_warmup=1, n_repeat=3,
    )
    for name, r in rdkit_results.items():
        print(f"  {name:15s}: {r['time']:.4f}s  {r['success']:2d}/{N_CONFS} confs  "
              f"({r['confs_per_sec']:.0f} confs/s)")
    print(f"  {'TOTAL':15s}: {rdkit_total:.4f}s  {rdkit_confs} confs")

    print("\n--- Metal ETKDG (per-molecule) ---")
    metal_results, metal_total, metal_confs = bench_metal(
        BENCHMARK_MOLECULES, N_CONFS, n_warmup=1, n_repeat=3,
    )
    for name, r in metal_results.items():
        print(f"  {name:15s}: {r['time']:.4f}s  {r['success']:2d}/{N_CONFS} confs  "
              f"({r['confs_per_sec']:.0f} confs/s)")
    print(f"  {'TOTAL':15s}: {metal_total:.4f}s  {metal_confs} confs")

    print("\n--- Metal ETKDG (all molecules batched) ---")
    batched_time, batched_confs = bench_metal_batched(
        BENCHMARK_MOLECULES, N_CONFS, n_warmup=1, n_repeat=3,
    )
    print(f"  All {len(BENCHMARK_MOLECULES)} molecules × {N_CONFS} confs: "
          f"{batched_time:.4f}s  {batched_confs} confs")

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  RDKit CPU:     {rdkit_total:.4f}s total, {rdkit_confs} confs")
    print(f"  Metal per-mol: {metal_total:.4f}s total, {metal_confs} confs")
    print(f"  Metal batched: {batched_time:.4f}s total, {batched_confs} confs")
    if metal_total > 0:
        print(f"  Metal/RDKit (per-mol): {rdkit_total/metal_total:.2f}x")
    if batched_time > 0:
        print(f"  Metal/RDKit (batched): {rdkit_total/batched_time:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
