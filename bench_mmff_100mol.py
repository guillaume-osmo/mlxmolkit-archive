"""Benchmark: 100 molecules x 10 conformers MMFF optimization.

Compares Metal BFGS / L-BFGS (sequential + batch) vs RDKit native.
"""
import os
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
from mlxmolkit.mmff_optimizer import mmff_optimize
from mlxmolkit.mmff_batch_optimizer import mmff_optimize_batch

SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C1=CC2=CC=CC=C2C=C1)C(=O)O",
    "OC(=O)CC1=CC=CC=C1NC1=C(Cl)C=CC=C1Cl", "CC(=O)NC1=CC=C(O)C=C1",
    "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O", "OC(=O)C1=CC=CC=C1O",
    "CC(=O)OC1=CC=CC=C1", "C1=CC=C(C=C1)C(=O)O",
    "CC(O)C1=CC=CC=C1", "OC1=CC=C(C=C1)C(O)=O",
    "CC1=CC=C(C=C1)NC(=O)C", "COC1=CC=C(C=C1)C(C)C(=O)O",
    "CC1=CC(=CC=C1)C(=O)O", "OC1=CC=CC=C1C=O",
    "CC(=O)C1=CC=CC=C1", "C1CCC(CC1)C(=O)O",
    "C1=CC=C(C=C1)CC(=O)O", "CC1=CC=CC=C1O",
    "OC(=O)C1CCCCC1", "CC(C)(C)C1=CC=C(C=C1)O",
    "C1=CC=C2C(=C1)C=CC=C2", "CC1=CC2=CC=CC=C2C=C1",
    "OC1=CC=C2C=CC=CC2=C1", "CC(=O)NC1=CC=CC=C1",
    "C1=CC=C(C=C1)NC(=O)CC(=O)O", "COC1=CC=CC=C1O",
    "CC1=CC=CC=C1", "C1CCCCC1O",
    "CC(C)C1=CC=CC=C1", "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    "C1=CC=C(C=C1)C(=O)CC(=O)C2=CC=CC=C2", "CC(=O)OCC",
    "CCOC(=O)C1=CC=CC=C1", "CC(C)CC(=O)O",
    "CCCCCC(=O)O", "CCCCCCCC(=O)O",
    "OC(=O)C=CC1=CC=CC=C1", "CC(=O)OC1=CC=C(C=C1)OC(=O)C",
    "OC1=C(O)C=CC=C1", "C1=CC=C(C=C1)C2=CC=CC=C2",
    "CC1=CC=CC=C1N", "NC1=CC=C(C=C1)O",
    "OC(=O)C1=CC=C(O)C(=C1)O", "CC(C)C(=O)OCC",
    "CCCC(=O)OC", "C1=CC=C(C=C1)CO",
    "OCC1=CC=CC=C1O", "CC1=CC=C(C=C1)O",
    "C1=CC=C(C=C1)C#N", "CC(=O)CC1=CC=CC=C1",
    "OC(=O)C1=CC=CC=C1N", "CC(=O)C1=CC=C(C=C1)O",
    "COC1=CC=C(C=C1)C=O", "CC(=O)C1=CC=C(C=C1)OC",
    "OC(=O)C1=CC=CC=C1Cl", "OC(=O)C1=CC=C(C=C1)Cl",
    "OC(=O)C1=CC=C(C=C1)F", "OC(=O)C1=CC=C(C=C1)Br",
    "CC(C)CO", "CCCCO",
    "CCCCCO", "CC(C)(C)O",
    "C1CCOC1", "C1CCOCC1",
    "CC1=CC=C(C=C1)C(=O)C", "OC(=O)CC1=CC=C(O)C=C1",
    "CC(O)CC1=CC=CC=C1", "CCOC(=O)CC(=O)OCC",
    "OC(=O)C(O)C(O)C(=O)O", "OC(=O)CCC(=O)O",
    "OC(=O)CCCC(=O)O", "OC(=O)CC(=O)O",
    "OC(=O)CCCCC(=O)O", "CC(=O)NCCC1=CNC2=CC=CC=C12",
    "OC(=O)C1=CC=NC=C1", "OC(=O)C1=CN=CC=C1",
    "C1=CN=CC=C1CO", "C1=CC=NC=C1",
    "C1=CC=C(C=C1)N1C=CC=C1", "CC1=CC=C(C=C1)S(=O)(=O)N",
    "CC(=O)OC1CC2CCC1C2", "CC(C)=CCCC(C)=CCO",
    "CC(C)=CCCC(C)=CC=O", "CC1=CCC(CC1)C(C)=C",
    "C1CC2CCCCC2CC1", "OC(=O)C1CC1",
    "CC1(C)CC1C(=O)O", "OC1=CC=C(C=C1)CC1=CC=C(O)C=C1",
    "CC(=O)OCC=C", "C=CC(=O)OC",
    "CC=CC(=O)O", "C=CC(=O)O",
    "OC(=O)C=CC=CC(=O)O", "CC(=O)NC1=CC=C(C=C1)OC(=O)C",
    "COC1=CC(=CC(=C1O)OC)C=O", "COC1=CC(=CC=C1O)C=O",
    "COC1=CC(=CC=C1O)CC=C", "OC1=CC=CC2=CC=CC=C12",
]

N_CONF = 10


def prepare_molecules():
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True

    mols = []
    for smi in SMILES[:100]:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        m = Chem.AddHs(m)
        cids = AllChem.EmbedMultipleConfs(m, numConfs=N_CONF, params=params)
        if len(cids) == 0:
            continue
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(m)
        if props is None:
            continue
        mols.append(m)
    return mols


def bench_rdkit_native(mols, num_threads=1):
    copies = [Chem.RWMol(m) for m in mols]
    t0 = time.perf_counter()
    all_e = []
    for m in copies:
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            m, maxIters=200, numThreads=num_threads,
        )
        all_e.extend(e for (_, e) in results)
    dt = time.perf_counter() - t0
    return dt, all_e


def bench_metal_sequential(mols, method, scale_grads, label):
    copies = [Chem.RWMol(m) for m in mols]
    t0 = time.perf_counter()
    all_e = []
    n_conv = 0
    tot_it = 0
    for m in copies:
        for c in m.GetConformers():
            r = mmff_optimize(m, conf_id=c.GetId(), method=method,
                              max_iters=200, scale_grads=scale_grads)
            all_e.append(r.energy)
            n_conv += r.converged
            tot_it += r.n_iters
    dt = time.perf_counter() - t0
    return dt, all_e, n_conv, tot_it


def bench_metal_batch(mols, method, scale_grads, n_threads, label):
    copies = [Chem.RWMol(m) for m in mols]
    t0 = time.perf_counter()
    all_e = []
    n_conv = 0
    tot_it = 0
    for m in copies:
        r = mmff_optimize_batch(
            m, method=method, max_iters=200,
            scale_grads=scale_grads, n_threads=n_threads,
        )
        all_e.extend(r.energies.tolist())
        n_conv += int(np.sum(r.converged))
        tot_it += int(np.sum(r.per_conf_iters))
    dt = time.perf_counter() - t0
    return dt, all_e, n_conv, tot_it


def main():
    n_cpu = os.cpu_count() or 4
    print("Preparing molecules ...")
    t0 = time.perf_counter()
    mols = prepare_molecules()
    embed_t = time.perf_counter() - t0
    total_confs = sum(m.GetNumConformers() for m in mols)
    atoms = [m.GetNumAtoms() for m in mols]
    print(f"  {len(mols)} molecules, {total_confs} conformers")
    print(f"  Atoms: min={min(atoms)} max={max(atoms)} mean={np.mean(atoms):.0f}")
    print(f"  CPUs: {n_cpu}")
    print(f"  Embed: {embed_t:.2f}s\n")

    # ── RDKit native (1 thread) ──
    print("RDKit native (1 thread) ...")
    t_rdk1, e_rdk = bench_rdkit_native(mols, num_threads=1)
    print(f"  {t_rdk1:.2f}s  ({t_rdk1/total_confs*1000:.1f} ms/conf)\n")

    # ── RDKit native (N threads) ──
    print(f"RDKit native ({n_cpu} threads) ...")
    t_rdkN, e_rdkN = bench_rdkit_native(mols, num_threads=n_cpu)
    print(f"  {t_rdkN:.2f}s  ({t_rdkN/total_confs*1000:.1f} ms/conf)\n")

    results = {}

    # ── Sequential Metal variants ──
    seq_configs = [
        ("lbfgs", True, "Seq L-BFGS sg=T"),
    ]
    for method, sg, label in seq_configs:
        print(f"{label} ...")
        dt, e_list, n_conv, tot_it = bench_metal_sequential(mols, method, sg, label)
        results[label] = (dt, e_list, n_conv, tot_it)
        print(f"  {dt:.2f}s  ({dt/total_confs*1000:.1f} ms/conf)"
              f"  conv={n_conv}/{total_confs}  avg_it={tot_it/total_confs:.0f}\n")

    # ── Batch Metal variants ──
    batch_configs = [
        ("lbfgs", True, 4, "Batch L-BFGS 4T sg=T"),
        ("lbfgs", True, n_cpu, f"Batch L-BFGS {n_cpu}T sg=T"),
        ("lbfgs", False, n_cpu, f"Batch L-BFGS {n_cpu}T sg=F"),
        ("bfgs", True, n_cpu, f"Batch BFGS {n_cpu}T sg=T"),
    ]
    for method, sg, nt, label in batch_configs:
        print(f"{label} ...")
        dt, e_list, n_conv, tot_it = bench_metal_batch(mols, method, sg, nt, label)
        results[label] = (dt, e_list, n_conv, tot_it)
        print(f"  {dt:.2f}s  ({dt/total_confs*1000:.1f} ms/conf)"
              f"  conv={n_conv}/{total_confs}  avg_it={tot_it/total_confs:.0f}\n")

    # ── Energy agreement ──
    print("=" * 80)
    print("ENERGY AGREEMENT vs RDKit native (1 thread)")
    print("=" * 80)
    for label in results:
        dt, e_list, n_conv, tot_it = results[label]
        diffs = [abs(a - b) for a, b in zip(e_list, e_rdk)]
        n01 = sum(1 for d in diffs if d < 0.1)
        n1 = sum(1 for d in diffs if d < 1.0)
        print(f"  {label:<28} mean|dE|={np.mean(diffs):.4f}"
              f"  max|dE|={max(diffs):.4f}"
              f"  <0.1: {n01}/{total_confs}"
              f"  <1.0: {n1}/{total_confs}")
    print()

    # ── Summary table ──
    print("=" * 80)
    print(f"SUMMARY: {len(mols)} mols x {N_CONF} confs = {total_confs} optimizations")
    print("=" * 80)
    print(f"{'Method':<28} {'Total(s)':>9} {'ms/conf':>9}"
          f" {'Conv%':>7} {'AvgIt':>6} {'vs RDKit1':>9} {'vs RDKitN':>9}")
    print("-" * 80)
    print(f"{'RDKit native (1T)':<28} {t_rdk1:9.2f} {t_rdk1/total_confs*1000:9.1f}"
          f" {'---':>7} {'---':>6} {'1.00x':>9} {t_rdk1/t_rdkN:8.2f}x")
    print(f"{'RDKit native (' + str(n_cpu) + 'T)':<28} {t_rdkN:9.2f}"
          f" {t_rdkN/total_confs*1000:9.1f}"
          f" {'---':>7} {'---':>6} {t_rdkN/t_rdk1:8.2f}x {'1.00x':>9}")
    for label in results:
        dt, _, n_conv, tot_it = results[label]
        ratio1 = dt / t_rdk1
        ratioN = dt / t_rdkN
        avg_it = tot_it / total_confs if total_confs > 0 else 0
        print(f"{label:<28} {dt:9.2f} {dt/total_confs*1000:9.1f}"
              f" {100*n_conv/total_confs:6.1f}% {avg_it:6.0f}"
              f" {ratio1:8.2f}x {ratioN:8.2f}x")


if __name__ == "__main__":
    main()
