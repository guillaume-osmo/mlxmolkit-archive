"""Does molecular complexity predict MMFF iteration count, and can we batch on it?

Two questions, one script:

  1. Is the number of L-BFGS iterations to convergence predictable from cheap
     2D descriptors?  (Yes: rotatable bonds r=0.81, rings *negatively*.)
  2. Can that prediction be used to sort molecules into difficulty-matched
     batches and win wall-clock?  (No: bucketing loses 2-3x.)

Run:  PYTHONPATH=. python experiments/iteration_count_vs_flexibility.py

Note on methodology: the optimizers mutate the RDKit molecules in place, so
every timed run rebuilds its inputs from SMILES.  Reusing molecules across
budgets silently feeds each run the previous run's optimised geometry and
produces iteration counts that *fall* as the budget rises.
"""
from __future__ import annotations

import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdDistGeom
from rdkit.Chem import rdMolDescriptors as rdmd

from mlxmolkit.mmff_batch_optimizer import mmff_optimize_molecules_batch
from mlxmolkit.mmff_metal_optimizer import mmff_optimize_metal_multi_mol

RDLogger.DisableLog("rdApp.*")

SMILES = [
    "CCO", "CC(C)O", "CCCCCC", "CCCCCCCCCC", "CCCCCCCCCCCCCCCC",
    "c1ccccc1", "Cc1ccccc1", "c1ccc2ccccc2c1",
    "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1O", "NC(Cc1ccccc1)C(=O)O", "CC(C)(C)c1ccc(O)cc1",
    "C1CCCCC1", "C1CCC2CCCCC2C1", "OCC(O)CO", "OCC(O)C(O)C(O)C(O)CO",
    "CCOC(=O)c1ccccc1", "CC1=CC(=O)CC(C)(C)C1", "CC(C)=CCCC(C)=CCO",
    "CC(C)=CCC/C(C)=C/CO", "O=C(O)CCCCCCCCCCCCCCC",
    "c1ccc(-c2ccccc2)cc1", "c1ccc(Oc2ccccc2)cc1", "CN1CCC[C@H]1c1cccnc1",
    "CC(N)C(=O)O", "NCCCCC(N)C(=O)O", "OC(=O)CCC(N)C(=O)O",
    "c1cnc2[nH]ccc2c1", "CCN(CC)CCOC(=O)c1ccccc1N", "COc1ccc(CC=C)cc1",
    "COc1cc(C=O)ccc1O", "O=Cc1ccccc1", "CSc1ccccc1", "Clc1ccccc1",
    "CC(C)CCOC(C)=O", "CCCCOC(=O)CCCCC", "O=C1CCCCCCCCCCCCCC1",
]

DESCRIPTORS = {
    "n_atoms_H": lambda m, mm: m.GetNumAtoms(),
    "heavy": lambda m, mm: mm.GetNumAtoms(),
    "rot_bonds": lambda m, mm: rdmd.CalcNumRotatableBonds(mm),
    "rings": lambda m, mm: rdmd.CalcNumRings(mm),
    "mol_wt": lambda m, mm: Descriptors.MolWt(mm),
    "tpsa": lambda m, mm: rdmd.CalcTPSA(mm),
    "frac_csp3": lambda m, mm: rdmd.CalcFractionCSP3(mm),
}


def build(seed: int = 42) -> list[Chem.Mol]:
    """Fresh molecules — never reuse these across timed runs (see module doc)."""
    mols = []
    for smi in SMILES:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if rdDistGeom.EmbedMultipleConfs(mol, 1, randomSeed=seed):
            mols.append(mol)
    return mols


def difficulty(mol: Chem.Mol) -> float:
    """Cheap 2D proxy for iteration count, from the correlations below."""
    heavy = Chem.RemoveHs(mol)
    return (
        mol.GetNumAtoms()
        + 8 * rdmd.CalcNumRotatableBonds(heavy)
        - 6 * rdmd.CalcNumRings(heavy)
    )


def correlate() -> None:
    mols = build()
    results = mmff_optimize_molecules_batch(
        mols, method="lbfgs", max_iters=3000, n_threads=8
    )
    iters = np.array([r.n_iters for r in results], dtype=float)

    print(f"n={len(mols)}  iterations: min {iters.min():.0f}  "
          f"median {np.median(iters):.0f}  max {iters.max():.0f}  "
          f"({iters.max() / max(iters.min(), 1):.0f}x spread)")
    print(f"\n{'descriptor':12s} {'pearson r':>10s}")
    for name, fn in DESCRIPTORS.items():
        vals = np.array(
            [fn(m, Chem.RemoveHs(m)) for m in mols], dtype=float
        )
        print(f"{name:12s} {np.corrcoef(vals, iters)[0, 1]:10.3f}")


def bucketing() -> None:
    """One big dispatch vs difficulty-sorted buckets, on the Metal path."""
    def run(groups):
        start = time.perf_counter()
        for group in groups:
            mmff_optimize_metal_multi_mol(
                [(m, None) for m in group], max_iters=1000
            )
        return (time.perf_counter() - start) * 1e3

    run([build()])  # warm

    n_buckets = 4
    per = len(build()) // n_buckets
    rng = np.random.default_rng(0)
    order = rng.permutation(len(build()))

    def buckets(key):
        pool = sorted(build(), key=key) if key else [
            build()[i] for i in order
        ]
        return [pool[i * per:(i + 1) * per] for i in range(n_buckets)]

    one = min(run([build()]) for _ in range(3))
    strategies = {
        "random": None,
        "by atom count": lambda m: m.GetNumAtoms(),
        "by difficulty": difficulty,
    }
    print(f"\n{'ONE dispatch of 40':28s} {one:8.1f} ms   1.00x")
    for label, key in strategies.items():
        t = min(run(buckets(key)) for _ in range(3))
        print(f"{n_buckets} buckets, {label:16s} {t:8.1f} ms   {one / t:.2f}x")


if __name__ == "__main__":
    correlate()
    bucketing()
