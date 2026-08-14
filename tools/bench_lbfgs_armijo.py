"""Measure the MMFF GPU optimizers against RDKit's float64 MMFF.

Exists to quantify one specific defect: the Armijo test in the threadgroup
BFGS/L-BFGS kernels compares a trial energy computed by a 32-way tree reduction
against an accepted energy computed by a serial thread-0 loop. Differencing two
summation orders of the same function puts a float32 noise floor under the
sufficient-decrease test, which near convergence either accepts noise or burns
MAX_LS_ITERS into a null step.

Reports, per optimizer: energy gap to RDKit, convergence rate, and how often the
optimizer leaves the geometry worse than it found it.

    python tools/bench_lbfgs_armijo.py [--json out.json]
"""
import argparse
import json

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdForceFieldHelpers

from mlxmolkit.mmff_minimize import mmff_minimize_nk
from mlxmolkit.mmff_params import extract_mmff_params

RDLogger.DisableLog("rdApp.*")

SMILES = [
    ("benzene", "c1ccccc1"),
    ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("diazepam", "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("testosterone", "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O"),
    ("glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ("tyrosine", "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
    ("tryptophan", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
    ("penicillinG", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),
    ("morphine", "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5"),
    ("nicotine", "CN1CCCC1c1cccnc1"),
    ("paracetamol2", "CC(=O)Nc1ccccc1"),
    ("salicylic", "OC(=O)c1ccccc1O"),
    ("phenylalanine", "N[C@@H](Cc1ccccc1)C(=O)O"),
    ("dopamine", "NCCc1ccc(O)c(O)c1"),
    ("serotonin", "NCCc1c[nH]c2ccc(O)cc12"),
]

MAX_ITERS = 200
GRAD_TOL = 1e-4

# `converged` from mmff_minimize_nk is `status == 0`, which conflates three very
# different exits: a genuine gradient-norm convergence, a TOLX step-too-small
# bail-out, and slope >= 0 (an ascent direction, i.e. a failure). Raising the
# iteration budget separates "stalled early" from "still making progress".


def build(smiles, seed=0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(mol, p) != 0:
        return None
    return mol


def rdkit_energy(mol, optimize):
    m = Chem.Mol(mol)
    props = AllChem.MMFFGetMoleculeProperties(m)
    if props is None:
        return None
    if optimize:
        rdForceFieldHelpers.MMFFOptimizeMolecule(m, maxIters=2000)
        props = AllChem.MMFFGetMoleculeProperties(m)
    ff = AllChem.MMFFGetMoleculeForceField(m, props)
    return ff.CalcEnergy()


def run(mol, use_lbfgs, max_iters=MAX_ITERS):
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    params = extract_mmff_params(mol)
    _, energies, converged = mmff_minimize_nk(
        [params], [1], pos.reshape(-1),
        max_iters=max_iters, grad_tol=GRAD_TOL, use_lbfgs=use_lbfgs,
    )
    return float(energies[0]), bool(converged[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = []
    for name, smi in SMILES:
        mol = build(smi)
        if mol is None:
            continue
        e_start = rdkit_energy(mol, optimize=False)
        e_ref = rdkit_energy(mol, optimize=True)
        if e_start is None or e_ref is None:
            continue
        row = {"name": name, "e_start": e_start, "e_rdkit": e_ref}
        for tag, lb in (("bfgs", False), ("lbfgs", True)):
            try:
                e, conv = run(mol, lb, args.max_iters)
            except Exception as exc:  # noqa: BLE001
                row[tag] = {"error": str(exc)[:80]}
                continue
            row[tag] = {"e": e, "converged": conv, "gap": e - e_ref,
                        "worse_than_start": e > e_start + 1e-3}
        rows.append(row)

    print(f"[{args.label or 'run'}]  max_iters={args.max_iters}")
    print(f"{'molecule':<15} {'RDKit':>10} | {'BFGS':>10} {'gap':>9} {'cv':>3} | "
          f"{'L-BFGS':>10} {'gap':>9} {'cv':>3}")
    print("-" * 82)
    for r in rows:
        b, l = r.get("bfgs", {}), r.get("lbfgs", {})
        def fmt(d):
            if "e" not in d:
                return f"{'ERR':>10} {'-':>9} {'-':>3}"
            return f"{d['e']:10.3f} {d['gap']:9.3f} {'Y' if d['converged'] else 'N':>3}"
        print(f"{r['name']:<15} {r['e_rdkit']:10.3f} | {fmt(b)} | {fmt(l)}")

    print()
    for tag in ("bfgs", "lbfgs"):
        ok = [r[tag] for r in rows if "e" in r.get(tag, {})]
        if not ok:
            continue
        gaps = np.array([d["gap"] for d in ok])
        nconv = sum(d["converged"] for d in ok)
        nworse = sum(d["worse_than_start"] for d in ok)
        print(f"{tag:>6}: n={len(ok):2d}  converged={nconv:2d}/{len(ok):<2d}  "
              f"worse-than-start={nworse:2d}  "
              f"gap mean={gaps.mean():8.3f}  median={np.median(gaps):8.3f}  "
              f"max={gaps.max():8.3f}  kcal/mol")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
