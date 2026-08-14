"""Cost/benefit curve for the MMFF iteration budget.

Answers "why 200 and not 500?" by measuring accuracy against RDKit's float64 MMFF
and wall time at several budgets, so the default can be picked from the knee of
the curve rather than by convention.

Wall time is measured on the whole batch in one kernel dispatch, which is the
number that matters: each molecule occupies its own threadgroup and stops on its
own status flag, but the dispatch does not return until the slowest one is done.

    python tools/sweep_mmff_iters.py
"""
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdForceFieldHelpers

from mlxmolkit.mmff_minimize import mmff_minimize_nk
from mlxmolkit.mmff_params import extract_mmff_params

RDLogger.DisableLog("rdApp.*")

from tools.bench_lbfgs_armijo import SMILES  # noqa: E402

BUDGETS = [100, 200, 500, 1000, 2000]
GRAD_TOL = 1e-4
REPEATS = 5


def prepare():
    mols, params, refs = [], [], []
    for _, smi in SMILES:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = 0xC0FFEE
        if AllChem.EmbedMolecule(m, p) != 0:
            continue
        r = Chem.Mol(m)
        rdForceFieldHelpers.MMFFOptimizeMolecule(r, maxIters=2000)
        props = AllChem.MMFFGetMoleculeProperties(r)
        if props is None:
            continue
        refs.append(AllChem.MMFFGetMoleculeForceField(r, props).CalcEnergy())
        mols.append(m)
        params.append(extract_mmff_params(m))
    pos = np.concatenate([
        np.asarray(m.GetConformer().GetPositions(), dtype=np.float32).reshape(-1)
        for m in mols])
    return params, pos, np.array(refs), len(mols)


def main():
    params, pos, refs, n = prepare()
    print(f"{n} molecules, one batched dispatch per timing\n")
    print(f"{'method':<7} {'iters':>6} {'ms/batch':>9} {'vs 200':>7} | "
          f"{'mean gap':>9} {'median':>8} {'max':>8} | {'converged':>9}")
    print("-" * 78)

    for use_lbfgs in (False, True):
        tag = "lbfgs" if use_lbfgs else "bfgs"
        base_t = None
        for it in BUDGETS:
            kw = dict(max_iters=it, grad_tol=GRAD_TOL, use_lbfgs=use_lbfgs)
            for _ in range(2):
                mmff_minimize_nk(params, [1] * n, pos, **kw)
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                _, e, c = mmff_minimize_nk(params, [1] * n, pos, **kw)
            ms = (time.perf_counter() - t0) / REPEATS * 1e3
            if it == 200:
                base_t = ms
            gaps = np.asarray(e) - refs
            rel = f"{ms / base_t:6.2f}x" if base_t else "     -"
            print(f"{tag:<7} {it:>6} {ms:9.2f} {rel:>7} | {gaps.mean():9.3f} "
                  f"{np.median(gaps):8.3f} {gaps.max():8.3f} | {int(np.sum(c)):>4}/{n}")
        print()


if __name__ == "__main__":
    main()
