"""
Benchmark: Metal BFGS vs SciPy BFGS for 3D molecular distance geometry.

Measures per-molecule optimization time for increasing complexity.
"""
import time

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from scipy.optimize import minimize as scipy_minimize

from mlxmolkit.bfgs_metal import bfgs_minimize
from mlxmolkit.energy_distgeom import make_distgeom_energy_grad

MOLECULES = {
    "ethanol": "CCO",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "diazepam": "O=C1CN=C(C2=CC=CC=C2)C2=C(N1C)C=CC(=C2)Cl",
    "taxol_core": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
}


def setup_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    pairs, targets = [], []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            lb, ub = bounds[j][i], bounds[i][j]
            if 0 < ub < 100:
                pairs.append([i, j])
                targets.append((lb + ub) / 2.0)
    return n_atoms, np.array(pairs, dtype=np.int32), np.array(targets, dtype=np.float32)


def bench_metal(n_atoms, pairs, targets, seed=42):
    fn = make_distgeom_energy_grad(pairs, targets, n_atoms)
    np.random.seed(seed)
    x0 = mx.array(np.random.randn(n_atoms * 3).astype(np.float32) * 0.5)
    t0 = time.perf_counter()
    result = bfgs_minimize(x0, fn, max_iters=500, grad_tol=1e-4)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def bench_scipy(n_atoms, pairs, targets, seed=42):
    np.random.seed(seed)
    x0 = np.random.randn(n_atoms * 3) * 0.5

    def energy_grad(x):
        pos = x.reshape(-1, 3)
        e = 0.0
        g = np.zeros_like(pos)
        for k in range(len(targets)):
            i, j = pairs[k]
            diff = pos[i] - pos[j]
            d = np.linalg.norm(diff) + 1e-12
            dd = d - targets[k]
            e += dd * dd
            deriv = 2.0 * dd * diff / d
            g[i] += deriv
            g[j] -= deriv
        return e, g.flatten()

    t0 = time.perf_counter()
    res = scipy_minimize(
        lambda x: energy_grad(x)[0],
        x0,
        jac=lambda x: energy_grad(x)[1],
        method="BFGS",
        options={"maxiter": 500, "gtol": 1e-4},
    )
    elapsed = time.perf_counter() - t0
    return res, elapsed


def main():
    print("=" * 85)
    print(f"{'Molecule':<15} {'Atoms':>5} {'Pairs':>6} {'Dim':>5}  "
          f"{'Metal(s)':>8} {'SciPy(s)':>8} {'Metal E':>10} {'SciPy E':>10} {'Iters':>6}")
    print("=" * 85)

    for name, smiles in MOLECULES.items():
        n_atoms, pairs, targets = setup_molecule(smiles)
        dim = n_atoms * 3

        r_metal, t_metal = bench_metal(n_atoms, pairs, targets)
        r_scipy, t_scipy = bench_scipy(n_atoms, pairs, targets)

        print(
            f"{name:<15} {n_atoms:>5} {len(targets):>6} {dim:>5}  "
            f"{t_metal:>8.3f} {t_scipy:>8.3f} {r_metal.energy:>10.2f} "
            f"{r_scipy.fun:>10.2f} {r_metal.n_iters:>3}/{r_scipy.nit:<3}"
        )

    print("=" * 85)
    print("\nNote: Metal overhead dominates for single small molecules.")
    print("Real benefit comes from batched optimization (many molecules simultaneously).")


if __name__ == "__main__":
    main()
