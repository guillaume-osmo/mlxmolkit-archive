"""Tests for BFGS, L-BFGS, and batched optimizers on Metal."""
import numpy as np
import mlx.core as mx
import pytest
from scipy.optimize import minimize as scipy_minimize

from mlxmolkit.bfgs_metal import (
    bfgs_minimize, lbfgs_minimize,
    _set_hessian_identity, _matvec_neg,
)
from mlxmolkit.bfgs_batch_metal import (
    lbfgs_minimize_batch, bfgs_minimize_batch,
    make_batch_distgeom_energy_grad,
)
from mlxmolkit.energy_distgeom import (
    make_distgeom_energy_grad,
    make_rosenbrock_energy_grad,
)


class TestMetalKernels:
    """Unit tests for individual Metal kernels."""

    def test_hessian_identity(self):
        for dim in [2, 5, 16, 64]:
            H = _set_hessian_identity(dim)
            mx.eval(H)
            H_np = np.array(H).reshape(dim, dim)
            np.testing.assert_allclose(H_np, np.eye(dim), atol=1e-6)

    def test_matvec_neg(self):
        dim = 8
        H = mx.array(np.eye(dim, dtype=np.float32).flatten())
        g = mx.array(np.ones(dim, dtype=np.float32))
        d = _matvec_neg(H, g, dim)
        mx.eval(d)
        np.testing.assert_allclose(np.array(d), -np.ones(dim), atol=1e-6)


class TestRosenbrock:
    """Test BFGS convergence on Rosenbrock function."""

    def test_converges_to_minimum(self):
        fn = make_rosenbrock_energy_grad()
        x0 = mx.array([-1.0, 1.0], dtype=mx.float32)
        result = bfgs_minimize(x0, fn, max_iters=500, grad_tol=1e-5)
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-3)
        assert result.energy < 1e-4

    def test_different_starting_points(self):
        fn = make_rosenbrock_energy_grad()
        for x_start in [[-2.0, 2.0], [0.0, 0.0], [3.0, -1.0]]:
            x0 = mx.array(x_start, dtype=mx.float32)
            result = bfgs_minimize(x0, fn, max_iters=500, grad_tol=1e-4)
            assert result.converged, f"Failed from {x_start}"
            np.testing.assert_allclose(result.x, [1.0, 1.0], atol=5e-2)


class TestDistanceGeometry:
    """Test BFGS on distance geometry energy."""

    def test_gradient_correctness(self):
        """Verify Metal gradient matches numerical gradient."""
        import mlxmolkit.energy_distgeom as edg
        edg._dg_kernel = None

        n_atoms = 4
        pairs = np.array([[0, 1], [0, 2], [1, 2], [2, 3]], dtype=np.int32)
        targets = np.array([1.5, 2.0, 1.3, 1.8], dtype=np.float32)
        fn = make_distgeom_energy_grad(pairs, targets, n_atoms)

        np.random.seed(42)
        x0 = np.random.randn(n_atoms * 3).astype(np.float32) * 0.5

        e, g = fn(mx.array(x0))
        mx.eval(e, g)
        g_metal = np.array(g)

        eps = 1e-3
        g_num = np.zeros_like(x0)
        for i in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += eps
            xm[i] -= eps
            ep, _ = fn(mx.array(xp))
            em, _ = fn(mx.array(xm))
            mx.eval(ep, em)
            g_num[i] = (float(ep.item()) - float(em.item())) / (2 * eps)

        np.testing.assert_allclose(g_metal, g_num, atol=1e-2, rtol=1e-2)

    def test_pentagon_reconstruction(self):
        """Reconstruct a regular pentagon from distance bounds."""
        n_atoms = 5
        angles = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
        target_coords = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros(n_atoms)], axis=1
        )
        pairs, targets = [], []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                pairs.append([i, j])
                targets.append(np.linalg.norm(target_coords[i] - target_coords[j]))
        pairs = np.array(pairs, dtype=np.int32)
        targets = np.array(targets, dtype=np.float32)
        fn = make_distgeom_energy_grad(pairs, targets, n_atoms)

        np.random.seed(42)
        x0 = mx.array(np.random.randn(n_atoms * 3).astype(np.float32) * 0.5)
        result = bfgs_minimize(x0, fn, max_iters=500, grad_tol=1e-6)

        assert result.converged
        assert result.energy < 1e-4

        final = result.x.reshape(n_atoms, 3)
        for k, (i, j) in enumerate(pairs):
            d = np.linalg.norm(final[i] - final[j])
            assert abs(d - targets[k]) < 0.01, (
                f"Pair ({i},{j}): target={targets[k]:.4f}, got={d:.4f}"
            )

    def test_matches_scipy(self):
        """Metal BFGS should converge to same energy as scipy BFGS."""
        n_atoms = 6
        pairs_list = [
            [0, 1], [0, 2], [1, 2], [1, 3], [2, 3],
            [3, 4], [4, 5], [3, 5], [0, 5],
        ]
        np.random.seed(7)
        target_vals = 1.0 + np.random.rand(len(pairs_list)).astype(np.float32)
        pairs = np.array(pairs_list, dtype=np.int32)

        fn = make_distgeom_energy_grad(pairs, target_vals, n_atoms)
        np.random.seed(42)
        x0 = np.random.randn(n_atoms * 3).astype(np.float32) * 0.5
        result = bfgs_minimize(mx.array(x0), fn, max_iters=500, grad_tol=1e-5)

        def scipy_energy(x):
            pos = x.reshape(-1, 3)
            e = 0.0
            for k, (i, j) in enumerate(pairs_list):
                d = np.linalg.norm(pos[i] - pos[j]) + 1e-12
                e += (d - target_vals[k]) ** 2
            return e

        res = scipy_minimize(
            scipy_energy, x0.astype(np.float64),
            method="BFGS", options={"maxiter": 500, "gtol": 1e-5},
        )
        np.testing.assert_allclose(result.energy, res.fun, atol=0.1)


class TestBFGSRDKit:
    """Test with RDKit molecules (requires rdkit)."""

    @pytest.fixture
    def aspirin_setup(self):
        from rdkit import Chem
        from rdkit.Chem import rdDistGeom

        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
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
        return {
            "n_atoms": n_atoms,
            "pairs": np.array(pairs, dtype=np.int32),
            "targets": np.array(targets, dtype=np.float32),
        }

    def test_aspirin_converges(self, aspirin_setup):
        fn = make_distgeom_energy_grad(
            aspirin_setup["pairs"],
            aspirin_setup["targets"],
            aspirin_setup["n_atoms"],
        )
        np.random.seed(42)
        x0 = np.random.randn(aspirin_setup["n_atoms"] * 3).astype(np.float32) * 0.5
        result = bfgs_minimize(mx.array(x0), fn, max_iters=500, grad_tol=1e-3)
        assert result.converged
        assert result.n_iters < 200


class TestLBFGS:
    """Test L-BFGS optimizer."""

    def test_rosenbrock(self):
        fn = make_rosenbrock_energy_grad()
        x0 = mx.array([-1.0, 1.0], dtype=mx.float32)
        result = lbfgs_minimize(x0, fn, max_iters=1000, grad_tol=1e-5)
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-2)

    def test_pentagon_lbfgs(self):
        n_atoms = 5
        angles = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
        target_coords = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros(n_atoms)], axis=1
        )
        pairs, targets = [], []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                pairs.append([i, j])
                targets.append(np.linalg.norm(target_coords[i] - target_coords[j]))
        pairs = np.array(pairs, dtype=np.int32)
        targets = np.array(targets, dtype=np.float32)
        fn = make_distgeom_energy_grad(pairs, targets, n_atoms)

        np.random.seed(42)
        x0 = mx.array(np.random.randn(n_atoms * 3).astype(np.float32) * 0.5)
        result = lbfgs_minimize(x0, fn, max_iters=500, grad_tol=1e-6, m=10)
        assert result.converged
        assert result.energy < 1e-4

    def test_lbfgs_matches_bfgs_energy(self):
        """L-BFGS should reach similar energy as full BFGS."""
        n_atoms = 6
        pairs_list = [
            [0, 1], [0, 2], [1, 2], [1, 3], [2, 3],
            [3, 4], [4, 5], [3, 5], [0, 5],
        ]
        np.random.seed(7)
        target_vals = 1.0 + np.random.rand(len(pairs_list)).astype(np.float32)
        pairs = np.array(pairs_list, dtype=np.int32)
        fn = make_distgeom_energy_grad(pairs, target_vals, n_atoms)

        np.random.seed(42)
        x0 = np.random.randn(n_atoms * 3).astype(np.float32) * 0.5
        x0_mx = mx.array(x0)

        rb = bfgs_minimize(x0_mx, fn, max_iters=500, grad_tol=1e-5)
        rl = lbfgs_minimize(x0_mx, fn, max_iters=500, grad_tol=1e-5, m=10)
        np.testing.assert_allclose(rl.energy, rb.energy, atol=1.0)


class TestBatchedDGKernel:
    """Test the batched distance geometry energy/gradient kernel."""

    def test_batch_gradient_correctness(self):
        """Batched kernel should produce correct gradients."""
        n_atoms_list = [3, 4]
        all_pairs = [
            np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
            np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
        ]
        all_targets = [
            np.array([1.5, 2.0, 1.3], dtype=np.float32),
            np.array([1.0, 1.5, 2.0], dtype=np.float32),
        ]

        batch_fn, atom_starts, total_atoms, n_mols = make_batch_distgeom_energy_grad(
            all_pairs, all_targets, n_atoms_list,
        )

        np.random.seed(42)
        x0 = np.random.randn(total_atoms * 3).astype(np.float32) * 0.5
        x0_mx = mx.array(x0)

        ep, g = batch_fn(x0_mx)
        mx.eval(ep, g)
        g_metal = np.array(g)

        # Numerical gradient
        eps = 1e-3
        g_num = np.zeros_like(x0)
        for i in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += eps
            xm[i] -= eps
            ep_p, _ = batch_fn(mx.array(xp))
            ep_m, _ = batch_fn(mx.array(xm))
            mx.eval(ep_p, ep_m)
            e_p = float(mx.sum(ep_p).item())
            e_m = float(mx.sum(ep_m).item())
            g_num[i] = (e_p - e_m) / (2 * eps)

        np.testing.assert_allclose(g_metal, g_num, atol=1e-2, rtol=1e-2)


class TestBatchedOptimizers:
    """Test batched BFGS and L-BFGS optimizers."""

    @pytest.fixture
    def small_batch_setup(self):
        """3 small molecules for batch testing."""
        n_atoms_list = [4, 5, 4]
        all_pairs = [
            np.array([[0, 1], [0, 2], [1, 2], [2, 3], [0, 3], [1, 3]], dtype=np.int32),
            np.array(
                [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [0, 4], [2, 4], [1, 4], [0, 3]],
                dtype=np.int32,
            ),
            np.array([[0, 1], [0, 2], [1, 2], [2, 3], [0, 3], [1, 3]], dtype=np.int32),
        ]
        np.random.seed(7)
        all_targets = [
            1.0 + np.random.rand(6).astype(np.float32),
            1.0 + np.random.rand(10).astype(np.float32),
            1.0 + np.random.rand(6).astype(np.float32),
        ]
        return all_pairs, all_targets, n_atoms_list

    def test_batched_lbfgs_converges(self, small_batch_setup):
        all_pairs, all_targets, n_atoms_list = small_batch_setup
        result = lbfgs_minimize_batch(
            all_pairs, all_targets, n_atoms_list,
            max_iters=500, grad_tol=1e-4,
        )
        assert result.n_molecules == 3
        assert np.all(result.converged)
        assert np.all(result.energies < 10.0)

    def test_batched_bfgs_converges(self, small_batch_setup):
        all_pairs, all_targets, n_atoms_list = small_batch_setup
        result = bfgs_minimize_batch(
            all_pairs, all_targets, n_atoms_list,
            max_iters=500, grad_tol=1e-4,
        )
        assert result.n_molecules == 3
        assert np.all(result.converged)

    def test_batch_matches_sequential(self, small_batch_setup):
        """Batched energies should be close to sequential single-molecule results."""
        all_pairs, all_targets, n_atoms_list = small_batch_setup

        batch_result = lbfgs_minimize_batch(
            all_pairs, all_targets, n_atoms_list,
            max_iters=500, grad_tol=1e-4, seed=42,
        )

        for i in range(len(n_atoms_list)):
            fn = make_distgeom_energy_grad(
                all_pairs[i], all_targets[i], n_atoms_list[i],
            )
            np.random.seed(42)
            x0 = mx.array(
                np.random.randn(n_atoms_list[i] * 3).astype(np.float32) * 0.5
            )
            single = lbfgs_minimize(x0, fn, max_iters=500, grad_tol=1e-4, m=10)
            np.testing.assert_allclose(
                batch_result.energies[i], single.energy, atol=5.0,
            )

    def test_batched_with_rdkit(self):
        """Batched optimization with RDKit molecules."""
        from rdkit import Chem
        from rdkit.Chem import rdDistGeom

        smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
        all_pairs, all_targets, atom_counts = [], [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            n = mol.GetNumAtoms()
            atom_counts.append(n)
            bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
            pairs, targets = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    lb, ub = bounds[j][i], bounds[i][j]
                    if 0 < ub < 100:
                        pairs.append([i, j])
                        targets.append((lb + ub) / 2.0)
            all_pairs.append(np.array(pairs, dtype=np.int32))
            all_targets.append(np.array(targets, dtype=np.float32))

        result = lbfgs_minimize_batch(
            all_pairs, all_targets, atom_counts,
            max_iters=300, grad_tol=1e-3,
        )
        assert result.n_molecules == 3
        assert np.all(result.converged)


class TestMMFFOptimizer:
    """Test MMFF optimization matching RDKit native results."""

    def test_aspirin_mmff_bfgs(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
        from mlxmolkit.mmff_optimizer import mmff_optimize

        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)

        mol_ref = Chem.RWMol(mol)
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol_ref, maxIters=200)
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_ref)
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol_ref, props)
        e_rdkit = ff.CalcEnergy()

        r = mmff_optimize(mol, method="bfgs", max_iters=200, scale_grads=True)
        assert r.converged
        np.testing.assert_allclose(r.energy, e_rdkit, atol=0.01)

    def test_aspirin_mmff_lbfgs(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
        from mlxmolkit.mmff_optimizer import mmff_optimize

        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)

        r = mmff_optimize(mol, method="lbfgs", max_iters=200, scale_grads=True)
        assert r.converged
        assert r.energy < 20.0

    def test_ibuprofen_mmff(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
        from mlxmolkit.mmff_optimizer import mmff_optimize

        mol = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)

        mol_ref = Chem.RWMol(mol)
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol_ref, maxIters=200)
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_ref)
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol_ref, props)
        e_rdkit = ff.CalcEnergy()

        r = mmff_optimize(mol, method="bfgs", max_iters=200, scale_grads=True)
        assert r.converged
        np.testing.assert_allclose(r.energy, e_rdkit, atol=0.1)


class TestBatchMMFFOptimizer:
    """Test batched MMFF optimization with thread-parallel energy/gradient."""

    @pytest.fixture
    def aspirin_10confs(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDistGeom

        mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
        return mol

    def test_batch_lbfgs_converges(self, aspirin_10confs):
        from mlxmolkit.mmff_batch_optimizer import mmff_optimize_batch

        r = mmff_optimize_batch(
            aspirin_10confs, method="lbfgs", max_iters=500,
            grad_tol=1e-3, scale_grads=True, n_threads=4,
        )
        assert r.n_conformers == 10
        assert np.sum(r.converged) >= 8
        assert np.all(r.energies < 100.0)

    def test_batch_bfgs_converges(self, aspirin_10confs):
        from mlxmolkit.mmff_batch_optimizer import mmff_optimize_batch

        r = mmff_optimize_batch(
            aspirin_10confs, method="bfgs", max_iters=200,
            scale_grads=True, n_threads=4,
        )
        assert r.n_conformers == 10
        assert np.sum(r.converged) >= 8

    def test_batch_matches_sequential(self, aspirin_10confs):
        """Batch energies should be close to sequential mmff_optimize."""
        from rdkit import Chem
        from mlxmolkit.mmff_optimizer import mmff_optimize
        from mlxmolkit.mmff_batch_optimizer import mmff_optimize_batch

        mol_seq = Chem.RWMol(aspirin_10confs)
        seq_energies = []
        for conf in mol_seq.GetConformers():
            r = mmff_optimize(mol_seq, conf_id=conf.GetId(),
                              method="lbfgs", max_iters=200, scale_grads=True)
            seq_energies.append(r.energy)

        mol_batch = Chem.RWMol(aspirin_10confs)
        batch_r = mmff_optimize_batch(
            mol_batch, method="lbfgs", max_iters=200,
            scale_grads=True, n_threads=4,
        )

        for k in range(len(seq_energies)):
            np.testing.assert_allclose(
                batch_r.energies[k], seq_energies[k], atol=0.5,
                err_msg=f"Conformer {k}: batch={batch_r.energies[k]:.4f} "
                        f"seq={seq_energies[k]:.4f}",
            )

    def test_batch_matches_rdkit_native(self, aspirin_10confs):
        """Batch energies should be close to RDKit native."""
        from rdkit import Chem
        from rdkit.Chem import rdForceFieldHelpers
        from mlxmolkit.mmff_batch_optimizer import mmff_optimize_batch

        mol_rdk = Chem.RWMol(aspirin_10confs)
        rdk_results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            mol_rdk, maxIters=200, numThreads=1,
        )
        rdk_energies = [e for (_, e) in rdk_results]

        mol_batch = Chem.RWMol(aspirin_10confs)
        batch_r = mmff_optimize_batch(
            mol_batch, method="lbfgs", max_iters=200,
            scale_grads=True, n_threads=4,
        )

        for k in range(len(rdk_energies)):
            np.testing.assert_allclose(
                batch_r.energies[k], rdk_energies[k], atol=1.0,
                err_msg=f"Conformer {k}: batch={batch_r.energies[k]:.4f} "
                        f"rdkit={rdk_energies[k]:.4f}",
            )

    def test_molecules_batch(self):
        """Test mmff_optimize_molecules_batch with multiple molecules."""
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDistGeom
        from mlxmolkit.mmff_batch_optimizer import mmff_optimize_molecules_batch

        smiles_list = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        ]
        mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            AllChem.EmbedMultipleConfs(mol, numConfs=3, params=params)
            mols.append(mol)

        results = mmff_optimize_molecules_batch(
            mols, method="lbfgs", max_iters=200, n_threads=4,
        )
        assert len(results) == 3
        for r in results:
            assert r.n_conformers == 3
            assert np.sum(r.converged) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
