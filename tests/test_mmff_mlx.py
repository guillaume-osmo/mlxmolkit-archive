"""Tests for MLX Metal MMFF energy, gradient, and batched optimizer."""
import numpy as np
import mlx.core as mx
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from mlxmolkit.mmff_params import extract_mmff_params
from mlxmolkit.mmff_energy_vectorized import mmff_energy_grad_batch
from mlxmolkit.mmff_energy_mlx import params_to_mlx, mmff_energy_batch, make_energy_and_grad_fn
from mlxmolkit.mmff_metal_kernel import pack_params_for_metal, mmff_energy_grad_metal
from mlxmolkit.mmff_mlx_optimizer import mmff_optimize_mlx_batch
from mlxmolkit.mmff_metal_optimizer import mmff_optimize_metal_batch

ETKDG = rdDistGeom.ETKDGv3()
ETKDG.randomSeed = 42


def _make_mol(smiles: str, n_confs: int = 1) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=ETKDG)
    return mol


def _get_positions(mol: Chem.Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    C = mol.GetNumConformers()
    pos = np.zeros((C, n, 3), dtype=np.float64)
    for k in range(C):
        conf = mol.GetConformer(k)
        for i in range(n):
            p = conf.GetAtomPosition(i)
            pos[k, i] = [p.x, p.y, p.z]
    return pos


class TestMLXEnergyMatchesNumPy:
    """Validate MLX energy matches NumPy vectorized reference."""

    @pytest.mark.parametrize(
        "smiles,name",
        [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),
            ("CCO", "ethanol"),
        ],
    )
    def test_energy_parity(self, smiles, name):
        mol = _make_mol(smiles, n_confs=5)
        pos = _get_positions(mol)

        np_params = extract_mmff_params(mol, conf_id=0)
        mlx_params = params_to_mlx(np_params)

        e_np, _ = mmff_energy_grad_batch(np_params, pos)
        e_mlx = mmff_energy_batch(mlx_params, mx.array(pos.astype(np.float32)))
        mx.eval(e_mlx)

        np.testing.assert_allclose(
            np.array(e_mlx),
            e_np.astype(np.float32),
            rtol=1e-5,
            err_msg=f"{name}: MLX energy doesn't match NumPy reference",
        )


class TestMLXAutoGrad:
    """Validate MLX auto-diff gradients match numerical and NumPy analytic."""

    def test_gradient_matches_numpy_analytic(self):
        mol = _make_mol("CC(=O)OC1=CC=CC=C1C(=O)O", n_confs=3)
        pos = _get_positions(mol)

        np_params = extract_mmff_params(mol, conf_id=0)
        mlx_params = params_to_mlx(np_params)

        _, g_np = mmff_energy_grad_batch(np_params, pos)
        energy_and_grad = make_energy_and_grad_fn(mlx_params)
        _, g_mlx = energy_and_grad(mx.array(pos.astype(np.float32)))
        mx.eval(g_mlx)

        g_np32 = g_np.astype(np.float32)
        g_mlx_flat = np.array(g_mlx).reshape(3, -1)

        for k in range(3):
            cos_sim = np.dot(g_np32[k], g_mlx_flat[k]) / (
                np.linalg.norm(g_np32[k]) * np.linalg.norm(g_mlx_flat[k]) + 1e-30
            )
            assert cos_sim > 0.9999, (
                f"Conf {k}: cosine similarity {cos_sim:.6f} too low"
            )

    def test_gradient_matches_numerical(self):
        mol = _make_mol("CCO", n_confs=1)
        pos = _get_positions(mol)

        np_params = extract_mmff_params(mol, conf_id=0)
        mlx_params = params_to_mlx(np_params)

        n = mol.GetNumAtoms()
        pos_mlx = mx.array(pos.astype(np.float32))

        energy_and_grad = make_energy_and_grad_fn(mlx_params)
        _, g_ad = energy_and_grad(pos_mlx)
        mx.eval(g_ad)
        g_ad_np = np.array(g_ad)

        eps = 1e-3
        g_num = np.zeros_like(g_ad_np)
        for i in range(n):
            for d in range(3):
                pp = np.array(pos_mlx, copy=True)
                pm = np.array(pos_mlx, copy=True)
                pp[0, i, d] += eps
                pm[0, i, d] -= eps
                ep = float(mx.sum(mmff_energy_batch(mlx_params, mx.array(pp))))
                em = float(mx.sum(mmff_energy_batch(mlx_params, mx.array(pm))))
                g_num[0, i, d] = (ep - em) / (2 * eps)

        np.testing.assert_allclose(
            g_ad_np[0], g_num[0], atol=0.05, rtol=0.02,
            err_msg="Auto-diff gradient doesn't match numerical gradient",
        )


class TestMLXBatchOptimizer:
    """Test the MLX Metal batched optimizer."""

    def test_aspirin_converges(self):
        mol = _make_mol("CC(=O)OC1=CC=CC=C1C(=O)O", n_confs=10)
        mol_work = Chem.RWMol(mol)
        r = mmff_optimize_mlx_batch(mol_work, max_iters=200)
        assert r.n_conformers == 10
        assert np.all(r.energies < 100.0)
        assert np.all(np.isfinite(r.energies))

    def test_close_to_rdkit(self):
        mol = _make_mol("CC(=O)OC1=CC=CC=C1C(=O)O", n_confs=10)

        mol_rdk = Chem.RWMol(mol)
        rdk_res = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            mol_rdk, maxIters=200, numThreads=1
        )
        rdk_e = np.array([r[1] for r in rdk_res])

        mol_mlx = Chem.RWMol(mol)
        mlx_r = mmff_optimize_mlx_batch(mol_mlx, max_iters=200)

        np.testing.assert_allclose(
            mlx_r.energies,
            rdk_e,
            atol=2.0,
            err_msg="MLX optimizer energies differ from RDKit by more than 2 kcal/mol",
        )

    def test_multiple_molecules(self):
        for smi in ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]:
            mol = _make_mol(smi, n_confs=5)
            mol_work = Chem.RWMol(mol)
            r = mmff_optimize_mlx_batch(mol_work, max_iters=200)
            assert np.all(np.isfinite(r.energies)), f"NaN for {smi}"
            assert np.all(r.energies < 500.0), f"Energy too high for {smi}"

    def test_large_batch(self):
        mol = _make_mol("CC(=O)OC1=CC=CC=C1C(=O)O", n_confs=100)
        mol_work = Chem.RWMol(mol)
        r = mmff_optimize_mlx_batch(mol_work, max_iters=100)
        assert r.n_conformers == 100
        assert np.all(np.isfinite(r.energies))


class TestFusedMetalKernel:
    """Test the fused Metal kernel for energy + gradient."""

    @pytest.mark.parametrize(
        "smiles",
        [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CCO",
        ],
    )
    def test_energy_matches_mlx(self, smiles):
        mol = _make_mol(smiles, n_confs=5)
        pos = _get_positions(mol).astype(np.float32)
        np_params = extract_mmff_params(mol, conf_id=0)
        mlx_params = params_to_mlx(np_params)
        idx_buf, param_buf, meta = pack_params_for_metal(np_params)

        e_mlx = mmff_energy_batch(mlx_params, mx.array(pos))
        e_metal, _ = mmff_energy_grad_metal(idx_buf, param_buf, meta, mx.array(pos))
        mx.eval(e_mlx, e_metal)

        np.testing.assert_allclose(
            np.array(e_metal), np.array(e_mlx), rtol=1e-5,
        )

    def test_gradient_matches_mlx_autograd(self):
        mol = _make_mol("CC(=O)OC1=CC=CC=C1C(=O)O", n_confs=3)
        pos = _get_positions(mol).astype(np.float32)
        np_params = extract_mmff_params(mol, conf_id=0)
        mlx_params = params_to_mlx(np_params)
        idx_buf, param_buf, meta = pack_params_for_metal(np_params)

        eg_fn = make_energy_and_grad_fn(mlx_params)
        _, g_mlx = eg_fn(mx.array(pos))
        _, g_metal = mmff_energy_grad_metal(idx_buf, param_buf, meta, mx.array(pos))
        mx.eval(g_mlx, g_metal)

        g_mlx_np = np.array(g_mlx).reshape(3, -1)
        g_metal_np = np.array(g_metal)

        for k in range(3):
            cos_sim = np.dot(g_mlx_np[k], g_metal_np[k]) / (
                np.linalg.norm(g_mlx_np[k]) * np.linalg.norm(g_metal_np[k]) + 1e-30
            )
            assert cos_sim > 0.9999, f"Conf {k}: cos_sim={cos_sim:.6f}"


class TestMetalBatchOptimizer:
    """Test the fused Metal batched optimizer."""

    def test_aspirin_converges(self):
        mol = _make_mol("CC(=O)OC1=CC=CC=C1C(=O)O", n_confs=10)
        mol_work = Chem.RWMol(mol)
        r = mmff_optimize_metal_batch(mol_work, max_iters=200)
        assert r.n_conformers == 10
        assert np.all(np.isfinite(r.energies))
        assert np.all(r.energies < 100.0)

    def test_close_to_rdkit(self):
        mol = _make_mol("CC(=O)OC1=CC=CC=C1C(=O)O", n_confs=10)

        mol_rdk = Chem.RWMol(mol)
        rdk_res = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            mol_rdk, maxIters=200, numThreads=1
        )
        rdk_e = np.array([r[1] for r in rdk_res])

        mol_metal = Chem.RWMol(mol)
        r = mmff_optimize_metal_batch(mol_metal, max_iters=200)

        np.testing.assert_allclose(
            r.energies, rdk_e, atol=2.0,
            err_msg="Metal optimizer energies differ from RDKit by > 2 kcal/mol",
        )

    def test_large_batch(self):
        mol = _make_mol("CC(=O)OC1=CC=CC=C1C(=O)O", n_confs=100)
        mol_work = Chem.RWMol(mol)
        r = mmff_optimize_metal_batch(mol_work, max_iters=100)
        assert r.n_conformers == 100
        assert np.all(np.isfinite(r.energies))
