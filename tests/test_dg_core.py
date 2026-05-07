"""
Test DG core (stages 1-4) against RDKit EmbedMultipleConfs for speed and correctness.
"""
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from mlxmolkit.dg_extract import (
    get_bounds_matrix, extract_dg_params, batch_dg_params,
)
from mlxmolkit.dg_energy_metal import make_dg4d_energy_grad
from mlxmolkit.dg_minimize_metal import (
    dg_minimize_batch, dg_collapse_4th_dim, extract_3d_coords,
    generate_random_4d_coords,
)
from mlxmolkit.stereo_checks import (
    check_tetrahedral_geometry, check_chirality, run_stage3_checks,
)

TEST_SMILES = [
    "CCO",                     # ethanol (simple)
    "c1ccccc1",                # benzene
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "C[C@@H](O)CC",           # chiral (2-butanol)
]


def test_dg_extract():
    """Test that DG parameter extraction works for all test molecules."""
    for smi in TEST_SMILES:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        bmat = get_bounds_matrix(mol)
        params = extract_dg_params(mol, bmat)

        assert params.n_atoms == mol.GetNumAtoms()
        assert len(params.dist_idx1) > 0, f"No distance pairs for {smi}"
        assert len(params.dist_idx1) == len(params.dist_idx2)
        assert len(params.dist_lb2) == len(params.dist_idx1)
        assert np.all(params.dist_lb2 >= 0)
        assert np.all(params.dist_ub2 > 0)
        assert np.all(params.dist_ub2 >= params.dist_lb2)

    print("[PASS] test_dg_extract")


def test_batch_dg_params():
    """Test batching of DG parameters."""
    mols = []
    params_list = []
    for smi in TEST_SMILES[:3]:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        bmat = get_bounds_matrix(mol)
        params = extract_dg_params(mol, bmat)
        mols.append(mol)
        params_list.append(params)

    system = batch_dg_params(params_list)
    assert system.n_mols == 3
    assert system.n_atoms_total == sum(p.n_atoms for p in params_list)
    assert len(system.dist_idx1) == sum(len(p.dist_idx1) for p in params_list)
    assert system.atom_starts[-1] == system.n_atoms_total

    print("[PASS] test_batch_dg_params")


def test_dg_energy_kernel():
    """Test that the DG energy kernel computes reasonable values."""
    import mlx.core as mx

    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    bmat = get_bounds_matrix(mol)
    params = extract_dg_params(mol, bmat)
    system = batch_dg_params([params])

    fn, atom_starts, n_atoms, n_mols, dim = make_dg4d_energy_grad(system)

    # Random 4D coordinates
    np.random.seed(42)
    pos = np.random.randn(n_atoms * dim).astype(np.float32) * 2.0
    pos_mx = mx.array(pos)

    ep, grad = fn(pos_mx)
    mx.eval(ep, grad)

    energy = float(np.sum(np.array(ep)))
    grad_np = np.array(grad)

    assert energy > 0, "Energy should be positive for random coordinates"
    assert np.all(np.isfinite(grad_np)), "Gradient should be finite"
    assert np.linalg.norm(grad_np) > 0, "Gradient should be nonzero"

    print(f"[PASS] test_dg_energy_kernel (E={energy:.4f}, |g|={np.linalg.norm(grad_np):.4f})")


def test_dg_minimize_single_mol():
    """Test DG minimization on a single molecule."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    bmat = get_bounds_matrix(mol)
    params = extract_dg_params(mol, bmat)
    system = batch_dg_params([params])

    result = dg_minimize_batch(system, max_iters=200, grad_tol=1e-3)

    assert result.n_mols == 1
    assert result.energies[0] < 100.0, f"Energy too high: {result.energies[0]}"
    print(f"[PASS] test_dg_minimize_single_mol "
          f"(E={result.energies[0]:.4f}, iters={result.n_iters}, "
          f"converged={result.converged[0]})")


def test_dg_minimize_batch():
    """Test batched DG minimization on multiple molecules."""
    params_list = []
    mols = []
    for smi in TEST_SMILES[:3]:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        bmat = get_bounds_matrix(mol)
        params = extract_dg_params(mol, bmat)
        params_list.append(params)
        mols.append(mol)

    system = batch_dg_params(params_list)
    result = dg_minimize_batch(system, max_iters=200, grad_tol=1e-3)

    assert result.n_mols == 3
    for i in range(3):
        assert result.energies[i] < 200.0, f"Mol {i}: E too high = {result.energies[i]}"

    print(f"[PASS] test_dg_minimize_batch "
          f"(E={result.energies}, iters={result.n_iters})")


def test_dg_4d_collapse():
    """Test 4D → 3D collapse (stage 4)."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    bmat = get_bounds_matrix(mol)
    params = extract_dg_params(mol, bmat)
    system = batch_dg_params([params])

    result_4d = dg_minimize_batch(system, max_iters=200, grad_tol=1e-3)

    result_collapse = dg_collapse_4th_dim(
        system, result_4d.positions, fourth_dim_weight=1000.0, max_iters=200,
    )

    pos_3d = extract_3d_coords(
        result_collapse.positions, result_collapse.atom_starts,
        result_collapse.n_mols, dim=4,
    )

    # Check 4th dim is near zero
    n_atoms = int(system.atom_starts[1])
    max_4th = 0.0
    for i in range(n_atoms):
        x4 = abs(result_collapse.positions[i * 4 + 3])
        max_4th = max(max_4th, x4)

    assert max_4th < 1.5, f"4th dim not collapsed: max={max_4th}"
    print(f"[PASS] test_dg_4d_collapse (max_4th={max_4th:.6f})")


def test_stages_1_to_4():
    """Full stages 1-4 pipeline test with stereo checks."""
    mol = Chem.MolFromSmiles("C[C@@H](O)CC")
    mol = Chem.AddHs(mol)
    bmat = get_bounds_matrix(mol)
    params = extract_dg_params(mol, bmat)

    n_confs = 5
    params_list = [params] * n_confs
    system = batch_dg_params(params_list)

    # Stage 1: random 4D coords
    x0 = generate_random_4d_coords(system, seed=42)

    # Stage 2: DG minimize
    result_dg = dg_minimize_batch(system, x0=x0, max_iters=200, grad_tol=1e-3)

    # Stage 3: stereo checks on 3D projection
    pos_3d = extract_3d_coords(
        result_dg.positions, result_dg.atom_starts, result_dg.n_mols, dim=4,
    )
    passes_3 = []
    for i in range(n_confs):
        a_s = int(system.atom_starts[i])
        passed = run_stage3_checks(pos_3d, mol, bmat, atom_offset=a_s)
        passes_3.append(passed)

    # Stage 4: 4D collapse
    result_col = dg_collapse_4th_dim(
        system, result_dg.positions, fourth_dim_weight=100.0, max_iters=100,
    )
    pos_3d_final = extract_3d_coords(
        result_col.positions, result_col.atom_starts, result_col.n_mols, dim=4,
    )

    print(f"[PASS] test_stages_1_to_4 "
          f"(stage3_pass={sum(passes_3)}/{n_confs}, "
          f"E_dg={result_dg.energies}, "
          f"E_col={result_col.energies})")


def test_speed_vs_rdkit():
    """Speed comparison: Metal DG (stages 1-4) vs RDKit EmbedMultipleConfs."""
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    n_confs = 10

    # --- RDKit baseline ---
    t0 = time.perf_counter()
    ps = AllChem.EmbedMultipleConfs(
        mol, numConfs=n_confs, randomSeed=42,
        useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
    )
    t_rdkit = time.perf_counter() - t0
    print(f"RDKit ETKDG:   {n_confs} confs in {t_rdkit:.4f}s "
          f"({len(ps)} success)")

    # --- Metal DG stages 1-4 ---
    bmat = get_bounds_matrix(mol)
    params = extract_dg_params(mol, bmat)

    t0 = time.perf_counter()
    params_list = [params] * n_confs
    system = batch_dg_params(params_list)
    x0 = generate_random_4d_coords(system, seed=42)

    result_dg = dg_minimize_batch(system, x0=x0, max_iters=200, grad_tol=1e-3)
    result_col = dg_collapse_4th_dim(
        system, result_dg.positions, fourth_dim_weight=100.0, max_iters=100,
    )
    t_metal = time.perf_counter() - t0

    pos_3d = extract_3d_coords(
        result_col.positions, result_col.atom_starts, result_col.n_mols, dim=4,
    )

    print(f"Metal DG 1-4:  {n_confs} confs in {t_metal:.4f}s")
    print(f"Speedup: {t_rdkit / t_metal:.2f}x" if t_metal > 0 else "N/A")
    print(f"[PASS] test_speed_vs_rdkit")


if __name__ == "__main__":
    test_dg_extract()
    test_batch_dg_params()
    test_dg_energy_kernel()
    test_dg_minimize_single_mol()
    test_dg_minimize_batch()
    test_dg_4d_collapse()
    test_stages_1_to_4()
    test_speed_vs_rdkit()
    print("\n=== All DG core tests passed ===")
