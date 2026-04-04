"""Tests for N×k parallel conformer generation with shared constraints."""
import numpy as np
import mlx.core as mx
import pytest

from mlxmolkit.dg_extract import DGParams, extract_dg_params, get_bounds_matrix
from mlxmolkit.shared_batch import (
    SharedConstraintBatch,
    pack_shared_dg_batch,
    init_random_positions,
)
from mlxmolkit.conformer_metal import dg_minimize_shared


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_all_pairs_dg(n_atoms: int, lb: float = 1.0, ub: float = 4.0) -> DGParams:
    """Create DGParams with all-pair distance constraints."""
    pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    n_dist = len(pairs)
    ub2 = ub * ub
    lb2 = lb * lb
    return DGParams(
        n_atoms=n_atoms,
        dist_idx1=np.array([p[0] for p in pairs], dtype=np.int32),
        dist_idx2=np.array([p[1] for p in pairs], dtype=np.int32),
        dist_lb2=np.full(n_dist, lb2, dtype=np.float32),
        dist_ub2=np.full(n_dist, ub2, dtype=np.float32),
        dist_weight=np.full(n_dist, 1.0 / max(ub2 - lb2, 1e-8), dtype=np.float32),
        chiral_idx1=np.zeros(0, dtype=np.int32),
        chiral_idx2=np.zeros(0, dtype=np.int32),
        chiral_idx3=np.zeros(0, dtype=np.int32),
        chiral_idx4=np.zeros(0, dtype=np.int32),
        chiral_vol_lower=np.zeros(0, dtype=np.float32),
        chiral_vol_upper=np.zeros(0, dtype=np.float32),
        fourth_idx=np.arange(n_atoms, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# SharedConstraintBatch packing tests
# ---------------------------------------------------------------------------

class TestSharedBatchPacking:

    def test_single_mol_single_conf(self):
        dg = _make_all_pairs_dg(5)
        batch = pack_shared_dg_batch([dg], [1], dim=4)
        assert batch.n_mols == 1
        assert batch.n_confs_total == 1
        assert batch.n_atoms_total == 5
        assert batch.conf_to_mol.tolist() == [0]
        assert batch.conf_atom_starts.tolist() == [0, 5]

    def test_single_mol_multi_conf(self):
        dg = _make_all_pairs_dg(8)
        batch = pack_shared_dg_batch([dg], [10], dim=4)
        assert batch.n_confs_total == 10
        assert batch.n_atoms_total == 80  # 10 × 8
        assert all(batch.conf_to_mol == 0)
        # Constraints stored once: 8*(8-1)/2 = 28 pairs
        assert len(batch.dist_idx1) == 28

    def test_multi_mol_multi_conf(self):
        dg1 = _make_all_pairs_dg(5)
        dg2 = _make_all_pairs_dg(3)
        batch = pack_shared_dg_batch([dg1, dg2], [3, 2], dim=4)
        assert batch.n_mols == 2
        assert batch.n_confs_total == 5
        assert batch.n_atoms_total == 3 * 5 + 2 * 3  # 21
        assert batch.conf_to_mol.tolist() == [0, 0, 0, 1, 1]
        # Dist constraints: mol0=10 + mol1=3 = 13
        assert len(batch.dist_idx1) == 13
        # All indices are LOCAL
        assert batch.dist_idx1.max() < 5

    def test_constraints_are_local(self):
        """Constraint indices must be LOCAL [0, n_atoms_mol), NOT global."""
        dg1 = _make_all_pairs_dg(5)
        dg2 = _make_all_pairs_dg(10)
        batch = pack_shared_dg_batch([dg1, dg2], [1, 1], dim=4)
        # mol1 has 10 atoms → max local index = 9
        mol1_start = batch.dist_term_starts[1]
        mol1_end = batch.dist_term_starts[2]
        mol1_indices = np.concatenate([
            batch.dist_idx1[mol1_start:mol1_end],
            batch.dist_idx2[mol1_start:mol1_end],
        ])
        assert mol1_indices.max() == 9
        assert mol1_indices.min() == 0

    def test_memory_savings(self):
        """Shared batch should use less memory than duplicated approach."""
        dg = _make_all_pairs_dg(30)  # 435 pairs
        k = 50
        batch = pack_shared_dg_batch([dg], [k], dim=4)
        # Constraints stored once: 435 pairs
        n_constraints = len(batch.dist_idx1)
        assert n_constraints == 435  # NOT 435 × 50
        # Positions: 50 × 30 = 1500 atoms
        assert batch.n_atoms_total == 1500


# ---------------------------------------------------------------------------
# DG minimize tests — serial gradient (default)
# ---------------------------------------------------------------------------

class TestDGMinimizeSerial:

    def test_all_converge(self):
        dg = _make_all_pairs_dg(10)
        batch = pack_shared_dg_batch([dg], [5], dim=4)
        pos = init_random_positions(batch, seed=42)
        _, energies, statuses = dg_minimize_shared(batch, pos, max_iters=200)
        assert np.all(statuses == 0), f"Not all converged: {statuses}"

    def test_energy_near_zero(self):
        dg = _make_all_pairs_dg(8, lb=1.0, ub=4.0)
        batch = pack_shared_dg_batch([dg], [3], dim=4)
        pos = init_random_positions(batch, seed=99)
        _, energies, _ = dg_minimize_shared(batch, pos, max_iters=200)
        assert np.all(energies < 1.0), f"Energies too high: {energies}"

    def test_conformers_are_distinct(self):
        """Different random seeds → different optimized geometries."""
        dg = _make_all_pairs_dg(10)
        batch = pack_shared_dg_batch([dg], [3], dim=4)
        pos = init_random_positions(batch, seed=42)
        out_pos, _, _ = dg_minimize_shared(batch, pos, max_iters=200)
        dim = 4
        c0 = out_pos[batch.conf_atom_starts[0] * dim:batch.conf_atom_starts[1] * dim]
        c1 = out_pos[batch.conf_atom_starts[1] * dim:batch.conf_atom_starts[2] * dim]
        assert np.max(np.abs(c0 - c1)) > 0.01, "Conformers are identical"

    def test_multi_molecule_partition(self):
        """Each conformer of each molecule should converge independently."""
        dg1 = _make_all_pairs_dg(5)
        dg2 = _make_all_pairs_dg(8)
        batch = pack_shared_dg_batch([dg1, dg2], [3, 2], dim=4)
        pos = init_random_positions(batch, seed=7)
        _, energies, statuses = dg_minimize_shared(batch, pos, max_iters=200)
        assert np.all(statuses == 0)
        # All 5 conformers (3 for mol0, 2 for mol1) should have reasonable energy
        assert len(energies) == 5

    def test_fourth_dim_penalty(self):
        """Fourth dimension should shrink toward zero."""
        dg = _make_all_pairs_dg(6)
        batch = pack_shared_dg_batch([dg], [2], dim=4)
        pos = init_random_positions(batch, seed=55)
        out_pos, _, _ = dg_minimize_shared(
            batch, pos, max_iters=300, fourth_dim_weight=1.0,
        )
        # Extract 4th coordinate of all atoms
        positions_4d = out_pos.reshape(-1, 4)[:, 3]
        max_4th = np.max(np.abs(positions_4d))
        assert max_4th < 2.0, f"4th dim too large: {max_4th}"

    def test_empty_constraints(self):
        """Molecule with zero constraints should still run without error."""
        dg = DGParams(
            n_atoms=3,
            dist_idx1=np.zeros(0, dtype=np.int32),
            dist_idx2=np.zeros(0, dtype=np.int32),
            dist_lb2=np.zeros(0, dtype=np.float32),
            dist_ub2=np.zeros(0, dtype=np.float32),
            dist_weight=np.zeros(0, dtype=np.float32),
            chiral_idx1=np.zeros(0, dtype=np.int32),
            chiral_idx2=np.zeros(0, dtype=np.int32),
            chiral_idx3=np.zeros(0, dtype=np.int32),
            chiral_idx4=np.zeros(0, dtype=np.int32),
            chiral_vol_lower=np.zeros(0, dtype=np.float32),
            chiral_vol_upper=np.zeros(0, dtype=np.float32),
            fourth_idx=np.arange(3, dtype=np.int32),
        )
        batch = pack_shared_dg_batch([dg], [2], dim=4)
        pos = init_random_positions(batch, seed=1)
        _, energies, statuses = dg_minimize_shared(batch, pos, max_iters=50)
        assert len(energies) == 2


# ---------------------------------------------------------------------------
# DG minimize tests — parallel gradient (gather mode)
# ---------------------------------------------------------------------------

class TestDGMinimizeParallelGrad:

    def test_parallel_matches_serial_small(self):
        """Parallel gather gradient on small molecule (no buffer aliasing)."""
        dg = _make_all_pairs_dg(20)
        batch = pack_shared_dg_batch([dg], [10], dim=4)
        pos = init_random_positions(batch, seed=42)

        _, e_serial, s_serial = dg_minimize_shared(
            batch, pos, max_iters=200, parallel_grad=False,
        )
        _, e_parallel, s_parallel = dg_minimize_shared(
            batch, pos, max_iters=200, parallel_grad=True,
        )
        # Energy should match (gradient computation is mathematically equivalent)
        np.testing.assert_allclose(e_serial, e_parallel, atol=1e-4)

    @pytest.mark.xfail(reason="MLX buffer aliasing: lbfgs_history_starts and out_statuses may overlap when same shape/dtype")
    def test_parallel_status_values(self):
        """Check status values are correct (not aliased with lbfgs starts)."""
        dg = _make_all_pairs_dg(15)
        batch = pack_shared_dg_batch([dg], [5], dim=4)
        pos = init_random_positions(batch, seed=42)
        _, _, statuses = dg_minimize_shared(
            batch, pos, max_iters=300, parallel_grad=True,
        )
        # Statuses should be 0 or 1, not large offset values
        assert np.all(statuses <= 1), f"Status values look aliased: {statuses}"


# ---------------------------------------------------------------------------
# Scale tests
# ---------------------------------------------------------------------------

class TestDGMinimizeScale:

    def test_many_conformers(self):
        """N=3, k=20 → 60 conformers should all converge."""
        dg1 = _make_all_pairs_dg(8)
        dg2 = _make_all_pairs_dg(6)
        dg3 = _make_all_pairs_dg(10)
        batch = pack_shared_dg_batch([dg1, dg2, dg3], [20, 20, 20], dim=4)
        assert batch.n_confs_total == 60
        pos = init_random_positions(batch, seed=42)
        _, energies, statuses = dg_minimize_shared(batch, pos, max_iters=200)
        converged = np.sum(statuses == 0)
        assert converged >= 55, f"Only {converged}/60 converged"

    @pytest.mark.slow
    def test_medium_scale(self):
        """N=10, k=10 → 100 conformers with varying molecule sizes."""
        dg_list = [_make_all_pairs_dg(n) for n in [5, 8, 10, 6, 12, 7, 9, 4, 11, 8]]
        batch = pack_shared_dg_batch(dg_list, [10] * 10, dim=4)
        assert batch.n_confs_total == 100
        pos = init_random_positions(batch, seed=42)
        _, energies, statuses = dg_minimize_shared(batch, pos, max_iters=200)
        converged = np.sum(statuses == 0)
        assert converged >= 90, f"Only {converged}/100 converged"


# ---------------------------------------------------------------------------
# RDKit integration test (requires RDKit)
# ---------------------------------------------------------------------------

class TestDGMinimizeRDKit:

    @pytest.mark.xfail(reason="Large molecule needs gradient scaling (nvMolKit 0.1x) not yet in TPM kernel")
    def test_aspirin(self):
        """Full pipeline: SMILES → RDKit extract → shared batch → DG minimize."""
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not available")

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        mol = Chem.AddHs(mol)
        bmat = get_bounds_matrix(mol)
        dg_params = extract_dg_params(mol, bmat, dim=4)

        k = 5
        batch = pack_shared_dg_batch([dg_params], [k], dim=4)
        pos = init_random_positions(batch, seed=42)
        out_pos, energies, statuses = dg_minimize_shared(
            batch, pos, max_iters=500, fourth_dim_weight=0.1,
        )
        # Aspirin with hydrogens is large; convergence depends on random init
        converged = np.sum(statuses == 0)
        assert converged >= 1, f"No conformers converged for aspirin (statuses={statuses})"
        assert batch.n_confs_total == k
        # Constraints stored once
        assert len(batch.dist_idx1) == len(dg_params.dist_idx1)

    def test_two_molecules(self):
        """Two different molecules, each with k conformers."""
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not available")

        mol1 = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))   # benzene
        mol2 = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))    # acetic acid

        bmat1 = get_bounds_matrix(mol1)
        bmat2 = get_bounds_matrix(mol2)
        dg1 = extract_dg_params(mol1, bmat1)
        dg2 = extract_dg_params(mol2, bmat2)

        batch = pack_shared_dg_batch([dg1, dg2], [3, 4], dim=4)
        assert batch.n_confs_total == 7
        assert batch.conf_to_mol.tolist() == [0, 0, 0, 1, 1, 1, 1]

        pos = init_random_positions(batch, seed=42)
        _, energies, statuses = dg_minimize_shared(
            batch, pos, max_iters=300, fourth_dim_weight=0.1,
        )
        assert len(energies) == 7
        converged = np.sum(statuses == 0)
        assert converged >= 5, f"Only {converged}/7 converged"
