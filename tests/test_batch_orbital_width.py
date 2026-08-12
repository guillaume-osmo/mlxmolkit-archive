"""The batch two-centre store is packed per pair, and d orbitals go through it.

History, because the shape of this file changed twice:

1. `prepare_batch` allocated `(N, MA, MA, 256)` with 256 = 4**4 — an sp
   assumption, so a 9-orbital atom could not be batched at all.
2. That became `(N, MA, MA, MO**4)` sized by the widest atom in the batch,
   which let d in but made a single sulfur inflate *every* pair, C-H included,
   by 9**4/4**4 = 25.6x. A 100-molecule batch cost ~5 GB.
3. Now each pair is stored lower-triangle packed at its own size, with an
   offset table: 1 entry per centre for hydrogen, 10 for sp, 45 with d.

The packings nest — below 4 orbitals the 9-basis index equals the 4-basis one —
so sp is not a special case, it is the small case, and there is one code path.

These tests pin the layout, the storage win, and the property that actually
matters: batch and sequential must agree, for sp *and* d.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem

from mlxmolkit.nddo.batch import prepare_batch
from mlxmolkit.nddo.fock_metal import build_fock_batch_cpu
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.packing import pack, unpack, pack_index, packed_size
from mlxmolkit.nddo.scf import (_build_basis_info, _build_core_hamiltonian,
                                _build_fock)

SP_ONLY = ["C", "CCO", "O=Cc1ccccc1", "CC1CCC(C(C)C)CC1O"]
WITH_D = ["CS", "CCS", "Clc1ccccc1", "CSc1ccccc1", "Brc1ccccc1"]


def geometry(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return ([a.GetAtomicNum() for a in mol.GetAtoms()],
            np.asarray(mol.GetConformer().GetPositions(), dtype=float))


def batch_of(smiles: str):
    return prepare_batch([geometry(smiles)], param_dict=get_params("PM6"),
                         method="PM6")


# ---------------------------------------------------------------------------
# The packing itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_orbitals,expected", [(1, 1), (4, 10), (9, 45)])
def test_packed_size_matches_the_distinct_pair_count(n_orbitals, expected):
    assert packed_size(n_orbitals) == expected


def test_the_sp_and_d_packings_nest():
    """Below 4 orbitals both bases must give the same index, or sp blocks
    could not be read as the leading corner of a d block."""
    for i in range(4):
        for j in range(4):
            assert pack_index(i, j) == pack_index(j, i)
            assert pack_index(i, j) < packed_size(4)


@pytest.mark.parametrize("n_a,n_b", [(1, 1), (1, 4), (4, 1), (4, 4),
                                     (4, 9), (9, 4), (9, 9)])
def test_pack_round_trips_exactly(n_a, n_b):
    rng = np.random.default_rng(0)
    dense = rng.normal(size=(n_a, n_a, n_b, n_b))
    dense = 0.5 * (dense + dense.transpose(1, 0, 2, 3))
    dense = 0.5 * (dense + dense.transpose(0, 1, 3, 2))
    assert np.array_equal(unpack(pack(dense, n_a, n_b), n_a, n_b), dense)


# ---------------------------------------------------------------------------
# The batch layout
# ---------------------------------------------------------------------------

def test_pair_blocks_are_sized_by_their_own_two_atoms():
    """Methane: C-H pairs are 10x1, H-H pairs 1x1. Nothing is padded to 9."""
    batch = batch_of("C")
    norb = batch.atom_norb[0]
    for a in range(5):
        for b in range(5):
            offset = int(batch.pair_offset[0, a, b])
            if a == b:
                assert offset == -1, "an atom pairs with itself"
            else:
                assert offset >= 0

    # 4 C-H pairs x 2 orderings x (10*1), plus 6 H-H x 2 x (1*1)
    assert batch.w.shape[1] == 4 * 2 * 10 + 6 * 2 * 1 == 92


def test_a_single_d_atom_does_not_inflate_the_sp_pairs():
    """The regression that motivated packing: one sulfur must not cost every
    C-H pair 25x."""
    sp = batch_of("CCO")
    with_d = batch_of("CCS")
    uniform_would_be = len(with_d.atom_norb[0]) ** 2 * with_d.max_orb ** 4
    assert with_d.w.shape[1] < uniform_would_be / 100, (
        f"packed {with_d.w.shape[1]} vs uniform {uniform_would_be} — "
        "the pairs are being padded to the widest atom again"
    )
    # And the sp-only batch stays small in absolute terms.
    assert sp.w.shape[1] < 2000


def test_one_centre_d_integrals_are_precomputed_only_for_d_atoms():
    batch = batch_of("CS")
    norb = batch.atom_norb[0]
    for a, n in enumerate(norb[:len(batch.atoms_list[0])]):
        has_w = np.abs(batch.atom_w[0, a]).max() > 0
        assert has_w == (n == 9), f"atom {a} with {n} orbitals: atom_w={has_w}"


# ---------------------------------------------------------------------------
# The property that matters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", SP_ONLY + WITH_D)
def test_batch_fock_matches_the_sequential_fock(smiles):
    """Same density in, same Fock out — whatever the storage looks like."""
    atoms, coords = geometry(smiles)
    params = get_params("PM6")
    info = _build_basis_info(atoms, params, molecular_charge=0.0)
    n = info['n_basis']

    rng = np.random.default_rng(0)
    P = rng.normal(size=(n, n))
    P = 0.5 * (P + P.T)

    H = _build_core_hamiltonian(atoms, coords, info)
    reference = _build_fock(H, P, info, atoms, coords)

    batch = prepare_batch([(atoms, coords)], param_dict=params, method="PM6")
    assert np.abs(batch.H_core[0][:n, :n] - H).max() < 1e-10, "H_core diverged"
    batch.P = P[None, :, :].copy()
    got = build_fock_batch_cpu(batch)[0][:n, :n]

    assert np.abs(got - reference).max() < 1e-9, (
        f"{smiles}: batch Fock differs from sequential"
    )


@pytest.mark.parametrize("smiles", WITH_D)
def test_d_orbital_molecules_can_be_batched_at_all(smiles):
    """Used to raise IndexError before the d integrals were routed in."""
    batch = batch_of(smiles)
    assert batch.max_orb == 9
    assert np.isfinite(batch.H_core).all()
    assert np.isfinite(batch.w).all()


# ---------------------------------------------------------------------------
# The GPU path
# ---------------------------------------------------------------------------
# Worth stating why these exist: during this change the Metal kernel was left
# reading the old dense layout while the store had become packed, and it
# returned values around 1e34 — yet the whole suite still passed, because
# nothing compared the GPU path against a reference. These close that.

@pytest.mark.parametrize("smiles", SP_ONLY + WITH_D)
def test_metal_fock_matches_the_cpu_reference(smiles):
    """The kernel indexes the packed store itself; it must agree with numpy."""
    mx = pytest.importorskip("mlx.core")
    from mlxmolkit.nddo.fock_metal import build_fock_batch_metal

    batch = batch_of(smiles)
    n = int(batch.n_basis_arr[0])
    rng = np.random.default_rng(0)
    P = rng.normal(size=(batch.max_basis, batch.max_basis))
    P = 0.5 * (P + P.T)
    batch.P = P[None, :, :].copy()

    cpu = build_fock_batch_cpu(batch)[0][:n, :n]
    gpu = build_fock_batch_metal(batch)[0][:n, :n]
    scale = max(np.abs(cpu).max(), 1e-30)
    assert np.abs(gpu - cpu).max() / scale < 1e-5, (
        f"{smiles}: Metal Fock disagrees with the CPU reference beyond float32"
    )


@pytest.mark.parametrize("smiles", ["CCO", "CCS", "Clc1ccccc1"])
def test_batched_scf_energy_matches_the_sequential_solver(smiles):
    """End to end, through the GPU, including d."""
    from mlxmolkit.nddo import nddo_energy, nddo_energy_batch

    atoms, coords = geometry(smiles)
    one = nddo_energy(atoms, coords, method="PM6")["energy_eV"]
    many = nddo_energy_batch([(atoms, coords)], method="PM6")[0]["energy_eV"]
    assert abs(many - one) < 1e-2, f"{smiles}: batched SCF diverged"


def test_d_molecules_are_not_quietly_routed_to_the_sequential_solver():
    """A mixed batch must be solved as one batch.

    d molecules used to be split out and solved one at a time, which made every
    batch-vs-sequential comparison vacuous for exactly the molecules that
    needed checking.
    """
    import inspect
    from mlxmolkit.nddo import scf

    source = inspect.getsource(scf.nddo_energy_batch)
    assert "d_positions" not in source, (
        "the sequential fallback for d-orbital molecules is back"
    )
