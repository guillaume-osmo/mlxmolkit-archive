"""The batched CPU Fock build must equal the scalar one it replaced.

`build_fock_batch_cpu_reference` is five nested Python loops over
(molecule, pair, mu, nu, lam, sig) — slow enough to be unusable (57 s for a
200-molecule PM6 batch, against 2.5 s for simply looping `nddo_energy`) and
simple enough to read and believe. `build_fock_batch_cpu` groups every pair in
the batch by orbital shape and contracts each group once.

The bug this caught while being written is the reason the file exists: writing
the p-block diagonal as `F[pk, pk]` with `pk` a *slice* fills all nine entries
of the 3x3 instead of its three diagonal ones. The result was still smooth,
still symmetric, and wrong by 4.18 — invisible without an element-wise
comparison against the scalar form.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from mlxmolkit.nddo.batch import prepare_batch  # noqa: E402
from mlxmolkit.nddo.fock_metal import (  # noqa: E402
    build_fock_batch_cpu,
    build_fock_batch_cpu_reference,
)

RDLogger.DisableLog("rdApp.*")

# Mixed sizes and elements, including hydrogen-only atoms (n_orb == 1) so the
# shape grouping has to handle more than the 4x4 case.
SMILES = [
    "CCO",
    "O=Cc1ccccc1",
    "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O",
    "CSc1ccccc1",
    "ClCCl",
    "CC(=O)OCC",
    "[HH]",
    "O",
]


def batch_with_random_density(smiles_list, method="PM6", seed=0):
    mols = []
    for smi in smiles_list:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            continue
        mols.append(([a.GetAtomicNum() for a in mol.GetAtoms()],
                     np.asarray(mol.GetConformer().GetPositions())))
    if not mols:
        pytest.skip("no molecules embedded")

    batch = prepare_batch(mols, method=method)
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(batch.n_mols, batch.max_basis, batch.max_basis)) * 0.1
    # A density matrix is symmetric, and the reference exploits that.
    batch.P = 0.5 * (P + np.swapaxes(P, 1, 2))
    return batch


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_build_matches_the_scalar_reference(seed):
    batch = batch_with_random_density(SMILES, seed=seed)

    reference = build_fock_batch_cpu_reference(batch)
    batched = build_fock_batch_cpu(batch)

    worst = float(np.max(np.abs(batched - reference)))
    assert worst < 1e-10, (
        f"batched CPU Fock differs from the scalar reference by {worst:.3e}"
    )


def test_the_cached_plan_does_not_go_stale_across_densities():
    """The plan is cached on the batch, so a second density must not reuse
    anything density-dependent."""
    batch = batch_with_random_density(SMILES, seed=3)

    first = build_fock_batch_cpu(batch)          # builds and caches the plan
    assert hasattr(batch, "_fock_cpu_plan")

    rng = np.random.default_rng(99)
    P = rng.normal(size=batch.P.shape) * 0.1
    batch.P = 0.5 * (P + np.swapaxes(P, 1, 2))

    second = build_fock_batch_cpu(batch)         # same plan, new density
    expected = build_fock_batch_cpu_reference(batch)

    assert not np.allclose(first, second), "the new density changed nothing"
    assert float(np.max(np.abs(second - expected))) < 1e-10


def test_a_single_molecule_batch_is_handled():
    """Degenerate batch size: no pairs across molecules to group."""
    batch = batch_with_random_density(["CCO"], seed=4)

    assert float(np.max(np.abs(
        build_fock_batch_cpu(batch) - build_fock_batch_cpu_reference(batch)
    ))) < 1e-10


def test_padding_outside_each_molecule_stays_zero():
    """Molecules are padded to the batch's widest basis.

    The scatter runs on flat indices into (N, MB, MB); an off-by-one in the
    molecule stride would write into a neighbour's padding — or worse, into a
    neighbour's real block — and the shorter molecules are where that shows.
    """
    batch = batch_with_random_density(SMILES, seed=5)
    batched = build_fock_batch_cpu(batch)

    for mol in range(batch.n_mols):
        n_bas = batch.n_basis_arr[mol]
        if n_bas == batch.max_basis:
            continue
        pad = batched[mol, n_bas:, :]
        assert np.all(pad == 0.0), (
            f"molecule {mol} wrote {np.count_nonzero(pad)} nonzero entries "
            f"into rows past its {n_bas}-orbital basis"
        )
        pad = batched[mol, :, n_bas:]
        assert np.all(pad == 0.0)
