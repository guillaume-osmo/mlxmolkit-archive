"""The incremental gradient must equal the full-rebuild one it replaces.

Displacing one atom only changes the pairs that touch it, so the frozen-density
gradient rebuilds just those and patches the reference matrices instead of
reassembling H and F from scratch 6N times.

Correctness here is not a judgement call: the previous implementation is still
present as ``_energy_frozen_density``, so the old gradient can be reconstructed
exactly and differenced. Any disagreement beyond round-off means a pair was
missed.

The asymmetric term is the one to watch. Electron-nuclear attraction for the
ordered pair (i, j) lands on atom i's diagonal block and (j, i) on atom j's, so
moving one atom perturbs *every* atom's diagonal block. Missing the second
ordering leaves a gradient that still looks plausible.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem

from mlxmolkit.nddo import nddo_energy, nddo_gradient
from mlxmolkit.nddo.anal_grad import _energy_frozen_density
from mlxmolkit.nddo.methods import get_params

SP_ONLY = ["CCO", "O=Cc1ccccc1"]
WITH_D = ["CCS", "CS", "CSC", "CCCl"]          # ethanethiol first: minimal d case


def geometry(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return ([a.GetAtomicNum() for a in mol.GetAtoms()],
            np.asarray(mol.GetConformer().GetPositions(), dtype=float))


def full_rebuild_gradient(atoms, coords, method="PM6", step=1e-5):
    """The previous implementation: rebuild H and F in full at every step."""
    P = nddo_energy(atoms, coords, method=method,
                    max_iter=200, conv_tol=1e-8)['density']
    params = get_params(method)
    grad = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[a, d] += step
            minus[a, d] -= step
            grad[a, d] = (
                _energy_frozen_density(atoms, plus, P, params, method)
                - _energy_frozen_density(atoms, minus, P, params, method)
            ) / (2.0 * step)
    return grad


@pytest.mark.parametrize("smiles", SP_ONLY + WITH_D)
def test_incremental_matches_the_full_rebuild(smiles):
    atoms, coords = geometry(smiles)
    _, incremental = nddo_gradient(atoms, coords, method="PM6", analytical=True)
    reference = full_rebuild_gradient(atoms, coords)
    assert np.abs(incremental - reference).max() < 1e-6, (
        f"{smiles}: incremental gradient diverges from the full rebuild"
    )


@pytest.mark.parametrize("smiles", ["CCS", "CS"])
def test_d_orbital_molecules_are_handled(smiles):
    """Sulfur carries d orbitals, so it exercises the 9x9 attraction path."""
    atoms, coords = geometry(smiles)
    assert any(z in (15, 16, 17, 35, 53) for z in atoms), "no d atom in this test"
    _, incremental = nddo_gradient(atoms, coords, method="PM6", analytical=True)
    reference = full_rebuild_gradient(atoms, coords)
    assert np.abs(incremental - reference).max() < 1e-6


@pytest.mark.parametrize("smiles", ["CCO", "CCS"])
def test_gradient_still_matches_central_differences(smiles):
    """The physics check, independent of either implementation."""
    atoms, coords = geometry(smiles)
    _, analytic = nddo_gradient(atoms, coords, method="PM6", analytical=True)
    _, numeric = nddo_gradient(atoms, coords, method="PM6", analytical=False)
    assert np.abs(analytic - numeric).max() < 1e-4


def test_both_attraction_orderings_are_applied():
    """Guards the asymmetric term: (i,j) and (j,i) hit different blocks.

    A heteronuclear pair is required — on a symmetric one the two orderings
    coincide and dropping one would go unnoticed.
    """
    atoms, coords = geometry("CS")
    _, incremental = nddo_gradient(atoms, coords, method="PM6", analytical=True)
    reference = full_rebuild_gradient(atoms, coords)
    per_atom = np.abs(incremental - reference).max(axis=1)
    assert per_atom.max() < 1e-6, (
        f"per-atom gradient error {per_atom}; an attraction ordering is missing"
    )
