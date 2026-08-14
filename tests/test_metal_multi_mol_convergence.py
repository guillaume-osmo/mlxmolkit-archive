"""Per-conformer convergence tracking on the Metal multi-molecule optimizer.

Before this, ``mmff_optimize_metal_multi_mol`` ran every one of ``max_iters``
iterations unconditionally and reported ``n_iters=max_iters`` — the cap, not
the work done. Raising the cap was therefore a linear cost even when every
conformer had long since converged (measured 160/390/780/1540 ms at
200/500/1000/2000 on five molecules that all converge under 100 iterations).
"""
from __future__ import annotations

import numpy as np
import pytest

from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.mmff_metal_optimizer import mmff_optimize_metal_multi_mol

# Two rigid molecules (converge in tens of iterations) and one floppy chain
# (hundreds), so the batch exercises both the early-exit and the straggler.
EASY = ["c1ccccc1", "Clc1ccccc1"]
HARD = ["CCCCCCCCCCCCCCCC"]


def _build(smiles, n_confs=2, seed=42):
    mols = []
    for smi in smiles:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if rdDistGeom.EmbedMultipleConfs(mol, n_confs, randomSeed=seed) == 0:
            pytest.skip(f"embedding failed for {smi}")
        mols.append(mol)
    return mols


def _run(mols, **kw):
    return mmff_optimize_metal_multi_mol([(m, None) for m in mols], **kw)


def test_per_conf_iters_is_recorded_and_below_the_cap():
    res = _run(_build(EASY), max_iters=2000)
    for mol_res in res.mol_results:
        assert mol_res.per_conf_iters is not None
        assert mol_res.converged is not None
        assert mol_res.converged.all()
        # Rigid aromatics converge in tens of iterations, so a recorded count
        # anywhere near the cap would mean the count is the cap in disguise.
        assert (mol_res.per_conf_iters < 500).all(), mol_res.per_conf_iters
        assert (mol_res.per_conf_iters > 0).all()


def test_converged_flag_agrees_with_reported_grad_norms():
    """The tolerance must apply to the same quantity the result reports.

    grad_norms comes from the *scaled* gradient; testing convergence against
    the unscaled one let a conformer come back with grad_norms=1.6e-5,
    converged=False at grad_tol=1e-4.
    """
    tol = 1e-4
    res = _run(_build(EASY), max_iters=2000, grad_tol=tol)
    for mol_res in res.mol_results:
        for conv, gnorm in zip(mol_res.converged, mol_res.grad_norms):
            assert conv == (gnorm <= tol), (conv, gnorm)


def test_flexible_molecule_needs_more_iterations_than_a_rigid_one():
    """Iterations track flexibility, which is why one chain sets batch cost."""
    res = _run(_build(EASY + HARD), max_iters=2000)
    rigid = np.concatenate(
        [r.per_conf_iters for r in res.mol_results[: len(EASY)]]
    )
    floppy = res.mol_results[-1].per_conf_iters
    assert floppy.min() > rigid.max(), (floppy, rigid)


def test_raising_the_cap_does_not_change_the_answer():
    """max_iters is a cap, not a cost: past convergence it buys nothing.

    Guards the early exit against stopping somewhere the extra iterations
    would still have moved the geometry.
    """
    e_1000 = _run(_build(EASY), max_iters=1000).mol_results
    e_4000 = _run(_build(EASY), max_iters=4000).mol_results
    for a, b in zip(e_1000, e_4000):
        np.testing.assert_allclose(a.energies, b.energies, atol=1e-4)
        np.testing.assert_array_equal(a.per_conf_iters, b.per_conf_iters)


def test_grad_tol_zero_disables_the_exit():
    """Unreachable tolerance reproduces the old always-run-max_iters path."""
    res = _run(_build(EASY), max_iters=100, grad_tol=0.0)
    for mol_res in res.mol_results:
        assert not mol_res.converged.any()
        assert (mol_res.per_conf_iters == 100).all()
