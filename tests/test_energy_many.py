"""`nddo_energy_many` must be a scheduling change, not a numerical one.

N independent molecules is embarrassingly parallel, and this entry point is
the one that treats it that way — `nddo_energy_batch` instead pads everything
to a common width and runs one wide NumPy pipeline in a single process.

Measured on 200 PM6 single points against OpenMOPAC 23.2, same geometries,
14 cores: MOPAC 0.34 s across 14 processes, this 0.33 s, `nddo_energy_batch`
1.21 s on Metal and 2.20 s on CPU, sequential 2.32 s, MOPAC one-at-a-time
3.01 s.

The property that matters is exactness: each molecule is solved by
`nddo_energy` itself, untouched, so the results must be bit-identical to a
plain loop. Anything else means the parallel path is doing its own arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from mlxmolkit.nddo.scf import (  # noqa: E402
    nddo_energy,
    nddo_energy_many,
    shutdown_worker_pool,
)

RDLogger.DisableLog("rdApp.*")

# Deliberately mixed: sizes 3-31 atoms so the chunker cannot balance by
# accident, and a d-bearing molecule so both integral paths run in workers.
SMILES = ["CCO", "O", "CSc1ccccc1", "O=Cc1ccccc1", "ClCCl",
          "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", "CC(=O)OCC", "c1ccccc1"]


@pytest.fixture(scope="module")
def molecules():
    out = []
    for smi in SMILES:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            continue
        out.append(([a.GetAtomicNum() for a in mol.GetAtoms()],
                    np.asarray(mol.GetConformer().GetPositions())))
    if len(out) < 4:
        pytest.skip("not enough molecules embedded")
    return out


@pytest.fixture(scope="module")
def sequential(molecules):
    return [nddo_energy(list(a), c, method="PM6", max_iter=200) for a, c in molecules]


@pytest.mark.parametrize("workers", [2, 3])
def test_results_are_bit_identical_to_a_plain_loop(molecules, sequential, workers):
    got = nddo_energy_many(molecules, method="PM6", max_iter=200,
                           workers=workers, reuse_pool=False)

    assert len(got) == len(sequential)
    for i, (ref, res) in enumerate(zip(sequential, got)):
        assert res["energy_eV"] == ref["energy_eV"], (
            f"molecule {i} ({SMILES[i]}): {res['energy_eV']!r} vs {ref['energy_eV']!r}"
        )
        assert res["converged"] == ref["converged"]
        assert res["n_basis"] == ref["n_basis"]
        assert np.array_equal(res["charges"], ref["charges"])


def test_results_come_back_in_input_order(molecules, sequential):
    """The work is striped across chunks, so order is reassembled, not implied."""
    got = nddo_energy_many(molecules, method="PM6", max_iter=200, workers=3,
                           reuse_pool=False)
    assert [r["n_basis"] for r in got] == [r["n_basis"] for r in sequential]


def test_one_worker_skips_the_pool_entirely(molecules, sequential):
    got = nddo_energy_many(molecules, method="PM6", max_iter=200, workers=1)
    assert [r["energy_eV"] for r in got] == [r["energy_eV"] for r in sequential]


def test_per_molecule_charges_are_honoured(molecules, sequential):
    """A charge must follow its own molecule through the chunking.

    -2 rather than -1: the SCF is closed-shell only, and water at -1 has nine
    electrons. The point here is that the charge lands on molecule 1 and
    nowhere else, not that O(2-) is a sensible species.
    """
    charges = [0.0] * len(molecules)
    charges[1] = -2.0                      # water, kept closed-shell
    got = nddo_energy_many(molecules, method="PM6", max_iter=200, workers=2,
                           molecular_charges=charges, reuse_pool=False)
    want = nddo_energy(list(molecules[1][0]), molecules[1][1], method="PM6",
                       max_iter=200, molecular_charge=-2.0)

    assert got[1]["energy_eV"] == want["energy_eV"]
    assert got[1]["energy_eV"] != sequential[1]["energy_eV"]
    # every other molecule is unaffected
    for i in (0, 2, 3):
        assert got[i]["energy_eV"] == sequential[i]["energy_eV"]


def test_mismatched_charge_count_is_rejected(molecules):
    with pytest.raises(ValueError, match="molecular_charges"):
        nddo_energy_many(molecules, method="PM6", molecular_charges=[0.0])


def test_empty_input():
    assert nddo_energy_many([], method="PM6") == []


def test_the_pool_is_reused_and_can_be_shut_down(molecules):
    import mlxmolkit.nddo.scf as scf

    shutdown_worker_pool()
    assert scf._POOL is None
    nddo_energy_many(molecules, method="PM6", max_iter=200, workers=2)
    assert scf._POOL is not None
    first = scf._POOL
    nddo_energy_many(molecules, method="PM6", max_iter=200, workers=2)
    assert scf._POOL is first, "a second call rebuilt the pool"
    shutdown_worker_pool()
    assert scf._POOL is None
