"""The gradient must be exact for every method family, not just PM6.

PM6 and its variants take the PWCCT core-core path; AM1, RM1 and PM3 take a
different one. A gradient rewrite once got the PM6 path right and left the
other completely wrong — every pair's core-core repulsion was computed with
the parameters of atoms 0 and 1 — and it went unnoticed because the exactness
check only covered PM6. The damage was invisible in the gradient's own
benchmark and showed up only as `nddo_optimize` running to its iteration cap
with grad_rms 4.59 instead of converging at 0.004.

Central differences of the energy are the oracle: whatever the gradient
claims, differentiating `nddo_energy` has to agree with it.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from mlxmolkit.nddo.anal_grad import analytical_gradient
from mlxmolkit.nddo.gradient import nddo_optimize
from mlxmolkit.nddo.scf import nddo_energy

RDLogger.DisableLog("rdApp.*")

# One from each core-core family, plus a d-bearing molecule for the spd path.
METHODS = ["RM1", "AM1", "PM3", "PM6"]
STEP = 1e-4
# Central differences carry their own O(step^2) truncation error, so this is a
# bound on agreement, not on the analytic gradient alone.
TOL = 1e-5


def geometry(smiles, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        pytest.skip(f"embedding failed for {smiles}")
    return ([a.GetAtomicNum() for a in mol.GetAtoms()],
            np.asarray(mol.GetConformer().GetPositions()))


def numerical_gradient(atoms, coords, method):
    grad = np.zeros_like(coords)
    for i in range(len(atoms)):
        for j in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[i, j] += STEP
            minus[i, j] -= STEP
            grad[i, j] = (
                nddo_energy(atoms, plus, method=method)["energy_eV"]
                - nddo_energy(atoms, minus, method=method)["energy_eV"]
            ) / (2.0 * STEP)
    return grad


@pytest.mark.parametrize("method", METHODS)
def test_gradient_matches_central_differences(method):
    atoms, coords = geometry("CCO")
    _result, analytic = analytical_gradient(atoms, coords.copy(), method=method)
    numeric = numerical_gradient(atoms, coords, method)

    rms = float(np.sqrt(np.mean((analytic - numeric) ** 2)))
    assert rms < TOL, (
        f"{method}: gradient disagrees with central differences by {rms:.3e} "
        f"eV/A rms"
    )


def test_gradient_is_exact_on_a_d_bearing_molecule():
    """Sulfur routes through the 9x9 Wigner-D path rather than the sp one."""
    atoms, coords = geometry("CSc1ccccc1")
    _result, analytic = analytical_gradient(atoms, coords.copy(), method="PM6")
    numeric = numerical_gradient(atoms, coords, "PM6")

    assert float(np.sqrt(np.mean((analytic - numeric) ** 2))) < TOL


@pytest.mark.parametrize("method", METHODS)
def test_optimizer_converges_under_every_method(method):
    """The end-to-end symptom the exactness check above would have caught.

    A wrong gradient need not look wrong — it can still be smooth and
    plausible — but it will not drive the optimizer to a stationary point.
    """
    atoms, coords = geometry("CCO")
    res = nddo_optimize(atoms, coords.copy(), method=method)

    assert res["converged"] is True, (
        f"{method}: stopped at {res['n_iter']} iterations with "
        f"grad_rms={res['grad_rms']:.5f}"
    )
    assert res["grad_rms"] < 0.005
