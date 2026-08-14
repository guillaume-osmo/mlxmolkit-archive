"""nddo_optimize must report its *own* convergence, not the SCF's.

The return dict was built as

    {..., 'opt_converged': False, 'converged': True, ...,
     **{k: v for k, v in result.items() if k != 'coords'}}

with the SCF `result` splatted **last**, so its `converged`, `n_iter` and
`method` overwrote the optimizer's. Two consequences:

1. `converged` was the SCF's flag for the final single-point — essentially
   always True — so a geometry optimization that exhausted `max_iter`
   reported success. The non-converged branch also hardcoded `'converged':
   True` outright.
2. `mlxmolkit.nddo.pipeline` read `opt_result['converged']` and
   `opt_result['n_iter']` into its own `opt_converged`/`opt_n_iter`, so the
   public pipeline surfaced the SCF's convergence and iteration count as the
   geometry optimization's.

Measured on menthol at the old default of max_iter=50: the optimizer stopped
at grad_rms=0.014 (~3x grad_tol) while the energy was still falling, and
reported converged=True. See #28.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from mlxmolkit.nddo.gradient import nddo_optimize

RDLogger.DisableLog("rdApp.*")

# Flexible enough that it cannot converge in a couple of iterations, small
# enough to stay quick.
FLEXIBLE = "CCCCO"


def geometry(smiles, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        pytest.skip(f"embedding failed for {smiles}")
    return ([a.GetAtomicNum() for a in mol.GetAtoms()],
            mol.GetConformer().GetPositions())


def test_exhausting_max_iter_reports_not_converged():
    """The bug in one assertion: starved budget must not report success."""
    atoms, coords = geometry(FLEXIBLE)
    res = nddo_optimize(atoms, coords, max_iter=2, grad_tol=1e-8)

    assert res["converged"] is False
    assert res["opt_converged"] is False
    assert res["n_iter"] == 2
    assert res["opt_n_iter"] == 2
    assert res["grad_rms"] > 1e-8


def test_scf_convergence_is_preserved_under_its_own_key():
    """Fixing `converged` must not throw the SCF's flag away."""
    atoms, coords = geometry(FLEXIBLE)
    res = nddo_optimize(atoms, coords, max_iter=2, grad_tol=1e-8)

    # The SCF converges at every geometry even when the optimizer does not.
    assert res["scf_converged"] is True
    assert res["converged"] is not res["scf_converged"]
    assert isinstance(res["scf_n_iter"], (int, np.integer))


def test_converged_run_agrees_across_both_key_spellings():
    atoms, coords = geometry("CCO")
    res = nddo_optimize(atoms, coords)

    assert res["converged"] is True
    assert res["converged"] == res["opt_converged"]
    assert res["n_iter"] == res["opt_n_iter"]
    assert res["grad_rms"] < 0.005
    # n_iter is the optimizer's, and the two counters are independent: the
    # SCF's is per single-point and much smaller than the optimizer's total.
    assert res["n_iter"] != res["scf_n_iter"]


def test_pipeline_reports_the_optimizers_convergence_not_the_scfs():
    """The pipeline read the SCF keys, so opt_converged was always True."""
    from mlxmolkit.nddo.pipeline import rm1_from_smiles

    res = rm1_from_smiles(FLEXIBLE, optimize=True, opt_max_iter=2,
                          opt_grad_tol=1e-8)
    if res is None:
        pytest.skip("pipeline declined this molecule")

    assert res["opt_converged"] is False
    assert res["opt_n_iter"] == 2
    # The SCF converges at every geometry, so a pipeline that echoed the SCF
    # flag here would report True no matter how starved the optimizer was.
    assert res["converged"] is True


@pytest.mark.slow
def test_menthol_converges_within_the_default_budget():
    """The molecule that exposed the too-small default cap.

    It needs 94 iterations under RM1, so the old default of 50 stopped it
    short at grad_rms ~0.014. Against MOPAC's own PM6 minimum the agreement
    improves from 0.4646 to 0.2009 kcal/mol once it is allowed to finish.
    """
    atoms, coords = geometry("CC(C)C1CCC(C)CC1O")
    res = nddo_optimize(atoms, coords)

    assert res["converged"] is True, (
        f"stopped at {res['n_iter']} iterations, grad_rms={res['grad_rms']:.5f}"
    )
    assert res["grad_rms"] < 0.005
    assert res["n_iter"] > 50, (
        "menthol converging in under 50 would mean this regression test no "
        "longer covers the case that motivated the larger default"
    )


def test_easy_molecule_still_exits_early():
    """The bigger cap is a bound, not a cost — the loop must still return
    as soon as grad_tol is met."""
    atoms, coords = geometry("Clc1ccccc1")
    res = nddo_optimize(atoms, coords)

    assert res["converged"] is True
    assert res["n_iter"] < 50, f"took {res['n_iter']} of a 200 cap"


def test_single_and_batch_agree_on_the_same_molecule():
    """One molecule, two entry points, one answer.

    nddo_optimize_batch defaulted to max_iter=50 while nddo_optimize used 200,
    so menthol came out 0.385 kcal/mol apart depending on which you called —
    9x the MOPAC agreement #28 requires be preserved. The batch path is not
    approximate: it runs the same L-BFGS per molecule and skips a molecule
    once it converges, so the only thing that differed was the budget.
    """
    from mlxmolkit.nddo.gradient import nddo_optimize_batch

    atoms, coords = geometry("CCO")
    single = nddo_optimize(atoms, coords.copy())
    batch = nddo_optimize_batch([(atoms, coords.copy())])[0]

    assert single["converged"] is True
    assert batch["opt_converged"] is True
    assert abs(single["heat_of_formation_kcal"]
               - batch["heat_of_formation_kcal"]) < 0.042, (
        "single and batch disagree by more than the MOPAC agreement tolerance"
    )


def test_optimizer_defaults_match_across_entry_points():
    """A third set of defaults is how the paths drifted apart in the first
    place — the deprecated alias used to hardcode 100 / 0.01."""
    import inspect

    from mlxmolkit.nddo.gradient import nddo_optimize_batch
    from mlxmolkit.nddo.pipeline import rm1_from_smiles

    single = inspect.signature(nddo_optimize).parameters
    batch = inspect.signature(nddo_optimize_batch).parameters
    pipeline = inspect.signature(rm1_from_smiles).parameters

    assert single["max_iter"].default == batch["max_iter"].default
    assert single["grad_tol"].default == batch["grad_tol"].default
    assert pipeline["opt_max_iter"].default == single["max_iter"].default
    assert pipeline["opt_grad_tol"].default == single["grad_tol"].default


def test_deprecated_alias_delegates_its_defaults():
    """rm1_optimize hardcoded grad_tol=0.01, twice as loose as either path."""
    import inspect

    from mlxmolkit.nddo.gradient import rm1_optimize

    params = inspect.signature(rm1_optimize).parameters
    assert params["max_iter"].default is None
    assert params["grad_tol"].default is None


# --- eigenvector following ------------------------------------------------

def test_ef_reaches_the_same_minimum_as_lbfgs():
    """EF is an optimization strategy, not a different model.

    Both must land on the same stationary point; the MOPAC agreement #28
    requires be preserved is 0.042 kcal/mol, so the two paths have to agree
    at least that well or one of them is finding a different minimum.
    """
    atoms, coords = geometry("CCO")
    lbfgs = nddo_optimize(atoms, coords.copy(), method="PM6")
    ef = nddo_optimize(atoms, coords.copy(), method="PM6", optimizer="ef")

    assert ef["converged"] is True
    assert abs(ef["heat_of_formation_kcal"]
               - lbfgs["heat_of_formation_kcal"]) < 0.042


def test_ef_reports_convergence_the_same_way():
    """EF goes through _optimize_result too, so the SCF keys cannot leak."""
    atoms, coords = geometry(FLEXIBLE)
    res = nddo_optimize(atoms, coords, max_iter=2, grad_tol=1e-8,
                        optimizer="ef")

    assert res["converged"] is False
    assert res["opt_converged"] is False
    assert res["n_iter"] == 2
    assert res["scf_converged"] is True


def test_ef_needs_fewer_gradients_on_a_flexible_molecule():
    """Where EF earns its keep. Geraniol: 133 gradient calls under L-BFGS,
    91 under EF on the held-out measurement.

    Rigid molecules go the other way (indole 17 -> 24), which is why EF is
    opt-in rather than the default.
    """
    import mlxmolkit.nddo.anal_grad as anal_grad

    original = anal_grad.analytical_gradient
    calls = {"n": 0}

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    atoms, coords = geometry("CC(C)=CCC/C(C)=C/CO")
    anal_grad.analytical_gradient = counted
    try:
        calls["n"] = 0
        nddo_optimize(atoms, coords.copy(), method="PM6")
        n_lbfgs = calls["n"]
        calls["n"] = 0
        nddo_optimize(atoms, coords.copy(), method="PM6", optimizer="ef")
        n_ef = calls["n"]
    finally:
        anal_grad.analytical_gradient = original

    assert n_ef < n_lbfgs, f"EF used {n_ef} gradients, L-BFGS {n_lbfgs}"


def test_unknown_optimizer_is_rejected():
    atoms, coords = geometry("CCO")
    with pytest.raises(ValueError, match="unknown optimizer"):
        nddo_optimize(atoms, coords, optimizer="newton")
