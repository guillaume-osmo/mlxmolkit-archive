"""Batch NDDO SCF must agree with the single-molecule solver.

`nddo_energy` is the reference — it is what the single-molecule pipeline uses
and it was correct throughout. `nddo_energy_batch` is a throughput
optimisation over it, and had drifted in three ways:

  - the core-core repulsion branch for PM6 variants existed in `nddo_energy`
    and `anal_grad` but was missing from `prepare_batch`, so every batched
    PM6/PM6_SP/PM6_D energy used the AM1-style term — several eV out, and
    ~100-260 kcal/mol on the heat of formation, while the density and charges
    were correct;
  - the batched integral layout is sp-only, so PM6_D on P/S/Cl/Br/I raised
    IndexError instead of computing anything;
  - the batch SCF ran undamped past iteration 2 where `nddo_energy` always
    applies adaptive mixing.

These tests pin the agreement so the two paths cannot drift apart again.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.scf import nddo_energy, nddo_energy_batch

SP_ONLY = ["O", "CCO", "c1ccccc1", "CC(=O)OCC"]

# Derived from the registry rather than hardcoded, so adding a method to
# METHOD_PARAMS automatically brings it under this parity check.
METHODS = sorted(get_params.__globals__["METHOD_PARAMS"])
D_ORBITAL = ["CSC", "CCS", "ClCCl"]


def _embed(smiles, seed=42):
    from mlxmolkit.nddo.pipeline import _smiles_to_3d

    result = _smiles_to_3d(smiles, seed=seed)
    if result is None:
        pytest.skip(f"could not embed {smiles}")
    atoms, coords = result[0], result[1]
    return atoms, coords


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("smiles", SP_ONLY)
def test_batch_matches_single_molecule_energy(method, smiles):
    atoms, coords = _embed(smiles)
    one = nddo_energy(atoms, coords, method=method)
    many = nddo_energy_batch([(atoms, coords)], method=method, use_metal=False)[0]

    assert many["converged"] == one["converged"]
    assert many["energy_eV"] == pytest.approx(one["energy_eV"], abs=1e-4)
    assert many["nuclear_eV"] == pytest.approx(one["nuclear_eV"], abs=1e-9)
    assert many["heat_of_formation_kcal"] == pytest.approx(
        one["heat_of_formation_kcal"], abs=1e-2)


@pytest.mark.parametrize("smiles", D_ORBITAL)
def test_pm6d_d_orbital_molecules_batch_correctly(smiles):
    """d-orbital molecules must go through the batch path, not around it.

    These raised IndexError out of prepare_batch originally. Then they were
    routed to the sequential solver instead, which made this comparison
    vacuous — "batch" *was* the sequential code, so it agreed to the last bit.
    Now the batch store is packed per pair and the d integrals are wired in, so
    this compares two genuinely independent implementations.

    Hence two tolerances rather than one. The energy is variational, so an
    error in the converged density shows up only at second order and the two
    paths still agree to ~1e-12 eV. Charges are first order in that same
    density error, so they agree to ~1e-6. Tightening the charge bound would be
    asking the SCF to converge further than conv_tol promises.
    """
    atoms, coords = _embed(smiles)
    one = nddo_energy(atoms, coords, method="PM6_D")
    many = nddo_energy_batch([(atoms, coords)], method="PM6_D", use_metal=False)[0]

    assert many["energy_eV"] == pytest.approx(one["energy_eV"], abs=1e-9)
    assert many["charges"] == pytest.approx(np.asarray(one["charges"]), abs=1e-5)


def test_mixed_sp_and_d_batch_keeps_input_order():
    """A batch mixing both kinds must come back in the order it went in."""
    smis = ["O", "CSC", "CCO", "ClCCl"]
    mols = [_embed(s) for s in smis]
    many = nddo_energy_batch(mols, method="PM6_D", use_metal=False)

    assert len(many) == len(smis)
    for (atoms, coords), res in zip(mols, many):
        one = nddo_energy(atoms, coords, method="PM6_D")
        assert res["energy_eV"] == pytest.approx(one["energy_eV"], abs=1e-4)
        assert res["n_basis"] == one["n_basis"]


def test_pm6_core_core_is_not_the_am1_form():
    """Guard the specific regression: PM6 must not fall back to AM1 core-core.

    The two differ by several eV, so if prepare_batch ever loses the method
    again this fails loudly instead of returning a plausible-looking number.
    """
    from mlxmolkit.nddo.integrals import (compute_nuclear_repulsion,
                                         nuclear_repulsion_for_method)

    atoms, coords = _embed("CCO")
    params = get_params("PM6_D")

    pm6 = nuclear_repulsion_for_method(atoms, coords, params, "PM6_D")
    am1_style = compute_nuclear_repulsion(atoms, coords, param_dict=params)

    assert abs(pm6 - am1_style) > 1.0, (
        "PM6 core-core is indistinguishable from the AM1-style term")

    batched = nddo_energy_batch([(atoms, coords)], method="PM6_D",
                                use_metal=False)[0]
    assert batched["nuclear_eV"] == pytest.approx(pm6, abs=1e-9)
