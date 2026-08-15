"""The gradient serves its displaced pairs two different ways; they must agree.

An sp-only molecule computes the six displaced geometries' rotations and
overlaps in one batch each and hands `_pair_energy_many` the rows positionally.
A d-bearing molecule instead declares every geometry to `pair_cache` and each
row is looked up by a key built from its coordinate bytes.

The positional path is the faster one and the easier one to get quietly wrong:
slicing the wrong geometry's block still yields a smooth, plausible gradient,
just one that belongs to a different displacement. So the two paths are pinned
against each other here rather than each against itself.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from mlxmolkit.nddo.anal_grad import _pair_energy_many, analytical_gradient
from mlxmolkit.nddo.d_two_center import pair_cache
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.overlap_batch import overlap_pairs
from mlxmolkit.nddo.rotation_batch import rotate_pairs
from mlxmolkit.nddo.scf import _build_basis_info, nddo_energy

RDLogger.DisableLog("rdApp.*")

STEP = 1e-5


def setup(smiles, method="PM6"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        pytest.skip(f"embedding failed for {smiles}")
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    coords = np.asarray(mol.GetConformer().GetPositions())
    info = _build_basis_info(atoms, get_params(method), molecular_charge=0.0)
    return atoms, coords, info


def shifts():
    out = []
    for d in range(3):
        for sign in (1.0, -1.0):
            delta = np.zeros(3)
            delta[d] = sign * STEP
            out.append(delta)
    return out


def test_positional_and_keyed_paths_agree_on_every_displacement():
    atoms, coords, info = setup("CCO")
    params, starts = info["params"], info["atom_basis_start"]
    n_basis = info["n_basis"]
    n_atoms = len(atoms)
    all_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    P = nddo_energy(atoms, coords, method="PM6")["density"]

    ref_specs = [(params[i], params[j], coords[i], coords[j])
                 for i, j in all_pairs]
    shifted_specs = []
    for delta in shifts():
        shifted_specs.extend((params[i], params[j], coords[i], coords[j] + delta)
                             for i, j in all_pairs)
    ws = rotate_pairs([(a, b) for a, b, _c, _d in shifted_specs],
                      [(c, d) for _a, _b, c, d in shifted_specs])
    S = overlap_pairs(shifted_specs)

    n_pairs = len(all_pairs)
    for g, delta in enumerate(shifts()):
        block = slice(g * n_pairs, (g + 1) * n_pairs)
        with pair_cache(ref_specs):
            positional = _pair_energy_many(
                params, coords, all_pairs, starts, P, n_basis, shift=delta,
                ws_all=ws[block], S_all=S[block])

        # The keyed path needs this geometry declared, which is what a
        # d-bearing molecule's gradient does for all six at once.
        with pair_cache(ref_specs + shifted_specs[block]):
            keyed = _pair_energy_many(params, coords, all_pairs, starts, P,
                                      n_basis, shift=delta)

        assert positional.keys() == keyed.keys()
        for pair in all_pairs:
            assert positional[pair] == pytest.approx(keyed[pair], rel=1e-12), (
                f"displacement {g} pair {pair}: positional {positional[pair]!r} "
                f"vs keyed {keyed[pair]!r}"
            )


def test_slicing_the_wrong_displacement_is_detectable():
    """Guards the guard: the assertion above must be able to fail.

    Every displacement is a 1e-5 A shift of the same geometry, so the six
    blocks are numerically close. If the comparison were too loose to separate
    them, the test above would pass against a gradient that mixed them up.
    """
    atoms, coords, info = setup("CCO")
    params, starts = info["params"], info["atom_basis_start"]
    n_atoms = len(atoms)
    all_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    P = nddo_energy(atoms, coords, method="PM6")["density"]

    ref_specs = [(params[i], params[j], coords[i], coords[j])
                 for i, j in all_pairs]
    deltas = shifts()
    shifted_specs = []
    for delta in deltas:
        shifted_specs.extend((params[i], params[j], coords[i], coords[j] + delta)
                             for i, j in all_pairs)
    ws = rotate_pairs([(a, b) for a, b, _c, _d in shifted_specs],
                      [(c, d) for _a, _b, c, d in shifted_specs])
    S = overlap_pairs(shifted_specs)

    n_pairs = len(all_pairs)
    with pair_cache(ref_specs):
        right = _pair_energy_many(params, coords, all_pairs, starts, P,
                                  info["n_basis"], shift=deltas[0],
                                  ws_all=ws[0:n_pairs], S_all=S[0:n_pairs])
        # Same shift, but fed the +y block instead of the +x one.
        wrong = _pair_energy_many(params, coords, all_pairs, starts, P,
                                  info["n_basis"], shift=deltas[0],
                                  ws_all=ws[2 * n_pairs:3 * n_pairs],
                                  S_all=S[2 * n_pairs:3 * n_pairs])

    assert any(right[pair] != pytest.approx(wrong[pair], rel=1e-12)
               for pair in all_pairs), (
        "a mismatched displacement block produced identical energies, so the "
        "agreement test above proves nothing"
    )


@pytest.mark.parametrize("smiles,has_d", [("CCO", False), ("CSc1ccccc1", True)])
def test_both_routings_give_an_exact_gradient(smiles, has_d):
    """The routing is by element, so each molecule exercises exactly one path."""
    atoms, coords, info = setup(smiles)
    assert any(p.n_basis == 9 for p in info["params"]) is has_d

    _result, analytic = analytical_gradient(atoms, coords.copy(), method="PM6")

    h = 1e-4
    numeric = np.zeros_like(coords)
    for i in range(len(atoms)):
        for j in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[i, j] += h
            minus[i, j] -= h
            numeric[i, j] = (
                nddo_energy(atoms, plus, method="PM6")["energy_eV"]
                - nddo_energy(atoms, minus, method="PM6")["energy_eV"]
            ) / (2.0 * h)

    assert float(np.sqrt(np.mean((analytic - numeric) ** 2))) < 1e-5
