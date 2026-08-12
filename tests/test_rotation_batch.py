"""The vectorised rotation must equal the scalar one it replaces.

`rotate_integrals_to_molecular_frame` walks a four-deep Python loop over the
100 distinct orbital-pair combinations of an sp pair. Every element of the
result is a linear combination of the local-frame integrals with coefficients
polynomial in the rotation vectors, so the loop is replaceable by broadcasting
over a pair axis — but only if the replacement is exact, since every energy in
the library flows through it.

The scalar routine is the reference. These tests compare against it directly
rather than against stored numbers, so they stay honest if the physics changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.rotation import rotate_integrals_to_molecular_frame
from mlxmolkit.nddo.rotation_batch import rotate_pairs

PARAMS = get_params("PM6")
SP_ELEMENTS = [1, 6, 7, 8, 9]


def random_pairs(n, rng, elements=SP_ELEMENTS):
    pair_params, pair_coords = [], []
    for _ in range(n):
        za, zb = int(rng.choice(elements)), int(rng.choice(elements))
        origin = rng.normal(size=3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        distance = 0.9 + 2.5 * rng.random()
        pair_params.append((PARAMS[za], PARAMS[zb]))
        pair_coords.append((origin, origin + direction * distance))
    return pair_params, pair_coords


def test_matches_the_scalar_routine_over_random_pairs():
    rng = np.random.default_rng(0)
    pair_params, pair_coords = random_pairs(400, rng)
    reference = np.array([rotate_integrals_to_molecular_frame(pA, pB, rA, rB)[0]
                          for (pA, pB), (rA, rB) in zip(pair_params, pair_coords)])
    assert np.array_equal(rotate_pairs(pair_params, pair_coords), reference)


@pytest.mark.parametrize("za,zb,kind", [
    (6, 6, "XX"), (6, 8, "XX"), (7, 8, "XX"),
    (6, 1, "XH"), (8, 1, "XH"),
    (1, 6, "HX"), (1, 8, "HX"),      # hydrogen first: solved swapped, transposed
    (1, 1, "HH"),
])
def test_every_pair_type_is_exact(za, zb, kind):
    """Each type takes a different branch; HX is the one most easily got wrong."""
    rng = np.random.default_rng(1)
    pA, pB = PARAMS[za], PARAMS[zb]
    for _ in range(25):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        rA = np.zeros(3)
        rB = direction * (0.9 + 2.0 * rng.random())
        got = rotate_pairs([(pA, pB)], [(rA, rB)])[0]
        want = rotate_integrals_to_molecular_frame(pA, pB, rA, rB)[0]
        assert np.array_equal(got, want), f"{kind} pair diverged"


def test_the_attraction_blocks_are_slices_of_the_rotated_tensor():
    """The batched path returns only w and derives e1b/e2a from it.

    e1b = -Z_B * w[:, :, 0, 0] and e2a = -Z_A * w[0, 0, :, :]. If that ever
    stopped holding, the gradient would silently lose its attraction terms.
    """
    rng = np.random.default_rng(2)
    pair_params, pair_coords = random_pairs(60, rng)
    for (pA, pB), (rA, rB) in zip(pair_params, pair_coords):
        w, e1b, e2a = rotate_integrals_to_molecular_frame(pA, pB, rA, rB)
        assert np.abs(e1b + float(pB.n_valence) * w[:, :, 0, 0]).max() < 1e-12
        assert np.abs(e2a + float(pA.n_valence) * w[0, 0, :, :]).max() < 1e-12


def test_coincident_atoms_give_a_zero_block():
    """Guard the R -> 0 branch, which the scalar routine short-circuits."""
    got = rotate_pairs([(PARAMS[6], PARAMS[6])], [(np.zeros(3), np.zeros(3))])
    assert np.array_equal(got[0], np.zeros((4, 4, 4, 4)))


def test_an_empty_pair_list_is_allowed():
    """A molecule with a single atom has no pairs; that must not raise."""
    assert rotate_pairs([], []).shape == (0, 4, 4, 4, 4)
