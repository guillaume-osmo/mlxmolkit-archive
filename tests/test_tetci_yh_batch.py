"""The batched d-orbital attraction must equal the scalar one it replaces.

`yh_e1b_contribution` builds the 9x9 electron-nuclear attraction block on a
d-bearing atom by rotating 45 local-frame integrals with a 15x45 Wigner-D
coefficient matrix. `generate_rotation_matrix` already accepts a pair axis, so
the scalar routine was rebuilding that machinery for a single row per call —
589 ms of an 800-molecule `prepare_batch`.

`yh_e1b_batch` lifts the whole path to arrays. Nothing about the arithmetic
changes, so the scalar is the reference and these compare against it directly
rather than against stored numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.tetci_yh import yh_e1b_batch, yh_e1b_contribution

PARAMS = get_params("PM6")
D_ELEMENTS = [Z for Z, p in PARAMS.items() if p.n_basis == 9]
PARTNERS = [1, 6, 7, 8, 9, 16, 17]


def scalar_stack(pair_params, pair_coords):
    return np.array([yh_e1b_contribution(pA, pB, rA, rB)
                     for (pA, pB), (rA, rB) in zip(pair_params, pair_coords)])


def test_matches_the_scalar_routine_over_random_pairs():
    """Agreement is to 1e-14 on blocks of order 90 eV — 1e-16 relative.

    Not bit-exact, and deliberately so: the batch evaluates the same arithmetic
    on an array layout, and a few pairs land one ULP apart. That is eleven
    orders below the float32 Metal kernel this feeds.
    """
    rng = np.random.default_rng(0)
    pair_params, pair_coords = [], []
    for _ in range(2000):
        za, zb = int(rng.choice(D_ELEMENTS)), int(rng.choice(PARTNERS))
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        origin = rng.normal(size=3)
        pair_params.append((PARAMS[za], PARAMS[zb]))
        pair_coords.append((origin, origin + direction * (0.9 + 2.5 * rng.random())))

    got = yh_e1b_batch(pair_params, pair_coords)
    assert np.abs(got - scalar_stack(pair_params, pair_coords)).max() < 1e-12


@pytest.mark.parametrize("zb", PARTNERS + [15, 35, 53])
def test_every_partner_shell_is_exact(zb):
    """The partner enters only through rho0_B and the Z_B prefactor, but it also
    picks the pair type inside two_center_integrals — H (1 orbital), sp (4) and
    spd (9) take three different branches there."""
    pair_params = [(PARAMS[za], PARAMS[zb]) for za in D_ELEMENTS]
    pair_coords = [(np.zeros(3), np.array([0.3, -0.7, 1.5]))] * len(pair_params)
    got = yh_e1b_batch(pair_params, pair_coords)
    assert np.abs(got - scalar_stack(pair_params, pair_coords)).max() < 1e-12


def test_mixed_partner_shells_in_one_call():
    """Pairs of different partner width share a call, so the grouping inside
    two_center_integrals_batch must put each row back where it came from. A
    transposed regroup is invisible when every pair has the same shape."""
    rng = np.random.default_rng(1)
    pair_params, pair_coords = [], []
    for k in range(300):
        zb = PARTNERS[k % len(PARTNERS)]
        pair_params.append((PARAMS[D_ELEMENTS[k % len(D_ELEMENTS)]], PARAMS[zb]))
        origin = rng.normal(size=3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        pair_coords.append((origin, origin + direction * 1.8))

    got = yh_e1b_batch(pair_params, pair_coords)
    assert np.abs(got - scalar_stack(pair_params, pair_coords)).max() < 1e-12


def test_the_block_is_symmetric():
    """H_core is read with UPLO='L' by numpy and UPLO='U' by the Metal kernel,
    so a block that filled only one triangle would give two different answers.
    """
    pair_params = [(PARAMS[16], PARAMS[6]), (PARAMS[17], PARAMS[1])]
    pair_coords = [(np.zeros(3), np.array([1.1, 0.4, -0.9]))] * 2
    got = yh_e1b_batch(pair_params, pair_coords)
    assert np.abs(got - np.swapaxes(got, 1, 2)).max() == 0.0


def test_an_empty_pair_list_is_allowed():
    """An sp-only batch reaches this with nothing to do."""
    assert yh_e1b_batch([], []).shape == (0, 9, 9)


def test_poij_deduplication_matches_the_per_element_search():
    """POIJ's golden-section search is deduplicated; it must not change answers.

    The search depends only on (L, d^2, fg) — element parameters, never on
    geometry — so `POIJ` now solves each distinct pair once and indexes the
    result back. This re-runs the original per-element loop directly and
    compares, including the fg == 0 branch that zeroes the result and the
    repeated values the deduplication collapses.
    """
    from mlxmolkit.nddo._pyseqm_port.cal_par_np import POIJ, _poij_one

    rng = np.random.default_rng(3)
    for L in (1, 2):
        d = np.abs(rng.normal(size=40)) + 0.05
        fg = np.abs(rng.normal(size=40)) + 0.01
        d[:8] = d[0]                       # duplicates: what dedup collapses
        fg[:8] = fg[0]
        fg[9] = 0.0                        # the zeroing branch
        got = POIJ(L, d, fg)
        want = np.array([_poij_one(L, float(a * a), float(b)) for a, b in zip(d, fg)])
        assert np.array_equal(got, want)
        assert got[9] == 0.0
