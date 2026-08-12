"""The table-driven overlap must reproduce the scalar one it replaces.

`overlap_batch.TABLE` encodes six jcall branches as coefficient triples. A
transposed index there is invisible by inspection and would surface only as a
wrong energy in the fourth decimal, so the table is checked against the scalar
routine over every element pair it claims to cover.

Agreement is to float64 machine epsilon, not bit-exact: the evaluator sums the
triples in table order while the scalar sums in its written parenthesisation.
Summing the same triples forward and reversed differs by the same ~1e-15, and
against float128 the table's order is exact — so the residual is associativity,
not arithmetic.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.overlap import overlap_molecular_frame
from mlxmolkit.nddo.overlap_batch import TABLE, overlap_pairs

PARAMS = get_params("PM6")
COVERED = [1, 6, 7, 8, 9]          # the table's elements: qn <= 3 and sp only
FALLBACK = [16, 17, 35, 53]        # d orbitals (S, Cl) or qn > 3 (Br, I)
EPS = 1e-14


def specs(za, zb, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        origin = rng.normal(size=3) * 0.4
        out.append((PARAMS[za], PARAMS[zb], origin,
                    origin + v * (0.7 + 2.5 * rng.random())))
    return out


@pytest.mark.parametrize("za,zb", list(itertools.product(COVERED, COVERED)))
def test_matches_the_scalar_for_every_covered_pair(za, zb):
    """Both orderings of every covered element pair."""
    pair_specs = specs(za, zb, 120, seed=za * 100 + zb)
    got = overlap_pairs(pair_specs)
    for spec, batch in zip(pair_specs, got):
        reference = overlap_molecular_frame(*spec)
        assert batch.shape == reference.shape
        assert np.abs(batch - reference).max() < EPS


@pytest.mark.parametrize("axis", [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
                                  (-1, 1e-9, 0), (0, 1, 0)])
def test_degenerate_directions(axis):
    """The quaternion is ill-conditioned near v = (-1,0,0); the scalar takes a
    separate antipodal branch and the batch must take it in the same lanes."""
    direction = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    pair_specs = [(PARAMS[a], PARAMS[b], np.zeros(3), direction * 1.5)
                  for a, b in [(6, 6), (6, 1), (1, 6), (8, 8), (7, 1)]]
    for spec, batch in zip(pair_specs, overlap_pairs(pair_specs)):
        assert np.abs(batch - overlap_molecular_frame(*spec)).max() < EPS


@pytest.mark.parametrize("za", FALLBACK)
def test_uncovered_pairs_route_to_the_scalar(za):
    """d orbitals and qn > 3 must fall back, not be guessed at."""
    pair_specs = specs(za, 6, 40, seed=za) + specs(6, za, 40, seed=za + 1)
    for spec, batch in zip(pair_specs, overlap_pairs(pair_specs)):
        assert np.array_equal(batch, overlap_molecular_frame(*spec))


def test_mixed_batch_keeps_input_order():
    """Pairs are grouped by jcall internally; results must come back in order."""
    pair_specs = []
    for za, zb in [(1, 1), (6, 1), (6, 6), (16, 1), (6, 16), (8, 8), (35, 6)]:
        pair_specs.extend(specs(za, zb, 5, seed=za + zb))
    for spec, batch in zip(pair_specs, overlap_pairs(pair_specs)):
        assert np.abs(batch - overlap_molecular_frame(*spec)).max() < EPS


def test_the_table_covers_every_jcall():
    assert set(TABLE) == {2, 3, 4, 431, 5, 6}
    for jcall, terms in TABLE.items():
        expected = {"S111"} if jcall == 2 else (
            {"S111", "S211"} if jcall in (3, 431) else
            {"S111", "S211", "S121", "S221", "S222"})
        assert set(terms) == expected, f"jcall {jcall}"


def test_empty_input():
    assert overlap_pairs([]) == []
