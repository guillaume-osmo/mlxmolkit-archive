"""The fixed Gauss grid must reproduce the adaptive quadrature it replaces."""
import numpy as np
import pytest

from mlxmolkit.nddo.slater_overlap_ref import (
    reduced_overlap, reduced_overlap_quadrature,
)

# (na, la, nb, lb, m, za, zb, R) -- iodine-like n=5 sigma/pi/delta, plus qn<=4 controls
CASES = [
    (5, 0, 5, 0, 0, 2.7, 2.7, 5.0), (5, 0, 5, 1, 0, 2.7, 2.1, 5.0),
    (5, 1, 5, 1, 1, 2.1, 2.1, 5.0), (5, 2, 5, 2, 2, 3.0, 3.0, 5.0),
    (5, 0, 5, 2, 0, 2.7, 3.0, 5.0), (5, 1, 5, 2, 1, 2.1, 3.0, 5.0),
    (2, 0, 2, 1, 0, 1.8, 1.6, 2.6), (3, 2, 3, 2, 0, 2.2, 2.2, 3.8),
    (4, 1, 5, 2, 1, 1.9, 3.0, 4.2),
]


@pytest.mark.parametrize("case", CASES)
def test_grid_matches_quadrature(case):
    ref = reduced_overlap_quadrature(*case)
    got = reduced_overlap(*case)
    assert got == pytest.approx(ref, rel=1e-9, abs=1e-14)


def test_vanishes_when_m_exceeds_l():
    assert reduced_overlap(5, 0, 5, 1, 1, 2.7, 2.1, 5.0) == 0.0
