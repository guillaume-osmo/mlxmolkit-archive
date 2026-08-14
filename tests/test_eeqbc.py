import numpy as np

from mlxmolkit.xtb.eeqbc import (
    eeqbc_capacitance_matrix,
    eeqbc_capacitance_pair,
    eeqbc_charges,
    eeqbc_pair_rvdw_matrix_ang,
    eeqbc_solve,
)


WATER_ATOMS = np.array([8, 1, 1], dtype=np.intp)
WATER_COORDS = np.array(
    [
        [0.0, 0.0, 0.117790],
        [0.0, 0.755453, -0.471160],
        [0.0, -0.755453, -0.471160],
    ],
    dtype=np.float64,
)


def test_eeqbc_capacitance_pair_is_bounded_by_geometric_cap():
    c_same = eeqbc_capacitance_pair(0.5, 2.0, 4.0, 9.0)
    c_far = eeqbc_capacitance_pair(20.0, 2.0, 4.0, 9.0)

    assert 0.0 <= c_far < c_same <= 6.0


def test_eeqbc_capacitance_matrix_is_laplacian_augmented():
    cmat = eeqbc_capacitance_matrix(WATER_ATOMS, WATER_COORDS)

    assert cmat.shape == (4, 4)
    np.testing.assert_allclose(cmat, cmat.T, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.sum(cmat[:-1, :-1], axis=1), 0.0, atol=1.0e-14)
    assert cmat[-1, -1] == 1.0


def test_eeqbc_pair_rvdw_matrix_uses_arithmetic_scale_and_angstrom_units():
    pair = eeqbc_pair_rvdw_matrix_ang(np.array([1, 8], dtype=np.intp))

    assert pair[0, 1] == pair[1, 0]
    assert pair[0, 1] > 2.0
    assert pair[0, 1] < 3.0


def test_eeqbc_water_solve_matches_formula_snapshot():
    result = eeqbc_solve(WATER_ATOMS, WATER_COORDS)

    np.testing.assert_allclose(
        result.charges,
        [-0.399162952686, 0.199581476343, 0.199581476343],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(result.charges.sum(), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.amat, result.amat.T, rtol=0.0, atol=0.0)
    assert result.energy < 0.0


def test_eeqbc_charged_species_sum_to_total_charge():
    atoms = np.array([7, 1, 1, 1, 1], dtype=np.intp)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.6, 0.6],
            [-0.6, -0.6, 0.6],
            [-0.6, 0.6, -0.6],
            [0.6, -0.6, -0.6],
        ],
        dtype=np.float64,
    )

    q = eeqbc_charges(atoms, coords, total_charge=1.0)

    np.testing.assert_allclose(np.sum(q), 1.0, atol=1.0e-12)
