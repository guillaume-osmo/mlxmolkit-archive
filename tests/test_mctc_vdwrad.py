import numpy as np
import pytest

from mlxmolkit.xtb.mctc_vdwrad import (
    MAX_Z,
    load_mctc_vdwrad_packed,
    mctc_vdw_pair_matrix_bohr,
    mctc_vdw_pair_radius_bohr,
)


def test_mctc_vdwrad_packed_shape_and_known_head_values():
    packed = load_mctc_vdwrad_packed()

    assert packed.shape == (MAX_Z * (MAX_Z + 1) // 2,)
    assert packed[0] == pytest.approx(4.123949321759188)
    assert packed[1] == pytest.approx(3.5048750433335307)
    assert packed[2] == pytest.approx(3.2781079083790776)


def test_mctc_vdw_pair_radius_uses_triangular_binary_indexing():
    # H-H, H-He, He-He are the first three packed entries.
    assert mctc_vdw_pair_radius_bohr(1, 1) == pytest.approx(4.123949321759188)
    assert mctc_vdw_pair_radius_bohr(1, 2) == pytest.approx(3.5048750433335307)
    assert mctc_vdw_pair_radius_bohr(2, 1) == pytest.approx(3.5048750433335307)
    assert mctc_vdw_pair_radius_bohr(2, 2) == pytest.approx(3.2781079083790776)


def test_mctc_vdw_pair_matrix_is_symmetric():
    atoms = np.array([8, 1, 6, 16], dtype=np.intp)
    matrix = mctc_vdw_pair_matrix_bohr(atoms)

    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=0.0)
    assert matrix[0, 1] == pytest.approx(mctc_vdw_pair_radius_bohr(8, 1))
    assert matrix[2, 3] == pytest.approx(mctc_vdw_pair_radius_bohr(6, 16))
