"""Tests for the GFN0-xTB coordination-number primitive."""

import mlx.core as mx
import numpy as np
import pytest


def _h2o_geom():
    atoms = mx.array([8, 1, 1], dtype=mx.int32)
    coords = mx.array(
        [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ],
        dtype=mx.float32,
    )
    return atoms, coords


def _ch4_geom():
    atoms = mx.array([6, 1, 1, 1, 1], dtype=mx.int32)
    coords = mx.array(
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
        dtype=mx.float32,
    )
    return atoms, coords


def test_h2o_cn():
    """Water O ≈ 2.0, H ≈ 1.0 each (chemically expected)."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    atoms, coords = _h2o_geom()
    CN = coordination_number_erf(coords, atoms)
    mx.eval(CN)
    assert float(CN[0]) == pytest.approx(2.0, abs=0.05)   # O bonded to 2 H
    assert float(CN[1]) == pytest.approx(1.0, abs=0.05)
    assert float(CN[2]) == pytest.approx(1.0, abs=0.05)


def test_ch4_cn():
    """Methane C ≈ 4.0, H ≈ 1.0 each."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    atoms, coords = _ch4_geom()
    CN = coordination_number_erf(coords, atoms)
    mx.eval(CN)
    assert float(CN[0]) == pytest.approx(4.0, abs=0.1)
    for i in range(1, 5):
        assert float(CN[i]) == pytest.approx(1.0, abs=0.05)


def test_2d_input_returns_1d_output():
    """Single-mol API: (n, 3) input → (n,) output, no batch dim."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    atoms, coords = _h2o_geom()
    CN = coordination_number_erf(coords, atoms)
    assert CN.shape == (3,)


def test_3d_input_returns_2d_output():
    """Batched: (B, n, 3) input → (B, n) output."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    atoms_b = mx.array(
        [[8, 1, 1, 0, 0], [6, 1, 1, 1, 1]], dtype=mx.int32
    )
    coords_b = mx.array(
        [
            [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469],
             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0],
             [0.629, 0.629, 0.629],
             [-0.629, -0.629, 0.629],
             [-0.629, 0.629, -0.629],
             [0.629, -0.629, -0.629]],
        ],
        dtype=mx.float32,
    )
    n_atoms = mx.array([3, 5], dtype=mx.int32)
    CN_b = coordination_number_erf(coords_b, atoms_b, n_atoms)
    mx.eval(CN_b)
    assert CN_b.shape == (2, 5)
    # H2O block (mol 0): real atoms in [0, 3), padded in [3, 5)
    assert float(CN_b[0, 0]) == pytest.approx(2.0, abs=0.05)
    assert float(CN_b[0, 3]) == 0.0          # padded
    assert float(CN_b[0, 4]) == 0.0          # padded
    # CH4 block: real atoms across all 5
    assert float(CN_b[1, 0]) == pytest.approx(4.0, abs=0.1)


def test_padding_does_not_leak_into_real_atoms():
    """A padded atom (Z=0) physically present in the coords array but
    masked by n_atoms_arr must not contribute to any real atom's CN."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    atoms_real_only = mx.array([[8, 1, 1, 0, 0]], dtype=mx.int32)
    atoms_with_phantom = mx.array([[8, 1, 1, 6, 6]], dtype=mx.int32)
    coords = mx.array(
        [[
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
            [10.0, 10.0, 10.0],   # would-be Z=6 atom (very far)
            [-10.0, 10.0, 10.0],
        ]],
        dtype=mx.float32,
    )
    n_atoms = mx.array([3], dtype=mx.int32)
    CN_real = coordination_number_erf(coords, atoms_real_only, n_atoms)
    CN_phantom = coordination_number_erf(coords, atoms_with_phantom, n_atoms)
    mx.eval(CN_real, CN_phantom)
    # First 3 entries (real atoms) must be identical regardless of what
    # sits in the padded slots — n_atoms_arr=3 should mask them out.
    np.testing.assert_allclose(
        np.array(CN_real[0, :3]), np.array(CN_phantom[0, :3]), atol=1e-6
    )


def test_isolated_atom_has_zero_cn():
    """A single atom in a batch returns CN=0 (no neighbors)."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    atoms = mx.array([6], dtype=mx.int32)
    coords = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)
    CN = coordination_number_erf(coords, atoms)
    mx.eval(CN)
    assert float(CN[0]) == 0.0


def test_far_atoms_have_negligible_cn():
    """Two atoms 100 Å apart give ~0 CN (no overlap of the erf curve)."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    atoms = mx.array([6, 6], dtype=mx.int32)
    coords = mx.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=mx.float32)
    CN = coordination_number_erf(coords, atoms)
    mx.eval(CN)
    assert float(CN[0]) < 1e-6
    assert float(CN[1]) < 1e-6


def test_close_atoms_saturate_at_max_cn():
    """An impossibly clashed cluster saturates at max_cn."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    n = 12
    atoms = mx.array([6] * n, dtype=mx.int32)
    coords = mx.array(
        np.tile([[0.0, 0.0, 0.0]], (n, 1)) + np.random.RandomState(0).randn(n, 3) * 0.01,
        dtype=mx.float32,
    )
    CN = coordination_number_erf(coords, atoms, max_cn=8.0)
    mx.eval(CN)
    for i in range(n):
        assert float(CN[i]) == pytest.approx(8.0, abs=1e-4)


def test_diagonal_does_not_self_count():
    """Atom A's distance to itself is 0; without the diagonal mask, it
    would contribute 1.0 erf-count and inflate every CN by exactly 1.0."""
    from mlxmolkit.xtb.cn import coordination_number_erf
    atoms, coords = _ch4_geom()
    CN = coordination_number_erf(coords, atoms)
    mx.eval(CN)
    # If the diagonal weren't zeroed, C would read ~5.0 (4 neighbors + 1 self).
    assert float(CN[0]) < 4.5
