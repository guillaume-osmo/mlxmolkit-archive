"""Tests for mlxmolkit.cosine_dense."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlxmolkit.cosine_dense import (
    cosine_matrix_dense,
    l2_normalize_rows,
    max_cosine_to_set,
)


class TestL2Normalize:
    def test_unit_norm(self):
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(8, 16).astype(np.float32))
        x_n = l2_normalize_rows(x)
        mx.eval(x_n)
        norms = np.linalg.norm(np.array(x_n), axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_zero_row_safe(self):
        x = mx.array(np.zeros((1, 8), dtype=np.float32))
        x_n = l2_normalize_rows(x)
        mx.eval(x_n)
        # With eps, a zero row stays at zero (no NaN/Inf)
        assert np.all(np.isfinite(np.array(x_n)))


class TestCosineMatrix:
    def test_self_similarity_diagonal_is_one(self):
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(4, 16).astype(np.float32))
        sims = cosine_matrix_dense(x)
        mx.eval(sims)
        diag = np.array(sims).diagonal()
        assert np.allclose(diag, 1.0, atol=1e-5)

    def test_known_pair(self):
        a = mx.array(np.array([[1, 0, 0]], dtype=np.float32))
        b = mx.array(np.array([[0, 1, 0]], dtype=np.float32))  # orthogonal
        sim = cosine_matrix_dense(a, b)
        mx.eval(sim)
        assert abs(float(sim[0, 0])) < 1e-5

    def test_antipode_minus_one(self):
        a = mx.array(np.array([[1, 0, 0]], dtype=np.float32))
        b = mx.array(np.array([[-1, 0, 0]], dtype=np.float32))
        sim = cosine_matrix_dense(a, b)
        mx.eval(sim)
        assert float(sim[0, 0]) < -0.999

    def test_asymmetric_shape(self):
        rng = np.random.RandomState(1)
        a = mx.array(rng.randn(3, 8).astype(np.float32))
        b = mx.array(rng.randn(5, 8).astype(np.float32))
        sim = cosine_matrix_dense(a, b)
        mx.eval(sim)
        assert sim.shape == (3, 5)


class TestMaxCosineToSet:
    def test_empty_reference_zero(self):
        x = mx.array(np.random.randn(4, 16).astype(np.float32))
        empty = mx.zeros((0, 16))
        out = max_cosine_to_set(x, empty)
        mx.eval(out)
        assert out.shape == (4,)
        assert np.allclose(np.array(out), 0.0)

    def test_self_against_self_is_one(self):
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(5, 16).astype(np.float32))
        out = max_cosine_to_set(x, x)
        mx.eval(out)
        # Each row's max neighbor is itself (cos = 1.0)
        assert np.allclose(np.array(out), 1.0, atol=1e-5)

    def test_no_overlap_returns_orthogonal_score(self):
        a = mx.array(np.array([[1, 0, 0, 0]], dtype=np.float32))
        b = mx.array(np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
        out = max_cosine_to_set(a, b)
        mx.eval(out)
        assert abs(float(out[0])) < 1e-5
