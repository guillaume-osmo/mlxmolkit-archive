import numpy as np
import pytest

from mlxmolkit.xtb.basis import build_basis
from mlxmolkit.xtb.multipole_grad import multipole_gradient
from mlxmolkit.xtb.multipole_integrals import multipole_matrices
from mlxmolkit.xtb.multipole_integrals_cpp import (
    CPP_AVAILABLE,
    mmompop_cpp,
    overlap_gradient_cpp,
    multipole_gradient_cpp,
    multipole_matrices_cpp,
)
from mlxmolkit.xtb.aes import mmompop
from mlxmolkit.xtb.overlap_grad import overlap_gradient
from mlxmolkit.xtb.params_gfn2 import GFN2_PARAMS
from mlxmolkit.xtb.scf_gfn2 import gfn2_n_gauss


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ multipole extension is not built")
def test_multipole_matrices_cpp_matches_python_reference():
    atoms = [8, 1, 1]
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )
    basis = build_basis(
        atoms,
        coords,
        params_dict=GFN2_PARAMS,
        n_gauss_fn=gfn2_n_gauss,
    )

    ref = multipole_matrices(basis)
    got = multipole_matrices_cpp(basis)

    for value, expected in zip(got, ref):
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ multipole extension is not built")
def test_multipole_gradient_cpp_matches_python_reference():
    atoms = [8, 1, 1]
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )
    basis = build_basis(
        atoms,
        coords,
        params_dict=GFN2_PARAMS,
        n_gauss_fn=gfn2_n_gauss,
    )

    ref = multipole_gradient(basis)
    got = multipole_gradient_cpp(basis)

    for value, expected in zip(got, ref):
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ multipole extension is not built")
def test_overlap_gradient_cpp_matches_python_reference():
    atoms = [8, 1, 1]
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )
    basis = build_basis(
        atoms,
        coords,
        params_dict=GFN2_PARAMS,
        n_gauss_fn=gfn2_n_gauss,
    )

    ref = overlap_gradient(basis)
    got = overlap_gradient_cpp(basis)

    for value, expected in zip(got, ref):
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ multipole extension is not built")
def test_mmompop_cpp_matches_python_reference():
    rng = np.random.default_rng(864)
    nao = 10
    nat = 5
    A = rng.normal(size=(nao, nao))
    P = A + A.T
    B = rng.normal(size=(nao, nao))
    S = B + B.T
    dpint = rng.normal(size=(3, nao, nao))
    qpint = rng.normal(size=(6, nao, nao))
    for k in range(3):
        dpint[k] = 0.5 * (dpint[k] + dpint[k].T)
    for k in range(6):
        qpint[k] = 0.5 * (qpint[k] + qpint[k].T)
    aoat = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    coords = rng.normal(size=(nat, 3))

    ref = mmompop(P, S, dpint, qpint, aoat, coords)
    got = mmompop_cpp(P, S, dpint, qpint, aoat, coords)

    for value, expected in zip(got, ref):
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
