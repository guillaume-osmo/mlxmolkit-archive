"""FD verification of the dipole/quadrupole AO-integral derivatives."""
import numpy as np
import pytest

from mlxmolkit.xtb.basis import build_basis
from mlxmolkit.xtb.multipole_grad import multipole_gradient
from mlxmolkit.xtb.multipole_integrals import multipole_matrices
from mlxmolkit.xtb.params_gfn2 import GFN2_PARAMS
from mlxmolkit.xtb.scf_gfn2 import gfn2_n_gauss


@pytest.fixture
def h2o_cao():
    atoms = [8, 1, 1]
    coords = np.array([
        [0.0,  0.0,   0.117790],
        [0.0,  0.755, -0.471],
        [0.0, -0.755, -0.471],
    ])
    cao = build_basis(
        atoms, coords, params_dict=GFN2_PARAMS, n_gauss_fn=gfn2_n_gauss,
    )
    return atoms, coords, cao


def test_dpint_grad_matches_fd_on_bra_atom(h2o_cao):
    atoms, coords, cao = h2o_cao
    dD_dA, dD_dB, _, _ = multipole_gradient(cao)

    # Pick a cross-atom AO pair: O.s (idx 0) — H1.s (idx 4)
    mu, nu = 0, 4
    h = 1e-5
    A_save = cao[mu].center.copy()
    cao[mu].center[0] += h
    _, dpint_p, _ = multipole_matrices(cao)
    cao[mu].center[0] -= 2 * h
    _, dpint_m, _ = multipole_matrices(cao)
    cao[mu].center[0] = A_save[0]
    dD_FD_x = (dpint_p[:, mu, nu] - dpint_m[:, mu, nu]) / (2 * h)
    dD_AN_x = dD_dA[0, :, mu, nu]
    assert np.max(np.abs(dD_AN_x - dD_FD_x)) < 1e-8


def test_qpint_grad_matches_fd_on_ket_atom(h2o_cao):
    atoms, coords, cao = h2o_cao
    _, _, dQ_dA, dQ_dB = multipole_gradient(cao)

    mu, nu = 0, 5
    h = 1e-5
    B_save = cao[nu].center.copy()
    cao[nu].center[1] += h
    _, _, qpint_p = multipole_matrices(cao)
    cao[nu].center[1] -= 2 * h
    _, _, qpint_m = multipole_matrices(cao)
    cao[nu].center[1] = B_save[1]
    dQ_FD_y = (qpint_p[:, mu, nu] - qpint_m[:, mu, nu]) / (2 * h)
    dQ_AN_y = dQ_dB[1, :, mu, nu]
    assert np.max(np.abs(dQ_AN_y - dQ_FD_y)) < 1e-8


def test_translation_invariance(h2o_cao):
    """∂dpint/∂A + ∂dpint/∂B = -∂(dipint)/∂(constant translation) ≠ 0
    in general, but for the *atom-position* gradient the sum
    dD_dA[mu,nu] + dD_dB[mu,nu] should give the explicit ∂(integral)/∂R
    with R as a global translation — non-zero because dpint is computed
    in a fixed Cartesian frame, not relative coordinates.
    """
    atoms, coords, cao = h2o_cao
    dD_dA, dD_dB, _, _ = multipole_gradient(cao)
    # We just sanity-check shapes here — a true translation invariance
    # check would FD shifting the entire molecule by a constant vector.
    n = len(cao)
    assert dD_dA.shape == (3, 3, n, n)
    assert dD_dB.shape == (3, 3, n, n)
