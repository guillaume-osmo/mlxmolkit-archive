import pytest
import numpy as np

from mlxmolkit.xtb.gxtb_acp import build_gxtb_acp_hamiltonian
from mlxmolkit.xtb.gxtb_basis import (
    build_gxtb_qvszp_basis,
    qvszp_qeff,
    qvszp_qeff_derivatives,
)
from mlxmolkit.xtb.hcore_gxtb import (
    GXTB_D_H0_SCALE_DAMP,
    _diat_scale,
    _diatomic_scaled_overlap_cao,
    build_hcore_gxtb,
    gxtb_shell_selfenergies,
)
from mlxmolkit.xtb.scf_gxtb import (
    GXTB_HALIDE_INCREMENT_CORRECTION,
    _coulomb_matrix,
    _first_order_offsite,
    _first_order_onsite,
    _halide_increment_correction,
    _mfx_fock_energy,
    _mfx_gamma_ao,
    _third_order_twobody,
    gxtb_energy,
    gxtb_gradient_numerical,
)


WATER_ATOMS = [8, 1, 1]
WATER_COORDS = np.array(
    [
        [0.0, 0.0, 0.117790],
        [0.0, 0.755453, -0.471160],
        [0.0, -0.755453, -0.471160],
    ],
    dtype=np.float64,
)
H2S_ATOMS = [16, 1, 1]
H2S_COORDS = np.array(
    [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.962000, 0.692000],
        [0.000000, -0.962000, 0.692000],
    ],
    dtype=np.float64,
)

import pathlib as _pathlib

try:
    from mlxmolkit.xtb import _gxtb_cpp as _cpp  # noqa: F401
    _HAVE_GXTB_CPP = True
except ImportError:
    _HAVE_GXTB_CPP = False

_ONECX = (_pathlib.Path(__file__).resolve().parent.parent
          / "data" / "gxtb_onecxints_extracted.npz")

# The extension needs `build_ext --inplace` and the .npz is untracked, so
# neither is present in a clean clone.
_needs_gxtb_cpp = pytest.mark.skipif(
    not _HAVE_GXTB_CPP or not _ONECX.exists(),
    reason=f"needs the _gxtb_cpp extension (built={_HAVE_GXTB_CPP}) and "
           f"data/gxtb_onecxints_extracted.npz (present={_ONECX.exists()})",
)



def test_qvszp_qeff_derivative_matches_central_difference():
    q = np.array([-0.4, 0.2])
    cn = np.array([1.5, 0.8])
    k0 = np.array([1.0, 0.9])
    k1 = np.array([0.01, 0.2])
    k2 = np.array([0.3, -0.1])
    k3 = np.array([0.05, 0.02])
    dq, dcn = qvszp_qeff_derivatives(q, cn, k0, k1, k2, k3)
    h = 1.0e-6
    np.testing.assert_allclose(
        dq,
        (qvszp_qeff(q + h, cn, k0, k1, k2, k3) - qvszp_qeff(q - h, cn, k0, k1, k2, k3)) / (2 * h),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dcn,
        (qvszp_qeff(q, cn + h, k0, k1, k2, k3) - qvszp_qeff(q, cn - h, k0, k1, k2, k3)) / (2 * h),
        rtol=1e-9,
        atol=1e-9,
    )


def test_gxtb_qvszp_active_basis_water_overlap_is_well_conditioned():
    basis = build_gxtb_qvszp_basis(WATER_ATOMS, WATER_COORDS)
    assert len(basis.sao_basis) == 6
    assert basis.shell_atom.tolist() == [0, 0, 1, 2]
    assert basis.shell_l.tolist() == [0, 1, 0, 0]
    np.testing.assert_allclose(np.diag(basis.S), 1.0, atol=2e-12)
    assert np.linalg.eigvalsh(basis.S).min() > 0.1
    np.testing.assert_allclose(basis.eeqbc_charges.sum(), 0.0, atol=1e-12)


def test_gxtb_hcore_water_is_symmetric_and_diagonal_matches_selfenergy():
    basis = build_gxtb_qvszp_basis(WATER_ATOMS, WATER_COORDS)
    H0, shell_self = build_hcore_gxtb(WATER_ATOMS, WATER_COORDS, basis)
    np.testing.assert_allclose(H0, H0.T, atol=1e-14)
    np.testing.assert_allclose(shell_self, gxtb_shell_selfenergies(WATER_ATOMS, basis))
    np.testing.assert_allclose(np.diag(H0), shell_self[basis.bf_to_shell], atol=1e-14)


def test_gxtb_hcore_scales_sulfur_d_overlap_blocks():
    basis = build_gxtb_qvszp_basis(H2S_ATOMS, H2S_COORDS)
    scaled = _diatomic_scaled_overlap_cao(
        np.asarray(H2S_ATOMS, dtype=np.intp),
        H2S_COORDS * 1.8897259886,
        basis,
    )
    sulfur_d = np.where((basis.cao_bf_to_shell == 2))[0]
    first_h = np.where((basis.cao_bf_to_shell == 3))[0]
    raw_block = basis.S_cao[np.ix_(sulfur_d, first_h)]
    scaled_block = scaled[np.ix_(sulfur_d, first_h)]
    np.testing.assert_allclose(
        scaled_block,
        raw_block * (1.0 + GXTB_D_H0_SCALE_DAMP * (_diat_scale(16, 1, 0) - 1.0)),
        rtol=1e-13,
        atol=1e-13,
    )


def test_gxtb_first_order_onsite_potential_matches_central_difference():
    basis = build_gxtb_qvszp_basis(WATER_ATOMS, WATER_COORDS)
    qsh = np.array([0.18, -0.31, 0.06, 0.07], dtype=np.float64)
    _, potential = _first_order_onsite(
        np.asarray(WATER_ATOMS, dtype=np.intp),
        basis.cn,
        basis.shell_atom,
        qsh,
    )
    h = 1.0e-6
    fd = np.zeros_like(qsh)
    for i in range(qsh.size):
        qp = qsh.copy()
        qm = qsh.copy()
        qp[i] += h
        qm[i] -= h
        ep, _ = _first_order_onsite(np.asarray(WATER_ATOMS, dtype=np.intp), basis.cn, basis.shell_atom, qp)
        em, _ = _first_order_onsite(np.asarray(WATER_ATOMS, dtype=np.intp), basis.cn, basis.shell_atom, qm)
        fd[i] = (ep - em) / (2.0 * h)
    np.testing.assert_allclose(potential, fd, rtol=1e-8, atol=1e-8)


def test_gxtb_halide_increment_correction_is_additive():
    atoms = np.asarray([6, 9, 17, 35, 53], dtype=np.intp)
    expected = (
        GXTB_HALIDE_INCREMENT_CORRECTION[9]
        + GXTB_HALIDE_INCREMENT_CORRECTION[17]
        + GXTB_HALIDE_INCREMENT_CORRECTION[35]
    )
    np.testing.assert_allclose(_halide_increment_correction(atoms), expected, atol=1e-14)


def test_gxtb_first_order_offsite_potential_matches_central_difference():
    basis = build_gxtb_qvszp_basis(WATER_ATOMS, WATER_COORDS)
    qsh = np.array([0.18, -0.31, 0.06, 0.07], dtype=np.float64)
    jmat = _coulomb_matrix(WATER_COORDS * 1.8897259886, basis.shell_atom, basis.shell_hardness)
    _, potential = _first_order_offsite(
        np.asarray(WATER_ATOMS, dtype=np.intp),
        basis.shell_atom,
        jmat,
        qsh,
    )
    h = 1.0e-6
    fd = np.zeros_like(qsh)
    for i in range(qsh.size):
        qp = qsh.copy()
        qm = qsh.copy()
        qp[i] += h
        qm[i] -= h
        ep, _ = _first_order_offsite(np.asarray(WATER_ATOMS, dtype=np.intp), basis.shell_atom, jmat, qp)
        em, _ = _first_order_offsite(np.asarray(WATER_ATOMS, dtype=np.intp), basis.shell_atom, jmat, qm)
        fd[i] = (ep - em) / (2.0 * h)
    np.testing.assert_allclose(potential, fd, rtol=1e-9, atol=1e-9)


def test_gxtb_mfx_fock_is_symmetric_and_attractive():
    basis = build_gxtb_qvszp_basis(WATER_ATOMS, WATER_COORDS)
    gamma = _mfx_gamma_ao(np.asarray(WATER_ATOMS, dtype=np.intp), WATER_COORDS, basis)
    rng = np.random.default_rng(123)
    P = rng.normal(size=basis.S.shape)
    P = 0.5 * (P + P.T)

    energy, fock = _mfx_fock_energy(P, basis.S, gamma)
    assert np.isfinite(energy)
    np.testing.assert_allclose(fock, fock.T, atol=1e-14)
    density = np.eye(basis.S.shape[0], dtype=np.float64)
    closed_shell_energy, _ = _mfx_fock_energy(density, basis.S, gamma)
    assert closed_shell_energy < 0.0


def test_gxtb_acp_hamiltonian_is_symmetric_and_attractive_for_water():
    basis = build_gxtb_qvszp_basis(WATER_ATOMS, WATER_COORDS)
    H_acp = build_gxtb_acp_hamiltonian(WATER_ATOMS, WATER_COORDS, basis)
    np.testing.assert_allclose(H_acp, H_acp.T, atol=1e-14)
    assert H_acp.shape == basis.S.shape
    assert np.trace(H_acp) < 0.0


@_needs_gxtb_cpp
def test_gxtb_twobody_third_order_potential_matches_central_difference():
    basis = build_gxtb_qvszp_basis(WATER_ATOMS, WATER_COORDS)
    qsh = np.array([0.18, -0.31, 0.06, 0.07], dtype=np.float64)
    _, potential = _third_order_twobody(basis, WATER_ATOMS, WATER_COORDS, qsh)
    h = 1.0e-6
    fd = np.zeros_like(qsh)
    for i in range(qsh.size):
        qp = qsh.copy()
        qm = qsh.copy()
        qp[i] += h
        qm[i] -= h
        ep, _ = _third_order_twobody(basis, WATER_ATOMS, WATER_COORDS, qp)
        em, _ = _third_order_twobody(basis, WATER_ATOMS, WATER_COORDS, qm)
        fd[i] = (ep - em) / (2.0 * h)
    np.testing.assert_allclose(potential, fd, rtol=1e-8, atol=1e-8)


@_needs_gxtb_cpp
def test_gxtb_energy_water_smoke():
    res = gxtb_energy(
        WATER_ATOMS,
        WATER_COORDS,
        max_iter=6,
        use_d4srev=False,
        use_pacp=False,
    )
    assert np.isfinite(res["energy_hartree"])
    assert res["method"] == "g-xTB-reconstructed"
    assert res["n_basis"] == 6
    np.testing.assert_allclose(np.sum(res["atom_charges"]), 0.0, atol=1e-8)


@_needs_gxtb_cpp
def test_gxtb_numerical_gradient_shape_smoke():
    grad = gxtb_gradient_numerical(
        WATER_ATOMS,
        WATER_COORDS,
        h=1.0e-3,
        max_iter=3,
        use_d4srev=False,
        use_pacp=False,
    )
    assert grad.shape == WATER_COORDS.shape
    assert np.all(np.isfinite(grad))
