import numpy as np
import mlx.core as mx
import pytest

from mlxmolkit.xtb.aes import fockelectro, get_radcn, mmomgabzero, mmompop, setvsdq
from mlxmolkit.xtb.aes_fast import (
    mmompop_vectorized,
    mulliken_shell_charges_vectorized,
    setvsdq_vectorized,
)
from mlxmolkit.xtb.aes_mlx import (
    fockelectro_mlx,
    get_radcn_mlx,
    mmomgabzero_mlx,
    mmompop_mlx,
)
from mlxmolkit.xtb.scf_gfn2_mlx import gfn2_energy_mlx


def test_gfn2_energy_mlx_requires_float32_opt_in():
    atoms = [8, 1, 1]
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="allow_float32=True"):
        gfn2_energy_mlx(atoms, coords)


def test_fockelectro_mlx_matches_numpy_reference():
    rng = np.random.default_rng(123)
    nao = 7
    nat = 3
    P = rng.normal(size=(nao, nao))
    S = rng.normal(size=(nao, nao))
    dpint = rng.normal(size=(3, nao, nao))
    qpint = rng.normal(size=(6, nao, nao))
    aoat = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    vs = rng.normal(size=nat)
    vd = rng.normal(size=(3, nat))
    vq = rng.normal(size=(6, nat))

    F_np, e_np = fockelectro(P, S, dpint, qpint, aoat, vs, vd, vq)
    F_mx, e_mx = fockelectro_mlx(
        mx.array(P),
        mx.array(S),
        mx.array(dpint),
        mx.array(qpint),
        mx.array(aoat),
        mx.array(vs),
        mx.array(vd),
        mx.array(vq),
    )
    mx.eval(F_mx, e_mx)

    np.testing.assert_allclose(np.asarray(F_mx), F_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(np.asarray(e_mx)), e_np, rtol=1e-6, atol=1e-6)


def test_mmomgabzero_mlx_matches_numpy_reference():
    atoms = [6, 8, 1, 1]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [-0.7, 0.8, 0.2],
            [-0.5, -0.9, -0.1],
        ],
        dtype=np.float64,
    )
    cn = np.array([3.1, 1.8, 0.9, 0.9], dtype=np.float64)

    rad_np = get_radcn(atoms, cn)
    gab3_np, gab5_np = mmomgabzero(coords, rad_np)

    atoms_mx = mx.array(np.asarray(atoms, dtype=np.int32))
    coords_mx = mx.array(coords)
    cn_mx = mx.array(cn)
    rad_mx = get_radcn_mlx(atoms_mx, cn_mx)
    gab3_mx, gab5_mx = mmomgabzero_mlx(coords_mx, rad_mx)
    mx.eval(rad_mx, gab3_mx, gab5_mx)

    np.testing.assert_allclose(np.asarray(rad_mx), rad_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gab3_mx), gab3_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gab5_mx), gab5_np, rtol=1e-6, atol=1e-6)


def test_mmompop_mlx_matches_numpy_reference():
    rng = np.random.default_rng(321)
    nao = 8
    nat = 4
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
    aoat = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    coords = rng.normal(size=(nat, 3))

    dip_np, qp_np = mmompop(P, S, dpint, qpint, aoat, coords)
    dip_mx, qp_mx = mmompop_mlx(
        mx.array(P),
        mx.array(S),
        mx.array(dpint),
        mx.array(qpint),
        mx.array(aoat),
        mx.array(coords),
    )
    mx.eval(dip_mx, qp_mx)

    np.testing.assert_allclose(np.asarray(dip_mx), dip_np, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(qp_mx), qp_np, rtol=1e-6, atol=1e-6)


def test_mmompop_vectorized_matches_numpy_reference():
    rng = np.random.default_rng(654)
    nao = 9
    nat = 4
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
    aoat = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    coords = rng.normal(size=(nat, 3))

    dip_np, qp_np = mmompop(P, S, dpint, qpint, aoat, coords)
    dip_fast, qp_fast = mmompop_vectorized(P, S, dpint, qpint, aoat, coords)

    np.testing.assert_allclose(dip_fast, dip_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qp_fast, qp_np, rtol=1e-12, atol=1e-12)


def test_mulliken_shell_charges_vectorized_matches_reference_loop():
    rng = np.random.default_rng(753)
    nao = 11
    n_shell = 6
    A = rng.normal(size=(nao, nao))
    P = A + A.T
    B = rng.normal(size=(nao, nao))
    S = B + B.T
    bf_to_shell = np.array([0, 0, 1, 2, 2, 3, 3, 3, 4, 5, 5], dtype=np.int64)
    z_ref = rng.random(n_shell)

    PS = P @ S
    pop = np.zeros(n_shell, dtype=np.float64)
    for mu in range(nao):
        pop[bf_to_shell[mu]] += PS[mu, mu]
    ref = z_ref - pop

    got = mulliken_shell_charges_vectorized(P, S, bf_to_shell, n_shell, z_ref)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_setvsdq_vectorized_matches_numpy_reference():
    rng = np.random.default_rng(987)
    atoms = [6, 8, 1, 1, 6]
    nat = len(atoms)
    coords = rng.normal(size=(nat, 3))
    q = rng.normal(size=nat)
    dipm = rng.normal(size=(3, nat))
    qp = rng.normal(size=(6, nat))
    gab3 = rng.random(size=(nat, nat))
    gab5 = rng.random(size=(nat, nat))
    gab3 = 0.5 * (gab3 + gab3.T)
    gab5 = 0.5 * (gab5 + gab5.T)
    np.fill_diagonal(gab3, 0.0)
    np.fill_diagonal(gab5, 0.0)

    ref = setvsdq(atoms, coords, q, dipm, qp, gab3, gab5)
    fast = setvsdq_vectorized(atoms, coords, q, dipm, qp, gab3, gab5)

    for got, expected in zip(fast, ref):
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
