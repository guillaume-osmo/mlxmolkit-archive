import numpy as np

from mlxmolkit.xtb.gradient_aes_gfn2 import aes_gradient_frozen_density
from mlxmolkit.xtb.gradient_gfn2 import _aes_full_energy_at, _fd_grad_scalar
from mlxmolkit.xtb.scf_gfn2_fast import gfn2_energy_fast


def test_aes_gradient_frozen_density_matches_full_aes_fd_h2o():
    atoms = [8, 1, 1]
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )
    res = gfn2_energy_fast(atoms, coords, conv_tol=1e-7, max_iter=200)
    P = res["density"]
    qsh = res["shell_charges"]
    shell_atom = res["shell_atom"]

    got = aes_gradient_frozen_density(atoms, coords, P, qsh, shell_atom, h_explicit=1e-4)
    expected = _fd_grad_scalar(
        atoms,
        coords,
        lambda c: _aes_full_energy_at(atoms, c, P, qsh, shell_atom),
        h=1e-4,
    )

    np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-7)
