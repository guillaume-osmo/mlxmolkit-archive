import pytest
import numpy as np

from mlxmolkit.xtb.dispersion_d4 import (
    d4_dispersion_gfn2,
    d4_dispersion_gradient_gfn2,
)
from mlxmolkit.xtb.gradient_gfn2 import _fd_grad_scalar

try:
    import dftd4 as _dftd4  # noqa: F401
    _HAVE_DFTD4 = True
except ImportError:
    _HAVE_DFTD4 = False

# 'dftd4' is not a declared dependency of this package. Absent optional
# tooling is a skip with a reason, not a failure that reads as broken science.
_needs_dftd4 = pytest.mark.skipif(
    not _HAVE_DFTD4,
    reason="needs the optional 'dftd4' package "
           "(conda install -c conda-forge dftd4-python)",
)



@_needs_dftd4
def test_d4_dispersion_gradient_matches_fd_h2o():
    atoms = [8, 1, 1]
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )

    got = d4_dispersion_gradient_gfn2(atoms, coords)
    expected = _fd_grad_scalar(
        atoms,
        coords,
        lambda c: d4_dispersion_gfn2(atoms, c),
        h=1e-4,
    )

    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)
