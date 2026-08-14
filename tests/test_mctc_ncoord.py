import numpy as np

from mlxmolkit.xtb.mctc_ncoord import erf_coordination_number
from mlxmolkit.xtb.params_gxtb import GXTB_PARAMS


def test_erf_coordination_number_matches_recovered_gxtb_water_formula():
    atoms = np.array([8, 1, 1], dtype=np.intp)
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )

    cn = erf_coordination_number(atoms, coords, GXTB_PARAMS["pa_cn_rcov"], k=2.068)

    np.testing.assert_allclose(cn, [1.579653994662, 0.838686777538, 0.838686777538])


def test_erf_coordination_number_is_translation_invariant():
    atoms = np.array([6, 8, 1], dtype=np.intp)
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.7, 0.8, 0.1]])

    cn = erf_coordination_number(atoms, coords, GXTB_PARAMS["pa_cn_rcov"], k=2.068)
    shifted = erf_coordination_number(
        atoms,
        coords + np.array([10.0, -2.0, 0.5]),
        GXTB_PARAMS["pa_cn_rcov"],
        k=2.068,
    )

    np.testing.assert_allclose(cn, shifted, rtol=0.0, atol=1.0e-15)
