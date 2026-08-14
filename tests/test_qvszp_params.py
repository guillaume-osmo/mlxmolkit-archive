import numpy as np

from mlxmolkit.xtb.qvszp_params import QVSZP_PARAMS


def test_qvszp_binary_table_shapes():
    assert QVSZP_PARAMS["cov_radii"].shape == (103,)
    assert QVSZP_PARAMS["nshell"].shape == (103,)
    assert QVSZP_PARAMS["ang_shell"].shape == (103, 4)
    assert QVSZP_PARAMS["n_prim"].shape == (103, 4)
    assert QVSZP_PARAMS["exponents"].shape == (103, 4, 12)
    assert QVSZP_PARAMS["coefficients"].shape == (103, 4, 12)
    assert QVSZP_PARAMS["coefficients_env"].shape == (103, 4, 12)


def test_qvszp_h_c_o_s_layout_from_binary():
    expected = {
        1: (2, [0, 1, 0, 0], [8, 3, 0, 0]),
        6: (3, [0, 1, 2, 0], [6, 6, 3, 0]),
        8: (3, [0, 1, 2, 0], [6, 6, 3, 0]),
        16: (3, [0, 1, 2, 0], [5, 5, 2, 0]),
    }
    for Z, (n_shell, ang, n_prim) in expected.items():
        idx = Z - 1
        assert int(QVSZP_PARAMS["nshell"][idx]) == n_shell
        np.testing.assert_array_equal(QVSZP_PARAMS["ang_shell"][idx], ang)
        np.testing.assert_array_equal(QVSZP_PARAMS["n_prim"][idx], n_prim)


def test_qvszp_known_head_values():
    h_s = QVSZP_PARAMS.shell(1, 0)
    assert h_s.l == 0
    assert h_s.n_prim == 8
    np.testing.assert_allclose(
        h_s.exponents[:3],
        [337.00501222, 53.31053105, 12.20825348],
        rtol=0.0,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        h_s.coefficients[:3],
        [0.0008798, 0.00691742, 0.03424541],
        rtol=0.0,
        atol=5e-9,
    )
