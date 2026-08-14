import numpy as np

from mlxmolkit.xtb.params_gxtb import (
    GXTB_PARAMS,
    GXTB_REPULSION_LITERAL_BY_ADDR,
    GXTB_REPULSION_LITERAL_SEQUENCE,
    SHELL_LABELS,
)


def test_gxtb_parameter_shapes_and_global_tables():
    p = GXTB_PARAMS
    assert p["ps_reference_occ"].shape == (103, 4)
    assert p["pa_rep_zeff"].shape == (103,)
    assert p["pa_nshell"].shape == (103,)
    assert p["pg_tb4_kshell"].shape == (4,)
    np.testing.assert_allclose(p["pg_tb4_kshell"], [1.0, 1.15, 1.3, 1.45])
    np.testing.assert_allclose(p["pg_h0_shpoly2"], [1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(p["pg_fock_kq"], [1.1, 0.55, 0.275, 0.1375])


def test_gxtb_repulsion_scalar_literals_from_add_repulsion():
    expected = {
        0x73B268: 1.5,
        0x73B270: 2.068,
        0x73B278: 2.0,
        0x73B280: 0.73,
        0x73B288: 0.0046511298,
        0x73B290: 0.011607795128002491,
        0x73B298: 0.011095539524126988,
        0x73B2A0: 0.012098131381864387,
        0x73B2A8: 0.008544252691968662,
    }

    assert GXTB_REPULSION_LITERAL_BY_ADDR == expected
    np.testing.assert_allclose(GXTB_REPULSION_LITERAL_SEQUENCE, tuple(expected.values()))


def test_gxtb_h_c_o_s_element_views():
    h = GXTB_PARAMS.element(1)
    c = GXTB_PARAMS.element(6)
    o = GXTB_PARAMS.element(8)
    s = GXTB_PARAMS.element(16)

    assert h.n_shell == 1
    assert c.n_shell == 2
    assert o.n_shell == 2
    assert s.n_shell == 3
    assert tuple(shell.label for shell in s.shells) == SHELL_LABELS[:3]

    np.testing.assert_allclose(h.reference_occ, [1.0])
    np.testing.assert_allclose(c.reference_occ, [1.03539398945965, 2.96460601049032])
    np.testing.assert_allclose(o.reference_occ, [1.67303036949422, 4.32696963045594])
    np.testing.assert_allclose(
        s.reference_occ,
        [1.75759024471622, 4.08981250079158, 0.15259725444982],
    )

    assert np.isclose(o.rep_zeff, 2.7937777512)
    assert np.isclose(s.aes_dip_scale, 0.1724210314)
