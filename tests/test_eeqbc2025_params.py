import numpy as np
import pytest

from mlxmolkit.xtb.eeqbc2025_params import (
    EEQBC2025_PARAMS,
    MAX_Z,
    PARAMETER_NAMES,
)


def test_eeqbc2025_parameter_shapes():
    p = EEQBC2025_PARAMS

    for name in PARAMETER_NAMES:
        assert p[name].shape == (MAX_Z,)
        assert np.all(np.isfinite(p[name]))


def test_eeqbc2025_known_binary_head_values():
    p = EEQBC2025_PARAMS

    np.testing.assert_allclose(p["rvdw_scale"][:8], [1.0, 1.0, 1.06, 1.0, 1.02, 1.0, 1.05, 1.08])
    np.testing.assert_allclose(
        p["rad"][:3],
        [0.9839410084, 0.0217134247, 0.1638324678],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        p["chi"][:3],
        [1.5191499066, -0.9209695912, -1.0984929066],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        p["avg_cn"][:3],
        [0.39211, 0.08106, 0.99101],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_eeqbc2025_element_view_for_h_c_o_s():
    h = EEQBC2025_PARAMS.element(1)
    c = EEQBC2025_PARAMS.element(6)
    o = EEQBC2025_PARAMS.element(8)
    s = EEQBC2025_PARAMS.element(16)

    assert h.rvdw_scale == pytest.approx(1.0)
    assert c.chi == pytest.approx(1.4041145025)
    assert o.cap == pytest.approx(7.3478606982)
    assert s.avg_cn == pytest.approx(1.02216)


def test_eeqbc2025_z_validation():
    with pytest.raises(ValueError):
        EEQBC2025_PARAMS.element(0)
    with pytest.raises(ValueError):
        EEQBC2025_PARAMS.atom_value("chi", MAX_Z + 1)
