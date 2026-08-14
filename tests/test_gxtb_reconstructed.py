import numpy as np
import pytest

from mlxmolkit.xtb.gxtb_cpp import CPP_AVAILABLE
from mlxmolkit.xtb.gxtb_reconstructed import (
    _gxtb_erf_coordination_number,
    gxtb_reconstructed_repulsion,
    repulsion_constants_from_binary,
)
from mlxmolkit.xtb.mctc_vdwrad import mctc_vdw_pair_radius_bohr
from mlxmolkit.xtb.params_gxtb import GXTB_PARAMS, GXTB_REPULSION_LITERAL_BY_ADDR


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_reconstructed_repulsion_water_is_finite_and_translationally_invariant():
    atoms = np.array([8, 1, 1], dtype=np.intp)
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )

    result = gxtb_reconstructed_repulsion(atoms, coords)

    assert np.isfinite(result.energy)
    assert result.energy > 0.0
    assert result.gradient.shape == coords.shape
    np.testing.assert_allclose(np.sum(result.gradient, axis=0), 0.0, atol=1e-12)
    assert result.metadata["complete_gxtb"] is False


def test_reconstructed_repulsion_constant_mapping_matches_binary_literals():
    constants = repulsion_constants_from_binary()
    lit = GXTB_REPULSION_LITERAL_BY_ADDR

    assert constants.stored_scalar_1p5 == lit[0x73B268]
    assert constants.exp_power_1 == lit[0x73B270]
    assert constants.exp_power_2 == lit[0x73B278]
    assert constants.exp2_scale == lit[0x73B280]
    assert constants.exp2_weight == lit[0x73B288]
    assert constants.quartic_coeff == lit[0x73B290]
    assert constants.cubic_coeff == lit[0x73B298]


def test_gxtb_reconstructed_cn_uses_recovered_erf_count():
    atoms = np.array([8, 1, 1], dtype=np.intp)
    coords = np.array(
        [
            [0.0, 0.0, 0.117790],
            [0.0, 0.755453, -0.471160],
            [0.0, -0.755453, -0.471160],
        ],
        dtype=np.float64,
    )

    cn = _gxtb_erf_coordination_number(atoms, coords)

    np.testing.assert_allclose(cn, [1.579653994662, 0.838686777538, 0.838686777538])


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_reconstructed_repulsion_builder_uses_geometric_rvdw_scale_and_k1_linear():
    atoms = np.array([1, 8], dtype=np.intp)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    result = gxtb_reconstructed_repulsion(atoms, coords, cn=np.array([1.0, 1.0]))

    scale = GXTB_PARAMS["pa_rvdw_scale"][atoms - 1]
    expected_pair_rvdw = mctc_vdw_pair_radius_bohr(1, 8) * np.sqrt(scale[0] * scale[1])
    expected_linear = 0.5 * (
        GXTB_PARAMS["pa_rep_k1"][atoms[0] - 1]
        + GXTB_PARAMS["pa_rep_k1"][atoms[1] - 1]
    )

    assert result.pair_rvdw[0, 1] == pytest.approx(expected_pair_rvdw)
    assert result.linear_coeff[0, 1] == pytest.approx(expected_linear)
