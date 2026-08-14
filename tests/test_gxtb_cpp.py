import math

import numpy as np
import pytest

from mlxmolkit.xtb.gxtb_cpp import (
    CPP_AVAILABLE,
    cn_scaled_parameter,
    disassembled_kernel_status,
    full_calculator_available,
    gxtb_energy_gradient,
    implementation_status,
    missing_calculator_blocks,
    multipole_damping_pair,
    multipole_damping_pair_derivs,
    multipole_mrad_pair,
    repulsion_cn_scaled_parameter,
    repulsion_descriptor_potential,
    repulsion_energy_from_matvec,
    repulsion_energy_gradient,
    repulsion_energy_gradient_asm,
    repulsion_energy_gradient_parameterized,
    repulsion_pair_roffset_arithmetic,
    repulsion_pair_matrix,
    repulsion_pair_matrix_asm,
    repulsion_pair_value,
    repulsion_pair_value_asm,
    repulsion_pair_value_asm_deriv,
    repulsion_pair_value_deriv,
    repulsion_q_potential,
    repulsion_scaled_zeff,
    repulsion_state,
    scaled_zeff,
)
from mlxmolkit.xtb.params_gxtb import GXTB_PARAMS


def test_gxtb_cpp_status_distinguishes_kernels_from_full_calculator():
    status = implementation_status()
    implemented = {block.name for block in status if block.status == "implemented"}
    missing = {block.name for block in status if block.status != "implemented"}

    assert "binary parameter tables" in implemented
    assert "repulsion scalar helpers" in implemented
    assert "SCF driver and analytic gradient assembly" in missing
    assert not full_calculator_available()
    assert missing_calculator_blocks()


def test_gxtb_disassembled_kernel_status_is_symbol_anchored():
    kernels = disassembled_kernel_status()
    wrappers = {kernel.wrapper for kernel in kernels}
    symbols = {kernel.symbol for kernel in kernels}

    assert "scaled_zeff / repulsion_scaled_zeff" in wrappers
    assert "repulsion_energy_from_matvec" in wrappers
    assert "tblite_repulsion_gxtb::get_scaled_zeff" in symbols
    assert any(kernel.status == "partial" for kernel in kernels)


def test_gxtb_energy_gradient_is_explicitly_not_available_yet():
    with pytest.raises(NotImplementedError, match="native C\\+\\+ g-xTB is not complete"):
        gxtb_energy_gradient([8, 1, 1], np.zeros((3, 3), dtype=np.float64))


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_scaled_zeff_matches_assembly_derived_formula():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    descriptor = np.array([0.2, -0.1, 0.35, 0.7], dtype=np.float64)
    zeff = GXTB_PARAMS["pa_rep_zeff"]
    scale = GXTB_PARAMS["pa_rep_q"]

    got = scaled_zeff(atomic_numbers, zeff, scale, descriptor)
    expected = zeff[atomic_numbers - 1] * (1.0 - scale[atomic_numbers - 1] * descriptor)

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_scaled_zeff_uses_binary_extracted_tables():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    descriptor = np.array([0.0, 1.0, -0.5, 0.25], dtype=np.float64)

    got = repulsion_scaled_zeff(atomic_numbers, descriptor)
    expected = GXTB_PARAMS["pa_rep_zeff"][atomic_numbers - 1] * (
        1.0 - GXTB_PARAMS["pa_rep_q"][atomic_numbers - 1] * descriptor
    )

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_cn_scaled_parameter_matches_repulsion_update_formula():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    cn = np.array([0.0, 1.4, 2.8, 4.2], dtype=np.float64)
    base = GXTB_PARAMS["pa_rep_alpha"]
    slope = GXTB_PARAMS["pa_rep_cn"]

    values, derivs = cn_scaled_parameter(atomic_numbers, base, slope, cn)
    root = np.sqrt(cn + 1.0e-12)
    expected_values = base[atomic_numbers - 1] * (
        1.0 + slope[atomic_numbers - 1] * (root - 1.0e-6)
    )
    expected_derivs = base[atomic_numbers - 1] * slope[atomic_numbers - 1] / (2.0 * root)

    np.testing.assert_allclose(values, expected_values, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(derivs, expected_derivs, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_cn_scaled_parameter_uses_binary_extracted_tables():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    cn = np.array([0.25, 1.0, 2.5, 3.5], dtype=np.float64)

    got, got_deriv = repulsion_cn_scaled_parameter(atomic_numbers, cn)
    expected, expected_deriv = cn_scaled_parameter(
        atomic_numbers,
        GXTB_PARAMS["pa_rep_alpha"],
        GXTB_PARAMS["pa_rep_cn"],
        cn,
    )

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(got_deriv, expected_deriv, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_state_applies_binary_tables_and_disassembly_formulas():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    descriptor = np.array([0.0, 0.2, -0.1, 0.35], dtype=np.float64)
    cn = np.array([0.25, 1.0, 2.5, 3.5], dtype=np.float64)

    state = repulsion_state(atomic_numbers, descriptor=descriptor, cn=cn)
    expected_scaled = repulsion_scaled_zeff(atomic_numbers, descriptor)
    expected_alpha, expected_dalpha = repulsion_cn_scaled_parameter(atomic_numbers, cn)
    expected_roffset = GXTB_PARAMS["pa_rep_roffset"][atomic_numbers - 1]
    expected_pair_roffset = 0.5 * (
        expected_roffset[:, None] + expected_roffset[None, :]
    )

    np.testing.assert_allclose(state.scaled_zeff, expected_scaled, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(state.alpha, expected_alpha, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(state.dalpha_dcn, expected_dalpha, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(state.atom_roffset, expected_roffset, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(state.pair_roffset, expected_pair_roffset, rtol=0.0, atol=0.0)


def test_repulsion_pair_roffset_arithmetic_uses_atom_table():
    atomic_numbers = np.array([1, 6, 8], dtype=np.intp)
    got = repulsion_pair_roffset_arithmetic(atomic_numbers)
    atom_roffset = GXTB_PARAMS["pa_rep_roffset"][atomic_numbers - 1]
    expected = 0.5 * (atom_roffset[:, None] + atom_roffset[None, :])

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_cn_scaled_parameter_derivative_matches_central_difference():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    cn = np.array([0.2, 1.4, 2.8, 4.2], dtype=np.float64)
    base = GXTB_PARAMS["pa_rep_alpha"]
    slope = GXTB_PARAMS["pa_rep_cn"]
    h = 1e-6

    _, derivs = cn_scaled_parameter(atomic_numbers, base, slope, cn)
    plus, _ = cn_scaled_parameter(atomic_numbers, base, slope, cn + h)
    minus, _ = cn_scaled_parameter(atomic_numbers, base, slope, cn - h)
    fd = (plus - minus) / (2.0 * h)

    np.testing.assert_allclose(derivs, fd, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_energy_from_matvec_contracts_scaled_zeff():
    scaled = np.array([1.2, -0.5, 0.7, 3.1], dtype=np.float64)
    matvec = np.array([0.3, 0.8, -1.4, 2.2], dtype=np.float64)

    got = repulsion_energy_from_matvec(scaled, matvec)
    expected = float(np.dot(scaled, matvec))

    assert got == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_descriptor_potential_matches_get_potential_formula():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    matvec = np.array([0.3, 0.8, -1.4, 2.2], dtype=np.float64)
    base = GXTB_PARAMS["pa_rep_zeff"]
    scale = GXTB_PARAMS["pa_rep_q"]

    got = repulsion_descriptor_potential(atomic_numbers, base, scale, matvec)
    expected = -base[atomic_numbers - 1] * scale[atomic_numbers - 1] * matvec

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_q_potential_uses_binary_extracted_tables():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    matvec = np.array([0.3, 0.8, -1.4, 2.2], dtype=np.float64)

    got = repulsion_q_potential(atomic_numbers, matvec)
    expected = repulsion_descriptor_potential(
        atomic_numbers,
        GXTB_PARAMS["pa_rep_zeff"],
        GXTB_PARAMS["pa_rep_q"],
        matvec,
    )

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_pair_value_matches_observed_pair_form():
    r = 2.4
    alpha_a = 1.35
    alpha_b = 1.61
    roffset = 0.12
    coeffs = np.array([0.2, -0.03, 0.004, 0.0007], dtype=np.float64)
    exp_power_1 = 1.2
    exp_power_2 = 2.1
    exp2_scale = 0.8
    exp2_weight = 0.35

    got = repulsion_pair_value(
        r,
        alpha_a,
        alpha_b,
        roffset,
        coeffs,
        exp_power_1,
        exp_power_2,
        exp2_scale,
        exp2_weight,
    )
    inv_r = 1.0 / r
    poly = (
        1.0
        + coeffs[0] * inv_r
        + coeffs[1] * inv_r**2
        + coeffs[2] * inv_r**3
        + coeffs[3] * inv_r**4
    )
    alpha = alpha_a * alpha_b / (alpha_a + alpha_b)
    rho = r + roffset
    expected = poly * (
        math.exp(-alpha * rho**exp_power_1)
        + exp2_weight * math.exp(-alpha * exp2_scale * rho**exp_power_2)
    )

    assert got == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_pair_derivative_matches_central_difference():
    r = 2.4
    alpha_a = 1.35
    alpha_b = 1.61
    roffset = 0.12
    coeffs = np.array([0.2, -0.03, 0.004, 0.0007], dtype=np.float64)
    exp_power_1 = 1.2
    exp_power_2 = 2.1
    exp2_scale = 0.8
    exp2_weight = 0.35
    h = 1.0e-6

    value, deriv = repulsion_pair_value_deriv(
        r,
        alpha_a,
        alpha_b,
        roffset,
        coeffs,
        exp_power_1,
        exp_power_2,
        exp2_scale,
        exp2_weight,
    )
    fd = (
        repulsion_pair_value(
            r + h,
            alpha_a,
            alpha_b,
            roffset,
            coeffs,
            exp_power_1,
            exp_power_2,
            exp2_scale,
            exp2_weight,
        )
        - repulsion_pair_value(
            r - h,
            alpha_a,
            alpha_b,
            roffset,
            coeffs,
            exp_power_1,
            exp_power_2,
            exp2_scale,
            exp2_weight,
        )
    ) / (2.0 * h)

    assert value == repulsion_pair_value(
        r,
        alpha_a,
        alpha_b,
        roffset,
        coeffs,
        exp_power_1,
        exp_power_2,
        exp2_scale,
        exp2_weight,
    )
    assert deriv == pytest.approx(fd, rel=1e-10, abs=1e-10)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-XTB extension is not built")
def test_repulsion_pair_value_asm_matches_rvdw_scaled_assembly_form():
    r = 2.4
    alpha_a = 1.35
    alpha_b = 1.61
    pair_rvdw = 3.2
    roffset = 0.12
    linear_coeff = 0.2
    quadratic_coeff = -0.03
    cubic_coeff = 0.004
    quartic_coeff = 0.0007
    exp_power_1 = 1.2
    exp_power_2 = 2.1
    exp2_scale = 0.8
    exp2_weight = 0.35

    got = repulsion_pair_value_asm(
        r,
        alpha_a,
        alpha_b,
        pair_rvdw,
        roffset,
        linear_coeff,
        quadratic_coeff,
        cubic_coeff,
        quartic_coeff,
        exp_power_1,
        exp_power_2,
        exp2_scale,
        exp2_weight,
    )
    inv_r = 1.0 / r
    x = pair_rvdw * inv_r
    poly = (
        1.0
        + linear_coeff * inv_r
        + quadratic_coeff * x**2
        + cubic_coeff * x**3
        + quartic_coeff * x**4
    )
    alpha = alpha_a * alpha_b / (alpha_a + alpha_b)
    rho = r + roffset
    expected = poly * (
        math.exp(-alpha * rho**exp_power_1)
        + exp2_weight * math.exp(-alpha * exp2_scale * rho**exp_power_2)
    )

    assert got == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-XTB extension is not built")
def test_repulsion_pair_value_asm_derivative_matches_central_difference():
    args = (
        1.35,
        1.61,
        3.2,
        0.12,
        0.2,
        -0.03,
        0.004,
        0.0007,
        1.2,
        2.1,
        0.8,
        0.35,
    )
    r = 2.4
    h = 1.0e-6

    value, deriv = repulsion_pair_value_asm_deriv(r, *args)
    fd = (
        repulsion_pair_value_asm(r + h, *args)
        - repulsion_pair_value_asm(r - h, *args)
    ) / (2.0 * h)

    assert value == repulsion_pair_value_asm(r, *args)
    assert deriv == pytest.approx(fd, rel=1e-10, abs=1e-10)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_pair_matrix_matches_pair_value_loop():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.4, 0.3],
            [-0.4, 0.3, 1.7],
        ],
        dtype=np.float64,
    )
    alpha = np.array([1.2, 1.4, 1.1, 1.8], dtype=np.float64)
    pair_roffset = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ],
        dtype=np.float64,
    )
    coeffs = np.array([0.2, -0.03, 0.004, 0.0007], dtype=np.float64)
    args = (coeffs, 1.2, 2.1, 0.8, 0.35)

    got = repulsion_pair_matrix(coords, alpha, pair_roffset, *args, cutoff=25.0)
    expected = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            r = np.linalg.norm(coords[i] - coords[j])
            expected[i, j] = expected[j, i] = repulsion_pair_value(
                r,
                alpha[i],
                alpha[j],
                pair_roffset[i, j],
                *args,
            )

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_pair_matrix_asm_matches_pair_value_loop():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.4, 0.3],
            [-0.4, 0.3, 1.7],
        ],
        dtype=np.float64,
    )
    alpha = np.array([1.2, 1.4, 1.1, 1.8], dtype=np.float64)
    pair_rvdw = np.array(
        [
            [0.0, 3.1, 3.2, 3.3],
            [3.1, 0.0, 3.4, 3.5],
            [3.2, 3.4, 0.0, 3.6],
            [3.3, 3.5, 3.6, 0.0],
        ],
        dtype=np.float64,
    )
    pair_roffset = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ],
        dtype=np.float64,
    )
    linear_coeff = np.full((4, 4), 0.2, dtype=np.float64)
    quadratic_coeff = np.full((4, 4), -0.03, dtype=np.float64)
    args = (0.004, 0.0007, 1.2, 2.1, 0.8, 0.35)

    got = repulsion_pair_matrix_asm(
        coords,
        alpha,
        pair_rvdw,
        pair_roffset,
        linear_coeff,
        quadratic_coeff,
        *args,
        cutoff=25.0,
    )
    expected = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            r = np.linalg.norm(coords[i] - coords[j])
            expected[i, j] = expected[j, i] = repulsion_pair_value_asm(
                r,
                alpha[i],
                alpha[j],
                pair_rvdw[i, j],
                pair_roffset[i, j],
                linear_coeff[i, j],
                quadratic_coeff[i, j],
                *args,
            )

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_energy_gradient_matches_matrix_and_fd():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.4, 0.3],
            [-0.4, 0.3, 1.7],
        ],
        dtype=np.float64,
    )
    scaled = np.array([1.0, 0.8, -0.4, 1.3], dtype=np.float64)
    alpha = np.array([1.2, 1.4, 1.1, 1.8], dtype=np.float64)
    pair_roffset = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ],
        dtype=np.float64,
    )
    coeffs = np.array([0.2, -0.03, 0.004, 0.0007], dtype=np.float64)
    args = (coeffs, 1.2, 2.1, 0.8, 0.35)

    energy, gradient, matvec = repulsion_energy_gradient(
        coords,
        scaled,
        alpha,
        pair_roffset,
        *args,
        cutoff=25.0,
    )
    matrix = repulsion_pair_matrix(coords, alpha, pair_roffset, *args, cutoff=25.0)

    np.testing.assert_allclose(matvec, matrix @ scaled, rtol=0.0, atol=1e-15)
    assert energy == pytest.approx(float(scaled @ matvec), rel=0.0, abs=1e-15)

    h = 1.0e-6
    fd = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for k in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[i, k] += h
            minus[i, k] -= h
            e_plus = repulsion_energy_gradient(plus, scaled, alpha, pair_roffset, *args)[0]
            e_minus = repulsion_energy_gradient(minus, scaled, alpha, pair_roffset, *args)[0]
            fd[i, k] = (e_plus - e_minus) / (2.0 * h)

    np.testing.assert_allclose(gradient, fd, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_energy_gradient_asm_matches_matrix_and_fd():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.4, 0.3],
            [-0.4, 0.3, 1.7],
        ],
        dtype=np.float64,
    )
    scaled = np.array([1.0, 0.8, -0.4, 1.3], dtype=np.float64)
    alpha = np.array([1.2, 1.4, 1.1, 1.8], dtype=np.float64)
    pair_rvdw = np.array(
        [
            [0.0, 3.1, 3.2, 3.3],
            [3.1, 0.0, 3.4, 3.5],
            [3.2, 3.4, 0.0, 3.6],
            [3.3, 3.5, 3.6, 0.0],
        ],
        dtype=np.float64,
    )
    pair_roffset = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ],
        dtype=np.float64,
    )
    linear_coeff = np.full((4, 4), 0.2, dtype=np.float64)
    quadratic_coeff = np.full((4, 4), -0.03, dtype=np.float64)
    args = (0.004, 0.0007, 1.2, 2.1, 0.8, 0.35)

    energy, gradient, matvec = repulsion_energy_gradient_asm(
        coords,
        scaled,
        alpha,
        pair_rvdw,
        pair_roffset,
        linear_coeff,
        quadratic_coeff,
        *args,
        cutoff=25.0,
    )
    matrix = repulsion_pair_matrix_asm(
        coords,
        alpha,
        pair_rvdw,
        pair_roffset,
        linear_coeff,
        quadratic_coeff,
        *args,
        cutoff=25.0,
    )

    np.testing.assert_allclose(matvec, matrix @ scaled, rtol=0.0, atol=1e-15)
    assert energy == pytest.approx(float(scaled @ matvec), rel=0.0, abs=1e-15)

    h = 1.0e-6
    fd = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for k in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[i, k] += h
            minus[i, k] -= h
            e_plus = repulsion_energy_gradient_asm(
                plus,
                scaled,
                alpha,
                pair_rvdw,
                pair_roffset,
                linear_coeff,
                quadratic_coeff,
                *args,
            )[0]
            e_minus = repulsion_energy_gradient_asm(
                minus,
                scaled,
                alpha,
                pair_rvdw,
                pair_roffset,
                linear_coeff,
                quadratic_coeff,
                *args,
            )[0]
            fd[i, k] = (e_plus - e_minus) / (2.0 * h)

    np.testing.assert_allclose(gradient, fd, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_repulsion_energy_gradient_parameterized_matches_manual_state_call():
    atomic_numbers = np.array([1, 6, 8, 16], dtype=np.intp)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.4, 0.3],
            [-0.4, 0.3, 1.7],
        ],
        dtype=np.float64,
    )
    descriptor = np.array([0.0, 0.2, -0.1, 0.35], dtype=np.float64)
    cn = np.array([0.25, 1.0, 2.5, 3.5], dtype=np.float64)
    coeffs = np.array([0.2, -0.03, 0.004, 0.0007], dtype=np.float64)
    args = (coeffs, 1.2, 2.1, 0.8, 0.35)

    energy, gradient, matvec, state = repulsion_energy_gradient_parameterized(
        atomic_numbers,
        coords,
        descriptor,
        cn,
        *args,
    )
    expected_state = repulsion_state(atomic_numbers, descriptor=descriptor, cn=cn)
    expected_energy, expected_gradient, expected_matvec = repulsion_energy_gradient(
        coords,
        expected_state.scaled_zeff,
        expected_state.alpha,
        expected_state.pair_roffset,
        *args,
    )

    assert energy == expected_energy
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(matvec, expected_matvec, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(state.scaled_zeff, expected_state.scaled_zeff, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_multipole_damping_pair_matches_erf_microkernel():
    amplitudes = np.array([0.7, 1.1, -0.3, 2.0], dtype=np.float64)
    betas = np.array([0.2, 0.5, 1.7, 3.0], dtype=np.float64)
    a = 1.25
    b = -0.4

    got = multipole_damping_pair(a, b, amplitudes, betas)
    expected = np.array(
        [0.5 * amp * (1.0 + math.erf((a - b) * beta)) for amp, beta in zip(amplitudes, betas)]
    )

    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=1e-15)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_multipole_mrad_pair_fetches_table_entry():
    table = np.arange(20, dtype=np.float64).reshape(4, 5) + 0.25

    assert multipole_mrad_pair(table, 2, 3) == table[2, 3]


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ g-xTB extension is not built")
def test_multipole_damping_derivative_matches_central_difference():
    amplitudes = np.array([0.7, 1.1, -0.3, 2.0], dtype=np.float64)
    betas = np.array([0.2, 0.5, 1.7, 3.0], dtype=np.float64)
    a = 0.37
    b = -0.11
    h = 1e-6

    values, d_delta = multipole_damping_pair_derivs(a, b, amplitudes, betas)
    expected_values = multipole_damping_pair(a, b, amplitudes, betas)
    fd = (
        multipole_damping_pair(a + h, b, amplitudes, betas)
        - multipole_damping_pair(a - h, b, amplitudes, betas)
    ) / (2.0 * h)

    np.testing.assert_allclose(values, expected_values, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(d_delta, fd, rtol=1e-10, atol=1e-10)
