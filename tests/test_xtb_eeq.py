"""Tests for the GFN0-xTB EEQ charge solver.

Parity baseline: ``xtb-python`` (conda-forge ``xtb-python`` 22.1) with
``Param.GFN0xTB``. Requires the env var ``XTBPATH`` pointing at the
xtb param directory (``$CONDA_PREFIX/share/xtb`` in the osmo env).
The fixture skips the test if xtb-python is unavailable so the suite
still runs on CI without the parity dep.
"""

import os

import mlx.core as mx
import numpy as np
import pytest


ANG_TO_BOHR = 1.8897259886


def _xtb_charges(atoms, coords_ang, charge=0):
    """Reference charges from xtb-python's GFN0-xTB Mulliken-style q."""
    xtb = pytest.importorskip("xtb")
    if "XTBPATH" not in os.environ:
        prefix = os.environ.get("CONDA_PREFIX", "")
        if prefix:
            os.environ["XTBPATH"] = os.path.join(prefix, "share", "xtb")
    from xtb.interface import Calculator, Param
    a = np.asarray(atoms, dtype=np.int32)
    c = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    cal = Calculator(Param.GFN0xTB, a, c, charge=float(charge))
    cal.set_verbosity(0)
    res = cal.singlepoint()
    return np.asarray(res.get_charges())


def _mol_charges(atoms, coords, charge=0):
    from mlxmolkit.xtb.eeq import eeq_charges
    a = mx.array(np.asarray(atoms, dtype=np.int32))
    c = mx.array(np.asarray(coords, dtype=np.float32))
    q = eeq_charges(c, a, total_charge=float(charge))
    mx.eval(q)
    return np.asarray(q)


# ---------------------------------------------------------------------------
# Single-molecule parity vs xtb-python
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,atoms,coords", [
    ("H2O",     [8, 1, 1], [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]),
    ("NH3",     [7, 1, 1, 1], [[0.0, 0, 0], [0.939, 0, -0.328], [-0.470, 0.814, -0.328], [-0.470, -0.814, -0.328]]),
    ("CO2",     [6, 8, 8], [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]),
    ("CH4",     [6, 1, 1, 1, 1], [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629], [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]]),
    ("Ethanol", [6, 6, 8, 1, 1, 1, 1, 1, 1],
        [[1.21, -0.39, 0], [0, 0.45, 0], [-1.13, -0.32, 0],
         [1.21, -1.03, 0.88], [1.21, -1.03, -0.88], [2.10, 0.25, 0],
         [0, 1.10, 0.88], [0, 1.10, -0.88], [-1.91, 0.21, 0]]),
    ("Benzene", [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
        [[1.39, 0, 0], [0.695, 1.204, 0], [-0.695, 1.204, 0],
         [-1.39, 0, 0], [-0.695, -1.204, 0], [0.695, -1.204, 0],
         [2.475, 0, 0], [1.238, 2.143, 0], [-1.238, 2.143, 0],
         [-2.475, 0, 0], [-1.238, -2.143, 0], [1.238, -2.143, 0]]),
])
def test_neutral_parity_vs_xtb(name, atoms, coords):
    q_ref = _xtb_charges(atoms, coords)
    q_mol = _mol_charges(atoms, coords)
    np.testing.assert_allclose(q_mol, q_ref, atol=0.05)


@pytest.mark.parametrize("name,atoms,coords,charge", [
    ("NH4+", [7, 1, 1, 1, 1], [[0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [-0.6, 0.6, -0.6], [0.6, -0.6, -0.6]], 1),
    ("OH-",  [8, 1], [[0, 0, 0], [0.96, 0, 0]], -1),
])
def test_charged_species_parity(name, atoms, coords, charge):
    q_ref = _xtb_charges(atoms, coords, charge=charge)
    q_mol = _mol_charges(atoms, coords, charge=charge)
    np.testing.assert_allclose(q_mol, q_ref, atol=0.05)


# ---------------------------------------------------------------------------
# Constraint: total charge sums to q_total over real atoms
# ---------------------------------------------------------------------------


def test_neutral_charge_sums_to_zero():
    q = _mol_charges([8, 1, 1], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]], charge=0)
    assert abs(q.sum()) < 1e-5


def test_cation_charge_sums_to_one():
    q = _mol_charges(
        [7, 1, 1, 1, 1],
        [[0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [-0.6, 0.6, -0.6], [0.6, -0.6, -0.6]],
        charge=+1,
    )
    assert abs(q.sum() - 1.0) < 1e-5


def test_anion_charge_sums_to_negative_one():
    q = _mol_charges([8, 1], [[0, 0, 0], [0.96, 0, 0]], charge=-1)
    assert abs(q.sum() + 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Shape conventions: single-mol vs batched
# ---------------------------------------------------------------------------


def test_single_mol_returns_1d():
    q = _mol_charges([8, 1, 1], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
    assert q.ndim == 1 and q.shape == (3,)


def test_batched_input_returns_2d():
    """Batched (B, max_atoms, 3) → (B, max_atoms); padded slots zero."""
    from mlxmolkit.xtb.eeq import eeq_charges
    atoms_b = mx.array([
        [8, 1, 1, 0, 0, 0, 0, 0, 0],   # H2O padded to 9
        [6, 1, 1, 1, 1, 0, 0, 0, 0],   # CH4 padded to 9
    ], dtype=mx.int32)
    coords_b = mx.array([
        [[0., 0., 0.117], [0., 0.757, -0.469], [0., -0.757, -0.469],
         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
         [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629],
         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ], dtype=mx.float32)
    n_atoms = mx.array([3, 5], dtype=mx.int32)
    q_b = eeq_charges(coords_b, atoms_b, n_atoms_arr=n_atoms)
    mx.eval(q_b)
    q_b_np = np.asarray(q_b)
    assert q_b_np.shape == (2, 9)
    # Per-mol total over real atoms is 0 (neutral).
    assert abs(q_b_np[0, :3].sum()) < 1e-5
    assert abs(q_b_np[1, :5].sum()) < 1e-5
    # Padded slots are exactly zero.
    np.testing.assert_array_equal(q_b_np[0, 3:], 0.0)
    np.testing.assert_array_equal(q_b_np[1, 5:], 0.0)


def test_batch_matches_single_mol():
    """Same molecule processed one-at-a-time vs in-batch must give the
    same charges (modulo padding)."""
    from mlxmolkit.xtb.eeq import eeq_charges
    atoms_h2o = [8, 1, 1]
    coords_h2o = [[0., 0., 0.117], [0., 0.757, -0.469], [0., -0.757, -0.469]]
    q_single = _mol_charges(atoms_h2o, coords_h2o)

    atoms_b = mx.array([atoms_h2o + [0, 0]], dtype=mx.int32)
    coords_b = mx.array([coords_h2o + [[0, 0, 0], [0, 0, 0]]], dtype=mx.float32)
    n_atoms = mx.array([3], dtype=mx.int32)
    q_b = eeq_charges(coords_b, atoms_b, n_atoms_arr=n_atoms)
    mx.eval(q_b)
    np.testing.assert_allclose(np.asarray(q_b)[0, :3], q_single, atol=1e-5)
