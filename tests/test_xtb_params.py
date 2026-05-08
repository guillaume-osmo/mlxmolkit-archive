"""Tests for the GFN0-xTB parameter loader."""

import pytest


def test_globals_pinned_values():
    """The 20 globals from $globpar must match the file header byte-for-byte."""
    from mlxmolkit.xtb.params_gfn0 import GFN0_GLOBALS as g
    assert g.ks == pytest.approx(2.0000000)
    assert g.kp == pytest.approx(2.4868000)
    assert g.kd == pytest.approx(2.2700000)
    assert g.kdiff == pytest.approx(1.1241000)
    assert g.s8 == pytest.approx(2.8500000)
    assert g.s9 == pytest.approx(0.0)              # GFN0: no ATM
    assert g.a1 == pytest.approx(0.8000000)
    assert g.a2 == pytest.approx(4.6000000)
    assert g.kexp == pytest.approx(1.5)
    assert g.kexplight == pytest.approx(1.5)
    assert g.renscale == pytest.approx(-0.0900000)


def test_full_element_coverage():
    """All Z = 1..86 must load without errors."""
    from mlxmolkit.xtb.params_gfn0 import GFN0_PARAMS
    assert set(GFN0_PARAMS.keys()) == set(range(1, 87))


def test_hydrogen_shells():
    """H has the auxiliary 2s shell beyond its main 1s — confirm both load."""
    from mlxmolkit.xtb.params_gfn0 import GFN0_PARAMS
    H = GFN0_PARAMS[1]
    assert H.symbol == "H"
    assert len(H.shells) == 2
    assert H.shells[0].n == 1 and H.shells[0].l == 0   # 1s
    assert H.shells[1].n == 2 and H.shells[1].l == 0   # 2s (auxiliary)
    assert H.shells[0].h == pytest.approx(-11.9223639)
    assert H.shells[1].h == pytest.approx(-2.8061095)
    assert H.en == pytest.approx(1.92)
    assert H.eeq_chi == pytest.approx(1.25)


def test_carbon_shells():
    from mlxmolkit.xtb.params_gfn0 import GFN0_PARAMS
    C = GFN0_PARAMS[6]
    assert C.symbol == "C"
    assert len(C.shells) == 2
    assert C.shells[0].l == 0 and C.shells[0].n == 2   # 2s
    assert C.shells[1].l == 1 and C.shells[1].n == 2   # 2p
    # Per-shell parameters
    assert C.shells[0].h == pytest.approx(-15.7545853)
    assert C.shells[1].h == pytest.approx(-9.7975356)
    assert C.shells[0].zeta == pytest.approx(1.9915841)
    assert C.shells[1].zeta == pytest.approx(1.7845353)
    # K_cn and K_q are shell-resolved:
    assert C.shells[0].k_cn == pytest.approx(-5.5477603)   # KCNS
    assert C.shells[1].k_cn == pytest.approx(1.5631408)    # KCNP
    # k_q2 is atom-resolved (single value)
    assert C.k_q2 == pytest.approx(0.0908886)
    assert C.en == pytest.approx(2.48)


def test_oxygen_en_fallback():
    """O has no EN= in the file; must fall back to the hardcoded array."""
    from mlxmolkit.xtb.params_gfn0 import GFN0_PARAMS
    O = GFN0_PARAMS[8]
    assert O.symbol == "O"
    assert O.en == pytest.approx(3.44)


def test_iodine_three_shells():
    """I has 5s + 5p + 5d — three shells."""
    from mlxmolkit.xtb.params_gfn0 import GFN0_PARAMS
    I = GFN0_PARAMS[53]
    assert I.symbol == "I"
    assert len(I.shells) == 3
    assert [s.l for s in I.shells] == [0, 1, 2]
    assert all(s.n == 5 for s in I.shells)
    # EN falls back from array (Z=53 → 2.66)
    assert I.en == pytest.approx(2.66)


def test_method_registry_normalization():
    """Method-name lookup is case- and dash-insensitive."""
    from mlxmolkit.xtb.methods import get_xtb_params, XTB_METHOD_PARAMS
    p1 = get_xtb_params("GFN0")
    p2 = get_xtb_params("gfn0")
    p3 = get_xtb_params("GFN0-xTB")
    p4 = get_xtb_params("gfn0xtb")
    assert p1 is p2 is p3 is p4 is XTB_METHOD_PARAMS["GFN0"]


def test_method_registry_unknown_raises():
    from mlxmolkit.xtb.methods import get_xtb_params
    with pytest.raises(ValueError, match="Unknown xTB method"):
        get_xtb_params("PM7")


def test_params_are_frozen():
    """Element params should be immutable to avoid silent mutation bugs."""
    from mlxmolkit.xtb.params_gfn0 import GFN0_PARAMS
    H = GFN0_PARAMS[1]
    with pytest.raises(Exception):  # FrozenInstanceError
        H.en = 99.0


def test_no_None_in_shell_fields():
    """Every shell must have all numeric fields populated (zeros for missing
    KCN*/POLY*/KQ* are fine, but no None leaking through to consumers)."""
    from mlxmolkit.xtb.params_gfn0 import GFN0_PARAMS
    for Z, p in GFN0_PARAMS.items():
        for s in p.shells:
            assert s.h is not None
            assert s.zeta is not None
            assert s.k_cn is not None
            assert s.k_poly is not None
            assert s.k_q1 is not None
