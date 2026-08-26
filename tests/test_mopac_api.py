"""The ctypes binding to libmopac must agree with MOPAC's own text output.

`tools/mopac_api.py` declares three C structs by hand. If a field is missing,
misordered, or misaligned — or if a future MOPAC changes the layout — ctypes
will happily read whatever is at that offset and return a plausible-looking
number. Nothing else in the tree would notice.

So the binding is checked against `tests/_mopac_ref_generated.py`, which was
produced the old way: by running the `mopac` executable and scraping its
printed report. Same MOPAC, same frozen geometries, a completely different
path in and out. Agreement to ~1e-4 is only possible if the struct layout is
right.

These are MOPAC-against-MOPAC tolerances, three orders of magnitude tighter
than the mlxmolkit-against-MOPAC ones in test_mopac_parity.py (0.5 kcal/mol,
0.002 e), and they are not the same claim.
"""
from __future__ import annotations

import numpy as np
import pytest

mopac_api = pytest.importorskip("tools.mopac_api")
from tests._mopac_ref_generated import MOPAC_REF  # noqa: E402

try:
    mopac_api._find_libmopac()
except FileNotFoundError:
    pytest.skip("libmopac not installed", allow_module_level=True)

# The residual between the two paths, measured across all 18 molecules: at most
# 4e-05 kcal/mol and 5e-04 e. Coordinate precision is not the cause — the .mop
# file carried 6 decimals and re-running at 6 decimals moves nothing (0.000036
# against 0.000040) — so it is MOPAC's own printed precision plus whatever the
# report rounds. Far below anything either code is claimed to resolve.
HEAT_TOL_KCAL = 1e-3
CHARGE_TOL_E = 2e-3


@pytest.mark.parametrize("name", sorted(MOPAC_REF))
def test_binding_reproduces_the_scraped_reference(name):
    ref = MOPAC_REF[name]
    result = mopac_api.scf(ref["atoms"], np.asarray(ref["coords"]), model="PM6")

    assert result["heat_kcal"] == pytest.approx(ref["hf_kcal"], abs=HEAT_TOL_KCAL), (
        f"{name}: API {result['heat_kcal']:.5f} vs scraped {ref['hf_kcal']:.5f} "
        f"kcal/mol — a struct-layout error reads the wrong bytes for `heat`"
    )
    charges = np.asarray(result["charges"])
    assert charges.shape == (len(ref["atoms"]),)
    assert np.max(np.abs(charges - np.asarray(ref["charges"]))) <= CHARGE_TOL_E


def test_charges_sum_to_the_net_charge():
    """A pointer read past the end of `charge` would still look like numbers."""
    ref = MOPAC_REF["ethanol"]
    result = mopac_api.scf(ref["atoms"], np.asarray(ref["coords"]))
    assert float(np.sum(result["charges"])) == pytest.approx(0.0, abs=1e-4)


def test_a_net_charge_changes_the_answer():
    """`charge` is an int field early in the struct; if its offset were wrong
    the calculation would silently ignore it."""
    ref = MOPAC_REF["ethanol"]
    neutral = mopac_api.scf(ref["atoms"], np.asarray(ref["coords"]))
    anion = mopac_api.scf(ref["atoms"], np.asarray(ref["coords"]), charge=-2)

    assert anion["heat_kcal"] != neutral["heat_kcal"]
    assert float(np.sum(anion["charges"])) == pytest.approx(-2.0, abs=1e-3)


def test_the_model_field_selects_the_hamiltonian():
    """PM6 and AM1 must not return the same number."""
    ref = MOPAC_REF["ethanol"]
    pm6 = mopac_api.scf(ref["atoms"], np.asarray(ref["coords"]), model="PM6")
    am1 = mopac_api.scf(ref["atoms"], np.asarray(ref["coords"]), model="AM1")
    assert abs(pm6["heat_kcal"] - am1["heat_kcal"]) > 0.1


def test_an_unknown_model_is_rejected_before_reaching_fortran():
    with pytest.raises(ValueError, match="unknown model"):
        mopac_api.scf([1, 1], np.zeros((2, 3)), model="B3LYP")


def test_relax_lowers_the_energy_and_moves_the_atoms():
    ref = MOPAC_REF["ethanol"]
    coords = np.asarray(ref["coords"])
    single = mopac_api.scf(ref["atoms"], coords)
    relaxed = mopac_api.relax(ref["atoms"], coords)

    assert relaxed["heat_kcal"] <= single["heat_kcal"] + 1e-6
    assert relaxed["coords"].shape == coords.shape
    assert np.max(np.abs(relaxed["coords"] - coords)) > 1e-4


def test_repeated_calls_stay_stable():
    """Each call allocates in Fortran and frees through the API's destructors.

    A missed free leaks; a double free or a stale pointer corrupts the next
    call. Either shows up as drift or a crash across a run of calls, not on
    the first one.
    """
    ref = MOPAC_REF["dimethyl_sulfide"]
    first = mopac_api.scf(ref["atoms"], np.asarray(ref["coords"]))
    for _ in range(25):
        again = mopac_api.scf(ref["atoms"], np.asarray(ref["coords"]))
        assert again["heat_kcal"] == first["heat_kcal"]
        assert np.array_equal(again["charges"], first["charges"])
