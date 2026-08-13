"""The d-pair cache must change speed and nothing else.

`d_pair_cache` precomputes the three geometry-dependent d-pair quantities — the
TETCI w tensor, the molecular-frame overlap, and the 9x9 Wigner-D attraction —
in one batched call each, and serves the scalar entry points from memory. Every
one of those routines is written for a batch and was being called a pair at a
time, which is why a 16-atom thioanisole gradient cost more than a 31-atom
menthol one.

A cache is only ever a speed change, so these tests are about the two ways it
can stop being one: returning the wrong entry, and outliving its geometry.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from mlxmolkit.nddo import d_two_center as D
from mlxmolkit.nddo.anal_grad import analytical_gradient
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.pipeline import _smiles_to_3d
from mlxmolkit.nddo.tetci_yh import yh_e1b_contribution

PARAMS = get_params("PM6")


def geometry(smiles):
    result = _smiles_to_3d(smiles, seed=42)
    if result is None:
        pytest.skip(f"could not embed {smiles}")
    return result[0], result[1]


@pytest.mark.parametrize("smiles", [
    "CSc1ccccc1",        # S with an aromatic ring — the original 1440 ms case
    "Clc1ccccc1",
    "COP(=O)(OC)OC",     # P, and four oxygens
    "ClC(Cl)(Cl)Cl",     # every heavy atom carries d
    "c1ccsc1",
    "CCO",               # no d at all: the cache must be a no-op, not a crash
])
def test_the_gradient_is_unchanged_by_the_cache(smiles):
    """Uncached is the reference. Agreement is to round-off, not bit-exact.

    The batched routines evaluate the same arithmetic on an array layout, so a
    pair can land one ULP from its scalar value. A central difference over a
    1e-5 step divides by 2e-5, which turns 1e-12 eV of round-off into ~5e-8
    eV/A. That is the scale seen here, and it is well below the 1e-4 eV/A at
    which this gradient is checked against full-SCF central differences.
    """
    atoms, coords = geometry(smiles)
    _res, cached = analytical_gradient(atoms, coords, method="PM6")

    real = D.d_pair_cache
    from contextlib import contextmanager

    @contextmanager
    def disabled(_specs):
        yield
    D.d_pair_cache = disabled
    try:
        _res, plain = analytical_gradient(atoms, coords, method="PM6")
    finally:
        D.d_pair_cache = real

    assert np.abs(cached - plain).max() < 1e-6


def test_an_entry_is_returned_for_the_pair_it_was_computed_for():
    """The attraction is direction-dependent: e1b(A from B) is not e1b(B from A).

    Keying it the way the TETCI tensor is keyed — Z-sorted, so either argument
    order collapses onto one entry — would hand back the wrong block. This is
    the failure that a gradient still runs happily with.
    """
    atoms, coords = geometry("CSc1ccccc1")
    s = next(i for i, z in enumerate(atoms) if z == 16)
    c = next(i for i, z in enumerate(atoms) if z == 6)
    pS, pC = PARAMS[16], PARAMS[6]
    rS, rC = coords[s], coords[c]

    want = yh_e1b_contribution(pS, pC, rS, rC)
    with D.d_pair_cache([(pS, pC, rS, rC)]):
        assert np.array_equal(yh_e1b_contribution(pS, pC, rS, rC), want)


def test_the_cache_does_not_outlive_its_geometry():
    """A stale cache is silently wrong, not slow, so unwinding has to be exact."""
    atoms, coords = geometry("CSc1ccccc1")
    s = next(i for i, z in enumerate(atoms) if z == 16)
    c = next(i for i, z in enumerate(atoms) if z == 6)
    spec = (PARAMS[16], PARAMS[6], coords[s], coords[c])

    assert D._TETCI_CACHE is None
    with D.d_pair_cache([spec]):
        assert D._TETCI_CACHE is not None
        with D.d_pair_cache([spec]):          # nesting restores the outer one
            pass
        assert D._TETCI_CACHE is not None
    assert D._TETCI_CACHE is None

    with pytest.raises(RuntimeError):
        with D.d_pair_cache([spec]):
            raise RuntimeError("boom")
    assert D._TETCI_CACHE is None, "an exception left a stale geometry installed"


def test_an_empty_spec_list_installs_an_empty_cache():
    """An sp-only molecule reaches this with nothing to precompute."""
    with D.d_pair_cache([]):
        assert D._TETCI_CACHE == {}
        assert D._OVERLAP_CACHE == {}
        assert D._E1B_CACHE == {}
    assert D._TETCI_CACHE is None
