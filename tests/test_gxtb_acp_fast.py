"""The direct AO x aux ACP block must reproduce the concatenated-overlap reference."""
import numpy as np
import pytest

from mlxmolkit.xtb.gxtb_acp import (
    build_gxtb_acp_hamiltonian, build_gxtb_acp_hamiltonian_reference,
)
from mlxmolkit.xtb.gxtb_basis import build_gxtb_qvszp_basis

MOLECULES = [
    ([8, 1, 1], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    ([6, 1, 1, 1, 1], [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                       [0.63, -0.63, -0.63], [-0.63, 0.63, -0.63]]),
    ([16, 1, 1], [[0, 0, 0], [1.34, 0, 0], [-0.4, 1.28, 0]]),
    ([6, 8, 1, 1], [[0, 0, 0], [1.21, 0, 0], [-0.57, 0.94, 0], [-0.57, -0.94, 0]]),
]


@pytest.mark.parametrize("atoms,coords", MOLECULES)
def test_matches_reference(atoms, coords):
    xyz = np.asarray(coords, dtype=np.float64)
    basis = build_gxtb_qvszp_basis(np.asarray(atoms, dtype=int), xyz, total_charge=0.0)
    ref = build_gxtb_acp_hamiltonian_reference(atoms, xyz, basis, enabled=True)
    got = build_gxtb_acp_hamiltonian(atoms, xyz, basis, enabled=True)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) < 1e-12


def test_disabled_returns_zeros():
    atoms, coords = MOLECULES[0]
    xyz = np.asarray(coords, dtype=np.float64)
    basis = build_gxtb_qvszp_basis(np.asarray(atoms, dtype=int), xyz, total_charge=0.0)
    assert not np.any(build_gxtb_acp_hamiltonian(atoms, xyz, basis, enabled=False))
