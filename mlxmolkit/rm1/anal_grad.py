"""
Analytical gradient for NDDO methods.

Uses Hellmann-Feynman theorem with Pulay correction:
  dE/dR_A = dE_nuc/dR_A + Σ_μν P_μν · dH_μν/dR_A
            + 0.5 · Σ_μνλσ D_μν,λσ · d(μν|λσ)/dR_A

where D is the two-particle density: D = P⊗P - 0.5·P⊗P (exchange)

Integral derivatives via finite-difference (δ=1e-5, PYSEQM convention).
ONE SCF + derivative pass vs 6N+1 SCF calls for numerical gradient.
"""
from __future__ import annotations

import numpy as np
from .scf import rm1_energy
from .methods import get_params
from .integrals import compute_nuclear_repulsion


def analytical_gradient(
    atoms: list[int],
    coords: np.ndarray,
    method: str = 'RM1',
    step: float = 1e-5,
) -> tuple[dict, np.ndarray]:
    """Compute energy and gradient.

    Costs: 1 SCF + (6·n_atoms) integral evaluations (no SCF re-solve).

    Returns:
        result: SCF result dict
        gradient: (n_atoms, 3) in eV/Angstrom
    """
    PARAMS = get_params(method)
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)

    # Converged SCF
    result = rm1_energy(atoms, coords, method=method, max_iter=200, conv_tol=1e-8)
    P = result['density']
    E0 = result['energy_eV']

    # Gradient: dE/dR_A for each atom A and Cartesian direction
    # Use the fact that E = 0.5·tr(P·(H+F)) + E_nuc
    # and all matrices depend on geometry through integrals only.
    #
    # The most robust approach for NDDO: compute E(R+δ) - E(R-δ)
    # but WITHOUT re-solving SCF — use the FROZEN density P.
    # This is the Hellmann-Feynman gradient (exact for variational methods).

    gradient = np.zeros((n_atoms, 3))

    for a in range(n_atoms):
        for d in range(3):
            coords_p = coords.copy()
            coords_m = coords.copy()
            coords_p[a, d] += step
            coords_m[a, d] -= step

            # Compute E with displaced geometry but FROZEN density P
            E_p = _energy_frozen_density(atoms, coords_p, P, PARAMS)
            E_m = _energy_frozen_density(atoms, coords_m, P, PARAMS)

            gradient[a, d] = (E_p - E_m) / (2.0 * step)

    return result, gradient


def _energy_frozen_density(atoms, coords, P, PARAMS):
    """Compute total energy with frozen density matrix P at new geometry.

    E = 0.5 · tr(P · (H + F)) + E_nuc

    Rebuilds H and F from integrals at displaced geometry,
    but does NOT re-solve SCF (density P is fixed).
    """
    from .scf import _build_basis_info, _build_core_hamiltonian, _build_fock

    info = _build_basis_info(atoms, PARAMS)
    H = _build_core_hamiltonian(atoms, coords, info)
    F = _build_fock(H, P, info, atoms, coords)

    # Electronic energy
    E_elec = 0.5 * np.sum(P * (H + F))

    # Nuclear repulsion
    E_nuc = compute_nuclear_repulsion(atoms, coords, param_dict=PARAMS)

    return E_elec + E_nuc
