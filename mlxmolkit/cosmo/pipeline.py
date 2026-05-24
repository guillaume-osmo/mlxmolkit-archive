"""
End-to-end COSMO-RS pipeline: SMILES → activity coefficients.

Pipeline:
  SMILES → RDKit 3D → RM1 SCF → Mulliken charges
  → COSMO cavity → surface charges → sigma profile
  → COSMO-RS → activity coefficients, solvation energy
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def smiles_to_cosmo(
    smiles: str,
    method: str = 'RM1',
    n_surface_points: int = 194,
    epsilon: float = 78.39,
    seed: int = 42,
) -> Optional[dict]:
    """Generate COSMO surface from SMILES.

    Pipeline: SMILES → RDKit 3D → RM1 SCF → COSMO cavity → sigma profile

    Args:
        smiles: SMILES string
        method: NDDO method ('RM1', 'AM1', 'AM1_STAR', 'RM1_STAR')
        n_surface_points: Lebedev points per atom (110 or 194)
        epsilon: dielectric constant (78.39 for water)
        seed: random seed for 3D generation

    Returns:
        dict with COSMO surface data and sigma profile, or None if failed
    """
    from ..rm1.pipeline import _smiles_to_3d
    from ..rm1.scf import rm1_energy
    from ..rm1.methods import get_params
    from .cavity import cosmo_surface
    from .sigma import full_sigma_analysis

    # Step 1: Generate 3D
    result_3d = _smiles_to_3d(smiles, seed=seed)
    if result_3d is None:
        return None

    atoms, coords = result_3d
    PARAMS = get_params(method)
    if any(z not in PARAMS for z in atoms):
        return None

    # Step 2: RM1 SCF
    scf_result = rm1_energy(atoms, coords, method=method)
    if not scf_result['converged']:
        return None

    # Step 3: COSMO surface
    cosmo_result = cosmo_surface(
        atoms, coords, scf_result['density'],
        n_points=n_surface_points, epsilon=epsilon,
    )

    # Step 4: Sigma profile
    sigma_result = full_sigma_analysis(cosmo_result, atoms)

    return {
        'smiles': smiles,
        'atoms': atoms,
        'coords': coords,
        'n_atoms': len(atoms),
        'method': method,
        'energy_eV': scf_result['energy_eV'],
        'heat_of_formation_kcal': scf_result['heat_of_formation_kcal'],
        **cosmo_result,
        **sigma_result,
    }


def activity_coefficients_from_smiles(
    smiles_list: list[str],
    x: np.ndarray,
    T: float = 298.15,
    method: str = 'RM1',
    epsilon: float = 78.39,
) -> dict:
    """Compute activity coefficients for a mixture from SMILES.

    Args:
        smiles_list: list of SMILES strings (one per component)
        x: (n_mol,) mole fractions (must sum to 1)
        T: temperature in Kelvin
        method: NDDO method
        epsilon: dielectric constant

    Returns:
        dict with activity coefficients and component data
    """
    from .cosmors import activity_coefficients

    n_mol = len(smiles_list)
    x = np.asarray(x, dtype=np.float64)

    # Generate COSMO for each molecule
    cosmo_data = []
    for smi in smiles_list:
        result = smiles_to_cosmo(smi, method=method, epsilon=epsilon)
        if result is None:
            raise ValueError(f"Failed to compute COSMO for {smi}")
        cosmo_data.append(result)

    # Activity coefficients
    lng = activity_coefficients(cosmo_data, x, T=T)
    gamma = np.exp(lng)

    return {
        'smiles': smiles_list,
        'x': x,
        'T': T,
        'lng': lng,
        'gamma': gamma,
        'cosmo_data': cosmo_data,
    }
