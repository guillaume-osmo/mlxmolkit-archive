"""
Batch COSMO pipeline: N molecules SMILES → sigma profiles in one call.

Uses RM1 batch SCF (Metal GPU) then vectorized COSMO cavity + sigma.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def batch_smiles_to_cosmo(
    smiles_list: list[str],
    method: str = 'RM1',
    n_surface_points: int = 194,
    epsilon: float = 78.39,
    use_metal: bool = True,
    verbose: bool = False,
) -> list[Optional[dict]]:
    """Generate COSMO surfaces for N molecules from SMILES (batched).

    Pipeline:
      1. RDKit 3D generation (per molecule)
      2. RM1 batch SCF (Metal GPU, all N at once)
      3. COSMO cavity + sigma profile (per molecule, vectorized numpy)

    Args:
        smiles_list: list of SMILES strings
        method: NDDO method ('RM1', 'AM1', 'PM3', etc.)
        n_surface_points: Lebedev points per atom
        epsilon: dielectric constant
        use_metal: use Metal GPU for SCF
        verbose: print progress

    Returns:
        list of COSMO result dicts (None for failed molecules)
    """
    from ..rm1.pipeline import _smiles_to_3d
    from ..rm1.scf import rm1_energy_batch
    from ..rm1.methods import get_params
    from .cavity import cosmo_surface, cosmo_surface_batch
    from .sigma import full_sigma_analysis

    N = len(smiles_list)
    PARAMS = get_params(method)

    if verbose:
        print(f"Batch COSMO: {N} molecules, method={method}")

    # Step 1: Generate 3D with dedup cache (RDKit is 40% of total time)
    _3d_cache = {}
    mol_data = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        if smi in _3d_cache:
            atoms, coords = _3d_cache[smi]
        else:
            r = _smiles_to_3d(smi, seed=42 + i)
            if r is None:
                continue
            atoms, coords = r
            if any(z not in PARAMS for z in atoms):
                continue
            _3d_cache[smi] = (atoms, coords)
        mol_data.append((atoms, coords.copy()))
        valid_idx.append(i)

    if verbose:
        print(f"  3D generated: {len(mol_data)}/{N}")

    if not mol_data:
        return [None] * N

    # Step 2: Batch RM1 SCF (Metal GPU)
    scf_results = rm1_energy_batch(mol_data, method=method, use_metal=use_metal)

    if verbose:
        n_conv = sum(1 for r in scf_results if r['converged'])
        print(f"  SCF converged: {n_conv}/{len(mol_data)}")

    # Step 3: COSMO cavity + sigma — Metal GPU batch or sequential CPU
    cosmo_inputs = []
    cosmo_map = []
    for j, idx in enumerate(valid_idx):
        if scf_results[j]['converged']:
            atoms, coords = mol_data[j]
            cosmo_inputs.append((atoms, coords, scf_results[j]['density']))
            cosmo_map.append((j, idx))

    results = [None] * N
    n_cosmo = 0

    if cosmo_inputs:
        try:
            cosmo_results = cosmo_surface_batch(
                cosmo_inputs, n_points=n_surface_points,
                epsilon=epsilon, use_metal=use_metal,
            )
            for k, (j, idx) in enumerate(cosmo_map):
                atoms, coords = mol_data[j]
                sigma_result = full_sigma_analysis(cosmo_results[k], atoms)
                results[idx] = {
                    'smiles': smiles_list[idx],
                    'atoms': atoms, 'coords': coords,
                    'n_atoms': len(atoms), 'method': method,
                    'energy_eV': scf_results[j]['energy_eV'],
                    'heat_of_formation_kcal': scf_results[j]['heat_of_formation_kcal'],
                    **cosmo_results[k], **sigma_result,
                }
                n_cosmo += 1
        except Exception:
            pass

    if verbose:
        print(f"  COSMO computed: {n_cosmo}/{len(mol_data)}")

    return results


def batch_activity_coefficients(
    smiles_list: list[str],
    x: np.ndarray,
    T: float = 298.15,
    method: str = 'RM1',
    epsilon: float = 78.39,
    use_metal: bool = True,
) -> dict:
    """Compute activity coefficients for a mixture (batch COSMO).

    Args:
        smiles_list: SMILES for each component
        x: mole fractions
        T: temperature (K)
        method: NDDO method
        epsilon: dielectric constant
        use_metal: Metal GPU for SCF

    Returns:
        dict with gamma, lng, cosmo_data
    """
    from .cosmors import activity_coefficients

    cosmo_data = batch_smiles_to_cosmo(smiles_list, method=method,
                                        epsilon=epsilon, use_metal=use_metal)

    # Check all succeeded
    for i, r in enumerate(cosmo_data):
        if r is None:
            raise ValueError(f"Failed to compute COSMO for {smiles_list[i]}")

    x = np.asarray(x, dtype=np.float64)
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
