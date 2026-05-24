"""
xTB backend for COSMO-RS: use GFN2-xTB charges for sigma profiles.

GFN2-xTB charges are MORE polarized than RM1 (closer to DFT),
giving better sigma profiles and activity coefficients.

Pipeline: SMILES → RDKit 3D → xTB charges → COSMO cavity → sigma profile

Requires: conda install -c conda-forge xtb-python
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def xtb_charges(
    atoms: list[int],
    coords: np.ndarray,
    solvent: str = None,
) -> tuple[float, np.ndarray]:
    """Compute GFN2-xTB energy and Mulliken charges.

    Args:
        atoms: atomic numbers
        coords: (n_atoms, 3) in Angstrom
        solvent: ALPB solvent name (None for gas phase)

    Returns:
        energy: total energy in Hartree
        charges: (n_atoms,) Mulliken charges
    """
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_MUTED

    numbers = np.array(atoms)
    positions = np.array(coords) / 0.529177  # Angstrom → Bohr

    calc = Calculator(Param.GFN2xTB, numbers, positions)
    calc.set_verbosity(VERBOSITY_MUTED)
    if solvent:
        calc.set_solvent(solvent)

    res = calc.singlepoint()
    return res.get_energy(), res.get_charges()


def smiles_to_cosmo_xtb(
    smiles: str,
    n_surface_points: int = 194,
    epsilon: float = 78.39,
    seed: int = 42,
) -> Optional[dict]:
    """Generate COSMO surface from SMILES using xTB charges.

    Pipeline: SMILES → RDKit 3D → GFN2-xTB charges → COSMO cavity → sigma

    More polarized charges than RM1 → better sigma profiles.
    """
    from ..rm1.pipeline import _smiles_to_3d
    from .cavity import build_cavity, compute_cosmo_charges
    from .sigma import full_sigma_analysis

    result_3d = _smiles_to_3d(smiles, seed=seed)
    if result_3d is None:
        return None

    atoms, coords = result_3d

    # GFN2-xTB charges
    try:
        energy_hartree, charges = xtb_charges(atoms, coords)
    except Exception:
        return None

    # COSMO cavity + charges
    seg_pos, seg_area, seg_normal, seg_atom = build_cavity(
        atoms, coords, n_points_per_atom=n_surface_points,
    )

    seg_charge = compute_cosmo_charges(
        atoms, coords, charges, seg_pos, seg_area, epsilon=epsilon,
    )

    seg_sigma = seg_charge / seg_area
    cavity_area = np.sum(seg_area)
    r_dot_n = np.sum((seg_pos - np.mean(coords, axis=0)) * seg_normal, axis=1)
    cavity_volume = np.abs(np.sum(r_dot_n * seg_area) / 3.0)

    cosmo_result = {
        'seg_pos': seg_pos, 'seg_area': seg_area,
        'seg_charge': seg_charge, 'seg_sigma': seg_sigma,
        'seg_normal': seg_normal, 'seg_atom': seg_atom,
        'mulliken_charges': charges,
        'cavity_area': cavity_area, 'cavity_volume': cavity_volume,
        'n_seg': len(seg_pos),
    }

    sigma_result = full_sigma_analysis(cosmo_result, atoms)

    return {
        'smiles': smiles,
        'atoms': atoms, 'coords': coords,
        'n_atoms': len(atoms),
        'method': 'GFN2-xTB',
        'energy_hartree': energy_hartree,
        'energy_eV': energy_hartree * 27.2114,
        **cosmo_result, **sigma_result,
    }


def batch_smiles_to_cosmo_xtb(
    smiles_list: list[str],
    n_surface_points: int = 194,
    epsilon: float = 78.39,
) -> list[Optional[dict]]:
    """Batch COSMO with xTB charges."""
    return [smiles_to_cosmo_xtb(s, n_surface_points, epsilon, seed=42+i)
            for i, s in enumerate(smiles_list)]
