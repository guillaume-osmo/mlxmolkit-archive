"""
COSMO-RS solvation model on Apple Metal GPU.

Pipeline: SMILES → RDKit 3D → RM1 SCF → COSMO cavity → sigma profile
→ COSMO-RS thermodynamics → activity coefficients, solvation energies

Based on openCOSMO-RS (TUHH-TVT) with RM1 replacing DFT/ORCA.

Usage:
    from mlxmolkit.cosmo import smiles_to_cosmo, activity_coefficients_from_smiles

    # Single molecule COSMO surface
    result = smiles_to_cosmo("CCO")
    print(result['sigma_profile'])

    # Activity coefficients for water-ethanol mixture
    result = activity_coefficients_from_smiles(
        ["O", "CCO"], x=[0.5, 0.5], T=298.15
    )
    print(result['gamma'])
"""

from .pipeline import smiles_to_cosmo, activity_coefficients_from_smiles
from .batch import batch_smiles_to_cosmo, batch_activity_coefficients
from .cavity import cosmo_surface, build_cavity
from .ddcosmo import ddcosmo_surface, ddcosmo_charges
from .sigma import full_sigma_analysis, compute_sigma_profile
from .cosmors import activity_coefficients, cosmospace
