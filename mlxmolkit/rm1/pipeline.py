"""
SMILES → RDKit 3D → RM1 energy pipeline.

Usage:
    from mlxmolkit.rm1.pipeline import rm1_from_smiles, rm1_from_smiles_batch

    result = rm1_from_smiles("c1ccccc1")  # benzene
    results = rm1_from_smiles_batch(["O", "C", "CC", "c1ccccc1"])
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def _smiles_to_3d(smiles: str, n_confs: int = 1, seed: int = 42) -> Optional[tuple]:
    """Convert SMILES to 3D coordinates using RDKit ETKDG.

    Returns:
        (atoms, coords) tuple or None if conversion fails
        atoms: list of atomic numbers
        coords: (n_atoms, 3) array in Angstrom
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit required: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # Fallback: try without ETKDG
        status = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if status != 0:
            return None

    # MMFF optimize
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass  # Use unoptimized geometry

    conf = mol.GetConformer()
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    return atoms, coords


def rm1_from_smiles(
    smiles: str,
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    seed: int = 42,
) -> Optional[dict]:
    """Compute RM1 energy from SMILES string.

    Pipeline: SMILES → RDKit AddHs + ETKDG + MMFF → RM1 SCF

    Args:
        smiles: SMILES string
        max_iter: max SCF iterations
        conv_tol: convergence threshold
        seed: random seed for 3D embedding

    Returns:
        Result dict with RM1 energies, or None if 3D generation fails.
        Extra keys: 'smiles', 'atoms', 'coords', 'n_atoms'
    """
    from .scf import rm1_energy
    from .params import RM1_PARAMS

    result_3d = _smiles_to_3d(smiles, seed=seed)
    if result_3d is None:
        return None

    atoms, coords = result_3d

    # Check all elements are supported
    for z in atoms:
        if z not in RM1_PARAMS:
            return None  # unsupported element

    result = rm1_energy(atoms, coords, max_iter=max_iter, conv_tol=conv_tol)
    result['smiles'] = smiles
    result['atoms'] = atoms
    result['coords'] = coords
    result['n_atoms'] = len(atoms)
    return result


def rm1_from_smiles_batch(
    smiles_list: list[str],
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    use_metal: bool = True,
    seed: int = 42,
    verbose: bool = False,
) -> list[Optional[dict]]:
    """Compute RM1 energies for a batch of SMILES strings.

    Pipeline: SMILES → RDKit 3D → RM1 batch SCF (Metal GPU)

    Args:
        smiles_list: list of SMILES strings
        max_iter: max SCF iterations
        conv_tol: convergence threshold
        use_metal: use Metal GPU for Fock matrix build
        seed: random seed for 3D embedding
        verbose: print progress

    Returns:
        List of result dicts (None for failed molecules).
    """
    from .scf import rm1_energy_batch
    from .params import RM1_PARAMS

    if verbose:
        print(f"RM1 batch: {len(smiles_list)} SMILES")

    # Step 1: Generate 3D coordinates for all molecules
    mol_data = []       # (atoms, coords) for valid molecules
    valid_indices = []   # indices into smiles_list
    failed_indices = []  # indices that failed

    for i, smi in enumerate(smiles_list):
        result_3d = _smiles_to_3d(smi, seed=seed + i)
        if result_3d is None:
            failed_indices.append(i)
            continue
        atoms, coords = result_3d
        # Check elements
        if any(z not in RM1_PARAMS for z in atoms):
            failed_indices.append(i)
            continue
        mol_data.append((atoms, coords))
        valid_indices.append(i)

    if verbose:
        print(f"  3D generated: {len(mol_data)}/{len(smiles_list)} "
              f"({len(failed_indices)} failed)")

    if len(mol_data) == 0:
        return [None] * len(smiles_list)

    # Step 2: Batch RM1 SCF
    batch_results = rm1_energy_batch(
        mol_data, max_iter=max_iter, conv_tol=conv_tol,
        use_metal=use_metal, verbose=verbose,
    )

    # Step 3: Assemble results
    results = [None] * len(smiles_list)
    for j, idx in enumerate(valid_indices):
        r = batch_results[j]
        r['smiles'] = smiles_list[idx]
        r['atoms'] = mol_data[j][0]
        r['coords'] = mol_data[j][1]
        r['n_atoms'] = len(mol_data[j][0])
        results[idx] = r

    return results
