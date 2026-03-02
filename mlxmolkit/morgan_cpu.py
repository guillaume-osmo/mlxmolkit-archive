"""
Morgan fingerprint adapter (CPU via RDKit) that outputs uint8 packed bit-vectors.

This keeps chemistry on CPU (RDKit) and produces MLX-friendly tensors for Tanimoto/Butina.

For parity with the nvMolKit blog and RDKit ClusterData, use the same generator API:
  rdFingerprintGenerator.GetMorganGenerator(radius=..., fpSize=...) and export to bytes.
"""
from __future__ import annotations
from typing import List, Sequence

import numpy as np
import mlx.core as mx

def morgan_fp_bytes_from_mols(
    mols: Sequence,
    *,
    radius: int = 3,
    nbits: int = 1024,
    use_count_simulation: bool = True,
) -> mx.array:
    """
    Same Morgan API as the nvMolKit blog: GetMorganGenerator(radius=, fpSize=).GetFingerprints(mols).
    Export to (N, nbytes) uint8 packed little-endian for Metal Tanimoto.
    use_count_simulation: True to match RDKit blog default; False matches GetMorganFingerprintAsBitVect.
    """
    try:
        from rdkit import DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except Exception as e:
        raise ImportError("RDKit is required for Morgan fingerprints.") from e

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nbits,
        countSimulation=use_count_simulation,
    )
    fps = generator.GetFingerprints(mols)
    nbytes = (nbits + 7) // 8
    out = np.zeros((len(mols), nbytes), dtype=np.uint8)
    for i, bv in enumerate(fps):
        bits = np.zeros((nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, bits)
        packed = np.packbits(bits, bitorder="little")
        out[i] = packed[:nbytes]
    return mx.array(out, dtype=mx.uint8)


def morgan_fp_bytes_from_smiles(
    smiles: Sequence[str],
    *,
    radius: int = 2,
    nbits: int = 2048,
    use_chirality: bool = True,
    use_rdkit_generator: bool = False,
) -> mx.array:
    """
    Morgan fps from SMILES. By default uses GetMorganFingerprintAsBitVect (legacy).
    Set use_rdkit_generator=True to use GetMorganGenerator (same as blog) for RDKit parity.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception as e:
        raise ImportError("RDKit is required for Morgan fingerprints in this adapter.") from e

    mols: List = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi}")
        mols.append(mol)

    if use_rdkit_generator:
        return morgan_fp_bytes_from_mols(
            mols,
            radius=radius,
            nbits=nbits,
            use_count_simulation=True,
        )

    nbytes = (nbits + 7) // 8
    out = np.zeros((len(smiles), nbytes), dtype=np.uint8)
    for i, mol in enumerate(mols):
        bv = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nbits, useChirality=use_chirality
        )
        bits = np.zeros((nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, bits)
        packed = np.packbits(bits, bitorder="little")
        out[i] = packed[:nbytes]
    return mx.array(out, dtype=mx.uint8)
