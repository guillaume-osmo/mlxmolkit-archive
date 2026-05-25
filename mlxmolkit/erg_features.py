"""
ERG (Extended Reduced Graph) fingerprint adapter for the mlxmolkit pipeline.

The ERG fingerprint (Stiefl et al., *J. Chem. Inf. Model.* 46:208, 2006) encodes
pharmacophore-pair counts at varying topological distances on a reduced graph
(node types = pharmacophoric features rather than atom types). It's a 315-dim
dense float vector and captures *scaffold-hopping*-friendly similarity: two
molecules with very different skeletons but the same pharmacophore arrangement
score high.

This module wraps RDKit's `rdReducedGraphs.GetErGFingerprint` (CPU; per-mol cost
dominated by graph reduction) and returns an MLX-resident `(N, 315)` float32
matrix so downstream similarity (cosine via :mod:`mlxmolkit.cosine_dense`) and
clustering can run on the Metal GPU without further conversion.

Public API:

- :func:`erg_fp_from_mols`    — RDKit mols → mx.array (N, 315) float32
- :func:`erg_fp_from_smiles`  — SMILES list → (mx.array (N_valid, 315), idx_map)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import mlx.core as mx


_ERG_DIM = 315


def erg_fp_from_mols(
    mols: Sequence,
    *,
    atom_types: int = 0,
    fuzz_increment: float = 0.3,
    min_path: int = 1,
    max_path: int = 15,
) -> mx.array:
    """ERG fingerprint of N pre-parsed RDKit mols → mx.array (N, 315) float32.

    Default parameters match the original Stiefl 2006 setup
    (``atom_types=0`` keeps the standard pharmacophore typing).
    """
    try:
        from rdkit.Chem import rdReducedGraphs
    except Exception as e:
        raise ImportError("RDKit is required for ERG fingerprints.") from e

    if len(mols) == 0:
        return mx.zeros((0, _ERG_DIM), dtype=mx.float32)

    out = np.zeros((len(mols), _ERG_DIM), dtype=np.float32)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = rdReducedGraphs.GetErGFingerprint(
            mol, atomTypes=atom_types, fuzzIncrement=fuzz_increment,
            minPath=min_path, maxPath=max_path,
        )
        out[i] = np.asarray(fp, dtype=np.float32)
    return mx.array(out, dtype=mx.float32)


def erg_fp_from_smiles(
    smiles: Sequence[str],
    *,
    atom_types: int = 0,
    fuzz_increment: float = 0.3,
    min_path: int = 1,
    max_path: int = 15,
) -> Tuple[mx.array, List[int]]:
    """ERG fingerprint of N SMILES, with implicit validity filtering.

    Returns
    -------
    fp : mx.array (N_valid, 315) float32
        Each row is a valid molecule's ERG fingerprint. Empty if all invalid.
    idx_map : list[int]
        For each row in `fp`, the corresponding index into the input SMILES list.
        Invalid SMILES (RDKit returns None) are skipped.
    """
    try:
        from rdkit.Chem import MolFromSmiles
    except Exception as e:
        raise ImportError("RDKit is required for ERG fingerprints.") from e

    valid_mols, idx_map = [], []
    for i, smi in enumerate(smiles):
        if not smi:
            continue
        mol = MolFromSmiles(smi)
        if mol is None:
            continue
        valid_mols.append(mol)
        idx_map.append(i)

    fp = erg_fp_from_mols(
        valid_mols,
        atom_types=atom_types,
        fuzz_increment=fuzz_increment,
        min_path=min_path,
        max_path=max_path,
    )
    return fp, idx_map
