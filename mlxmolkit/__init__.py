"""
mlxmolkit — GPU-accelerated molecular toolkit on Apple Silicon.

Pipelines:
  1. Conformer generation: DG (4D) → ETK (3D) → MMFF94 optimization
  2. Binary-FP clustering: Morgan FP → Tanimoto → Butina
  3. Dense-FP similarity: ERG fingerprint → cosine (Metal-backed via matmul)
"""

__version__ = "0.4.0"

# --- Binary-FP pipeline (Morgan / Tanimoto / Butina) ---
from mlxmolkit.tanimoto_metal_u32 import tanimoto_matrix_metal_u32
from mlxmolkit.fused_tanimoto_nlist import fused_neighbor_list_metal
from mlxmolkit.tanimoto_blockwise import tanimoto_neighbors_blockwise
from mlxmolkit.fp_uint32 import fp_uint8_to_uint32
from mlxmolkit.butina import (
    ButinaResult,
    butina_from_neighbor_list_csr,
    butina_from_similarity_matrix,
    butina_tanimoto_mlx,
)
from mlxmolkit.morgan_cpu import morgan_fp_bytes_from_mols, morgan_fp_bytes_from_smiles

# --- Dense-FP pipeline (ERG / cosine) ---
from mlxmolkit.erg_features import erg_fp_from_mols, erg_fp_from_smiles
from mlxmolkit.cosine_dense import (
    cosine_matrix_dense,
    l2_normalize_rows,
    max_cosine_to_set,
)

# --- Conformer generation ---
from mlxmolkit.conformer_pipeline_v2 import (
    generate_conformers_nk,
    ConformerResult,
    PipelineResult,
)

__all__ = [
    # Conformer generation
    "generate_conformers_nk",
    "ConformerResult",
    "PipelineResult",
    # Binary-FP clustering
    "tanimoto_matrix_metal_u32",
    "fused_neighbor_list_metal",
    "tanimoto_neighbors_blockwise",
    "fp_uint8_to_uint32",
    "butina_from_neighbor_list_csr",
    "butina_from_similarity_matrix",
    "butina_tanimoto_mlx",
    "ButinaResult",
    "morgan_fp_bytes_from_mols",
    "morgan_fp_bytes_from_smiles",
    # Dense-FP similarity
    "erg_fp_from_mols",
    "erg_fp_from_smiles",
    "cosine_matrix_dense",
    "l2_normalize_rows",
    "max_cosine_to_set",
]
