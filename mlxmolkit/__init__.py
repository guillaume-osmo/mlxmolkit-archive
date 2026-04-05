"""
mlxmolkit — GPU-accelerated molecular toolkit on Apple Silicon.

Two pipelines:
  1. Conformer generation: DG (4D) → ETK (3D) → MMFF94 optimization
  2. Molecular clustering: Morgan FP → Tanimoto → Butina
"""

__version__ = "0.4.0"

# --- Clustering ---
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
    # Clustering
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
]
