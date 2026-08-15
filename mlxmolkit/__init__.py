"""
mlxmolkit — GPU-accelerated molecular toolkit on Apple Silicon.

Pipelines:
  1. Conformer generation: DG (4D) → ETK (3D) → MMFF94 optimization
  2. Binary-FP clustering: Morgan FP → Tanimoto → Butina
  3. Dense-FP similarity: ERG fingerprint → cosine (Metal-backed via matmul)

Every public name below is bound on first use rather than at import (PEP 562).
The eager form pulled the whole Metal stack in through `tanimoto_metal_u32`,
so `import mlxmolkit` — or importing any submodule, which runs this file
first — paid for creating a Metal context whether or not the caller wanted the
GPU. That costs 87 ms in one process and does not parallelise: fourteen
processes importing `mlx.core` concurrently take 2.78 s against 0.20 s for
numpy, which was most of the warm-up for `nddo_energy_many`'s worker pool and
all of it wasted, since a worker running the sequential SCF never touches mlx.

`from mlxmolkit import X` and `mlxmolkit.X` both still work, and both return
the same object the eager import did — `tests/test_opencheese_namespace.py`
checks that identity across the two namespaces.
"""

__version__ = "0.4.0"

# Public name -> the module that defines it.
_LAZY_EXPORTS = {
    # --- Binary-FP pipeline (Morgan / Tanimoto / Butina) ---
    "tanimoto_matrix_metal_u32": "mlxmolkit.tanimoto_metal_u32",
    "fused_neighbor_list_metal": "mlxmolkit.fused_tanimoto_nlist",
    "tanimoto_neighbors_blockwise": "mlxmolkit.tanimoto_blockwise",
    "fp_uint8_to_uint32": "mlxmolkit.fp_uint32",
    "ButinaResult": "mlxmolkit.butina",
    "butina_from_neighbor_list_csr": "mlxmolkit.butina",
    "butina_from_similarity_matrix": "mlxmolkit.butina",
    "butina_tanimoto_mlx": "mlxmolkit.butina",
    "morgan_fp_bytes_from_mols": "mlxmolkit.morgan_cpu",
    "morgan_fp_bytes_from_smiles": "mlxmolkit.morgan_cpu",
    "ATOM_FEATURE_NAMES": "mlxmolkit.dipole_features",
    "DipoleFeatureTensors": "mlxmolkit.dipole_features",
    "dipole_atom_feature_tensors": "mlxmolkit.dipole_features",
    "export_dipole_atom_feature_dataset": "mlxmolkit.dipole_features",
    # --- Shape/electrostatic descriptors (CHEESE) ---
    # Re-exported so `mlxmolkit.cheese_batch` and `opencheese.cheese_batch` are
    # the same object. opencheese.descriptors imports these from
    # mlxmolkit.cheese, and tests/test_opencheese_namespace.py asserts the two
    # namespaces agree by identity.
    "CheeseBatch": "mlxmolkit.cheese",
    "cheese_batch": "mlxmolkit.cheese",
    "cheese_batch_from_rdkit_mols": "mlxmolkit.cheese",
    "cheese_similarity_matrix_mlx": "mlxmolkit.cheese",
    # --- Dense-FP pipeline (ERG / cosine) ---
    "erg_fp_from_mols": "mlxmolkit.erg_features",
    "erg_fp_from_smiles": "mlxmolkit.erg_features",
    "cosine_matrix_dense": "mlxmolkit.cosine_dense",
    "l2_normalize_rows": "mlxmolkit.cosine_dense",
    "max_cosine_to_set": "mlxmolkit.cosine_dense",
    # --- Conformer generation ---
    "generate_conformers_nk": "mlxmolkit.conformer_pipeline_v2",
    "ConformerResult": "mlxmolkit.conformer_pipeline_v2",
    "PipelineResult": "mlxmolkit.conformer_pipeline_v2",
}


def __getattr__(name):
    """Bind a public name, or a submodule, on first access."""
    import importlib

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value          # subsequent lookups skip this path
        return value

    # `import mlxmolkit; mlxmolkit.butina` used to work because the eager
    # imports above bound every submodule they touched as an attribute.
    if not name.startswith("_"):
        try:
            submodule = importlib.import_module(f"{__name__}.{name}")
        except ImportError:
            pass
        else:
            globals()[name] = submodule
            return submodule

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


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
    "ATOM_FEATURE_NAMES",
    "DipoleFeatureTensors",
    "dipole_atom_feature_tensors",
    "export_dipole_atom_feature_dataset",
]
