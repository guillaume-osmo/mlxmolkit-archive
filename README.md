# mlxmolkit — GPU-accelerated molecular toolkit on Apple Silicon

Port of [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (CUDA) to Apple Metal via [MLX](https://github.com/ml-explore/mlx). Two pipelines:

1. **Molecular Clustering** — Morgan FP → Tanimoto similarity → Butina clustering
2. **3D Conformer Generation** — DG (4D) → ETK (3D) → MMFF94 optimization

## Features

- **Conformer Generation** — Drop-in replacement for RDKit's ETKDG (`EmbedMolecules`). Supports ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG, ETDG, and pure DG.
- **MMFF94 Optimization** — GPU-accelerated force field optimization (`MMFFOptimizeMoleculesConfs`). All 7 MMFF energy terms with fused Metal kernel. Full BFGS or L-BFGS in-kernel (zero CPU round-trips).
- **Molecular Clustering** — Butina clustering at 150k+ molecules with divide-and-conquer memory management.
- **N x k Parallel** — Generate k conformers for N molecules simultaneously. Constraints shared across conformers (`conf_to_mol` indirection, 50% memory savings).

## Performance

### Conformer Generation (N=20 molecules, k=50 conformers = 1000 total)

| Pipeline | Time | Throughput | GPU Memory |
|----------|------|-----------|------------|
| DG only | 0.13s | 7,549 conf/s | 2.6 MB |
| DG + ETK | 0.16s | 6,228 conf/s | 2.6 MB |
| DG + ETK + MMFF | 0.52s | 1,908 conf/s | 5.1 MB |

### Conformer Memory Scaling (DG + ETK + MMFF, batch=500)

| Conformers | Batch | GPU (BFGS) | GPU (L-BFGS) | Time | Throughput |
|-----------|-------|-----------|-------------|------|-----------|
| 1,000 | 1000 | 5.1 MB | 2.9 MB | 0.43s | 2,342/s |
| 2,000 | 500 | 2.6 MB | 1.5 MB | 1.43s | 1,402/s |
| 4,000 | 500 | 2.6 MB | 1.5 MB | 1.91s | 2,094/s |
| 10,000 | 500 | **2.6 MB** | **1.5 MB** | 4.82s | 2,075/s |

GPU memory stays constant regardless of total conformers thanks to divide-and-conquer batching.

### Clustering (Enamine REAL subset, Apple M3 Max)

| N | Fused sim→CSR | Butina | **Total** | vs RDKit | Memory |
|---|---|---|---|---|---|
| 20k | 0.26s | 0.09s | **0.35s** | **152x** | 0.1 MB |
| 50k | 1.26s | 0.36s | **1.62s** | — | 0.5 MB |
| 100k | 4.87s | 0.97s | **5.84s** | — | 1.3 MB |
| 150k+ | blockwise | — | scales | — | bounded |

### ETKDG Variant Comparison (N=20, k=50)

| Variant | conf/s | Convergence |
|---------|--------|-------------|
| DG | 7,549 | 96.6% |
| KDG | 7,243 | 96.6% |
| ETDG | 1,844 | 96.6% |
| ETKDG | 6,064 | 96.6% |
| ETKDGv2 | 6,228 | 96.6% |
| ETKDGv3 | 6,636 | 96.6% |
| srETKDGv3 | 6,678 | 96.6% |

## Architecture

### Conformer Generation (N x k parallel)

```
SMILES x N
    |
[RDKit CPU] Extract params ONCE per molecule
    |
[Pack] SharedConstraintBatch (conf_to_mol indirection)
    |
+-- Stage 1: DG minimize (4D, Metal TPM=32) --------+
|   One threadgroup per conformer                    |
|   L-BFGS in-kernel, GPU-parallel line search       |
|   Shared constraints via conf_to_mol               |
+----------------------------------------------------+
    |
[Extract 3D] Drop 4th coordinate
    |
+-- Stage 2: ETK minimize (3D, Metal TPM=32) --------+
|   CSD torsion + improper + 1-4 distance             |
|   Optional parallel_grad for large molecules        |
+-----------------------------------------------------+
    |
+-- Stage 3: MMFF94 optimize (Metal, in-kernel) ------+
|   7 energy terms: bond, angle, stretch-bend,         |
|   OOP, torsion, vdW, electrostatic                   |
|   BFGS (default) or L-BFGS option                    |
+------------------------------------------------------+
    |
Optimized 3D conformers
```

### Clustering (divide-and-conquer for 150k+)

```
Morgan FP (RDKit CPU)
        |
   uint8 -> uint32 packing
        |
+-- N <= 100k: Fused Metal Kernel ------+
|   Single dispatch, no NxN matrix       |
+-- N > 100k: Blockwise D&C ------------+
|   Tile both dimensions (auto-sized)    |
|   mx.eval() between tiles (free GPU)  |
+----------------------------------------+
        |
   Butina greedy (CPU, numpy CSR)
        |
   Clusters
```

## Adaptive Iteration Scaling

Iterations auto-scale by molecule complexity (default). Small molecules converge early via in-kernel TOLX/gradient checks — no wasted GPU compute.

Formula: `max_iters = base + scale * max(n_atoms, sqrt(n_constraints))`

| Molecule | Atoms | Constraints | DG iters | ETK iters | MMFF iters |
|----------|-------|-------------|----------|-----------|------------|
| Methane | 5 | 10 | 400 | 200 | 275 |
| Benzene | 12 | 66 | 540 | 270 | 380 |
| Aspirin | 21 | 210 | 720 | 360 | 515 |
| Testosterone | 49 | 1176 | 1280 | 640 | 935 |
| 64-atom | 64 | 2016 | 1580 | 790 | 1160 |

Override with explicit values when needed:

```python
# Auto (default) — scales with molecule size
result = generate_conformers_nk(smiles_list, n_confs_per_mol=10)

# Fixed iterations for fine control
result = generate_conformers_nk(smiles_list, n_confs_per_mol=10,
    dg_max_iters=1000, etk_max_iters=500, mmff_max_iters=400)
```

## Optimization Options

| Option | Flag | Effect | When to use |
|--------|------|--------|-------------|
| Auto iterations (default) | `dg_max_iters=0` | Scales with molecule size | Always (default) |
| Warm-start retry | automatic | Re-runs non-converged with 2x iters | Always (automatic) |
| ETK parallel gradient | `parallel_grad=True` | 1.18x ETK speedup | Many distance constraints |
| DG parallel gradient | `parallel_grad=True` | Parallelizes dist gradient | >500 distance constraints |
| MMFF94s variant | `mmff_variant="MMFF94s"` | Softer torsion barriers | Conjugated/aromatic molecules |
| MMFF L-BFGS | `mmff_use_lbfgs=True` | 5x less memory | Molecules >50 atoms |
| MMFF BFGS (default) | `mmff_use_lbfgs=False` | 2x faster for small mols | Molecules <50 atoms |
| ETKDG variant | `variant="ETKDGv3"` | 7 variants supported | Choose per use case |

### MMFF94 Force Field Variants

| Variant | Flag | Torsion barriers | Best for |
|---------|------|-----------------|----------|
| MMFF94 (default) | `mmff_variant="MMFF94"` | Standard | General molecules |
| MMFF94s | `mmff_variant="MMFF94s"` | Softer for conjugated systems | Aromatic, planar, conjugated |

## Usage

### 3D Conformer Generation

```python
from mlxmolkit.conformer_pipeline_v2 import generate_conformers_nk

# Basic: 10 conformers per molecule, ETKDGv2
result = generate_conformers_nk(
    smiles_list=["c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"],
    n_confs_per_mol=10,
)
for mol in result.molecules:
    print(f"{mol.n_atoms} atoms, {len(mol.positions_3d)} conformers")

# Full pipeline with MMFF94
result = generate_conformers_nk(
    smiles_list=["c1ccccc1", "CC(=O)O"],
    n_confs_per_mol=50,
    variant="ETKDGv3",
    run_mmff=True,
    mmff_use_lbfgs=False,       # BFGS (default, fast for <50 atoms)
    max_confs_per_batch=500,    # divide-and-conquer batch size
)

# Pure distance geometry (no torsion refinement)
result = generate_conformers_nk(
    smiles_list=["c1ccccc1"],
    n_confs_per_mol=100,
    variant="DG",
)

# Large molecules: use L-BFGS for MMFF
result = generate_conformers_nk(
    smiles_list=[large_smiles],
    n_confs_per_mol=20,
    run_mmff=True,
    mmff_use_lbfgs=True,        # L-BFGS for >50 atoms
)
```

### Molecular Clustering

```python
from mlxmolkit import butina_tanimoto_mlx
import mlx.core as mx

# Automatic: fused kernel for N<=100k, blockwise for N>100k
result = butina_tanimoto_mlx(mx.array(fp_bytes), cutoff=0.4)
print(f"{len(result.clusters)} clusters")
```

### Low-level API

```python
from mlxmolkit import (
    fp_uint8_to_uint32,
    fused_neighbor_list_metal,
    tanimoto_neighbors_blockwise,
    butina_from_neighbor_list_csr,
)

fp_u32 = fp_uint8_to_uint32(mx.array(fp_bytes))

# Small N: fused single-dispatch
offsets, indices = fused_neighbor_list_metal(fp_u32, cutoff=0.4)

# Large N (150k+): divide-and-conquer blockwise
offsets, indices = tanimoto_neighbors_blockwise(fp_u32, cutoff=0.4)

result = butina_from_neighbor_list_csr(offsets, indices, N, cutoff=0.4)
```

## Tests

```bash
pip install -e .
pytest tests/ -v
```

## References

- [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) — NVIDIA's CUDA implementation (Apache 2.0)
- [shivampatel10/mlxmolkit](https://github.com/shivampatel10/mlxmolkit) — TPM threadgroup kernels and MMFF Metal implementation
- [RDKit blog: Butina clustering with nvMolKit](https://greglandrum.github.io/rdkit-blog/posts/2026-02-28-nvmolkit-clustering.html)
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework with Metal kernel support
- [MMFF94](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P) — Halgren, J. Comput. Chem. 1996
- [Butina, D. (1999)](https://doi.org/10.1021/ci9803381) — Performance of Kier-Hall and molecular connectivity indices
