# mlxmolkit

**GPU-accelerated molecular toolkit for Apple Silicon** — a port of
[nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (CUDA) to Apple Metal
via [MLX](https://github.com/ml-explore/mlx), extended with semi-empirical
quantum chemistry.

## What's in the box

| Area | What it does | Entry point |
|---|---|---|
| **Conformers** | Drop-in for RDKit `EmbedMolecules`: DG (4D) → ETK (3D) → MMFF94, all on Metal. 8 ETKDG variants. N×k parallel | `generate_conformers_nk` |
| **Clustering** | Morgan FP → Tanimoto → Butina, at 150k+ molecules with divide-and-conquer memory | `butina_tanimoto_mlx` |
| **NDDO semi-empirical** | 7 methods (RM1, AM1, PM3, PM6, PM6_SP, PM6_D, AM1\*) + PM6-D3H4 corrections, energies **and** gradients/geometry optimization | `mlxmolkit.nddo` |
| **xTB** | GFN0/1/2 and g-xTB energies, analytical gradients, ANCOPT geometry optimization, ALPB water solvation | `mlxmolkit.xtb` |
| **COSMO / COSMO-RS** | σ-profiles, σ-potentials, activity coefficients, solubility in solvent mixtures | `mlxmolkit.xtb` (σ), `mlxmolkit.cosmo` (ddCOSMO) |
| **Similarity & descriptors** | ERG fingerprints, dense cosine, CHEESE embeddings, Connolly surfaces, dipole atom features | top-level exports |

Measured tables — throughput, memory scaling, quality vs RDKit — are in
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

## Install

```bash
pip install mlxmolkit-rdkit
```

RDKit is needed for molecular input:

```bash
conda install -c conda-forge rdkit
pip install mlxmolkit-rdkit
```

**Requirements:** macOS on Apple Silicon (M1–M4).

## Quick start

```python
from mlxmolkit import generate_conformers_nk, butina_tanimoto_mlx
import mlx.core as mx

# 3D conformers — N molecules x k conformers, one GPU batch
result = generate_conformers_nk(
    smiles_list=["c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"],
    n_confs_per_mol=10,
    run_mmff=True,
)

# Clustering — 150k+ molecules
clusters = butina_tanimoto_mlx(mx.array(fp_bytes), cutoff=0.4)

# Semi-empirical SCF
from mlxmolkit.nddo import nddo_energy
e = nddo_energy(atoms=[8, 1, 1], coords=[[0,0,0], [0.97,0,0.93], [-0.97,0,0.93]],
                method="PM6_D")

# xTB
from mlxmolkit.xtb import gfn2_energy, ancopt
e = gfn2_energy(atoms, coords)
```

---

## Read this before you benchmark it

### MMFF on Metal is *not* faster than RDKit at matched accuracy

This is the number the throughput tables do not show, and it decides whether you
should use this path at all. 50 molecules × 50 conformers, M3 Max, quiet box,
fresh `Chem.Mol` copies per method:

| Method | ms/conf | vs RDKit 14T | mean abs ΔE | within 0.1 kcal/mol |
|---|---:|---:|---:|---:|
| RDKit 1 thread | 1.59 | 0.13× | — | — |
| **RDKit 14 threads** | **0.20** | 1.00× | 0.0000 | — |
| FUSED 200 it | **0.16** | **1.22×** | 1.742 | 800/2500 |
| FUSED 2000 it | 1.24 | 0.16× | 0.073 | 2326/2500 |
| MULTI 1000 it | 1.36 | 0.15× | **0.015** | **2470/2500** |

**The kernel is fast; the optimizer converges slowly.** At 200 iterations the
fused path genuinely beats 14-thread RDKit — but it needs ~2000 iterations to
match RDKit's energies, and cost is linear in iterations. So the throughput win
is spent on iteration count, and **at matched accuracy Metal is ~6× behind
RDKit-14T and roughly at parity with single-threaded RDKit.**

The lever is convergence per iteration — line search, preconditioning,
curvature — not kernel speed. Getting there at 200–500 iterations would beat
RDKit-14T outright; it is already within 1.22× at 200.

### Three ways to measure this wrong

1. **`mmff_optimize_batch(mol, conf_ids)` batches the conformers of *one*
   molecule** — batch width ~10, which measures kernel dispatch overhead rather
   than throughput. The real entry points are
   `mmff_optimize_metal_multi_mol` / `..._fused_multi_mol` in
   `mmff_metal_optimizer.py`, exercised by `bench_multi_mol.py`.
2. **`MMFFOptimizeMoleculeConfs` optimizes in place.** Reusing molecules across
   methods measures "re-converge something already at a minimum", which inflates
   whichever method runs second *and* its apparent accuracy. Always take a fresh
   `Chem.Mol(m)` per method.
3. **Benchmark on a quiet box.** A loaded machine flatters the GPU path against
   a 14-thread CPU baseline.

### Other choices that matter

| Choice | Use | Because |
|---|---|---|
| BFGS vs L-BFGS | **BFGS** below ~150 atoms (with H) | Better curvature ⇒ fewer iterations. Memory is O(n²) in the dense Hessian, so L-BFGS only wins when that exceeds ~1 MB/conformer. The pipeline auto-switches at 150+ (`mmff_use_lbfgs=None`) |
| Explicit hydrogens | **Always** — the pipeline calls `AddHs` for you | DG constraints are more complete and the bond/angle/torsion terms are fully defined; convergence is materially better |
| Batch size | Largest that fits (auto-sized by default) | Fewer kernel launches: 1,610 conf/s at batch 100 → 4,442 at batch 1000+ |
| Iterations | Auto (`dg_max_iters=0`) | Scales as `base + scale · max(n_atoms, √n_constraints)`; small molecules exit early via in-kernel TOLX/gradient checks |
| MMFF variant | `MMFF94s` for conjugated/aromatic | Softer torsion barriers |

---

## Semi-empirical NDDO

Seven methods plus PM6-D3H4 post-SCF corrections, **bit-exact to
[PYSEQM](https://github.com/lanl/PYSEQM) for PM6_D**, with no PYSEQM or PyTorch
runtime dependency.

| Method | Element coverage | HoF (H₂O, kcal/mol) |
|---|---|---:|
| RM1 | H, C, N, O, F, P, S, Cl, Br, I | −57.81 |
| AM1 | H, C, N, O | −59.22 |
| PM3 | H, C, N, O, F, P, S, Cl, Br, I | −53.19 |
| PM6 / PM6_SP | H, C, N, O, F, P, S, Cl, Br, I (sp-only) | −54.19 |
| PM6_D | + d-orbitals on P, S, Cl, Br | bit-exact vs PYSEQM |
| AM1\* / RM1\* | H, C, N, O (\*-variants) | −53.71 / −54.47 |
| PM6-D3H4 | D3 dispersion + H4 H-bond + HH repulsion | post-SCF correction |

- **Full d-orbital support** — P, S, Cl (qn=3) and Br (qn=4) via the 22-integral
  local frame, rotated to the molecular frame with Wigner D-matrices. Covers YH,
  YX and YY pair types.
- **Bit-exactness** — the per-pair W tensor matches PYSEQM to 2.66e-15 (machine
  epsilon); 27/27 SCF charge tests pass against frozen PYSEQM/MOPAC references,
  guarded by 23 frozen-reference regression tests.
- **DIIS + adaptive damping** — converges on hard cases (CCl₄, SF₆, DMSO) where
  plain mixing freezes in the wrong basin.

```python
from mlxmolkit.nddo import (
    nddo_energy, nddo_energy_batch,            # SCF
    nddo_gradient, nddo_optimize, nddo_optimize_batch,   # forces + geometry
    pm6_d3h4_correction, d3_energy, h4_energy, hh_repulsion,
    METHOD_PARAMS, get_params, ElementParams,
)
```

Bit-exact primitives vendored from PYSEQM live in
`mlxmolkit.nddo._pyseqm_port` (`diatom_overlap_matrixD`,
`two_elec_two_center_int`, `qn_int`, `qnD_int`).

## xTB and COSMO-RS

`mlxmolkit.xtb` exposes 51 public functions. The main groups:

| Group | Functions |
|---|---|
| Energies | `gfn0_energy`, `gfn1_energy`, `gfn2_energy`, `gxtb_energy` |
| Gradients | `gfn0_gradient`, `gfn1_gradient(_analytical)`, `gfn2_gradient(_analytical)`, `gxtb_energy_gradient` |
| Geometry optimization | `ancopt`, `gxtb_optimize_geometry`, `gfn2_alpb_water_optimize(_batch)` |
| Water solvation (ALPB) | `gfn2_energy_alpb_water`, `alpb_water_correction(_native)`, `gfn2_alpb_water_singlepoint` |
| σ-profiles | `sigma_profile_histogram`, `sigma_profile_klamt`, `klamt_average_sigmas`, `CosmoSegments`, `parse_xtb_cosmo`, `write_cosmo_file` |
| σ-potentials | `sigma_potential`, `sigma_potential_from_arrays`, `sigma_potential_ensemble`, `cosmors_sigma_potential_auto` |
| COSMO-RS | `make_cosmors`, `activity_coefficients`, `ActivityResult`, `OPENCOSMORS25A_PARAMS` |
| Solubility | `ideal_solid_solubility_ln_x`, `solubility_in_solvent_mixture`, `solute_solvent_mixture_x`, `estimate_delta_h_fusion_walden` |
| Tiered / hybrid | `hybrid_gxtb_gfn2_cosmo(_from_smiles)`, `tiered_gxtb_orca_cosmors(_from_smiles)`, `tiered_multiconformer_gxtb_orca`, `orca_cosmors_singlepoint` |

`mlxmolkit.cosmo` holds the ddCOSMO implementation itself (cavity construction,
Lebedev grids, spherical harmonics, σ-profiles). It exports no `__all__` — treat
it as internal to the σ-profile path rather than a stable API.

## Conformers and clustering

```python
from mlxmolkit.conformer_pipeline_v2 import generate_conformers_nk

result = generate_conformers_nk(
    smiles_list=[...],
    n_confs_per_mol=50,
    variant="ETKDGv3",          # DG, KDG, ETDG, ETDGv2, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3
    run_mmff=True,
    mmff_variant="MMFF94",      # or MMFF94s
    mmff_use_lbfgs=False,       # None = auto-switch at 150+ atoms
    max_confs_per_batch=500,    # divide-and-conquer; auto-sized by default
)
```

Clustering, low-level:

```python
from mlxmolkit import (
    fp_uint8_to_uint32, fused_neighbor_list_metal,
    tanimoto_neighbors_blockwise, butina_from_neighbor_list_csr,
)

fp_u32 = fp_uint8_to_uint32(mx.array(fp_bytes))
offsets, indices = fused_neighbor_list_metal(fp_u32, cutoff=0.4)   # N <= 100k
offsets, indices = tanimoto_neighbors_blockwise(fp_u32, cutoff=0.4)  # 150k+
result = butina_from_neighbor_list_csr(offsets, indices, N, cutoff=0.4)
```

Example scripts:

```bash
python examples/conf3d_example.py --n-mols 1000 --n-confs 10 --mmff
python examples/conf3d_example.py --smiles "c1ccccc1" "CC(=O)O" --n-confs 50 --mmff
```

## Architecture

### Conformer generation (N × k parallel)

```
SMILES x N
    |
[RDKit CPU] Extract params ONCE per molecule
    |
[Pack] SharedConstraintBatch (conf_to_mol indirection, ~50% memory saved)
    |
+-- Stage 1: DG minimize (4D, Metal TPM=32) ----------+
|   One threadgroup per conformer                      |
|   L-BFGS in-kernel, GPU-parallel line search         |
+-------------------------------------------------------+
    |
[Extract 3D] Drop 4th coordinate
    |
+-- Stage 2: ETK minimize (3D, Metal TPM=32) ---------+
|   CSD torsion + improper + 1-2/1-3/1-4 distance      |
+-------------------------------------------------------+
    |
+-- Stage 3: MMFF94 optimize (Metal, in-kernel) ------+
|   7 terms: bond, angle, stretch-bend, OOP,           |
|   torsion, vdW, electrostatic. BFGS or L-BFGS        |
+-------------------------------------------------------+
    |
Optimized 3D conformers
```

Constraints are shared across the conformers of a molecule via `conf_to_mol`
indirection, which is where the ~50% memory saving comes from.

### Clustering (divide-and-conquer for 150k+)

```
Morgan FP (RDKit CPU) -> uint8 -> uint32 packing
        |
+-- N <= 100k: fused Metal kernel, single dispatch, no NxN matrix
+-- N >  100k: blockwise D&C, tile both dims (auto-sized),
|              mx.eval() between tiles to free GPU
        |
Butina greedy (CPU, numpy CSR) -> clusters
```

## Tests

```bash
pip install -e .
pytest tests/ -v
```

905 tests across 85 files. Five modules under `tests/test_opencheese_*` and
`test_cheese_conformer_ensembles.py` currently fail to collect — they import
`tools.*` helpers that are not importable as a package.

## References

- [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) — NVIDIA's CUDA implementation (Apache 2.0)
- [shivampatel10/mlxmolkit](https://github.com/shivampatel10/mlxmolkit) — TPM threadgroup kernels and the MMFF Metal implementation
- [PYSEQM](https://github.com/lanl/PYSEQM) — LANL semi-empirical reference (BSD-3); the NumPy port in `mlxmolkit/nddo/_pyseqm_port/` is a mechanical torch→numpy translation of selected modules
- [RDKit blog: Butina clustering with nvMolKit](https://greglandrum.github.io/rdkit-blog/posts/2026-02-28-nvmolkit-clustering.html)
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework with Metal kernel support
- [MMFF94](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P) — Halgren, *J. Comput. Chem.* 1996
- [Butina, D. (1999)](https://doi.org/10.1021/ci9803381)

## Acknowledgements

Portions of the PM6_D / SCF / TETCI development were assisted by Claude
(Anthropic). All commits are authored by the maintainer; Claude was used as a
research and refactoring aide.
