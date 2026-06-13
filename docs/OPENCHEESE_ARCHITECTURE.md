# openCHEESE architecture

`openCHEESE` is the standalone project name for the open implementation of
3D shape/electrostatic similarity, conformer-ensemble teachers, and learned
projection embeddings.

`mlxmolkit` remains the Apple Silicon chemistry backend: conformer generation,
MMFF cleanup, AM1-BCC/QM charge baselines, Connolly grids, and MLX/Metal tensor
kernels. New user-facing CHEESE-like code should import from `opencheese`, not
from `mlxmolkit.cheese`.

## Boundary

The split should look like this:

| Layer | Belongs in openCHEESE | Belongs in mlxmolkit |
|---|---|---|
| Public descriptor API | yes | compatibility re-export only |
| Gaussian shape overlap | yes | MLX backend implementation until split |
| ESP-Sim analytic electrostatics | yes | MLX backend implementation until split |
| ROCS/Roshambo-style overlay | yes | backend kernels can stay optional |
| Conformer-ensemble teachers | yes | conformer generation provider |
| Graph/3D projection model | yes | MLX backend implementation until split |
| Training CLI for projection embeddings | yes | local scripts can remain during transition |
| RDKit molecule preparation | optional adapter | core dependency for mlxmolkit workflows |
| MLX/Metal kernels | backend package | yes |
| AM1-BCC/QM labels | data provider | yes |
| Connolly SES mesh generation | optional shape utility | yes |

The current repository contains the first extraction step:

```python
from opencheese import cheese_batch, cheese_similarity_matrix_mlx
from opencheese import CheeseGraphTransformer, cheese_embedding_batch
```

These imports currently delegate to `mlxmolkit` internals. That is deliberate:
it gives notebooks, scripts, and downstream users the future package namespace
now, while preserving the stable code path for the active fine-tuning work.

## Naming

Use these names in docs and public APIs:

- `openCHEESE`: project/product name
- `opencheese`: Python package name
- `openCHEESE descriptor`: exact shape/electrostatic scorer
- `openCHEESE teacher`: pairwise matrix generated from exact descriptors
- `openCHEESE projection`: learned embedding model trained to reproduce a
  teacher matrix
- `openCHEESE-MLX backend`: the current MLX/Metal implementation

Avoid using `CHEESE` alone for our code except when referring to the original
paper or the general method family.

## Current API

Exact descriptors:

```python
from opencheese import cheese_batch, cheese_similarity_matrix_mlx

batch = cheese_batch(atom_numbers, coords, charges)
scores = cheese_similarity_matrix_mlx(
    batch,
    shape_metric="carbo",
    electrostatic_metric="carbo",
)
```

Rigid overlay:

```python
from opencheese import CheeseAlignmentConfig, align_cheese_pair

result = align_cheese_pair(
    probe_atoms,
    probe_coords,
    reference_atoms,
    reference_coords,
    probe_charges=probe_charges,
    reference_charges=reference_charges,
    config=CheeseAlignmentConfig(start_mode="roshambo"),
)
```

Projection model:

```python
from opencheese import CheeseEmbeddingConfig, CheeseGraphTransformer

model = CheeseGraphTransformer(CheeseEmbeddingConfig(hidden_dim=128, n_layers=4))
```

Fine-tuning from an existing projection checkpoint:

```bash
python tools/train_cheese_projection.py \
  --teacher outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz \
  --teacher-channel shape \
  --loss-mode euclidean_dissimilarity \
  --init-weights outputs/cheese_projection/embedding_q_resp_shape_carbo_h128_l4_e30_steps64/best.safetensors \
  --out-dir outputs/cheese_projection/finetune_q_resp_shape_k10_bestpair_principal_h128_l4_e20_steps64 \
  --epochs 20 \
  --steps-per-epoch 64 \
  --batch-size 64
```

## Teacher levels

openCHEESE should expose several teacher-generation levels explicitly:

| Level | Meaning | Use |
|---|---|---|
| `canonical-single` | one conformer per molecule, centered/principal frame | fastest bootstrap |
| `ensemble-principal` | k conformers per molecule, best conformer pair, principal-frame canonicalization | current fine-tuning target |
| `ensemble-overlay-hardpairs` | exact overlay only for hard or high-value pairs | active learning refinement |
| `ensemble-overlay-full` | all conformer pairs with full rigid overlay | paper-faithful but expensive |

The current fine-tune target is `ensemble-principal`: 500 molecules, up to
10 conformers each, best conformer-pair score, shape/electrostatic Carbo.

Current local run:

| Artifact | Value |
|---|---:|
| Teacher file | `outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz` |
| Molecules | 500 |
| Max conformers per molecule | 10 |
| Teacher build time | 170.85 s |
| Shape score mean | 0.661863 |
| Electrostatic score mean | 0.065714 |
| Combined score mean | 0.587268 |
| Shape fine-tune best valid MSE | 0.003918 at epoch 2 |
| Shape final valid correlation | 0.552893 |
| Electrostatic fine-tune best valid MSE | 0.023319 at epoch 9 |
| Electrostatic final valid correlation | 0.976432 |

The teacher artifact above was generated before the metadata-format rename, so
its stored `format` is still `mlxmolkit.cheese_ensemble_pairwise_teacher`.
Newly generated teachers use `opencheese.ensemble_pairwise_teacher`.

## Improved shape training recipe

Shape needs a different training recipe than electrostatics. In the current
500-molecule teacher, the shape channel has a narrow score distribution, so
plain MSE can look good while top-k retrieval remains mediocre. The improved
recipe therefore uses:

- `anchor_topk` batches: every training step contains true shape-neighbors,
  hard negatives, and random negatives for each anchor.
- `hybrid_shape` loss: calibrated distance MSE + pairwise ranking loss +
  soft-neighborhood loss + optional InfoNCE/supervised contrastive positives.
- scaled target distances: `distance_scale * d(similarity)^distance_power`
  to avoid packing all shape distances into a tiny part of the unit sphere.
  `d` can be `1 - similarity` or `-log(similarity)`.
- teacher preprocessing: raw scores, per-anchor z-score scores, or
  size-residual scores after subtracting a fitted molecular-size baseline.
- MuonV2W optimizer: MuonV2/Polar Express for transformer hidden matrices and
  AdamW for embeddings, layer norms, biases, decoders, and small projections.
- retrieval metrics: Spearman, Recall@5/10, and NDCG@5/10, not MSE alone.
- `--no-charges` for pure shape-only models when electrostatic charge features
  may leak target information or add noise.

Raw-teacher fine-tune with ranking/InfoNCE, apples-to-apples against the
previous shape model:

```bash
python tools/train_cheese_projection.py \
  --teacher outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz \
  --teacher-channel shape \
  --teacher-transform raw \
  --loss-mode hybrid_shape \
  --distance-transform one_minus_similarity \
  --init-weights outputs/cheese_projection/embedding_q_resp_shape_carbo_h128_l4_e30_steps64/best.safetensors \
  --out-dir outputs/cheese_projection/finetune_shape_raw_hybrid_infonce_muonv2_h128_l4_e8_steps64 \
  --epochs 8 \
  --steps-per-epoch 64 \
  --batch-size 64 \
  --optimizer muonv2w \
  --lr 5e-5 \
  --muon-lr 4e-4 \
  --rank-weight 0.25 \
  --soft-neighborhood-weight 0.05 \
  --contrastive-weight 0.10 \
  --contrastive-positive-threshold 0.75 \
  --contrastive-positive-top-k 8 \
  --distance-scale 2.0 \
  --top-sim-weight 2.0 \
  --sampler anchor_topk \
  --anchor-rows 32 \
  --positive-k 16 \
  --hard-negative-k 128
```

Best local validation row:

| Run | Valid MSE | Pearson | Spearman | Recall@5 | Recall@10 | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| previous hybrid MuonV2 | 0.014743 | 0.7029 | 0.5805 | 0.348 | 0.472 | 0.9373 |
| raw + ranking + InfoNCE | 0.014540 | 0.7016 | 0.5829 | 0.360 | 0.476 | 0.9403 |
| anchor-zscore + neglog + InfoNCE | 0.260630 | 0.5481 | 0.5696 | 0.332 | 0.472 | 0.8185 |

The anchor-zscore run optimizes a transformed target, so its MSE is not directly
comparable to the raw-score runs. It is useful as a preprocessing experiment,
not yet the default.

Pure-shape smoke path:

```bash
python tools/train_cheese_projection.py \
  --teacher outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz \
  --teacher-channel shape \
  --teacher-transform size_residual \
  --loss-mode hybrid_shape \
  --distance-transform neglog_similarity \
  --optimizer muonv2w \
  --contrastive-weight 0.10 \
  --sampler anchor_topk \
  --no-charges \
  --out-dir outputs/cheese_projection/shape_size_residual_neglog_infonce_nocharges
```

Use this from scratch. A checkpoint trained with charge features has a
different module tree because `charge_projection` is absent in `--no-charges`
models.

## Better teacher refinement

The fast ensemble teacher is good for dense coverage, but it is still a
principal-frame approximation. For important shape pairs, refine the teacher by
running full conformer-pair overlay and replacing only those matrix entries:

```bash
python tools/refine_opencheese_teacher_hardpairs.py \
  --teacher outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz \
  --ensembles outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz \
  --select-channel shape \
  --top-k-per-row 5 \
  --min-score 0.70 \
  --max-pairs 100 \
  --out outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_refined_toppairs.npz
```

This creates a hybrid teacher: cheap labels everywhere and expensive
Roshambo-style labels where retrieval cares most.

For a paper-style conformer ensemble, regenerate with `--n-conformers 20` before
the teacher/refinement pass:

```bash
python tools/prepare_cheese_conformer_ensembles.py \
  --data data/espaloma_charge_zenodo_17308526/recalculated_charges/test_random1000_both_symmetrized_partial_bcc_fill/cheese_charge_training_am1bcc_resp.npz \
  --target q_resp \
  --n-conformers 20 \
  --out outputs/cheese_projection/cheese_ensembles_1000_k20_q_resp.npz
```

The ensemble builder now overgenerates candidates and performs post-MMFF
diversity selection before writing the cache. This matters: k=20 is useful only
when those conformers are genuinely different shape/energy basins, not twenty
copies of the same optimized minimum. Defaults are:

- generate `ceil(k * 4)` candidates, capped by `--max-candidates 200`
- sort by convergence and MMFF/UFF energy
- keep conformers within `--selection-energy-window 15.0` kcal/mol
- greedily select conformers separated by at least
  `--selection-rms-thresh 0.75` A heavy-atom aligned RMSD
- write fewer than k conformers if the molecule does not have k distinct
  low-energy basins

Use `--fill-to-n-conformers` only when a fixed tensor count matters more than
diversity. The cache stores per-molecule diagnostics:
`selection_min_pair_rmsd`, `selection_mean_pair_rmsd`,
`selection_max_relative_energy`, `selection_n_candidates`, and
`selection_n_energy_eligible`.

For a stricter k=20 teacher:

```bash
python tools/prepare_cheese_conformer_ensembles.py \
  --data data/espaloma_charge_zenodo_17308526/recalculated_charges/test_random1000_both_symmetrized_partial_bcc_fill/cheese_charge_training_am1bcc_resp.npz \
  --target q_resp \
  --n-conformers 20 \
  --candidate-multiplier 5 \
  --max-candidates 200 \
  --selection-rms-thresh 0.75 \
  --selection-energy-window 15 \
  --out outputs/cheese_projection/cheese_ensembles_1000_k20_diverse_q_resp.npz
```

## ESP-grid preprocessing

For electrostatic teachers, the next data upgrade is to cache surface ESP
values instead of relying only on analytic charge-overlap labels. The helper
below builds Connolly/MK-style grids for each selected conformer and
regenerates point-charge ESP values with the MLX/Metal grid kernel:

```bash
python tools/prepare_opencheese_esp_grid_cache.py \
  --ensembles outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz \
  --limit 500 \
  --max-conformers 10 \
  --point-density 0.35 \
  --out outputs/cheese_projection/opencheese_esp_grids_500_k10_q_resp.npz
```

The grid geometry is still CPU-side Connolly/MK point placement; potential
evaluation is GPU-side. This cache is the right substrate for a stricter
field/surface electrostatic teacher.

## Related ML Ideas

FuncMol is useful inspiration for future field-level priors, but it is not a
drop-in conformer initializer for a fixed input graph: it is an unconditional
all-atom molecular-field generator. Using it directly would require mapping a
generated field back onto the exact input atom order, bond graph, protonation,
and stereochemistry.

GraphMVP-style 2D/3D SSL is more directly useful for openCHEESE training. The
next model-training upgrade should add a 2D-only encoder branch and pretrain it
against the 3D branch with:

- contrastive 2D graph vs 3D conformer positives for the same molecule
- representation reconstruction, where 2D latents reconstruct stop-gradient 3D
  latents and vice versa
- conformer-view positives from the diverse k-conformer ensemble

This would make the projection model less brittle to any single conformer and
would let the 2D graph branch inherit 3D shape/electrostatic information.

## Independence plan

Step 1 is complete in this repository: create `opencheese` and switch training
and teacher scripts to import from it.

Step 2 should move implementation modules behind backend names:

```text
opencheese/
  descriptors.py
  embedding.py
  alignment.py
  teachers.py
  backends/
    mlx.py
    numpy.py
    rdkit.py
```

Step 3 should make `mlxmolkit` depend on `opencheese` instead of owning the
descriptor code. At that point:

- `opencheese` owns descriptors, teachers, model architecture, and training.
- `mlxmolkit` provides `OpenCheeseMlxBackend` plus conformer/charge providers.
- Existing `mlxmolkit.cheese` imports remain as deprecation-compatible aliases.

Step 4 should publish a separate package:

```toml
[project]
name = "opencheese"
```

The standalone package can keep optional extras:

```toml
opencheese[mlx]
opencheese[rdkit]
opencheese[train]
```

## Why not move everything immediately?

The code is still under active numerical validation:

- conformer parity with RDKit
- MLX paired shape/electrostatic scoring
- conformer-aware teacher generation
- projection fine-tuning
- AM1-BCC/RESP charge-label baselines

A physical move of large modules would mix extraction noise with scientific
changes. The safe sequence is:

1. create the public namespace
2. switch scripts/tests to the namespace
3. document the boundary
4. finish current fine-tuning and validation
5. move implementation modules in a focused extraction commit

That keeps the package direction clear without hiding real math changes inside
a rename.
