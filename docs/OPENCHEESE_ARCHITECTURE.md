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
