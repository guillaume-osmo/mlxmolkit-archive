# mlxmolkit and openCHEESE API logic

This document is the engineering map for `mlxmolkit` and the emerging
standalone `opencheese` package. It explains the public API shape, the internal
data contracts, and the reason each pipeline is wired the way it is. The goal
is to make the packages easier to extend without guessing which tensors live on
MLX, which steps still depend on RDKit, and where exact reference parity is
expected.

`openCHEESE` is the product/library name for the shape/electrostatic
similarity, conformer-ensemble teacher, and learned projection layers.
`mlxmolkit` is one backend/provider: Apple Silicon conformers, MMFF, QM charge
labels, Connolly grids, and MLX/Metal tensor kernels. New user code should use
`opencheese` imports for the CHEESE-like layers; existing `mlxmolkit.cheese`
imports remain compatibility aliases.

## Design rules

`mlxmolkit` is not a single descriptor function. It is a set of GPU-friendly
chemistry sessions:

- clustering: fingerprints -> similarity -> neighbor list -> Butina
- conformers: RDKit molecular graph -> DG/ETK constraints -> MLX/Metal
  minimization -> optional MMFF cleanup
- QM and charges: NDDO methods, AM1-BCC/QEq baselines, ESP/RESP label fitting,
  and learned charge prediction
- openCHEESE-style 3D: exact shape/electrostatic similarity first, learned
  projection embeddings second

The most important rule is that each layer keeps a clear contract:

- RDKit owns chemistry perception, atom order, hydrogens, bond orders, and
  stereochemistry labels.
- MLX/Metal owns dense numerical work once the graph has been converted into
  padded tensors or shared constraint buffers.
- Output files record enough metadata to reproduce the exact target:
  conformer count, charge target, canonicalization/alignment mode, metric, and
  model weights used for fine-tuning.
- Fast approximations must be named as approximations. For example, a
  principal-axis ensemble teacher is useful for 3D impact, but it is not the
  full Roshambo/ROCS all-pair rigid alignment teacher.

## Public module map

The root package exports the main workflow objects from `mlxmolkit/__init__.py`.
Use direct module imports for development, but keep user-facing scripts and
examples close to these public names.

| Area | Public entry points | Main implementation |
|---|---|---|
| Morgan/Tanimoto/Butina | `fp_uint8_to_uint32`, `tanimoto_matrix_metal_u32`, `tanimoto_neighbors_blockwise`, `butina_tanimoto_mlx` | `tanimoto_*`, `butina.py`, `butina_metal.py` |
| Conformers | `generate_conformers_nk`, `ConformerResult`, `PipelineResult` | `conformer_pipeline_v2.py`, `conformer_metal.py`, `shared_batch.py` |
| MMFF | `run_mmff=True` in conformer generation, optimizer modules | `mmff_*`, Metal kernels |
| QM / NDDO | `mlxmolkit.rm1.nddo_energy`, `nddo_energy_batch`, `pm6_d3h4_correction` | `mlxmolkit/rm1/*` |
| ESP/RESP labels | `connolly_surface_grid`, `fit_esp_charges_mlx`, `fit_resp_charges_mlx`, `pm6_esp_resp_charge_labels` | `esp_resp.py` |
| AM1-BCC | `am1_bcc_charges_from_rdkit_mol(s)`, `symmetrize_charges_by_topology` | `am1bcc.py` |
| Learned charge model | `GeometricChargePredictor`, `predict_partial_charges_*` | `charge_model.py`, `charge_training_dataset.py` |
| openCHEESE exact scoring | `opencheese.cheese_batch`, `opencheese.cheese_similarity_matrix_mlx`, `opencheese.cheese_similarity_pairs_mlx`, `opencheese.align_cheese_pair` | `opencheese`, backed by `mlxmolkit.cheese` during transition |
| openCHEESE projection | `opencheese.CheeseGraphTransformer`, `opencheese.cheese_embedding_batch` | `opencheese`, backed by `mlxmolkit.cheese_embedding` during transition |
| Connolly SES meshes | `connolly_ses_surface_mlx`, `connolly_ses_surface_from_atoms_mlx` | `connolly.py`, `tools/connolly_surface.py` |

## Conformer session

The intended user-level call is:

```python
from mlxmolkit import generate_conformers_nk

result = generate_conformers_nk(
    smiles_list=["c1ccccc1", "CC(=O)O"],
    n_confs_per_mol=10,
    variant="ETKDGv3",
    run_mmff=True,
    seed=20260613,
)
```

The conformer path is:

1. SMILES -> RDKit `MolFromSmiles`.
2. `Chem.AddHs` creates the explicit-H graph that all downstream tensors use.
3. RDKit bounds and stereochemistry perception provide the reference graph
   chemistry.
4. Distance-geometry terms are packed once per molecule into a
   `SharedConstraintBatch`.
5. One Metal threadgroup optimizes one conformer, while molecule-level
   constraints are shared across all conformers for that molecule.
6. The 4D DG coordinates are collapsed to 3D.
7. ET/K torsion terms are applied according to the requested variant.
8. Stereo and geometry checks reject bad conformers.
9. Optional MMFF94 cleanup uses the same hydrogenated graph and MMFF variant as
   the RDKit reference path.

Variant names are not cosmetic. They select different constraint families:

| Variant | Meaning |
|---|---|
| `DG` | Distance geometry only; no knowledge or torsion terms. |
| `KDG` | Adds knowledge-based distance preferences. |
| `ETDG` | Adds experimental torsion terms. |
| `ETDGv2` | ETDG with the v2 torsion parameterization. |
| `ETKDG` | Experimental torsions plus knowledge terms. |
| `ETKDGv2` | ETKDG with v2 torsion updates. |
| `ETKDGv3` | ETKDG with v3 macrocycle and small-ring handling. |
| `srETKDGv3` | Small-ring ETKDGv3 settings. |

When comparing to RDKit, compare apples to apples:

- same input SMILES and explicit-H graph
- same ETKDG variant
- same random seed policy
- same number of conformers
- same MMFF/UFF cleanup setting
- same stereochemistry validation
- same heavy-atom alignment and RMSD logic

The current RDKit-vs-mlxmolkit quality script is:

```bash
python tools/compare_rdkit_mlx_conformers.py \
  --limit 20 \
  --n-conformers 10 \
  --out outputs/cheese_projection/rdkit_vs_mlx_conformers_20_k10_batched_scoring_after_fixes.csv
```

That script uses batched Horn quaternion alignment for same-molecule
RDKit-vs-MLX conformer comparison, then evaluates shape Carbo and electrostatic
Carbo in MLX. Horn alignment is correct there because atom order and atom count
match. It is not a general cross-molecule shape overlay.

## QM and charge session

The current low-level API is function-oriented:

```python
from mlxmolkit.rm1 import nddo_energy, pm6_d3h4_correction

result = nddo_energy(atoms, coords, method="PM6_D", molecular_charge=0)
correction = pm6_d3h4_correction(atoms, coords)
```

The desired higher-level API is a `QMSession` facade:

```python
from mlxmolkit.qm import QMSession

qm = QMSession(method="PM6_D3H4", backend="mlx", charge_model="resp")
energy = qm.energy(atoms, coords, molecular_charge=0)
charges = qm.charges(atoms, coords, molecular_charge=0)
```

Until that facade lands, keep these invariants:

- Always pass `molecular_charge` explicitly for charged systems.
- `PM6_D` means the full d-orbital PM6 path for P, S, Cl, Br, and I.
- `PM6-D3H4` is a post-SCF correction on top of the PM6_D electronic result.
- OpenMOPAC parity tests should compare method semantics, not just names.
- PYSEQM parity tests guard primitive tensor math and parameter loading.

For ESP/RESP labels, the backend-neutral contract is:

```python
from mlxmolkit import fit_esp_charges_mlx, fit_resp_charges_mlx

esp = fit_esp_charges_mlx(atom_coords, grid_coords, esp_values, total_charge=0)
resp = fit_resp_charges_mlx(atom_coords, grid_coords, esp_values, total_charge=0, atoms=atoms)
```

Any external quantum backend can feed this contract if it produces grid
coordinates and electrostatic potential values. The RESP restraint is the
usual hyperbolic penalty:

```text
a * (sqrt(q_i^2 + b^2) - b)
```

The fit preserves the exact total molecular charge through a constrained
least-squares solve.

## AM1-BCC baseline

AM1-BCC is the practical open baseline for fast charge generation:

```python
from mlxmolkit import am1_bcc_charges_from_rdkit_mols

results = am1_bcc_charges_from_rdkit_mols(mols, method="AM1", total_charges=[0, 1])
```

The AM1-BCC logic has three layers:

1. Semi-empirical Mulliken-like seed charges from AM1/RM1/PM3/PM6.
2. Bond charge correction lookup from RDKit bond/atom environments.
3. Symmetry repair by topology classes, which is important for resonance
   groups such as nitro and carboxylate cases.

The AM1-BCC code is intentionally not tied to Amber runtime tools. MOPAC is a
reference/validation backend, not an inference dependency.

## Learned partial-charge model

`GeometricChargePredictor` is the fast inference model:

```python
from mlxmolkit import GeometricChargePredictor, ChargeModelConfig, predict_partial_charges_mlx

model = GeometricChargePredictor(ChargeModelConfig(readout="qeq"))
charges = predict_partial_charges_mlx(model, atoms, coords, bond_matrix, total_charge=0.0)
```

The model input contract is:

- `atomic_numbers`: padded integer tensor `(batch, atoms)`
- `coords`: padded float tensor `(batch, atoms, 3)`
- `bond_matrix`: padded integer tensor `(batch, atoms, atoms)`
- `mask`: padded atom mask `(batch, atoms)`
- `total_charge`: requested molecular charge

The direct readout predicts per-atom charges and projects them back to the
requested total charge. The QEq readout predicts electronegativity/hardness
parameters and solves the constrained charge-equilibration problem.

SMILES inference must generate 3D first:

```text
SMILES -> RDKit graph -> AddHs -> ETKDG/MMFF conformer -> padded MLX tensors -> model
```

This is different from graph-only charge models that ignore conformers.

## openCHEESE exact descriptor layer

The exact layer operates on conformers and charges in a common frame:

```python
from opencheese import cheese_batch, cheese_similarity_matrix_mlx

batch = cheese_batch(atom_numbers, coords, charges)
scores = cheese_similarity_matrix_mlx(
    batch,
    shape_metric="carbo",
    electrostatic_metric="carbo",
)
```

The score object carries:

- `shape`: Gaussian shape Carbo or Tanimoto
- `electrostatic`: ESP-Sim-style electrostatic Carbo or Tanimoto
- `combined`: weighted average after optional electrostatic remapping

Use the matrix scorer when every molecule has one conformer in a known common
frame. Use the paired scorer when you already have aligned pair tensors:

```python
from opencheese import cheese_similarity_pairs_mlx

pair_scores = cheese_similarity_pairs_mlx(probe_pairs, reference_pairs)
```

Use `align_cheese_pair` or `cheese_alignment_matrix` when the molecules are not
already aligned and a true cross-molecule overlay is required. That route is
the Roshambo/ROCS-style layer and is naturally more expensive than scoring a
fixed frame.

Important distinction:

- Same molecule, different toolkit conformers: Horn/Kabsch alignment is valid
  because atom order is shared.
- Different molecules: Horn/Kabsch atom mapping is not valid in general.
  Use shape overlay starts and Gaussian refinement instead.

## openCHEESE projection model

The projection model is a learned surrogate for pairwise openCHEESE comparison. It
does not replace the exact scorer during label generation; it learns an
embedding whose distances reproduce the exact pair matrix.

Inputs:

- atoms
- bond-state matrix with 13 states
- 3D coordinates
- partial charges, usually `q_resp` for the current experiments
- local chiral-volume feature

Model:

- atom embedding
- optional charge projection
- optional chirality projection
- pair features = bond embedding + radial basis encoding of 3D distances
- stacked graph/3D transformer blocks
- pooled molecular embedding
- auxiliary atom-identity and pair-distance reconstruction heads

Metric target:

```text
embedding_distance(i, j) ~= 1 - openCHEESE_similarity(i, j)
```

The projection trainer now supports fine-tuning from an existing checkpoint:

```bash
python tools/train_cheese_projection.py \
  --teacher outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz \
  --teacher-channel shape \
  --loss-mode euclidean_dissimilarity \
  --init-weights outputs/cheese_projection/embedding_q_resp_shape_carbo_h128_l4_e30_steps64/best.safetensors \
  --out-dir outputs/cheese_projection/finetune_q_resp_shape_k10_bestpair_principal_h128_l4 \
  --epochs 20 \
  --steps-per-epoch 64 \
  --batch-size 64
```

Use shape and electrostatic as separate heads/models when following the CHEESE
paper logic. A combined model is useful for retrieval experiments, but it hides
which signal improved.

## Conformer-aware openCHEESE teachers

The fastest bootstrap teacher is one conformer per molecule:

```bash
python tools/compute_cheese_pairwise_teacher.py \
  --data data/espaloma_charge_zenodo_17308526/recalculated_charges/test_random1000_both_symmetrized_partial_bcc_fill/cheese_charge_training_am1bcc_resp.npz \
  --target q_resp \
  --shape-metric carbo \
  --electrostatic-metric carbo \
  --out outputs/cheese_projection/cheese_teacher_1000_q_resp_carbo_canonical.npz
```

The conformer-aware teacher uses the cached `k=10` conformer ensembles:

```bash
python tools/compute_cheese_ensemble_teacher.py \
  --ensembles outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz \
  --canonicalize principal \
  --shape-metric carbo \
  --electrostatic-metric carbo \
  --limit 500 \
  --tile-size 8 \
  --out outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz
```

For each molecule pair, the script scores every conformer pair in the tile and
keeps the best shape/electrostatic/combined score. This introduces real 3D
conformer impact into the teacher. It is still a fast approximation because
the conformers are principal-axis canonicalized rather than fully
Roshambo-aligned.

Recommended fine-tuning sequence:

1. Fine-tune the shape projection from the canonical shape checkpoint using the
   conformer-aware shape channel.
2. Fine-tune the electrostatic projection from the canonical electrostatic
   checkpoint using the conformer-aware electrostatic channel.
3. Optionally fine-tune a combined retrieval embedding after the separate
   channels are stable.
4. Audit top disagreements with `cheese_alignment_matrix` to decide which
   pairs need expensive full-overlay labels.

## Connolly and surface logic

There are two related but separate surface utilities:

- `connolly_surface_grid()` in `esp_resp.py`: point grid for ESP/RESP fitting.
- `connolly_ses_surface_mlx()` in `connolly.py`: solvent-excluded surface mesh.

The mesh path computes the expensive SAS signed-distance field on MLX/Metal,
then applies exact EDT erosion and marching cubes. It is meant for geometry,
visualization, and shape experiments. The grid path is lighter and meant for
charge fitting.

## Verification commands

Focused tests for the recently touched layers:

```bash
pytest -q \
  tests/test_cheese.py \
  tests/test_cheese_embedding.py \
  tests/test_etk_metal_consistency.py \
  tests/test_mmff_minimize_nk.py \
  tests/test_conformer_metal.py
```

Conformer scoring parity:

```bash
python tools/compare_rdkit_mlx_conformers.py \
  --limit 20 \
  --n-conformers 10 \
  --out outputs/cheese_projection/rdkit_vs_mlx_conformers_20_k10_batched_scoring_after_fixes.csv
```

Conformer-aware teacher plus fine-tune:

```bash
python tools/compute_cheese_ensemble_teacher.py \
  --ensembles outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz \
  --canonicalize principal \
  --shape-metric carbo \
  --electrostatic-metric carbo \
  --limit 500 \
  --tile-size 8 \
  --out outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz

python tools/train_cheese_projection.py \
  --teacher outputs/cheese_projection/cheese_teacher_500_q_resp_k10_bestpair_principal_carbo.npz \
  --teacher-channel shape \
  --loss-mode euclidean_dissimilarity \
  --init-weights outputs/cheese_projection/embedding_q_resp_shape_carbo_h128_l4_e30_steps64/best.safetensors \
  --out-dir outputs/cheese_projection/finetune_q_resp_shape_k10_bestpair_principal_h128_l4 \
  --epochs 20 \
  --steps-per-epoch 64 \
  --batch-size 64
```

## Extension checklist

Before adding a new descriptor, model, or kernel, answer these questions in the
code or documentation:

- What is the molecule identity contract: SMILES, RDKit Mol, explicit-H Mol, or
  raw atoms/coords?
- Does the function require conformers? If yes, who generates them?
- Does the function preserve atom order?
- Are hydrogens included or ignored?
- Is the score exact parity, a fast approximation, or a learned surrogate?
- Which tensors are on MLX, and where does CPU preprocessing remain?
- Is the function batch-compatible?
- Does the output include enough metadata to reproduce the calculation?
- Which test proves the math did not silently change?

This checklist matters because most bugs in this package are not syntax bugs.
They are chemistry-contract bugs: wrong hydrogens, mismatched ETKDG variant,
implicit charge state, non-equivalent alignment, or comparing a learned
surrogate to an exact teacher without recording the teacher recipe.
