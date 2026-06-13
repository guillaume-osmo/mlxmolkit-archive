# openCHEESE Notes: LG-Mol, Force2Geo, and Surface ESP Clouds

## Why LG-Mol Matters

LG-Mol is the closest public code path to the surface/electrostatic branch we
want for openCHEESE. In their preprocessed LMDB, the field named `charge` is an
`(n_points, 4)` surface point cloud:

```text
x, y, z, electrostatic_potential
```

The LG-Mol preprocessing pipeline is:

1. RDKit conformer generation.
2. Antechamber AM1-BCC charges and radii.
3. PQR export.
4. MSMS surface generation with probe/density settings.
5. PyMesh mesh cleanup/remeshing.
6. APBS grid electrostatic potential.
7. `multivalue` interpolation from APBS `.dx` to surface vertices.
8. Store the final `[x,y,z,ESP]` cloud in LMDB and subsample 100-200 points at
   train time.

Their model then builds a kNN graph on the surface points, uses ESP as the point
feature, centers the point coordinates, and passes this graph through an
equivariant point-cloud encoder before fusing it with atom-level Uni-Mol
features.

## openCHEESE Port

We can reproduce the useful data contract without Amber, MSMS, or APBS:

1. Use the existing conformer ensemble cache.
2. Generate a Connolly solvent-excluded surface with
   `mlxmolkit.connolly.connolly_ses_surface_from_atoms_mlx`.
3. Evaluate AM1-BCC/qRESP point-charge ESP on the sampled surface with
   `opencheese.descriptors.electrostatic_potential_on_grid_metal`.
4. Save each conformer as an LG-Mol-compatible `[x,y,z,ESP]` cloud.

Implemented entry points:

```python
from opencheese.surface import lgmol_surface_cloud_from_atoms

cloud = lgmol_surface_cloud_from_atoms(
    atoms,
    coords,
    charges,
    method="ses",
    points_num=200,
    sampling="farthest",
).cloud
```

```bash
python tools/prepare_opencheese_surface_cloud_cache.py \
  --ensembles outputs/cheese_projection/cheese_ensembles_1000_k10_q_resp.npz \
  --out outputs/cheese_projection/opencheese_surface_clouds_1000_k10_q_resp.npz \
  --method ses \
  --points-num 200 \
  --sampling farthest
```

The `shell` method is kept as a fast smoke-test path. The `ses` method should be
used for LG-Mol/MSMS-style teacher quality.

## Why Farthest-Point Sampling

LG-Mol randomly samples surface rows during training. For small clouds this can
leave visible holes. openCHEESE defaults to deterministic farthest-point
sampling, which keeps the same tensor shape while covering the surface more
evenly. Random sampling is still available for exact ablation.

## Force2Geo Role

Force2Geo is not the projection model. It is a geometry-quality layer:

```text
2D graph / rough 3D -> MLIP relaxation -> better 3D conformer geometry
```

The practical openCHEESE use is to improve teacher and student input
geometries:

1. Generate diverse ETKDG/openCHEESE conformers.
2. Relax conformers with an MLIP model or a lightweight local surrogate.
3. Recompute Connolly surfaces and ESP clouds.
4. Train/fine-tune the shape and electrostatic projection losses on the
   improved teacher.

This is a higher-cost next step than the LG-Mol-style surface cache, but it is
the right place to use Force2Geo.

## GNN-MA Role

GNN-MA is complementary to the 3D CHEESE/openCHEESE path. It treats ligand-based
virtual screening as pairwise query-candidate retrieval using 2D molecular
graphs only. The useful pieces for us are:

1. Cross-graph node attention as a soft atom-alignment matrix.
2. Bond-aware atom updates and bond-to-atom aggregation.
3. A within-target hardest-negative ranking term after a short BCE warm-up.
4. EF@1-5% as the optimization/selection signal, not only ROC-AUC.

The full published implementation also has edge-to-edge cross attention over
flattened bond pairs. That is valuable for shortlist re-ranking but scales as
`N_A^2 * N_B^2`, so it should not be the first all-library openCHEESE path.

Implemented first step:

```python
from opencheese.gnn_ma import CrossGraphSoftAlignmentScorer

atom_a = encoder(...).atom_embeddings
atom_b = encoder(...).atom_embeddings
result = CrossGraphSoftAlignmentScorer(hidden_dim=128)(
    atom_a,
    mask_a,
    atom_b,
    mask_b,
)
score = result.logits
soft_atom_alignment = result.attention_ab
```

The strongest immediate experiment is a two-stage screen:

1. Use openCHEESE embeddings / Carbo teacher for fast global retrieval.
2. Re-rank the top few thousand with the GNN-MA-style pair scorer.
3. Train the pair scorer with BCE + hardest-negative ranking, or regress the
   openCHEESE teacher matrix and monitor Recall@k / NDCG@k / EF@1%.

This gives us an interpretable atom-alignment signal without replacing the 3D
shape/electrostatic teacher.
