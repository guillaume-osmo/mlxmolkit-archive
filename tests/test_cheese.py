import numpy as np

import mlx.core as mx

from mlxmolkit.cheese import (
    CheeseAlignmentConfig,
    align_cheese_pair,
    cheese_batch,
    cheese_alignment_matrix,
    cheese_similarity_pairs_mlx,
    cheese_similarity_matrix_mlx,
    electrostatic_carbo_matrix_mlx,
    electrostatic_overlap_matrix_mlx,
    electrostatic_potential_on_grid_metal,
    electrostatic_potential_on_grid_mlx,
    electrostatic_similarity_matrix_mlx,
    electrostatic_similarity_to_template_grids_mlx,
    electrostatic_similarity_to_template_grids_tiled_mlx,
    horn_align_pairwise_mlx,
    kabsch_align,
    mean_similarity_to_actives_mlx,
    rdkit_grid_shape_tanimoto_matrix_mlx,
    rdkit_shape_occupancy_mlx,
    rdkit_shape_tanimoto_from_mols,
    rdkit_shape_tanimoto_matrix_from_mols,
    shape_carbo_matrix_mlx,
    shape_tanimoto_matrix_mlx,
)


_ALIGN_ATOMS = [6, 7, 8, 16, 1]
_ALIGN_COORDS = np.asarray(
    [
        [0.00, 0.00, 0.00],
        [1.42, 0.13, -0.18],
        [-0.38, 1.21, 0.32],
        [0.23, -0.44, 1.71],
        [1.91, 0.78, 0.65],
    ],
    dtype=np.float64,
)
_ALIGN_CHARGES = np.asarray([-0.12, -0.34, -0.46, 0.28, 0.64], dtype=np.float64)


def _water_batch(charges=(-0.6, 0.3, 0.3)):
    atoms = [8, 1, 1]
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.9584, 0.0, 0.0],
            [-0.2396, 0.9275, 0.0],
        ],
        dtype=np.float32,
    )
    return cheese_batch([atoms], [coords], [np.asarray(charges, dtype=np.float32)])


def _row_rotation(axis: int, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    if axis == 0:
        col = np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == 1:
        col = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    else:
        col = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return col.T


_ESPSIM_REF_A = np.asarray(
    [
        [15.90600036, 3.95348310, 17.61453176],
        [3.95348310, 5.21580206, 1.91045387],
        [17.61453176, 1.91045387, 238.75820253],
    ],
    dtype=np.float64,
)
_ESPSIM_REF_B = np.asarray(
    [
        [-0.02495000, -0.04539319, -0.00247124],
        [-0.04539319, -0.25130000, -0.00258662],
        [-0.00247124, -0.00258662, -0.00130000],
    ],
    dtype=np.float64,
)


def _espsim_reference_overlap(coords_a, charges_a, coords_b, charges_b) -> float:
    dist2 = np.sum((coords_a[:, None, :] - coords_b[None, :, :]) ** 2, axis=-1)
    kernel = np.sum(
        _ESPSIM_REF_A.reshape(-1)[:, None, None]
        * np.exp(_ESPSIM_REF_B.reshape(-1)[:, None, None] * dist2[None, :, :]),
        axis=0,
    )
    return float(np.sum(charges_a[:, None] * charges_b[None, :] * kernel))


def _rdkit_mol_from_atoms_coords(atoms, coords):
    from rdkit import Chem, Geometry

    rw = Chem.RWMol()
    for z in atoms:
        rw.AddAtom(Chem.Atom(int(z)))
    mol = rw.GetMol()
    conf = Chem.Conformer(len(atoms))
    for i, xyz in enumerate(np.asarray(coords, dtype=np.float64)):
        conf.SetAtomPosition(i, Geometry.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf)
    return mol


def _rdkit_grid_coords(grid):
    coords = []
    for i in range(grid.GetSize()):
        p = grid.GetGridPointLoc(i)
        coords.append([p.x, p.y, p.z])
    return np.asarray(coords, dtype=np.float32)


def test_shape_tanimoto_self_is_one_and_symmetric():
    water = _water_batch()
    methane = cheese_batch(
        [[6, 1, 1, 1, 1]],
        [
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.63, 0.63, 0.63],
                    [-0.63, -0.63, 0.63],
                    [-0.63, 0.63, -0.63],
                    [0.63, -0.63, -0.63],
                ],
                dtype=np.float32,
            )
        ],
    )
    batch = cheese_batch(
        [
            np.asarray(water.atomic_numbers)[0, :3].tolist(),
            np.asarray(methane.atomic_numbers)[0, :5].tolist(),
        ],
        [
            np.asarray(water.coords)[0, :3],
            np.asarray(methane.coords)[0, :5],
        ],
    )

    sim = shape_tanimoto_matrix_mlx(batch)
    mx.eval(sim)
    sim_np = np.asarray(sim)

    assert np.allclose(np.diag(sim_np), 1.0, atol=1e-5)
    assert np.allclose(sim_np, sim_np.T, atol=1e-5)
    assert 0.0 <= sim_np[0, 1] <= 1.0


def test_shape_carbo_self_is_one_and_at_least_tanimoto_for_positive_overlaps():
    batch = cheese_batch(
        [[8, 1, 1], [8, 1, 1]],
        [
            np.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float32),
            np.asarray([[0.1, 0.0, 0.0], [1.06, 0.0, 0.0], [-0.14, 0.93, 0.0]], dtype=np.float32),
        ],
    )

    carbo = shape_carbo_matrix_mlx(batch)
    tanimoto = shape_tanimoto_matrix_mlx(batch)
    mx.eval(carbo, tanimoto)

    carbo_np = np.asarray(carbo)
    tanimoto_np = np.asarray(tanimoto)
    assert np.allclose(np.diag(carbo_np), 1.0, atol=1e-5)
    assert np.allclose(carbo_np, carbo_np.T, atol=1e-5)
    assert np.all((0.0 <= carbo_np) & (carbo_np <= 1.0))
    assert np.all(carbo_np + 1.0e-6 >= tanimoto_np)


def test_horn_pairwise_alignment_matches_kabsch_for_rigid_transform():
    base = _ALIGN_COORDS.astype(np.float32)
    weights = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    rotation_a = _row_rotation(0, 31.0) @ _row_rotation(2, -17.0)
    rotation_b = _row_rotation(1, -42.0) @ _row_rotation(2, 11.0)
    probes = np.stack(
        [
            base @ rotation_a + np.asarray([3.0, -1.0, 0.5], dtype=np.float32),
            base @ rotation_b + np.asarray([-2.0, 0.7, 1.2], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    refs = np.stack(
        [
            base,
            base @ _row_rotation(2, 12.0) + np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    aligned, rmsd, _ = horn_align_pairwise_mlx(probes, refs, weights=weights, power_iters=48)
    mx.eval(aligned, rmsd)
    aligned_np = np.asarray(aligned)
    rmsd_np = np.asarray(rmsd)

    for i in range(probes.shape[0]):
        for j in range(refs.shape[0]):
            expected = kabsch_align(probes[i], refs[j], weights=weights)
            assert np.allclose(aligned_np[i, j], expected, atol=2.0e-4)
            diff2 = np.sum((expected - refs[j]) ** 2, axis=1)
            expected_rmsd = np.sqrt(np.sum(diff2 * weights) / np.sum(weights))
            assert np.isclose(rmsd_np[i, j], expected_rmsd, atol=2.0e-4)


def test_cheese_similarity_pairs_matches_matrix_diagonal():
    atoms = [[8, 1, 1], [6, 1, 1, 1, 1]]
    coords_a = [
        np.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float32),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ],
            dtype=np.float32,
        ),
    ]
    coords_b = [coords_a[0] + np.asarray([0.15, -0.05, 0.08], dtype=np.float32), coords_a[1] * 1.02]
    charges = [
        np.asarray([-0.6, 0.3, 0.3], dtype=np.float32),
        np.asarray([-0.2, 0.05, 0.05, 0.05, 0.05], dtype=np.float32),
    ]
    probe = cheese_batch(atoms, coords_a, charges)
    ref = cheese_batch(atoms, coords_b, charges)

    pair_scores = cheese_similarity_pairs_mlx(probe, ref, shape_metric="carbo", electrostatic_metric="carbo")
    matrix_scores = cheese_similarity_matrix_mlx(probe, ref, shape_metric="carbo", electrostatic_metric="carbo")
    mx.eval(pair_scores.shape, pair_scores.electrostatic, pair_scores.combined, matrix_scores.shape, matrix_scores.electrostatic, matrix_scores.combined)

    assert np.allclose(np.asarray(pair_scores.shape), np.diag(np.asarray(matrix_scores.shape)), atol=1.0e-6)
    assert np.allclose(np.asarray(pair_scores.electrostatic), np.diag(np.asarray(matrix_scores.electrostatic)), atol=1.0e-6)
    assert np.allclose(np.asarray(pair_scores.combined), np.diag(np.asarray(matrix_scores.combined)), atol=1.0e-6)


def test_rdkit_shape_wrapper_matches_espsim_shape_definition():
    from rdkit.Chem import AllChem

    atoms = [6, 8, 7]
    coords_a = np.asarray([[0.0, 0.0, 0.0], [1.25, 0.2, 0.0], [-0.4, 1.1, 0.1]], dtype=np.float64)
    coords_b = coords_a + np.asarray([0.2, -0.1, 0.3])
    mol_a = _rdkit_mol_from_atoms_coords(atoms, coords_a)
    mol_b = _rdkit_mol_from_atoms_coords(atoms, coords_b)

    wrapped = rdkit_shape_tanimoto_from_mols(mol_a, mol_b)
    matrix = rdkit_shape_tanimoto_matrix_from_mols([mol_a], [mol_b])
    mx.eval(matrix)
    reference = 1.0 - AllChem.ShapeTanimotoDist(mol_a, mol_b)

    assert np.isclose(wrapped, reference, atol=1e-12)
    assert np.allclose(np.asarray(matrix), [[reference]], atol=1e-7)


def test_rdkit_shape_occupancy_mlx_matches_rdkit_grid_encoder():
    from rdkit import DataStructs, Geometry
    from rdkit.Chem import AllChem

    atoms = [8, 1]
    coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    mol = _rdkit_mol_from_atoms_coords(atoms, coords)
    grid = Geometry.UniformGrid3D(
        8.0,
        8.0,
        8.0,
        0.5,
        DataStructs.DiscreteValueType.TWOBITVALUE,
        Geometry.Point3D(-4.0, -4.0, -4.0),
    )
    AllChem.EncodeShape(mol, grid, ignoreHs=True, vdwScale=0.8, stepSize=0.25, maxLayers=-1)
    expected = np.asarray([grid.GetOccupancyVect()[i] for i in range(grid.GetSize())], dtype=np.float32)

    batch = cheese_batch([atoms], [coords])
    actual = rdkit_shape_occupancy_mlx(batch, _rdkit_grid_coords(grid))
    mx.eval(actual)

    assert np.array_equal(np.asarray(actual)[0], expected)


def test_rdkit_grid_shape_tanimoto_mlx_matches_rdkit_grid_distance():
    from rdkit import DataStructs, Geometry
    from rdkit.Chem import AllChem

    atoms = [8]
    coords_a = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    coords_b = np.asarray([[0.75, 0.0, 0.0]], dtype=np.float32)
    mol_a = _rdkit_mol_from_atoms_coords(atoms, coords_a)
    mol_b = _rdkit_mol_from_atoms_coords(atoms, coords_b)
    grid_kwargs = dict(
        dimX=8.0,
        dimY=8.0,
        dimZ=8.0,
        spacing=0.5,
        valType=DataStructs.DiscreteValueType.TWOBITVALUE,
        offSet=Geometry.Point3D(-4.0, -4.0, -4.0),
    )
    grid_a = Geometry.UniformGrid3D(**grid_kwargs)
    grid_b = Geometry.UniformGrid3D(**grid_kwargs)
    AllChem.EncodeShape(mol_a, grid_a, ignoreHs=True, vdwScale=0.8, stepSize=0.25, maxLayers=-1)
    AllChem.EncodeShape(mol_b, grid_b, ignoreHs=True, vdwScale=0.8, stepSize=0.25, maxLayers=-1)
    expected = 1.0 - Geometry.TanimotoDistance(grid_a, grid_b)

    batch_a = cheese_batch([atoms], [coords_a])
    batch_b = cheese_batch([atoms], [coords_b])
    actual = rdkit_grid_shape_tanimoto_matrix_mlx(batch_a, batch_b, grid_coords=_rdkit_grid_coords(grid_a))
    mx.eval(actual)

    assert np.allclose(np.asarray(actual), [[expected]], atol=1e-7)


def test_roshambo_style_alignment_recovers_rigid_transform():
    rotation = _row_rotation(2, 37.0) @ _row_rotation(0, -21.0)
    translation = np.asarray([4.2, -2.1, 1.3])
    probe_coords = _ALIGN_COORDS @ rotation + translation[None, :]

    result = align_cheese_pair(
        _ALIGN_ATOMS,
        probe_coords,
        _ALIGN_ATOMS,
        _ALIGN_COORDS,
        config=CheeseAlignmentConfig(start_mode="rotate_45", refine_top_k=16),
    )

    rmsd = np.sqrt(np.mean(np.sum((result.aligned_coords - _ALIGN_COORDS) ** 2, axis=1)))
    assert result.shape > 0.999
    assert rmsd < 5e-3
    assert result.n_starts > 24


def test_metal_start_scoring_matches_numpy_alignment_path():
    rotation = _row_rotation(2, -31.0) @ _row_rotation(1, 23.0)
    translation = np.asarray([-1.1, 2.7, -0.4])
    probe_coords = _ALIGN_COORDS @ rotation + translation[None, :]
    common = dict(
        start_mode="roshambo",
        refine_top_k=8,
        local_refine=False,
    )

    metal = align_cheese_pair(
        _ALIGN_ATOMS,
        probe_coords,
        _ALIGN_ATOMS,
        _ALIGN_COORDS,
        config=CheeseAlignmentConfig(**common, start_score_backend="metal"),
    )
    numpy = align_cheese_pair(
        _ALIGN_ATOMS,
        probe_coords,
        _ALIGN_ATOMS,
        _ALIGN_COORDS,
        config=CheeseAlignmentConfig(**common, start_score_backend="numpy"),
    )

    assert np.isclose(metal.shape, numpy.shape, atol=1e-5)
    assert np.allclose(metal.aligned_coords, numpy.aligned_coords, atol=2e-4)


def test_alignment_keeps_electrostatic_channel_after_overlay():
    rotation = _row_rotation(1, -44.0) @ _row_rotation(2, 19.0)
    probe_coords = _ALIGN_COORDS @ rotation + np.asarray([-2.3, 0.7, 3.1])[None, :]

    result = align_cheese_pair(
        _ALIGN_ATOMS,
        probe_coords,
        _ALIGN_ATOMS,
        _ALIGN_COORDS,
        probe_charges=-_ALIGN_CHARGES,
        reference_charges=_ALIGN_CHARGES,
        config=CheeseAlignmentConfig(start_mode="roshambo", refine_top_k=12),
    )

    assert result.shape > 0.995
    assert result.electrostatic < -0.99
    assert 0.45 < result.combined < 0.55


def test_alignment_matrix_identifies_transformed_reference():
    rotation = _row_rotation(0, 33.0) @ _row_rotation(2, -12.0)
    probe_coords = _ALIGN_COORDS @ rotation + np.asarray([1.5, -4.0, 2.0])[None, :]
    probe = cheese_batch([_ALIGN_ATOMS], [probe_coords], [_ALIGN_CHARGES])
    references = cheese_batch(
        [_ALIGN_ATOMS, [6, 1, 1, 1, 1]],
        [
            _ALIGN_COORDS,
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.63, 0.63, 0.63],
                    [-0.63, -0.63, 0.63],
                    [-0.63, 0.63, -0.63],
                    [0.63, -0.63, -0.63],
                ],
                dtype=np.float64,
            ),
        ],
        [_ALIGN_CHARGES, np.zeros(5)],
    )

    scores, alignments = cheese_alignment_matrix(
        probe,
        references,
        config=CheeseAlignmentConfig(start_mode="roshambo", refine_top_k=8),
        return_alignments=True,
    )
    mx.eval(scores.shape, scores.combined)

    assert np.asarray(scores.shape)[0, 0] > 0.995
    assert np.asarray(scores.shape)[0, 0] > np.asarray(scores.shape)[0, 1]
    assert alignments[0][0].reference_index == 0


def test_cocluster_starts_align_common_core_when_global_axes_shift():
    rotation = _row_rotation(0, -28.0) @ _row_rotation(2, 41.0)
    probe_coords = _ALIGN_COORDS @ rotation + np.asarray([2.2, -1.7, 0.4])[None, :]
    ref_atoms = _ALIGN_ATOMS + [6]
    ref_coords = np.vstack([_ALIGN_COORDS, np.asarray([[5.5, 5.2, -4.8]])])

    principal = align_cheese_pair(
        _ALIGN_ATOMS,
        probe_coords,
        ref_atoms,
        ref_coords,
        config=CheeseAlignmentConfig(
            start_mode="principal",
            local_refine=False,
            start_score_backend="numpy",
        ),
    )
    cocluster = align_cheese_pair(
        _ALIGN_ATOMS,
        probe_coords,
        ref_atoms,
        ref_coords,
        config=CheeseAlignmentConfig(
            start_mode="principal",
            local_refine=False,
            cocluster_starts=32,
            cocluster_pair_pool=18,
            start_score_backend="numpy",
        ),
    )

    core_rmsd = np.sqrt(np.mean(np.sum((cocluster.aligned_coords - _ALIGN_COORDS) ** 2, axis=1)))
    assert cocluster.shape > principal.shape + 0.05
    assert core_rmsd < 1e-3


def test_electrostatic_carbo_detects_inverted_charges():
    water = _water_batch()
    inverted = _water_batch(charges=(0.6, -0.3, -0.3))

    same = electrostatic_carbo_matrix_mlx(water, water)
    opposite = electrostatic_carbo_matrix_mlx(water, inverted)
    mx.eval(same, opposite)

    assert np.allclose(np.asarray(same), [[1.0]], atol=1e-5)
    assert np.asarray(opposite)[0, 0] < -0.999


def test_espsim_gaussian_electrostatics_matches_reference_formula():
    water = _water_batch()
    inverted = _water_batch(charges=(0.6, -0.3, -0.3))

    overlap = electrostatic_overlap_matrix_mlx(water, inverted)
    carbo = electrostatic_similarity_matrix_mlx(water, inverted, metric="carbo")
    tanimoto = electrostatic_similarity_matrix_mlx(water, inverted, metric="tanimoto")
    tanimoto_unit = electrostatic_similarity_matrix_mlx(water, inverted, metric="tanimoto", renormalize=True)
    mx.eval(overlap, carbo, tanimoto, tanimoto_unit)

    coords = np.asarray(water.coords)[0, :3].astype(np.float64)
    charges = np.asarray(water.charges)[0, :3].astype(np.float64)
    inv_charges = np.asarray(inverted.charges)[0, :3].astype(np.float64)
    ref_cross = _espsim_reference_overlap(coords, charges, coords, inv_charges)
    ref_self = _espsim_reference_overlap(coords, charges, coords, charges)
    ref_carbo = ref_cross / np.sqrt(ref_self * ref_self)
    ref_tanimoto = ref_cross / (ref_self + ref_self - ref_cross)

    assert np.allclose(np.asarray(overlap), [[ref_cross]], atol=1e-4)
    assert np.allclose(np.asarray(carbo), [[ref_carbo]], atol=1e-5)
    assert np.allclose(np.asarray(tanimoto), [[ref_tanimoto]], atol=1e-5)
    assert np.allclose(np.asarray(tanimoto_unit), [[0.0]], atol=1e-5)


def test_cheese_combined_similarity_maps_electrostatic_to_unit_interval():
    water = _water_batch()
    inverted = _water_batch(charges=(0.6, -0.3, -0.3))

    result = cheese_similarity_matrix_mlx(water, inverted)
    mx.eval(result.shape, result.electrostatic, result.combined)

    assert np.asarray(result.shape)[0, 0] > 0.999
    assert np.asarray(result.electrostatic)[0, 0] < -0.999
    assert 0.45 < np.asarray(result.combined)[0, 0] < 0.55


def test_deepfmpo_style_tanimoto_channel_uses_espsim_renormalization():
    water = _water_batch()
    inverted = _water_batch(charges=(0.6, -0.3, -0.3))

    result = cheese_similarity_matrix_mlx(water, inverted, electrostatic_metric="tanimoto")
    mx.eval(result.shape, result.electrostatic, result.combined)

    assert np.allclose(np.asarray(result.electrostatic), [[-1.0 / 3.0]], atol=1e-5)
    assert 0.45 < np.asarray(result.combined)[0, 0] < 0.55


def test_ultrafast_grid_esp_matches_reference_formula():
    water = _water_batch()
    grid = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    esp = electrostatic_potential_on_grid_mlx(water.coords, water.charges, grid, mask=water.mask)
    mx.eval(esp)

    coords = np.asarray(water.coords)[0, :3]
    charges = np.asarray(water.charges)[0, :3]
    delta = grid[:, None, :] - coords[None, :, :]
    ref = np.sum(charges[None, :] / np.linalg.norm(delta, axis=-1), axis=1)

    assert np.allclose(np.asarray(esp)[0], ref, atol=1e-5)


def test_metal_grid_esp_matches_mlx_matmul_path():
    water = _water_batch()
    grid = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [-2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mlx_values = electrostatic_potential_on_grid_mlx(water.coords, water.charges, grid, mask=water.mask)
    metal_values = electrostatic_potential_on_grid_metal(water.coords, water.charges, grid, mask=water.mask)
    mx.eval(mlx_values, metal_values)

    assert np.allclose(np.asarray(metal_values), np.asarray(mlx_values), atol=1e-5)


def test_tiled_template_grid_similarity_matches_full_path():
    water = _water_batch()
    grid = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [-2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )
    template_esp = electrostatic_potential_on_grid_mlx(water.coords, water.charges, grid, mask=water.mask)
    full = electrostatic_similarity_to_template_grids_mlx(water, grid, template_esp[0])
    tiled = electrostatic_similarity_to_template_grids_tiled_mlx(
        water,
        grid,
        template_esp[0],
        probe_tile_size=1,
        template_tile_size=1,
        grid_tile_size=2,
    )
    mx.eval(full, tiled)

    assert np.allclose(np.asarray(full), [[1.0]], atol=1e-5)
    assert np.allclose(np.asarray(tiled), np.asarray(full), atol=1e-5)


def test_mean_similarity_to_actives_matches_slide_equation_and_topk():
    sim = mx.array(
        [
            [0.2, 0.8, 0.4],
            [0.9, 0.1, 0.5],
        ],
        dtype=mx.float32,
    )
    mean = mean_similarity_to_actives_mlx(sim, active_mask=mx.array([1, 0, 1]))
    top1 = mean_similarity_to_actives_mlx(sim, active_mask=mx.array([1, 0, 1]), top_k=1)
    mx.eval(mean, top1)

    assert np.allclose(np.asarray(mean), [0.3, 0.7], atol=1e-6)
    assert np.allclose(np.asarray(top1), [0.4, 0.9], atol=1e-6)


def test_kabsch_align_recovers_rigid_transform():
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = coords @ rotation + np.asarray([2.0, -1.0, 0.5])

    aligned = kabsch_align(transformed, coords)

    assert np.allclose(aligned, coords, atol=1e-6)
