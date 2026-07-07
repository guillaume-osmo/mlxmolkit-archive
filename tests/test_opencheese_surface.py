import numpy as np

from opencheese.surface import lgmol_surface_cloud_from_atoms, surface_sample_indices


def test_surface_sample_indices_farthest_is_deterministic_and_unique():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    indices_a = surface_sample_indices(points, 4, mode="farthest", random_seed=1)
    indices_b = surface_sample_indices(points, 4, mode="farthest", random_seed=99)

    assert np.array_equal(indices_a, indices_b)
    assert len(np.unique(indices_a)) == 4


def test_lgmol_surface_cloud_shell_mode_returns_xyz_esp_columns():
    atoms = [8, 1, 1]
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.9584, 0.0, 0.0],
            [-0.2396, 0.9275, 0.0],
        ],
        dtype=np.float32,
    )
    charges = np.asarray([-0.6, 0.3, 0.3], dtype=np.float32)

    result = lgmol_surface_cloud_from_atoms(
        atoms,
        coords,
        charges,
        method="shell",
        points_num=32,
        sampling="farthest",
        point_density=0.1,
        min_points_per_shell=8,
        max_points_per_shell=16,
    )

    assert result.cloud.shape == (32, 4)
    assert result.points.shape == (32, 3)
    assert result.esp_values.shape == (32,)
    assert result.n_surface_vertices >= 32
    assert np.all(np.isfinite(result.cloud))
    assert np.std(result.cloud[:, 3]) > 0.0
