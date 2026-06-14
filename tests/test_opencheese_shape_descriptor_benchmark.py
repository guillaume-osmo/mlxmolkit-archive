import numpy as np
from rdkit.Chem import rdMolDescriptors

from tools.benchmark_opencheese_shape_descriptors import _usr_descriptor_from_points, _usr_score_matrix


def test_vectorized_usr_score_matches_rdkit_for_usrcat_length():
    rng = np.random.default_rng(123)
    left = rng.normal(size=(3, 60)).astype(np.float32)
    right = rng.normal(size=(4, 60)).astype(np.float32)

    score = _usr_score_matrix(left, right)

    for i in range(left.shape[0]):
        for j in range(right.shape[0]):
            expected = rdMolDescriptors.GetUSRScore(left[i].tolist(), right[j].tolist())
            assert np.isclose(score[i, j], expected, rtol=1.0e-6, atol=1.0e-6)


def test_surface_usr_descriptor_is_rotation_translation_invariant():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    moved = points @ rotation.T + np.asarray([4.0, -2.0, 7.0], dtype=np.float32)

    desc = _usr_descriptor_from_points(points)
    moved_desc = _usr_descriptor_from_points(moved)

    assert desc.shape == (12,)
    assert np.all(np.isfinite(desc))
    np.testing.assert_allclose(desc, moved_desc, rtol=1.0e-6, atol=1.0e-6)
