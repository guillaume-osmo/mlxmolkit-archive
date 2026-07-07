import numpy as np
import mlx.core as mx

import opencheese
from opencheese import cheese_batch, cheese_similarity_matrix_mlx
from mlxmolkit import cheese_batch as mlxmolkit_cheese_batch


def test_opencheese_namespace_exports_descriptor_api():
    atoms = [[8, 1, 1]]
    coords = [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9584, 0.0, 0.0],
                [-0.2396, 0.9275, 0.0],
            ],
            dtype=np.float32,
        )
    ]
    charges = [np.array([-0.834, 0.417, 0.417], dtype=np.float32)]

    batch = cheese_batch(atoms, coords, charges)
    scores = cheese_similarity_matrix_mlx(batch, shape_metric="carbo", electrostatic_metric="carbo")
    mx.eval(scores.shape, scores.electrostatic, scores.combined)

    assert opencheese.CheeseBatch is type(batch)
    assert cheese_batch is mlxmolkit_cheese_batch
    np.testing.assert_allclose(np.asarray(scores.shape), [[1.0]], atol=1.0e-5)
    np.testing.assert_allclose(np.asarray(scores.electrostatic), [[1.0]], atol=1.0e-5)
