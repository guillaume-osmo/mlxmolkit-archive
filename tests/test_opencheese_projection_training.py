import numpy as np
import mlx.core as mx
from mlx.utils import tree_map

from opencheese import CheeseEmbeddingConfig, CheeseGraphTransformer
from opencheese.optimizers import MuonV2W, polar_express
from tools.train_cheese_projection import (
    anchor_zscore_to_unit,
    projection_metric_loss,
    size_residual_teacher_to_unit,
    supervised_contrastive_loss_mlx,
    target_distance_mlx,
)


def test_teacher_transforms_keep_anchor_order_and_unit_range():
    teacher = np.asarray(
        [
            [1.0, 0.82, 0.30, 0.28],
            [0.82, 1.0, 0.42, 0.40],
            [0.30, 0.42, 1.0, 0.78],
            [0.28, 0.40, 0.78, 1.0],
        ],
        dtype=np.float32,
    )

    zscore = anchor_zscore_to_unit(teacher, temperature=1.0)
    residual, metadata = size_residual_teacher_to_unit(teacher, np.asarray([12, 13, 30, 31], dtype=np.float32))

    assert zscore.shape == teacher.shape
    assert residual.shape == teacher.shape
    assert np.all((zscore >= 0.0) & (zscore <= 1.0))
    assert np.all((residual >= 0.0) & (residual <= 1.0))
    assert zscore[0, 1] > zscore[0, 2]
    assert residual[0, 1] > residual[0, 2]
    assert "size_baseline_coefficients" in metadata


def test_neglog_distance_and_contrastive_loss_are_finite():
    embeddings = mx.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=mx.float32)
    target = mx.array(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
        dtype=mx.float32,
    )

    distance = target_distance_mlx(
        target,
        distance_transform="neglog_similarity",
        distance_scale=1.0,
        distance_power=1.0,
    )
    contrastive = supervised_contrastive_loss_mlx(
        embeddings,
        embeddings,
        target,
        temperature=0.07,
        positive_threshold=0.75,
        positive_top_k=1,
    )
    hybrid = projection_metric_loss(
        embeddings,
        embeddings,
        target,
        loss_mode="hybrid_shape",
        distance_transform="neglog_similarity",
        contrastive_weight=0.1,
        contrastive_positive_threshold=0.75,
        contrastive_positive_top_k=1,
    )
    mx.eval(distance, contrastive, hybrid)

    assert float(distance[0, 0]) < 1.0e-5
    assert np.isfinite(float(contrastive))
    assert np.isfinite(float(hybrid))


def test_muonv2w_updates_opencheese_model_parameters():
    model = CheeseGraphTransformer(
        CheeseEmbeddingConfig(hidden_dim=32, embedding_dim=16, bond_dim=8, n_layers=1, n_heads=4, n_rbf=8)
    )
    params = model.trainable_parameters()
    grads = tree_map(mx.ones_like, params)
    optimizer = MuonV2W(muon_lr=1.0e-4, adamw_lr=1.0e-4, filter_mode="opencheese_hidden")

    before = np.asarray(params["embedding_head"]["layers"][0]["weight"])
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    after = np.asarray(model.trainable_parameters()["embedding_head"]["layers"][0]["weight"])

    assert polar_express(mx.eye(4, dtype=mx.float32)).shape == (4, 4)
    assert not np.allclose(before, after)
