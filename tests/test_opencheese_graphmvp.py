import numpy as np
import mlx.core as mx

from opencheese import CheeseEmbeddingConfig, GraphMVPPretrainer, cheese_embedding_batch
from opencheese.graphmvp import (
    bidirectional_representation_reconstruction_loss_mlx,
    dual_info_nce_loss_mlx,
)


def _toy_batches():
    atoms = [[6, 6, 8], [6, 7, 1]]
    bonds = [
        np.asarray([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=np.int32),
        np.asarray([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.int32),
    ]
    coords = [
        np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.5, 0.4, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [-0.4, 0.9, 0.0]], dtype=np.float32),
    ]
    zeros = [np.zeros_like(xyz) for xyz in coords]
    batch_2d = cheese_embedding_batch(atoms, zeros, bonds, compute_chiral_features=False)
    batch_3d = cheese_embedding_batch(atoms, coords, bonds, compute_chiral_features=True)
    return batch_2d, batch_3d


def _config():
    return CheeseEmbeddingConfig(
        hidden_dim=32,
        embedding_dim=16,
        bond_dim=8,
        n_layers=1,
        n_heads=4,
        n_rbf=8,
        use_charges=False,
    )


def test_dual_info_nce_prefers_matching_views():
    view = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
    swapped = mx.array([[0.0, 1.0], [1.0, 0.0]], dtype=mx.float32)

    good, acc_ab, acc_ba = dual_info_nce_loss_mlx(view, view, temperature=0.1)
    bad, _, _ = dual_info_nce_loss_mlx(view, swapped, temperature=0.1)
    mx.eval(good, bad, acc_ab, acc_ba)

    assert float(good) < float(bad)
    assert float(acc_ab) == 1.0
    assert float(acc_ba) == 1.0


def test_graphmvp_pretrainer_loss_is_finite():
    batch_2d, batch_3d = _toy_batches()
    model = GraphMVPPretrainer(_config(), _config())

    loss = model(batch_2d, batch_3d, temperature=0.2)
    mx.eval(loss.total, loss.contrastive, loss.reconstruction)

    assert np.isfinite(float(loss.total))
    assert np.isfinite(float(loss.contrastive))
    assert np.isfinite(float(loss.reconstruction))


def test_representation_reconstruction_loss_modes_are_finite():
    batch_2d, batch_3d = _toy_batches()
    model = GraphMVPPretrainer(_config(), _config())
    emb_2d = model.encode_2d(batch_2d)
    emb_3d = model.encode_3d(batch_3d)

    losses = [
        bidirectional_representation_reconstruction_loss_mlx(
            emb_2d,
            emb_3d,
            model.project_2d_to_3d,
            model.project_3d_to_2d,
            loss=mode,
        )
        for mode in ("l1", "l2", "cosine")
    ]
    mx.eval(*losses)

    assert all(np.isfinite(float(loss)) for loss in losses)
