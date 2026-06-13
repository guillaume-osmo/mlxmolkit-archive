import numpy as np

import mlx.core as mx

from opencheese.gnn_ma import CrossGraphSoftAlignmentScorer, cross_graph_attention_weights_mlx


def test_cross_graph_attention_weights_respect_key_mask():
    query = mx.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=mx.float32)
    key = mx.array([[[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]], dtype=mx.float32)
    key_mask = mx.array([[1.0, 1.0, 0.0]], dtype=mx.float32)

    attn = cross_graph_attention_weights_mlx(query, key, key_mask)
    mx.eval(attn)

    attn_np = np.asarray(attn)
    assert attn_np.shape == (1, 2, 3)
    assert np.allclose(attn_np.sum(axis=-1), 1.0, atol=1e-6)
    assert np.allclose(attn_np[:, :, 2], 0.0, atol=1e-7)


def test_cross_graph_soft_alignment_scorer_outputs_pair_logits_and_attention():
    atom_a = mx.array(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=mx.float32,
    )
    atom_b = mx.array(
        [
            [[1.0, 0.1, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.2], [0.0, 0.0, 0.1, 1.0]],
        ],
        dtype=mx.float32,
    )
    mask_a = mx.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=mx.float32)
    mask_b = mx.array([[1.0, 1.0], [1.0, 1.0]], dtype=mx.float32)

    scorer = CrossGraphSoftAlignmentScorer(hidden_dim=4, pair_hidden_dim=8)
    result = scorer(atom_a, mask_a, atom_b, mask_b)
    mx.eval(result.logits, result.attention_ab, result.attention_ba, result.pooled_a, result.pooled_b)

    assert result.logits.shape == (2,)
    assert result.attention_ab.shape == (2, 3, 2)
    assert result.attention_ba.shape == (2, 2, 3)
    assert result.pooled_a.shape == (2, 4)
    assert result.pooled_b.shape == (2, 4)
    assert np.all(np.isfinite(np.asarray(result.logits)))
    assert np.allclose(np.asarray(result.attention_ba)[:, :, 2], 0.0, atol=1e-7)
