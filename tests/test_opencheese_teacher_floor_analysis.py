import numpy as np

from tools.analyze_opencheese_teacher_floor import topk_jaccard, topk_overlap


def test_topk_overlap_and_jaccard_exclude_self():
    reference = np.asarray(
        [
            [1.0, 0.9, 0.8, 0.1],
            [0.9, 1.0, 0.2, 0.8],
            [0.8, 0.2, 1.0, 0.7],
            [0.1, 0.8, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    candidate = np.asarray(
        [
            [1.0, 0.9, 0.1, 0.8],
            [0.9, 1.0, 0.8, 0.2],
            [0.1, 0.8, 1.0, 0.7],
            [0.8, 0.2, 0.7, 1.0],
        ],
        dtype=np.float32,
    )

    assert topk_overlap(reference, reference, k=2) == 1.0
    assert topk_jaccard(reference, reference, k=2) == 1.0
    assert 0.0 < topk_overlap(reference, candidate, k=2) < 1.0
    assert 0.0 < topk_jaccard(reference, candidate, k=2) < 1.0
