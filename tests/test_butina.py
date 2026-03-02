import numpy as np
import mlx.core as mx

from mlxmolkit.butina import (
    butina_from_neighbor_list_csr,
    butina_from_similarity_matrix,
    butina_tanimoto_mlx,
)
from mlxmolkit.fp_uint32 import fp_uint8_to_uint32
from mlxmolkit.fused_tanimoto_nlist import fused_neighbor_list_metal
from mlxmolkit.tanimoto_metal_u32 import tanimoto_matrix_metal_u32
from mlxmolkit.butina_metal import build_neighbor_list_metal

def test_butina_partition_and_no_dupes():
    rng = np.random.default_rng(0)
    N = 50
    x = rng.random((N, N), dtype=np.float32)
    sim = (x + x.T) / 2
    np.fill_diagonal(sim, 1.0)

    res = butina_from_similarity_matrix(sim, cutoff=0.8)
    all_idx = [i for c in res.clusters for i in c]
    assert len(all_idx) == N
    assert len(set(all_idx)) == N

def test_butina_csr_matches_dense():
    """butina_from_neighbor_list_csr gives same partition as butina_from_similarity_matrix."""
    rng = np.random.default_rng(42)
    N = 40
    sim = rng.random((N, N)).astype(np.float32)
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 1.0)
    cutoff = 0.7
    res_dense = butina_from_similarity_matrix(sim, cutoff=cutoff)
    # Build CSR by hand (same logic as GPU would)
    offsets = np.zeros((N + 1,), dtype=np.int32)
    for i in range(N):
        nbrs = [j for j in range(N) if i != j and sim[i, j] >= cutoff]
        offsets[i + 1] = offsets[i] + len(nbrs)
    indices = np.zeros((offsets[-1],), dtype=np.int32)
    for i in range(N):
        nbrs = [j for j in range(N) if i != j and sim[i, j] >= cutoff]
        indices[offsets[i] : offsets[i + 1]] = nbrs
    res_csr = butina_from_neighbor_list_csr(offsets, indices, N, cutoff=cutoff)
    all_dense = sorted([i for c in res_dense.clusters for i in c])
    all_csr = sorted([i for c in res_csr.clusters for i in c])
    assert all_dense == all_csr
    assert len(res_dense.clusters) == len(res_csr.clusters)


def test_butina_tanimoto_mlx_runs():
    """Full pipeline (nvMolKit-style): fp uint8 → Tanimoto Metal u32 → neighbor list → Butina Python."""
    rng = np.random.default_rng(2)
    N = 50
    nbytes = 128  # 1024 bits
    fp = rng.integers(0, 256, size=(N, nbytes), dtype=np.uint8)
    res = butina_tanimoto_mlx(mx.array(fp), cutoff=0.6)
    assert sum(len(c) for c in res.clusters) == N


def test_fused_matches_nonfused():
    """Fused Tanimoto→CSR gives identical neighbor lists as separate Tanimoto + threshold."""
    rng = np.random.default_rng(10)
    N = 200
    nbytes = 128
    cutoff = 0.4
    fp_u8 = rng.integers(0, 256, size=(N, nbytes), dtype=np.uint8)
    fp_u32 = fp_uint8_to_uint32(mx.array(fp_u8))

    sim = tanimoto_matrix_metal_u32(fp_u32)
    mx.eval(sim)
    offsets_ref, indices_ref = build_neighbor_list_metal(sim, cutoff)
    offsets_fused, indices_fused = fused_neighbor_list_metal(fp_u32, cutoff)

    np.testing.assert_array_equal(np.diff(offsets_ref), np.diff(offsets_fused))
    for i in range(N):
        ref_set = set(indices_ref[offsets_ref[i]:offsets_ref[i + 1]].tolist())
        fused_set = set(indices_fused[offsets_fused[i]:offsets_fused[i + 1]].tolist())
        assert ref_set == fused_set, f"Row {i}: ref={ref_set} fused={fused_set}"


def test_fused_butina_end_to_end():
    """Fused pipeline produces valid partition."""
    rng = np.random.default_rng(11)
    N = 100
    nbytes = 128
    fp_u8 = rng.integers(0, 256, size=(N, nbytes), dtype=np.uint8)
    fp_u32 = fp_uint8_to_uint32(mx.array(fp_u8))
    offsets, indices = fused_neighbor_list_metal(fp_u32, 0.4)
    res = butina_from_neighbor_list_csr(offsets, indices, N, 0.4)
    all_idx = [i for c in res.clusters for i in c]
    assert len(all_idx) == N
    assert len(set(all_idx)) == N
