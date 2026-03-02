import numpy as np
import mlx.core as mx

from mlxmolkit.fp_uint32 import fp_uint8_to_uint32
from mlxmolkit.tanimoto_metal_u32 import tanimoto_matrix_metal_u32


def _ref_tanimoto(a_bytes: np.ndarray, b_bytes: np.ndarray) -> np.ndarray:
    a_bits = np.unpackbits(a_bytes, bitorder="little", axis=1)
    b_bits = np.unpackbits(b_bytes, bitorder="little", axis=1)
    inter = (a_bits[:, None, :] & b_bits[None, :, :]).sum(axis=-1).astype(np.float32)
    union = (a_bits[:, None, :] | b_bits[None, :, :]).sum(axis=-1).astype(np.float32)
    return inter / (union + 1e-12)


def test_tanimoto_metal_u32_matches_ref():
    """uint32 Metal kernel matches numpy reference."""
    rng = np.random.default_rng(3)
    N, M = 20, 25
    nbytes = 128
    a = rng.integers(0, 256, size=(N, nbytes), dtype=np.uint8)
    b = rng.integers(0, 256, size=(M, nbytes), dtype=np.uint8)
    ref = _ref_tanimoto(a, b)
    au = fp_uint8_to_uint32(mx.array(a))
    bu = fp_uint8_to_uint32(mx.array(b))
    u32 = np.array(tanimoto_matrix_metal_u32(au, bu))
    np.testing.assert_allclose(u32, ref, rtol=1e-5, atol=1e-6)


def test_tanimoto_self_symmetry():
    """Self-similarity is symmetric with 1.0 diagonal."""
    rng = np.random.default_rng(1)
    N = 50
    nbytes = 128
    a = rng.integers(0, 256, size=(N, nbytes), dtype=np.uint8)
    au = fp_uint8_to_uint32(mx.array(a))
    sim = np.array(tanimoto_matrix_metal_u32(au))
    np.testing.assert_allclose(sim, sim.T, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.diag(sim), 1.0, rtol=1e-5, atol=1e-6)


def test_tanimoto_tiled_matches_naive():
    """Tiled kernel matches naive kernel for various N (regression for grid dispatch bug)."""
    rng = np.random.default_rng(4)
    nbytes = 128
    for N in [16, 32, 48, 64, 100]:
        fp_u8 = rng.integers(0, 256, size=(N, nbytes), dtype=np.uint8)
        fp_u32 = fp_uint8_to_uint32(mx.array(fp_u8))
        naive = np.array(tanimoto_matrix_metal_u32(fp_u32, use_tiled=False))
        tiled = np.array(tanimoto_matrix_metal_u32(fp_u32, use_tiled=True))
        np.testing.assert_allclose(tiled, naive, rtol=1e-5, atol=1e-6,
                                   err_msg=f"Tiled vs naive mismatch at N={N}")
