"""
Fingerprint packing: uint8 (nbytes) <-> uint32 (nwords) for Metal-friendly throughput.

32-bit reads reduce memory transactions and align with Apple GPU; use with tanimoto_metal_u32.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx


def fp_uint8_to_uint32(fp_bytes: mx.array) -> mx.array:
    """
    Convert (N, nbytes) uint8 packed bits to (N, nwords) uint32, little-endian.
    Pads so nbytes -> ceil(nbytes/4) words.
    """
    a = np.array(fp_bytes)
    N, nbytes = a.shape
    nwords = (nbytes + 3) // 4
    if nbytes % 4 == 0:
        out = a.view(np.uint32).reshape(N, nwords).copy()
    else:
        padded = np.zeros((N, nwords * 4), dtype=np.uint8)
        padded[:, :nbytes] = a
        out = padded.view(np.uint32).reshape(N, nwords).copy()
    return mx.array(out, dtype=mx.uint32)


def fp_uint32_to_uint8(fp_words: mx.array, nbytes: int) -> mx.array:
    """Convert (N, nwords) uint32 back to (N, nbytes) uint8 for APIs that expect bytes."""
    a = np.array(fp_words)
    N, nwords = a.shape
    raw = a.view(np.uint8).reshape(N, nwords * 4)
    return mx.array(raw[:, :nbytes].copy(), dtype=mx.uint8)
