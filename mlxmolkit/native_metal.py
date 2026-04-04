"""
Native Metal pipeline via ctypes — pre-compiled .metallib, zero JIT.

Loads libfused_metal.dylib which contains:
  - Pre-compiled Metal shader pipelines (loaded once from .metallib)
  - Direct Metal API dispatch (no MLX, no Python overhead)
  - C API: fused_tanimoto_pipeline() → CSR offsets + indices
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_lib = None


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib

    dylib_path = Path(__file__).parent.parent / "metal" / "libfused_metal.dylib"
    if not dylib_path.exists():
        raise FileNotFoundError(
            f"Native Metal library not found at {dylib_path}. "
            "Build it with: cd metal && clang++ -O2 -std=c++17 -shared "
            "-framework Metal -framework Foundation "
            "-o libfused_metal.dylib fused_pipeline.mm"
        )
    _lib = ctypes.cdll.LoadLibrary(str(dylib_path))

    _lib.fused_tanimoto_pipeline.restype = ctypes.c_int
    _lib.fused_tanimoto_pipeline.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # fp_u32
        ctypes.c_int,                     # N
        ctypes.c_int,                     # nwords
        ctypes.c_float,                   # cutoff
        ctypes.POINTER(ctypes.c_int32),   # offsets (out)
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),  # indices (out)
        ctypes.POINTER(ctypes.c_int),     # n_edges (out)
        ctypes.POINTER(ctypes.c_double),  # gpu_ms (out)
    ]

    _lib.free_indices.restype = None
    _lib.free_indices.argtypes = [ctypes.POINTER(ctypes.c_int32)]

    return _lib


def fused_neighbor_list_native(
    fp_u32: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fused Tanimoto + CSR neighbor list via native pre-compiled Metal.

    Args:
        fp_u32: (N, nwords) uint32 numpy array of packed fingerprints.
        cutoff: Tanimoto similarity threshold.

    Returns:
        (offsets, indices, gpu_ms) where offsets is (N+1,) int32,
        indices is (n_edges,) int32, and gpu_ms is GPU time in milliseconds.
    """
    lib = _get_lib()

    fp_u32 = np.ascontiguousarray(fp_u32, dtype=np.uint32)
    N, nwords = fp_u32.shape

    offsets = np.zeros(N + 1, dtype=np.int32)
    indices_ptr = ctypes.POINTER(ctypes.c_int32)()
    n_edges = ctypes.c_int(0)
    gpu_ms = ctypes.c_double(0.0)

    rc = lib.fused_tanimoto_pipeline(
        fp_u32.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        N,
        nwords,
        cutoff,
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(indices_ptr),
        ctypes.byref(n_edges),
        ctypes.byref(gpu_ms),
    )

    if rc != 0:
        raise RuntimeError(f"Native Metal pipeline failed with error code {rc}")

    total = n_edges.value
    if total > 0:
        buf = (ctypes.c_int32 * total).from_address(ctypes.addressof(indices_ptr.contents))
        indices = np.ctypeslib.as_array(buf).copy()
        lib.free_indices(indices_ptr)
    else:
        indices = np.array([], dtype=np.int32)

    return offsets, indices, gpu_ms.value
