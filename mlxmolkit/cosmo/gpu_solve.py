"""
GPU batched linear solver via MLX compiled graph.

Uses mx.compile to fuse the Gauss-Jordan elimination loop
into a single GPU execution graph — avoids per-column kernel overhead.
Falls back to numpy for small systems where LAPACK is faster.
"""
from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx

    def _gj_step(aug, col, N, n):
        """One Gauss-Jordan elimination step (MLX ops, compilable)."""
        ncols = n + 1

        # Scale pivot row
        pivot = aug[:, col, col:col+1]  # (N, 1)
        pivot = mx.where(mx.abs(pivot) > 1e-20, pivot, mx.ones_like(pivot))
        scaled_row = aug[:, col:col+1, :] / pivot[:, :, None]  # (N, 1, ncols)

        # Build elimination factors
        factors = aug[:, :, col:col+1]  # (N, n, 1)
        # Zero out pivot row's factor
        mask = mx.arange(n).reshape(1, -1, 1) != col  # (1, n, 1)
        factors = factors * mask

        # Eliminate + replace pivot row
        new_aug = aug - factors * scaled_row
        # Replace pivot row with scaled version
        row_mask = mx.arange(n).reshape(1, -1, 1) == col  # (1, n, 1)
        new_aug = mx.where(row_mask, mx.broadcast_to(scaled_row, new_aug.shape), new_aug)

        return new_aug

    def gpu_solve_batch(A: np.ndarray, b: np.ndarray, threshold: int = 999999) -> np.ndarray:
        """Solve A·x = b for N systems.

        numpy Accelerate (AMX) is fastest on Apple Silicon for n<1000.
        MLX GPU path available for future when mx.linalg.solve gets GPU support.

        Args:
            A: (N, n, n) float64
            b: (N, n) float64
            threshold: use GPU when N*n exceeds this

        Returns:
            x: (N, n) float64
        """
        N, n, _ = A.shape

        # For small systems, numpy LAPACK (Accelerate) is faster
        if N * n < threshold:
            return np.linalg.solve(A, b[:, :, np.newaxis])[:, :, 0]

        # Build augmented [A|b]
        ncols = n + 1
        aug_np = np.zeros((N, n, ncols), dtype=np.float32)
        aug_np[:, :, :n] = A.astype(np.float32)
        aug_np[:, :, n] = b.astype(np.float32)

        aug = mx.array(aug_np)

        # Gauss-Jordan elimination — MLX lazy graph
        for col in range(n):
            aug = _gj_step(aug, col, N, n)

        mx.eval(aug)
        result = np.array(aug)
        return result[:, :, n].astype(np.float64)


    def gpu_solve_single(A, b):
        return np.linalg.solve(A, b)  # always numpy for single

except ImportError:
    def gpu_solve_batch(A, b, threshold=1000):
        return np.linalg.solve(A, b[:, :, np.newaxis])[:, :, 0]
    def gpu_solve_single(A, b):
        return np.linalg.solve(A, b)
