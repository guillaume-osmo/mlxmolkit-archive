# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Experimental fast GFN2 gradient wrapper."""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from . import aes as _aes
from . import gradient_gfn2 as _grad
from . import multipole_integrals as _multipole_integrals
from . import scf_gfn2 as _scf
from .aes_fast import mmompop_vectorized, setvsdq_vectorized
from .gradient_aes_gfn2 import aes_gradient_frozen_density
from .multipole_integrals_cpp import CPP_AVAILABLE, multipole_matrices_cpp
from .scf_gfn2_fast import gfn2_energy_fast


def _aes_fd_coord_worker(args):
    atoms, coords, P_sao, qsh, shell_atom, atom_idx, axis, h = args
    cp = np.array(coords, dtype=np.float64, copy=True)
    cm = np.array(coords, dtype=np.float64, copy=True)
    cp[atom_idx, axis] += h
    cm[atom_idx, axis] -= h
    ep = _grad._aes_full_energy_at(atoms, cp, P_sao, qsh, shell_atom)
    em = _grad._aes_full_energy_at(atoms, cm, P_sao, qsh, shell_atom)
    return atom_idx, axis, (ep - em) / (2.0 * h)


def _parallel_aes_fd_gradient(
    atoms: list[int],
    coords_ang: np.ndarray,
    P_sao: np.ndarray,
    qsh: np.ndarray,
    shell_atom: np.ndarray,
    h: float,
    workers: int,
) -> np.ndarray:
    coords = np.asarray(coords_ang, dtype=np.float64)
    n_jobs = coords.shape[0] * 3
    workers = max(1, min(int(workers), n_jobs))
    jobs = [
        (atoms, coords, P_sao, qsh, shell_atom, i, ax, h)
        for i in range(coords.shape[0])
        for ax in range(3)
    ]
    grad = np.zeros((coords.shape[0], 3), dtype=np.float64)
    pool_kwargs: dict[str, Any] = {"max_workers": workers}
    if os.name == "posix":
        # macOS defaults to spawn, which cannot import heredoc/stdin
        # snippets. The worker is CPU-only and does not touch MLX state.
        pool_kwargs["mp_context"] = mp.get_context("fork")
    with ProcessPoolExecutor(**pool_kwargs) as ex:
        for i, ax, value in ex.map(_aes_fd_coord_worker, jobs):
            grad[i, ax] = value
    return grad


def _make_fast_fd_scalar(old_fd_scalar, fd_workers: int):
    def _fd_grad_scalar_fast(atoms, coords_ang, energy_fn, h=1e-3):
        code = getattr(energy_fn, "__code__", None)
        names = set(getattr(code, "co_names", ()))
        if "_aes_full_energy_at" not in names or energy_fn.__closure__ is None:
            return old_fd_scalar(atoms, coords_ang, energy_fn, h=h)

        closure = {
            name: cell.cell_contents
            for name, cell in zip(code.co_freevars, energy_fn.__closure__)
        }
        try:
            P_sao = closure["P_sao"]
            qsh = closure["qsh"]
            shell_atom = closure["shell_atom"]
        except KeyError:
            return old_fd_scalar(atoms, coords_ang, energy_fn, h=h)

        if CPP_AVAILABLE:
            return aes_gradient_frozen_density(
                list(atoms),
                coords_ang,
                P_sao,
                qsh,
                shell_atom,
                h_explicit=h,
            )

        if fd_workers <= 1:
            return old_fd_scalar(atoms, coords_ang, energy_fn, h=h)

        return _parallel_aes_fd_gradient(
            list(atoms), coords_ang, P_sao, qsh, shell_atom, h, fd_workers
        )

    return _fd_grad_scalar_fast


def gfn2_gradient_analytical_fast(
    atoms: list[int],
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    scf_kwargs: dict | None = None,
    fd_h_aux: float = 1e-3,
    fd_workers: int | None = None,
) -> dict:
    """Fast sidecar for :func:`gradient_gfn2.gfn2_gradient_analytical`.

    This swaps AES population/potential loops to vectorized float64
    implementations and uses :func:`gfn2_energy_fast` for the initial
    SCF state. The current AES and D4 finite-difference structure is
    otherwise unchanged.
    """

    if fd_workers is None:
        fd_workers = max(1, min(8, os.cpu_count() or 1))

    old_grad_energy = _grad.gfn2_energy
    old_fd_scalar = _grad._fd_grad_scalar
    old_scf_mmompop = _scf.mmompop
    old_scf_setvsdq = _scf.setvsdq
    old_aes_mmompop = _aes.mmompop
    old_aes_setvsdq = _aes.setvsdq
    old_scf_multipole_matrices = _scf.multipole_matrices
    old_multipole_matrices = _multipole_integrals.multipole_matrices
    _grad.gfn2_energy = gfn2_energy_fast
    _grad._fd_grad_scalar = _make_fast_fd_scalar(old_fd_scalar, fd_workers)
    _scf.mmompop = mmompop_vectorized
    _scf.setvsdq = setvsdq_vectorized
    _aes.mmompop = mmompop_vectorized
    _aes.setvsdq = setvsdq_vectorized
    if CPP_AVAILABLE:
        _scf.multipole_matrices = multipole_matrices_cpp
        _multipole_integrals.multipole_matrices = multipole_matrices_cpp
    try:
        return _grad.gfn2_gradient_analytical(
            atoms,
            coords_ang,
            charge=charge,
            scf_kwargs=scf_kwargs,
            fd_h_aux=fd_h_aux,
        )
    finally:
        _grad.gfn2_energy = old_grad_energy
        _grad._fd_grad_scalar = old_fd_scalar
        _scf.mmompop = old_scf_mmompop
        _scf.setvsdq = old_scf_setvsdq
        _aes.mmompop = old_aes_mmompop
        _aes.setvsdq = old_aes_setvsdq
        _scf.multipole_matrices = old_scf_multipole_matrices
        _multipole_integrals.multipole_matrices = old_multipole_matrices
