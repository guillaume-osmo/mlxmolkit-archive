"""Surface point-cloud utilities for openCHEESE.

LG-Mol stores each molecule's electrostatic surface as an ``(n, 4)`` array:
``x, y, z, ESP``. This module builds the same object with the local open stack:
MLX-assisted Connolly SES meshes for shape and MLX/Metal point-charge ESP for
the electrostatic channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import mlx.core as mx
import numpy as np

from mlxmolkit.connolly import ConnollySurfaceResult, connolly_ses_surface_from_atoms_mlx
from mlxmolkit.esp_resp import connolly_surface_grid
from opencheese.descriptors import electrostatic_potential_on_grid_metal


SurfaceMethod = Literal["ses", "shell"]
SurfaceSampling = Literal["none", "first", "random", "farthest"]


@dataclass(frozen=True)
class SurfaceCloudResult:
    """LG-Mol-style surface point cloud and diagnostics."""

    cloud: np.ndarray
    points: np.ndarray
    esp_values: np.ndarray
    sampling_indices: np.ndarray
    surface_area: float = float("nan")
    surface_volume: float = float("nan")
    euler_characteristic: int = 0
    n_surface_vertices: int = 0
    gpu_field_seconds: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


def lgmol_surface_cloud_from_atoms(
    atoms: Sequence[int],
    coords: Any,
    charges: Sequence[float] | np.ndarray | None = None,
    *,
    method: SurfaceMethod = "ses",
    points_num: int | None = 200,
    sampling: SurfaceSampling = "farthest",
    random_seed: int = 0,
    center: bool = False,
    min_distance: float = 1.0e-4,
    probe_radius: float = 1.4,
    grid_spacing: float = 0.35,
    margin: float = 2.0,
    chunk_size: int = 300_000,
    remove_cavities: bool = True,
    shell_factors: Sequence[float] = (1.4, 1.6, 1.8, 2.0),
    point_density: float = 0.35,
    min_points_per_shell: int = 16,
    max_points_per_shell: int = 128,
) -> SurfaceCloudResult:
    """Build an ``[x, y, z, ESP]`` surface cloud.

    Parameters
    ----------
    atoms, coords, charges
        Atomic numbers, Cartesian coordinates in Angstrom, and atom-centered
        partial charges. Missing charges are treated as zero for shape-only
        ablations.
    method
        ``"ses"`` uses the MLX Connolly solvent-excluded surface. ``"shell"``
        uses the older MK-style shell grid and is useful for very fast tests.
    points_num
        If provided, sample exactly this many points. When more points are
        requested than available, points are repeated deterministically.
    sampling
        ``"farthest"`` gives better surface coverage than LG-Mol's random
        sampling while preserving the same output contract.
    center
        If true, subtract the sampled point centroid from the returned xyz
        columns. ESP is always evaluated on the original coordinates.
    """

    atom_array = np.asarray(atoms, dtype=np.int32)
    coord_array = np.asarray(coords, dtype=np.float32)
    if coord_array.shape != (len(atom_array), 3):
        raise ValueError(f"coords must have shape ({len(atom_array)}, 3)")
    if charges is None:
        charge_array = np.zeros((len(atom_array),), dtype=np.float32)
    else:
        charge_array = np.asarray(charges, dtype=np.float32)
    if charge_array.shape != (len(atom_array),):
        raise ValueError(f"charges must have shape ({len(atom_array)},)")

    surface: ConnollySurfaceResult | None = None
    if method == "ses":
        surface = connolly_ses_surface_from_atoms_mlx(
            atom_array,
            coord_array,
            probe_radius=probe_radius,
            grid_spacing=grid_spacing,
            margin=margin,
            chunk_size=chunk_size,
            remove_cavities=remove_cavities,
        )
        raw_points = np.asarray(surface.vertices, dtype=np.float32)
    elif method == "shell":
        raw_points = connolly_surface_grid(
            atom_array,
            coord_array,
            shell_factors=shell_factors,
            point_density=point_density,
            min_points_per_shell=min_points_per_shell,
            max_points_per_shell=max_points_per_shell,
        ).astype(np.float32)
    else:
        raise ValueError(f"unknown surface method: {method!r}")

    if raw_points.ndim != 2 or raw_points.shape[1] != 3 or raw_points.shape[0] == 0:
        raise ValueError("surface generation produced no points")

    indices = surface_sample_indices(raw_points, points_num, mode=sampling, random_seed=random_seed)
    sampled_points = raw_points[indices].astype(np.float32, copy=False)
    esp_mx = electrostatic_potential_on_grid_metal(
        coord_array[None, :, :],
        charge_array[None, :],
        sampled_points,
        min_distance=min_distance,
    )
    mx.eval(esp_mx)
    esp = np.asarray(esp_mx[0], dtype=np.float32)
    out_points = sampled_points - sampled_points.mean(axis=0, keepdims=True) if center else sampled_points
    cloud = np.column_stack([out_points, esp]).astype(np.float32, copy=False)

    return SurfaceCloudResult(
        cloud=cloud,
        points=out_points.astype(np.float32, copy=False),
        esp_values=esp,
        sampling_indices=indices.astype(np.int64, copy=False),
        surface_area=float(surface.area) if surface is not None else float("nan"),
        surface_volume=float(surface.volume) if surface is not None else float("nan"),
        euler_characteristic=int(surface.euler_characteristic) if surface is not None else 0,
        n_surface_vertices=int(len(raw_points)),
        gpu_field_seconds=float(surface.gpu_field_seconds) if surface is not None else 0.0,
        metadata={
            "format": "opencheese.lgmol_surface_cloud",
            "method": method,
            "points_num": None if points_num is None else int(points_num),
            "sampling": sampling,
            "center": bool(center),
        },
    )


def surface_sample_indices(
    points: Any,
    points_num: int | None,
    *,
    mode: SurfaceSampling = "farthest",
    random_seed: int = 0,
) -> np.ndarray:
    """Return indices for a fixed-size surface cloud."""

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")
    n_points = int(pts.shape[0])
    if n_points == 0:
        raise ValueError("cannot sample an empty point cloud")
    if points_num is None or int(points_num) <= 0:
        return np.arange(n_points, dtype=np.int64)
    target = int(points_num)

    if mode == "none":
        base = np.arange(min(target, n_points), dtype=np.int64)
    elif mode == "first":
        base = np.arange(min(target, n_points), dtype=np.int64)
    elif mode == "random":
        rng = np.random.default_rng(int(random_seed))
        replace = target > n_points
        return rng.choice(n_points, size=target, replace=replace).astype(np.int64)
    elif mode == "farthest":
        base = _farthest_point_indices(pts, min(target, n_points))
    else:
        raise ValueError(f"unknown surface sampling mode: {mode!r}")

    if len(base) == target:
        return base.astype(np.int64, copy=False)
    rng = np.random.default_rng(int(random_seed))
    extra = rng.choice(base, size=target - len(base), replace=True)
    return np.concatenate([base, extra]).astype(np.int64, copy=False)


def _farthest_point_indices(points: np.ndarray, n_select: int) -> np.ndarray:
    n_points = int(points.shape[0])
    if n_select <= 0:
        return np.empty((0,), dtype=np.int64)
    centroid = points.mean(axis=0, keepdims=True)
    selected = np.empty((n_select,), dtype=np.int64)
    selected[0] = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    min_dist2 = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for out_index in range(1, n_select):
        selected[out_index] = int(np.argmax(min_dist2))
        dist2 = np.sum((points - points[selected[out_index]]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return selected


__all__ = [
    "SurfaceCloudResult",
    "lgmol_surface_cloud_from_atoms",
    "surface_sample_indices",
]
