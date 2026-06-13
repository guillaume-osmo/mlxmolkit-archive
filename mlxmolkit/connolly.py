"""GPU-assisted Connolly / solvent-excluded molecular surfaces.

This module computes a grid solvent-excluded surface (SES): the boundary of
the volume that a solvent probe cannot enter while rolling over vdW spheres.
The expensive ``grid points x atoms`` distance field is evaluated with MLX on
Metal; exact Euclidean distance transform and marching cubes are delegated to
SciPy/scikit-image when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import mlx.core as mx

from mlxmolkit.esp_resp import VDW_RADII


@dataclass(frozen=True)
class ConnollySurfaceResult:
    """Mesh and diagnostics for a solvent-excluded surface."""

    vertices: np.ndarray
    faces: np.ndarray
    area: float
    volume: float
    euler_characteristic: int
    grid_shape: tuple[int, int, int]
    n_voxels: int
    origin: np.ndarray
    spacing: float
    probe_radius: float
    gpu_field_seconds: float


def bondi_vdw_radii(atoms: Sequence[int], *, default_radius: float = 1.70) -> np.ndarray:
    """Return Bondi-like vdW radii for atomic numbers."""

    return np.asarray([VDW_RADII.get(int(z), float(default_radius)) for z in atoms], dtype=np.float32)


def connolly_ses_surface_mlx(
    coords: Any,
    radii: Sequence[float] | np.ndarray,
    *,
    probe_radius: float = 1.4,
    grid_spacing: float = 0.3,
    margin: float = 2.0,
    chunk_size: int = 300_000,
    remove_cavities: bool = True,
) -> ConnollySurfaceResult:
    """Compute a solvent-excluded Connolly surface mesh.

    Parameters
    ----------
    coords
        Atomic coordinates with shape ``(n_atoms, 3)`` in Angstrom.
    radii
        vdW radii, one per atom, in Angstrom.
    probe_radius
        Solvent probe radius, normally ``1.4`` A for water.
    grid_spacing
        Cartesian grid spacing in Angstrom. ``0.2``-``0.3`` is a practical
        accuracy/speed range for drug-like molecules.
    margin
        Extra padding outside inflated atoms.
    chunk_size
        Number of grid points per MLX distance-field chunk.
    remove_cavities
        If true, only the probe-center region connected to the grid boundary is
        treated as solvent, which seals internal cavities and avoids spurious
        inner surface sheets.
    """

    if grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive")
    if probe_radius <= 0:
        raise ValueError("probe_radius must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    try:
        from scipy import ndimage
        from skimage import measure
    except ImportError as exc:  # pragma: no cover - depends on optional extras.
        raise ImportError(
            "connolly_ses_surface_mlx requires scipy and scikit-image for EDT/marching cubes"
        ) from exc

    xyz = np.asarray(coords, dtype=np.float32)
    rad = np.asarray(radii, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if rad.shape != (xyz.shape[0],):
        raise ValueError("radii must have one value per atom")
    if len(rad) == 0:
        raise ValueError("at least one atom is required")

    spacing = float(grid_spacing)
    probe = float(probe_radius)
    pad = float(np.max(rad)) + probe + float(margin)
    origin = np.min(xyz, axis=0) - pad
    upper = np.max(xyz, axis=0) + pad
    grid_shape_np = np.ceil((upper - origin) / spacing).astype(np.int64) + 1
    grid_shape = tuple(int(v) for v in grid_shape_np)
    axes = [origin[axis] + spacing * np.arange(n, dtype=np.float32) for axis, n in enumerate(grid_shape)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32, copy=False)

    dsas = _sas_distance_field_mlx(
        points,
        xyz,
        rad + probe,
        chunk_size=int(chunk_size),
    ).reshape(grid_shape)

    infeasible = dsas < 0.0
    if remove_cavities:
        feasible = ~infeasible
        labels, _ = ndimage.label(feasible)
        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0].ravel(),
                    labels[-1].ravel(),
                    labels[:, 0].ravel(),
                    labels[:, -1].ravel(),
                    labels[:, :, 0].ravel(),
                    labels[:, :, -1].ravel(),
                ]
            )
        )
        border_labels = border_labels[border_labels != 0]
        outer_feasible = np.isin(labels, border_labels)
        infeasible = ~outer_feasible

    edt = ndimage.distance_transform_edt(infeasible) * spacing
    phi = probe - edt
    vertices, faces, _, _ = measure.marching_cubes(phi, level=0.0, spacing=(spacing, spacing, spacing))
    vertices = vertices.astype(np.float32, copy=False) + origin[None, :].astype(np.float32)
    faces = faces.astype(np.int32, copy=False)
    area = float(measure.mesh_surface_area(vertices, faces))
    volume = float(_mesh_volume(vertices, faces))
    euler = int(_mesh_euler_characteristic(faces, n_vertices=len(vertices)))

    return ConnollySurfaceResult(
        vertices=vertices,
        faces=faces,
        area=area,
        volume=volume,
        euler_characteristic=euler,
        grid_shape=grid_shape,
        n_voxels=int(points.shape[0]),
        origin=origin.astype(np.float32),
        spacing=spacing,
        probe_radius=probe,
        gpu_field_seconds=float(_sas_distance_field_mlx.last_seconds),
    )


def connolly_ses_surface_from_atoms_mlx(
    atoms: Sequence[int],
    coords: Any,
    *,
    probe_radius: float = 1.4,
    grid_spacing: float = 0.3,
    margin: float = 2.0,
    chunk_size: int = 300_000,
    remove_cavities: bool = True,
    default_radius: float = 1.70,
) -> ConnollySurfaceResult:
    """Compute SES mesh from atomic numbers and coordinates."""

    return connolly_ses_surface_mlx(
        coords,
        bondi_vdw_radii(atoms, default_radius=default_radius),
        probe_radius=probe_radius,
        grid_spacing=grid_spacing,
        margin=margin,
        chunk_size=chunk_size,
        remove_cavities=remove_cavities,
    )


def rdkit_mol_to_atoms_coords_radii(mol: Any, *, conf_id: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract atomic numbers, coordinates, and vdW radii from an RDKit molecule."""

    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit molecule has no conformer")
    atoms = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
    coords = np.asarray(mol.GetConformer(int(conf_id)).GetPositions(), dtype=np.float32)
    return atoms, coords, bondi_vdw_radii(atoms)


def _sas_distance_field_mlx(
    points: np.ndarray,
    coords: np.ndarray,
    inflated_radii: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    import time

    start = time.perf_counter()
    point_array = np.asarray(points, dtype=np.float32)
    coord_mx = mx.array(np.asarray(coords, dtype=np.float32))
    radius_mx = mx.array(np.asarray(inflated_radii, dtype=np.float32))
    out = np.empty((point_array.shape[0],), dtype=np.float32)
    for start_idx in range(0, point_array.shape[0], int(chunk_size)):
        end_idx = min(point_array.shape[0], start_idx + int(chunk_size))
        point_mx = mx.array(point_array[start_idx:end_idx])
        dist = mx.sqrt(mx.sum((point_mx[:, None, :] - coord_mx[None, :, :]) ** 2, axis=2))
        out[start_idx:end_idx] = np.asarray(mx.min(dist - radius_mx[None, :], axis=1), dtype=np.float32)
    mx.eval(coord_mx, radius_mx)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    _sas_distance_field_mlx.last_seconds = time.perf_counter() - start
    return out


_sas_distance_field_mlx.last_seconds = 0.0  # type: ignore[attr-defined]


def _mesh_euler_characteristic(faces: np.ndarray, *, n_vertices: int) -> int:
    edge_set: set[tuple[int, int]] = set()
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        edge_set.add(tuple(sorted((a, b))))
        edge_set.add(tuple(sorted((b, c))))
        edge_set.add(tuple(sorted((c, a))))
    return int(n_vertices - len(edge_set) + len(faces))


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    return abs(float(np.sum(signed) / 6.0))
