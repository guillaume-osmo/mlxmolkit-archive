"""Solvent-excluded (Connolly) molecular surface on the GPU.

The solvent-excluded surface (SES, a.k.a. the Connolly or molecular surface)
is the boundary swept by a probe sphere of radius ``r_probe`` (water ~1.4 A)
rolled over the van der Waals spheres of the atoms. This module computes it as
the zero level set of a signed distance field and extracts a watertight mesh
by marching cubes.

The whole *distance field* — the only part that scales as O(grid x atoms) and
was the bottleneck — runs entirely on the Metal GPU through MLX:

  1. SAS field      D(x) = min_i(|x - c_i| - (r_i + r_probe))        (broadcast min)
  2. cavity sealing flood-fill of the probe-feasible region from the box
     boundary, so buried pockets do not punch spurious holes             (iter dilation)
  3. EDT            Euclidean distance to the nearest feasible voxel via the
     Jump Flooding Algorithm (Rong & Tan 2006)                           (log2(N) gather passes)
  4. SES field      phi(x) = r_probe - EDT(x);  phi < 0 inside.

Marching-cubes mesh *extraction* (step 5) uses scikit-image on the CPU — that
is mesh extraction, not the distance computation, and runs on a thin band of
voxels. A pure-GPU enclosed volume (voxel count) is returned regardless, so
``build_mesh=False`` needs no scikit-image at all.

Distinct from :func:`mlxmolkit.esp_resp.connolly_surface_grid`, which returns
a Fibonacci point cloud on scaled vdW shells for ESP/RESP fitting rather than a
meshed surface with area and volume.

Reference: M. L. Connolly, *J. Appl. Crystallogr.* 16, 548 (1983);
G. Rong, T.-S. Tan, *Jump flooding in GPU* (I3D 2006).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import mlx.core as mx

# Bondi/standard van der Waals radii (Angstrom). Mirrors
# ``mlxmolkit.esp_resp.VDW_RADII``; inlined so this module stays import-light.
VDW_RADII = {
    1: 1.20, 2: 1.40, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52,
    9: 1.47, 10: 1.54, 11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80,
    16: 1.80, 17: 1.75, 18: 1.88, 35: 1.85, 53: 1.98,
}

__all__ = [
    "SESSurface",
    "ses_distance_field_mlx",
    "solvent_excluded_surface",
    "solvent_excluded_surface_from_rdkit_mol",
]

DEFAULT_PROBE_RADIUS = 1.4
DEFAULT_GRID_SPACING = 0.3
_DEFAULT_RADIUS = 1.80


@dataclass
class SESSurface:
    """A solvent-excluded surface and its scalar metrics.

    Attributes
    ----------
    area : surface area in A^2 (marching-cubes mesh; ``nan`` if no mesh built).
    volume : enclosed volume in A^3 (marching-cubes mesh; ``nan`` if no mesh).
    volume_voxel : enclosed volume in A^3 from the GPU voxel count (always set).
    vertices, faces : mesh arrays (``None`` if ``build_mesh=False``).
    origin, grid_spacing, grid_shape : grid geometry of the distance field.
    euler_characteristic : V - E + F of the mesh (2 per closed genus-0 shell).
    probe_radius : probe radius used (A).
    backend : compute backend tag.
    """

    area: float
    volume: float
    volume_voxel: float
    probe_radius: float
    grid_spacing: float
    grid_shape: tuple[int, int, int]
    origin: np.ndarray
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    euler_characteristic: int | None = None
    backend: str = "mlx_jfa"


# --------------------------------------------------------------------------- #
# GPU primitives                                                              #
# --------------------------------------------------------------------------- #
def _shift(a: mx.array, d: int, axis: int, fill: float) -> mx.array:
    """Shift ``a`` along ``axis`` so result[i] = a[i + d], out-of-range -> fill."""
    if d == 0:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    sl: list = [slice(None)] * a.ndim
    if d > 0:
        pad[axis] = (0, d)
        sl[axis] = slice(d, d + n)
    else:
        pad[axis] = (-d, 0)
        sl[axis] = slice(0, n)
    return mx.pad(a, pad, constant_values=fill)[tuple(sl)]


def _shift3(a: mx.array, dx: int, dy: int, dz: int, fill: float) -> mx.array:
    return _shift(_shift(_shift(a, dx, 0, fill), dy, 1, fill), dz, 2, fill)


def _index_grids(shape: tuple[int, int, int]) -> tuple[mx.array, mx.array, mx.array]:
    nx, ny, nz = shape
    ix = mx.broadcast_to(mx.arange(nx, dtype=mx.float32)[:, None, None], shape)
    iy = mx.broadcast_to(mx.arange(ny, dtype=mx.float32)[None, :, None], shape)
    iz = mx.broadcast_to(mx.arange(nz, dtype=mx.float32)[None, None, :], shape)
    return ix, iy, iz


def _outer_feasible_mask(feasible: mx.array, max_iter: int | None = None) -> mx.array:
    """Probe-feasible voxels reachable from the box boundary (6-connectivity).

    Geodesic flood-fill by iterated masked dilation — fully on GPU. Internal
    cavities (feasible components not touching the boundary) are dropped, so
    buried solvent pockets become solvent-excluded interior.
    """
    nx, ny, nz = feasible.shape
    feas = (feasible > 0.5).astype(mx.float32)
    ix, iy, iz = _index_grids(feasible.shape)
    boundary = (
        (ix == 0) | (ix == nx - 1)
        | (iy == 0) | (iy == ny - 1)
        | (iz == 0) | (iz == nz - 1)
    ).astype(mx.float32)
    outer = feas * boundary
    if max_iter is None:
        max_iter = nx + ny + nz
    prev_sum = -1.0
    for it in range(max_iter):
        dil = outer
        dil = mx.maximum(dil, _shift(outer, 1, 0, 0.0))
        dil = mx.maximum(dil, _shift(outer, -1, 0, 0.0))
        dil = mx.maximum(dil, _shift(outer, 1, 1, 0.0))
        dil = mx.maximum(dil, _shift(outer, -1, 1, 0.0))
        dil = mx.maximum(dil, _shift(outer, 1, 2, 0.0))
        dil = mx.maximum(dil, _shift(outer, -1, 2, 0.0))
        outer = feas * dil
        if it % 4 == 0:                              # bounded host syncs
            cur = float(mx.sum(outer).item())
            if cur == prev_sum:
                break
            prev_sum = cur
    return outer > 0.5


def _jfa_edt(seed_mask: mx.array, spacing: float, extra_passes: int = 2) -> mx.array:
    """Euclidean distance from every voxel to the nearest ``seed_mask`` voxel.

    Jump Flooding Algorithm: each voxel propagates its nearest-seed coordinate
    to neighbours at geometrically shrinking step sizes, so the transform
    converges in ~log2(N) fully-parallel passes. ``extra_passes`` step-1 passes
    at the end (JFA+k) remove the rare propagation gaps of plain JFA.
    """
    shape = seed_mask.shape
    nx, ny, nz = shape
    big = float(nx + ny + nz) * 4.0
    ix, iy, iz = _index_grids(shape)
    is_seed = seed_mask > 0.5
    big_arr = mx.array(big, dtype=mx.float32)
    sx = mx.where(is_seed, ix, big_arr)
    sy = mx.where(is_seed, iy, big_arr)
    sz = mx.where(is_seed, iz, big_arr)
    best = mx.where(is_seed, mx.zeros(shape, dtype=mx.float32),
                    mx.full(shape, big * big, dtype=mx.float32))

    K = 1
    while K < max(shape):
        K *= 2
    steps: list[int] = []
    k = K // 2
    while k >= 1:
        steps.append(k)
        k //= 2
    steps.extend([1] * extra_passes)

    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if not (dx == 0 and dy == 0 and dz == 0)]
    for k in steps:
        for dx, dy, dz in offsets:
            sxn = _shift3(sx, dx * k, dy * k, dz * k, big)
            syn = _shift3(sy, dx * k, dy * k, dz * k, big)
            szn = _shift3(sz, dx * k, dy * k, dz * k, big)
            cand = (ix - sxn) ** 2 + (iy - syn) ** 2 + (iz - szn) ** 2
            better = cand < best
            best = mx.where(better, cand, best)
            sx = mx.where(better, sxn, sx)
            sy = mx.where(better, syn, sy)
            sz = mx.where(better, szn, sz)
        mx.eval(sx, sy, sz, best)                    # bound graph size per pass
    return mx.sqrt(best) * spacing


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def _radii_for_atoms(atoms: Sequence[int]) -> np.ndarray:
    return np.array([VDW_RADII.get(int(z), _DEFAULT_RADIUS) for z in atoms], dtype=np.float64)


def ses_distance_field_mlx(
    atoms: Sequence[int],
    coords: Sequence[Sequence[float]] | np.ndarray,
    *,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    margin: float = 2.0,
    remove_cavities: bool = True,
    jfa_extra_passes: int = 2,
    point_chunk: int = 300_000,
) -> tuple[mx.array, np.ndarray, float]:
    """Signed SES distance field on a regular grid — 100% GPU (MLX/Metal).

    Returns ``(phi, origin, grid_spacing)`` where ``phi`` is an ``mx.array`` of
    shape ``(nx, ny, nz)`` with ``phi < 0`` inside the molecule (solvent
    excluded) and the Connolly surface at ``phi == 0``. ``origin`` is the
    physical coordinate of voxel ``(0, 0, 0)``.
    """
    coord_array = np.asarray(coords, dtype=np.float64)
    if coord_array.ndim != 2 or coord_array.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if len(atoms) != coord_array.shape[0]:
        raise ValueError("atoms and coords must have the same length")
    radii = _radii_for_atoms(atoms)
    rp = float(probe_radius)
    h = float(grid_spacing)

    pad = float(radii.max()) + rp + float(margin)
    lo = coord_array.min(0) - pad
    hi = coord_array.max(0) + pad
    nx, ny, nz = (np.ceil((hi - lo) / h).astype(int) + 1)
    nx, ny, nz = int(nx), int(ny), int(nz)
    shape = (nx, ny, nz)

    axes = [lo[d] + h * np.arange(n) for d, n in enumerate(shape)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
    n_pts = pts.shape[0]

    # --- GPU: SAS distance field, chunked over grid points ---
    centers = mx.array(coord_array.astype(np.float32))
    r_inflated = mx.array((radii + rp).astype(np.float32))
    dsas_parts = []
    for s in range(0, n_pts, point_chunk):
        block = mx.array(pts[s:s + point_chunk])                       # (m, 3)
        d = mx.sqrt(mx.sum((block[:, None, :] - centers[None, :, :]) ** 2, axis=2))
        dsas_parts.append(mx.min(d - r_inflated[None, :], axis=1))
    dsas = mx.concatenate(dsas_parts, axis=0).reshape(shape)

    feasible = dsas >= 0.0                                             # probe centre allowed
    if remove_cavities:
        seed = _outer_feasible_mask(feasible)
    else:
        seed = feasible
    edt = _jfa_edt(seed, spacing=h, extra_passes=jfa_extra_passes)
    phi = mx.array(rp, dtype=mx.float32) - edt
    mx.eval(phi)
    return phi, lo, h


def solvent_excluded_surface(
    atoms: Sequence[int],
    coords: Sequence[Sequence[float]] | np.ndarray,
    *,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    margin: float = 2.0,
    remove_cavities: bool = True,
    build_mesh: bool = True,
    jfa_extra_passes: int = 2,
    point_chunk: int = 300_000,
) -> SESSurface:
    """Compute the solvent-excluded (Connolly) surface of a molecule.

    The distance field is built on the GPU; with ``build_mesh=True`` a watertight
    triangle mesh is extracted by marching cubes (scikit-image) and its area and
    volume are returned. ``build_mesh=False`` skips scikit-image entirely and
    returns only the GPU voxel volume.
    """
    phi, origin, h = ses_distance_field_mlx(
        atoms, coords,
        probe_radius=probe_radius, grid_spacing=grid_spacing, margin=margin,
        remove_cavities=remove_cavities, jfa_extra_passes=jfa_extra_passes,
        point_chunk=point_chunk,
    )
    shape = tuple(int(s) for s in phi.shape)

    # Pure-GPU enclosed volume: count solvent-excluded voxels.
    volume_voxel = float(mx.sum((phi < 0.0).astype(mx.float32)).item()) * (h ** 3)

    area = float("nan")
    volume = float("nan")
    vertices = faces = None
    euler = None
    if build_mesh:
        try:
            from skimage import measure
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "marching-cubes meshing needs scikit-image; "
                "`pip install scikit-image`, or call with build_mesh=False."
            ) from exc
        phi_np = np.asarray(phi, dtype=np.float32)
        if phi_np.min() < 0.0 < phi_np.max():
            verts, faces_arr, _, _ = measure.marching_cubes(phi_np, level=0.0, spacing=(h, h, h))
            vertices = verts + origin[None, :]
            faces = faces_arr
            area = float(measure.mesh_surface_area(verts, faces_arr))
            tri = verts[faces_arr]
            volume = float(abs(
                np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
            ))
            euler = int(len(verts) - len(faces_arr) * 3 // 2 + len(faces_arr))
        else:
            area = 0.0
            volume = 0.0

    return SESSurface(
        area=area,
        volume=volume,
        volume_voxel=volume_voxel,
        probe_radius=float(probe_radius),
        grid_spacing=h,
        grid_shape=shape,
        origin=np.asarray(origin, dtype=np.float64),
        vertices=vertices,
        faces=faces,
        euler_characteristic=euler,
        backend="mlx_jfa",
    )


def solvent_excluded_surface_from_rdkit_mol(mol, *, conf_id: int = -1, **kwargs) -> SESSurface:
    """Connolly surface from an RDKit molecule that already has a 3D conformer."""
    if mol is None:
        raise ValueError("mol is None")
    conf = mol.GetConformer(conf_id)
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    coords = np.array(conf.GetPositions(), dtype=np.float64)
    return solvent_excluded_surface(atoms, coords, **kwargs)
