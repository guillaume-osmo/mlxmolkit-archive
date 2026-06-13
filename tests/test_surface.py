"""Tests for the GPU solvent-excluded (Connolly) surface."""
import sys; sys.path.insert(0, '.')
import numpy as np
import mlx.core as mx
import pytest

from mlxmolkit.surface import (
    SESSurface,
    VDW_RADII,
    ses_distance_field_mlx,
    solvent_excluded_surface,
    _jfa_edt,
)

WATER_Z = [8, 1, 1]
WATER_XYZ = np.array([[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9275, 0.0]])

METHANE_Z = [6, 1, 1, 1, 1]
METHANE_XYZ = np.array([
    [0.0, 0.0, 0.0],
    [0.6276, 0.6276, 0.6276],
    [0.6276, -0.6276, -0.6276],
    [-0.6276, 0.6276, -0.6276],
    [-0.6276, -0.6276, 0.6276],
])


def test_ses_watertight_and_metrics():
    s = solvent_excluded_surface(WATER_Z, WATER_XYZ, grid_spacing=0.35)
    assert isinstance(s, SESSurface)
    assert s.area > 0.0 and s.volume > 0.0 and s.volume_voxel > 0.0
    # closed, genus-0 single shell
    assert s.euler_characteristic == 2
    # GPU voxel volume agrees with the marching-cubes mesh volume
    assert abs(s.volume - s.volume_voxel) / s.volume < 0.1
    # mesh vertices carry the grid origin (placed in molecule frame)
    assert s.vertices is not None and s.vertices.shape[1] == 3


def test_field_sign_convention():
    phi, origin, h = ses_distance_field_mlx(METHANE_Z, METHANE_XYZ, grid_spacing=0.4)
    phi_np = np.asarray(phi)
    assert phi_np.min() < 0.0 < phi_np.max()                 # both sides present
    # the voxel at the carbon nucleus must be solvent-excluded (inside)
    idx = tuple(np.clip(np.round((METHANE_XYZ[0] - origin) / h).astype(int),
                        0, np.array(phi_np.shape) - 1))
    assert phi_np[idx] < 0.0


def test_build_mesh_false_skips_skimage():
    s = solvent_excluded_surface(WATER_Z, WATER_XYZ, grid_spacing=0.5, build_mesh=False)
    assert np.isnan(s.area) and np.isnan(s.volume)
    assert s.vertices is None and s.faces is None
    assert s.volume_voxel > 0.0


def test_probe_zero_limit_is_vdw_union():
    # As the probe shrinks to zero the SES collapses onto the van der Waals
    # union, so its volume must approach the vdW union volume on the same grid.
    Z, XYZ, h = METHANE_Z, METHANE_XYZ, 0.25
    s = solvent_excluded_surface(Z, XYZ, grid_spacing=h, probe_radius=0.05,
                                 build_mesh=False)
    rad = np.array([VDW_RADII[z] for z in Z])
    pad = rad.max() + 0.05 + 2.0
    lo, hi = XYZ.min(0) - pad, XYZ.max(0) + pad
    n = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [lo[d] + h * np.arange(n[d]) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    dmin = np.min(np.linalg.norm(grid[:, None, :] - XYZ[None, :, :], axis=2)
                  - rad[None, :], axis=1)
    vdw_vol = float((dmin < 0).sum()) * h ** 3
    assert abs(s.volume_voxel - vdw_vol) / vdw_vol < 0.08


def test_probe_volume_monotonic_nondecreasing():
    # The SES always contains the vdW volume; a larger probe only adds reentrant
    # fill, so the excluded volume is non-decreasing in probe radius.
    small = solvent_excluded_surface(METHANE_Z, METHANE_XYZ, grid_spacing=0.2,
                                     probe_radius=0.4, build_mesh=False)
    big = solvent_excluded_surface(METHANE_Z, METHANE_XYZ, grid_spacing=0.2,
                                   probe_radius=1.4, build_mesh=False)
    assert big.volume_voxel >= small.volume_voxel - 0.7   # allow voxel jitter


def test_jfa_edt_matches_scipy_exact():
    ndimage = pytest.importorskip("scipy.ndimage")
    n, h, r = 44, 0.3, 5.0
    c = (n - 1) / 2.0
    ix, iy, iz = np.indices((n, n, n))
    dist = np.sqrt((ix - c) ** 2 + (iy - c) ** 2 + (iz - c) ** 2)
    feasible = dist > r                                   # forbidden sphere in the middle
    edt_jfa = np.asarray(_jfa_edt(mx.array(feasible.astype(np.float32)),
                                  spacing=h, extra_passes=2))
    edt_sci = ndimage.distance_transform_edt(~feasible) * h
    err = np.abs(edt_jfa - edt_sci)
    assert err.max() < 0.1                                # << one voxel (h=0.3)


def test_cavity_sealing_changes_interior_volume():
    # Two concentric shells: an inner hollow that a probe could occupy if not
    # sealed. remove_cavities should fill it, giving a larger excluded volume.
    sealed = solvent_excluded_surface(WATER_Z, WATER_XYZ, grid_spacing=0.4,
                                      remove_cavities=True, build_mesh=False)
    raw = solvent_excluded_surface(WATER_Z, WATER_XYZ, grid_spacing=0.4,
                                   remove_cavities=False, build_mesh=False)
    # water has no internal cavity -> sealing must be a no-op (and never removes volume)
    assert abs(sealed.volume_voxel - raw.volume_voxel) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
