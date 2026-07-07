import numpy as np
import pytest

from mlxmolkit.connolly import bondi_vdw_radii, connolly_ses_surface_from_atoms_mlx


def test_connolly_ses_surface_mlx_returns_closed_water_mesh():
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")

    atoms = [8, 1, 1]
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.9584, 0.0, 0.0],
            [-0.2396, 0.9275, 0.0],
        ],
        dtype=np.float32,
    )

    surface = connolly_ses_surface_from_atoms_mlx(
        atoms,
        coords,
        probe_radius=1.4,
        grid_spacing=0.45,
        margin=1.5,
    )

    assert surface.vertices.shape[1] == 3
    assert surface.faces.shape[1] == 3
    assert surface.area > 10.0
    assert surface.volume > 1.0
    assert surface.euler_characteristic == 2
    assert surface.gpu_field_seconds >= 0.0


def test_bondi_vdw_radii_uses_default_for_unknown_atomic_numbers():
    radii = bondi_vdw_radii([1, 6, 999], default_radius=2.0)

    assert np.allclose(radii[:2], [1.2, 1.7], atol=1.0e-6)
    assert radii[2] == 2.0
