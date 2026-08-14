from pathlib import Path

import numpy as np
import pytest

from mlxmolkit.xtb.cosmo_sigma import cosmosegments_from_orcacosmo


BOHR = 0.52917721092


def test_orcacosmo_parser_units_indices_and_charges(tmp_path: Path) -> None:
    path = tmp_path / "mini.solute.orcacosmo"
    path.write_text(
        """
FINAL SINGLE POINT ENERGY     -75.123456789

#COSMO
2 # Number of atoms
3 # Number of surface points
100.0 # Volume
40.0 # Area
-0.0125 # CPCM dielectric energy

# CARTESIAN COORDINATES (A.U.) + RADII (A.U.) + ATOMIC NUMBER
0.0 0.0 0.0 1.20 8
1.0 0.0 0.0 0.80 1

# SURFACE POINTS (A.U.)
# X Y Z area potential charge w_leb Switch_F G_width atom
0.1 0.2 0.3 2.0 -0.40 0.020 1.0 1.0 0.50 0
0.4 0.5 0.6 3.0 -0.30 -0.030 1.0 1.0 0.50 1
0.7 0.8 0.9 5.0 -0.20 0.010 1.0 1.0 0.50 0

#COSMO_corrected
""".lstrip()
    )

    cosmo = cosmosegments_from_orcacosmo(path)

    assert cosmo.total_energy_hartree == pytest.approx(-75.123456789)
    assert cosmo.dielectric_energy_hartree == pytest.approx(-0.0125)
    assert cosmo.atom_z == [8, 1]
    np.testing.assert_allclose(cosmo.atom_coords_bohr, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(cosmo.atom_radii, np.array([1.20, 0.80]) * BOHR)

    assert cosmo.area == pytest.approx(40.0 * BOHR**2)
    assert cosmo.volume == pytest.approx(100.0 * BOHR**3)
    np.testing.assert_array_equal(cosmo.segments_atom, [1, 2, 1])
    np.testing.assert_allclose(cosmo.segments_xyz_bohr, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    np.testing.assert_allclose(cosmo.segments_area, np.array([2.0, 3.0, 5.0]) * BOHR**2)
    np.testing.assert_allclose(cosmo.segments_charge, [0.020, -0.030, 0.010])
    np.testing.assert_allclose(cosmo.segments_sigma, cosmo.segments_charge / cosmo.segments_area)
    assert cosmo.total_screening_charge == pytest.approx(0.0)
    np.testing.assert_allclose(cosmo.segments_potential, [-0.40, -0.30, -0.20])
    assert "#COSMO" in cosmo.cosmo_text


def test_orcacosmo_parser_rejects_missing_cosmo_block(tmp_path: Path) -> None:
    path = tmp_path / "bad.orcacosmo"
    path.write_text("FINAL SINGLE POINT ENERGY -1.0\n")

    with pytest.raises(ValueError, match="no #COSMO block"):
        cosmosegments_from_orcacosmo(path)
