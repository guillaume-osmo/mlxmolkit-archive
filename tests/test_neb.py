"""Tests for mlxmolkit.nddo.neb — NEB barriers and tautomer atom mapping."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mlxmolkit.nddo import neb


def _embed(smiles, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def test_kabsch_align_undoes_rotation_and_translation():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(8, 3))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0]])
    moved = target @ rot.T + np.array([3.0, -1.0, 2.0])
    recovered = neb.kabsch_align(moved, target)
    assert np.allclose(recovered, target, atol=1e-9)


def test_kabsch_align_does_not_mirror():
    """A reflection must not be used to fit; that would invert stereochemistry."""
    rng = np.random.default_rng(1)
    target = rng.normal(size=(6, 3))
    mirrored = target * np.array([1.0, 1.0, -1.0])
    recovered = neb.kabsch_align(mirrored, target)
    # a proper rotation cannot recover a mirror image, so a residual must remain
    assert not np.allclose(recovered, target, atol=1e-6)


def test_interpolate_path_endpoints_and_count():
    start = np.zeros((4, 3))
    end = np.ones((4, 3))
    band = neb.interpolate_path(start, end, 5)
    assert len(band) == 5
    assert np.allclose(band[0], start)
    # the far endpoint is aligned onto the near one first, so compare against that
    assert np.allclose(band[-1], neb.kabsch_align(end, start))
    spacing = [np.linalg.norm(band[i + 1] - band[i]) for i in range(4)]
    assert np.allclose(spacing, spacing[0])


def test_interpolate_path_rejects_degenerate_band():
    with pytest.raises(ValueError):
        neb.interpolate_path(np.zeros((2, 3)), np.ones((2, 3)), 2)


def test_improved_tangent_points_uphill_on_monotonic_rise():
    images = [np.zeros((1, 3)), np.array([[1.0, 0, 0]]), np.array([[2.0, 0, 0]])]
    tau = neb._improved_tangent(images, np.array([0.0, 1.0, 2.0]), 1)
    assert np.allclose(tau, np.array([[1.0, 0, 0]]))


# --------------------------------------------------------------------------------------
# tautomer atom mapping
# --------------------------------------------------------------------------------------
def test_map_tautomer_atoms_keto_enol():
    """Acetaldehyde and vinyl alcohol differ by one H moving from carbon to oxygen."""
    mol_a, mol_b = _embed("CC=O"), _embed("C=CO")
    perm = neb.map_tautomer_atoms(mol_a, mol_b)
    assert perm is not None
    assert sorted(perm) == list(range(mol_a.GetNumAtoms())), "mapping must be a permutation"
    elements_a = [a.GetAtomicNum() for a in mol_a.GetAtoms()]
    elements_b = [mol_b.GetAtomWithIdx(perm[i]).GetAtomicNum() for i in range(len(perm))]
    assert elements_a == elements_b, "mapped atoms must be the same elements"


def test_map_tautomer_atoms_alpha_beta_ionone():
    """The case this module exists for: an H moving between two ring carbons."""
    mol_a = _embed("CC(=O)C=CC1C(C)=CCCC1(C)C")
    mol_b = _embed("CC(=O)C=CC1=C(C)CCCC1(C)C")
    perm = neb.map_tautomer_atoms(mol_a, mol_b)
    assert perm is not None
    assert sorted(perm) == list(range(mol_a.GetNumAtoms()))
    elements_a = [a.GetAtomicNum() for a in mol_a.GetAtoms()]
    elements_b = [mol_b.GetAtomWithIdx(perm[i]).GetAtomicNum() for i in range(len(perm))]
    assert elements_a == elements_b


def test_map_tautomer_atoms_rejects_non_pair():
    """Different heavy-atom skeletons are not a tautomer pair."""
    assert neb.map_tautomer_atoms(_embed("CC=O"), _embed("CCC=O")) is None


def test_tautomer_barrier_rejects_non_pair():
    with pytest.raises(ValueError):
        neb.tautomer_barrier("CC=O", "CCC=O", method="AM1")


# --------------------------------------------------------------------------------------
# end-to-end NEB (slow: a real SCF per image per iteration)
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_neb_finds_a_barrier_between_formic_acid_rotamers():
    """A cheap 5-atom H-transfer band must produce a positive, finite barrier."""
    mol_a, mol_b = _embed("OC=O"), _embed("O=CO")
    perm = neb.map_tautomer_atoms(mol_a, mol_b)
    assert perm is not None
    atoms = [a.GetAtomicNum() for a in mol_a.GetAtoms()]
    pos_a = mol_a.GetConformer().GetPositions()
    pos_b = mol_b.GetConformer().GetPositions()
    coords_b = np.array([pos_b[perm[i]] for i in range(len(perm))])

    res = neb.estimate_barrier(atoms, pos_a, coords_b, method="AM1", n_images=5,
                               max_iter=15, optimize_endpoints=False)
    assert np.isfinite(res["barrier_forward_kcal"])
    assert res["barrier_forward_kcal"] >= 0.0
    assert res["energies_kcal"][0] == pytest.approx(0.0)
    assert 0 <= res["ts_index"] < 5
    assert len(res["images"]) == 5


@pytest.mark.slow
def test_neb_keto_enol_barrier_is_high_in_vacuum():
    """Gas-phase intramolecular keto-enol is NOT a low-barrier process.

    This is the result that stops a vacuum barrier being used on its own to decide
    which tautomers to merge: uncatalyzed keto-enol in vacuum is ~58 kcal/mol at PM6
    (~68 at CCSD(T)), i.e. locked, just like a carbon-to-carbon shift. A discriminating
    barrier needs an explicit proton shuttle in the band.
    """
    res = neb.tautomer_barrier("CC=O", "C=CO", method="PM6", n_images=7,
                               max_iter=60, force_tol=0.08)
    assert res["barrier_forward_kcal"] > 40.0


# --------------------------------------------------------------------------------------
# guards against physically impossible interpolated bands
# --------------------------------------------------------------------------------------
def test_min_interatomic_distance():
    coords = np.array([[0.0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
    assert neb.min_interatomic_distance(coords) == pytest.approx(1.5)


def test_check_band_geometry_accepts_a_sane_band():
    good = [np.array([[0.0, 0, 0], [1.5, 0, 0]]) for _ in range(3)]
    assert neb.check_band_geometry(good) is None


def test_check_band_geometry_flags_overlapping_atoms():
    bad = [np.array([[0.0, 0, 0], [1.5, 0, 0]]),
           np.array([[0.0, 0, 0], [0.1, 0, 0]])]
    problem = neb.check_band_geometry(bad)
    assert problem is not None
    assert "image 1" in problem


def test_scan_refuses_independently_embedded_ionone_conformers():
    """The real regression: alpha- and beta-ionone embedded separately differ in ring
    pucker and torsions, not just the moved H. Interpolating linearly between them drives
    atoms through each other and produced a nonsense 3.2e6 kcal/mol barrier. It must now
    refuse rather than return a number. Raises before any SCF, so this stays fast.
    """
    with pytest.raises(ValueError, match="not physical"):
        neb.tautomer_scan("CC(=O)C=CC1C(C)=CCCC1(C)C",
                          "CC(=O)C=CC1=C(C)CCCC1(C)C",
                          method="PM6", n_points=21)
