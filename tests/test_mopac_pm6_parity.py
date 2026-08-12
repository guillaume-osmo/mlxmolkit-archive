"""mlxmolkit's PM6 must be MOPAC's PM6.

This exists because a profiling table once printed mlxmolkit's total
electronic energy in eV next to MOPAC's heat of formation in kcal/mol under
the headers "mlx eV" and "mopac eV". They differed by a factor of ten and
looked like a physics failure. They are simply different quantities.

The comparison that means something is heat of formation against heat of
formation, on the *same* geometry, single point — no optimiser, no
convergence criteria, nothing between the two energy expressions.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem

from mlxmolkit.nddo import nddo_energy

MOPAC = Path.home() / "miniconda3/bin/mopac"
pytestmark = pytest.mark.skipif(not MOPAC.exists(), reason="MOPAC not installed")

# Spans sp only, aromatic, and d-bearing (S, Cl).
MOLECULES = ["C", "CCO", "c1ccccc1", "O=Cc1ccccc1", "CC(=O)C",
             "CCS", "Clc1ccccc1", "CSc1ccccc1"]

# Agreement measured across this set is 0.20 kcal/mol mean, 0.34 max. The
# bound is deliberately a little looser than the worst case so that ordinary
# numerical drift does not fail the suite, but tight enough that a genuine
# divergence in the energy expression — a wrong parameter, a dropped term —
# cannot hide.
TOLERANCE_KCAL = 0.6


def geometry(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return ([a.GetAtomicNum() for a in mol.GetAtoms()],
            np.asarray(mol.GetConformer().GetPositions(), dtype=float))


def mopac_heat_of_formation(atoms, coords, tmp_path) -> float:
    """MOPAC PM6, single point, coordinates frozen."""
    table = Chem.GetPeriodicTable()
    lines = ["PM6 1SCF", "parity", ""]
    for z, (x, y, z_) in zip(atoms, coords):
        lines.append(f"{table.GetElementSymbol(int(z)):<3} "
                     f"{x:12.6f} 0 {y:12.6f} 0 {z_:12.6f} 0")
    job = tmp_path / "parity.mop"
    job.write_text("\n".join(lines) + "\n")
    subprocess.run([str(MOPAC), job.name], cwd=tmp_path,
                   capture_output=True, timeout=600)
    out = job.with_suffix(".out")
    if not out.exists():
        pytest.skip("MOPAC produced no output")
    found = re.search(r"FINAL HEAT OF FORMATION =\s+(-?\d+\.\d+)\s*KCAL",
                      out.read_text())
    if found is None:
        pytest.skip("MOPAC did not report a heat of formation")
    return float(found.group(1))


@pytest.mark.parametrize("smiles", MOLECULES)
def test_pm6_heat_of_formation_matches_mopac(smiles, tmp_path):
    atoms, coords = geometry(smiles)
    mine = nddo_energy(atoms, coords, method="PM6")["heat_of_formation_kcal"]
    theirs = mopac_heat_of_formation(atoms, coords, tmp_path)
    assert abs(mine - theirs) < TOLERANCE_KCAL, (
        f"{smiles}: mlxmolkit {mine:.3f} vs MOPAC {theirs:.3f} kcal/mol"
    )


def test_the_two_energy_quantities_are_not_confused(tmp_path):
    """`energy_eV` and `heat_of_formation_kcal` are different things.

    Pinning the relationship stops anyone comparing the wrong one against
    MOPAC again — the electronic energy is roughly ten times larger and
    carries the isolated-atom reference, so it will never match a heat of
    formation. An order of magnitude apart is the point, not the exact ratio.
    """
    atoms, coords = geometry("CCO")
    result = nddo_energy(atoms, coords, method="PM6")
    # Ethanol: -618 eV electronic against -56 kcal/mol formation.
    assert abs(result["energy_eV"]) > 5 * abs(result["heat_of_formation_kcal"])
