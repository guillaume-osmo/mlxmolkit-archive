"""Cross-toolkit PM6 agreement — mlxmolkit vs an INDEPENDENT reference code.

MOPAC and our vendored primitives share Stewart/PYSEQM lineage, so the strongest
external check is a code with a separate implementation. PYSEQM (lanl/PYSEQM, the
PyTorch NDDO engine) is used here when importable; the full 4-engine matrix
(+ SCINE Sparrow via its own conda env) lives in tools/compare_cross_toolkit.py.

Skips cleanly when PYSEQM is not on the path, so this never breaks CI.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mlxmolkit.rm1 import nddo_energy

seqm = pytest.importorskip("seqm", reason="PYSEQM not installed")

_ns: dict = {}
exec((pathlib.Path(__file__).parent / "_mopac_ref_generated.py").read_text(), _ns)
MOPAC_REF = _ns["MOPAC_REF"]

# PYSEQM is mlxmolkit's PARAMETER SOURCE, so agreement is tight for qn<=4. Iodine (qn=5)
# is DELIBERATELY excluded: PYSEQM's n=5 two-center overlap is broken (CH2I2 Hf = -13937
# kcal in PYSEQM itself), so we fixed it from the exact spheroidal reference and now match
# MOPAC instead (see test_mopac_parity.py). Tracking PYSEQM on iodine would mean tracking a bug.
CASES = ["ethanol", "acetone", "dimethyl_sulfide", "thiophene", "chlorobenzene",
         "methyl_bromide", "ethyl_bromide"]
CHARGE_TOL = 0.005
HF_TOL_KCAL = 0.1   # we should reproduce our own parameter source to well under 0.1 kcal/mol
EV_PER_KCAL = 23.060547830619026


def _pyseqm(atoms, coords):
    import torch
    torch.set_default_dtype(torch.float64)
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.ElectronicStructure import Electronic_Structure

    order = np.argsort(-np.asarray(atoms))  # PYSEQM requires descending Z
    species = torch.as_tensor(np.asarray(atoms)[order][None], dtype=torch.int64)
    xyz = torch.tensor(np.asarray(coords)[order][None], dtype=torch.float64)
    p = {"method": "PM6", "scf_eps": 1e-8, "scf_converger": [2, 0.0],
         "sp2": [False, 1e-5], "elements": [0] + sorted(set(int(z) for z in atoms)),
         "learned": [], "pair_outer_cutoff": 1e10, "eig": True}
    mol = Molecule(Constants(), p, xyz, species)
    Electronic_Structure(p)(mol)
    q = np.empty(len(atoms))
    q[order] = mol.q[0].detach().numpy()[: len(atoms)]
    return float(mol.Hf[0]) * EV_PER_KCAL, q


@pytest.mark.parametrize("name", CASES)
def test_charges_match_pyseqm(name):
    ref = MOPAC_REF[name]
    ours = np.asarray(nddo_energy(ref["atoms"], np.array(ref["coords"]), method="PM6_D")["charges"])
    _, theirs = _pyseqm(ref["atoms"], ref["coords"])
    dq = np.abs(ours - theirs).max()
    assert dq <= CHARGE_TOL, f"{name}: mlxmolkit vs PYSEQM max|dq|={dq:.4f} e"


@pytest.mark.parametrize("name", CASES)
def test_heat_of_formation_matches_pyseqm(name):
    ref = MOPAC_REF[name]
    ours = nddo_energy(ref["atoms"], np.array(ref["coords"]), method="PM6_D")["heat_of_formation_kcal"]
    theirs, _ = _pyseqm(ref["atoms"], ref["coords"])
    assert abs(ours - theirs) <= HF_TOL_KCAL, (
        f"{name}: mlxmolkit Hf={ours:+.3f} vs PYSEQM {theirs:+.3f} (d={ours-theirs:+.3f} kcal/mol)"
    )
