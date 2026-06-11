"""Cross-toolkit PM6 agreement matrix on the frozen MOPAC-ref geometries.

Engines:
  * mlxmolkit  nddo_energy(method='PM6_D')          (this repo)
  * MOPAC      v23.2.5 `PM6 1SCF`                   (frozen in tests/_mopac_ref_generated.py)
  * PYSEQM     lanl/PYSEQM `method='PM6'`           (~/Github/PYSEQM, torch CPU)
  * Sparrow    SCINE sparrow 3.1 PM6                (dedicated conda env, subprocess)

Observables: Mulliken atomic charges (primary — the ODT-relevant output) and heat of
formation where the engine defines one (MOPAC, mlxmolkit, PYSEQM).

Run:  PYTHONPATH=.:/Users/guillaume-osmo/Github/PYSEQM python tools/compare_cross_toolkit.py
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mlxmolkit.rm1 import nddo_energy

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPARROW_PY = "/Users/guillaume-osmo/miniconda3/envs/sparrow/bin/python"

_ns: dict = {}
exec((ROOT / "tests" / "_mopac_ref_generated.py").read_text(), _ns)
MOPAC_REF = _ns["MOPAC_REF"]

MOLS = ["ethanol", "acetone", "dimethyl_sulfide", "thiophene", "chlorobenzene",
        "trimethyl_phosphate", "methyl_bromide", "ethyl_bromide", "methyl_iodide"]


def run_pyseqm(atoms, coords):
    import torch
    torch.set_default_dtype(torch.float64)
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.ElectronicStructure import Electronic_Structure

    order = np.argsort(-np.asarray(atoms))          # PYSEQM wants descending Z
    species = torch.as_tensor([np.asarray(atoms)[order]], dtype=torch.int64)
    xyz = torch.tensor(np.asarray(coords)[order][None], dtype=torch.float64)
    const = Constants()
    p = {
        "method": "PM6",
        "scf_eps": 1.0e-8,
        "scf_converger": [2, 0.0],
        "sp2": [False, 1.0e-5],
        "elements": [0] + sorted(set(int(z) for z in atoms)),
        "learned": [],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,             # single-point properties path (no force -> no grad needed)
    }
    mol = Molecule(const, p, xyz, species)
    Electronic_Structure(p)(mol)
    q = np.empty(len(atoms))
    q[order] = mol.q[0].detach().numpy()[: len(atoms)]
    hf_kcal = float(mol.Hf[0]) * 23.060547830619026  # PYSEQM Hf is in eV
    return hf_kcal, q


def run_sparrow(atoms, coords):
    payload = json.dumps({"atoms": [int(z) for z in atoms],
                          "coords": np.asarray(coords).tolist()})
    script = r"""
import json, sys
import scine_utilities as su
import scine_sparrow  # registers the module

import numpy as np
data = json.loads(sys.stdin.read())
calc = su.core.get_calculator("PM6", "sparrow")
elems = [su.ElementType(z) for z in data["atoms"]]
pos = np.ascontiguousarray(np.array(data["coords"]) * su.BOHR_PER_ANGSTROM, dtype=np.float64)
calc.structure = su.AtomCollection(elems, pos)
calc.set_required_properties([su.Property.Energy, su.Property.AtomicCharges])
res = calc.calculate()
print(json.dumps({"energy_hartree": res.energy,
                  "charges": list(res.atomic_charges)}))
"""
    out = subprocess.run([SPARROW_PY, "-c", script], input=payload,
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-400:])
    r = json.loads(out.stdout.strip().splitlines()[-1])
    return float(r["energy_hartree"]), np.array(r["charges"])


def main():
    print(f"{'molecule':20s} {'dHf(ours)':>9s} {'dHf(PYSEQM)':>11s}   "
          f"{'q: ours':>8s} {'PYSEQM':>8s} {'Sparrow':>8s}   (max|dq| vs MOPAC)")
    for name in MOLS:
        ref = MOPAC_REF[name]
        atoms, coords = ref["atoms"], np.array(ref["coords"])
        q_mop = np.asarray(ref["charges"])

        r = nddo_energy(atoms, coords, method="PM6_D")
        d_ours = r["heat_of_formation_kcal"] - ref["hf_kcal"]
        dq_ours = np.abs(np.asarray(r["charges"]) - q_mop).max()

        try:
            hf_p, q_p = run_pyseqm(atoms, coords)
            d_pyseqm = f"{hf_p - ref['hf_kcal']:+11.2f}"
            dq_pyseqm = f"{np.abs(q_p - q_mop).max():8.4f}"
        except Exception as e:
            d_pyseqm, dq_pyseqm = f"{'ERR':>11s}", f"{'ERR':>8s}"
            print(f"    [PYSEQM {name}: {type(e).__name__}: {str(e)[:90]}]")

        try:
            _, q_s = run_sparrow(atoms, coords)
            dq_sparrow = f"{np.abs(q_s - q_mop).max():8.4f}"
        except Exception as e:
            dq_sparrow = f"{'ERR':>8s}"
            print(f"    [Sparrow {name}: {type(e).__name__}: {str(e)[:90]}]")

        print(f"{name:20s} {d_ours:+9.2f} {d_pyseqm}   {dq_ours:8.4f} {dq_pyseqm} {dq_sparrow}")


if __name__ == "__main__":
    main()
