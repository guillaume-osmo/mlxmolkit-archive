"""Validate mlxmolkit's semiempirical (AM1/PM3/PM6/RM1) atomic charges against OpenMOPAC v23.2.5 at
IDENTICAL geometry, plus the CSV-augmentation element coverage (PM6 full d-orbital, PM3 full sp-only,
sp-only PM6 removed).

MOPAC is optional: set MOPAC_BIN to the openmopac binary (default: the conda `mopacenv` env). Parity tests
skip if it isn't present. The injected d-orbital elements (As/Si) are an @expectedFailure — mlxmolkit's
native d-two-center Fock is knowingly incomplete (see scf.py _pm6d_via_pyseqm); when that is completed the
xfail flips to XPASS, signalling the decorator should be removed.

Run:  python -m unittest tests.test_semiempirical_mopac_parity -v
"""
import os
import subprocess
import sys
import tempfile
import unittest

import shutil

import numpy as np

# use the repo's mlxmolkit, not a stale site-packages install (which shadows data files)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _find_mopac() -> str:
    """Locate MOPAC without hardcoding anyone's home directory.

    This was pinned to a contributor's own machine
    ("/Users/tgg/miniforge3/envs/mopacenv/bin/mopac"), so the tests skipped
    for every other checkout regardless of where MOPAC was installed — and a
    skip reads as "not applicable" rather than "misconfigured". PATH alone is
    not enough either: a conda *env* usually does not have the base env on
    PATH, which is exactly where a `conda install mopac` lands.
    """
    found = os.environ.get("MOPAC_BIN") or shutil.which("mopac")
    if found:
        return found
    for prefix in ("~/miniconda3", "~/miniforge3", "~/anaconda3", "/opt/homebrew"):
        candidate = os.path.expanduser(f"{prefix}/bin/mopac")
        if os.path.exists(candidate):
            return candidate
    return "mopac"


MOPAC_BIN = _find_mopac()

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from mlxmolkit.nddo.methods import METHOD_PARAMS, get_params
    from mlxmolkit.am1bcc import am1_bcc_charges_from_rdkit_mol
    _IMPORTS_OK = True
except Exception as _e:  # pragma: no cover
    _IMPORTS_OK = False
    _IMPORT_ERR = str(_e)

_HAVE_MOPAC = os.path.exists(MOPAC_BIN)

# The AM1-BCC correction table lives under mlxmolkit/data/, which carries no
# tracked files, so it is absent from every clean clone. Absent optional data
# is a skip with a reason, not six failures that look like broken code.
try:
    from importlib import resources as _resources

    _HAVE_BCC = os.path.exists(
        _resources.files("mlxmolkit").joinpath("data/bcc/original-am1-bcc.json")
    )
except Exception:
    _HAVE_BCC = False


# --------------------------------------------------------------------------- helpers
def _prep(smiles, seed=42):
    """RDKit mol with an MMFF-optimised 3D conformer (single geometry for both engines)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _mopac_charges(mol, method, workdir):
    """OpenMOPAC 1SCF NET ATOMIC CHARGES at the mol's current geometry, aligned to atom index."""
    conf = mol.GetConformer()
    mop = os.path.join(workdir, "m.mop")
    with open(mop, "w") as fh:
        fh.write(f"{method} 1SCF CHARGE={Chem.GetFormalCharge(mol)} PRECISE\nparity\n\n")
        for a in mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            fh.write(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}\n")
    for ext in ("out", "arc"):
        f = os.path.join(workdir, f"m.{ext}")
        if os.path.exists(f):
            os.remove(f)
    subprocess.run([MOPAC_BIN, "m.mop"], cwd=workdir, capture_output=True,
                   timeout=240, env=dict(os.environ, OMP_NUM_THREADS="2"))
    out = os.path.join(workdir, "m.out")
    if not os.path.exists(out):
        return None
    lines = open(out).read().splitlines()
    hdr = next((k for k, l in enumerate(lines) if "ATOM NO." in l and "CHARGE" in l), None)
    if hdr is None:
        return None
    q = {}
    for ln in lines[hdr + 1:]:
        p = ln.split()
        if len(p) >= 3 and p[0].isdigit():
            try:
                q[int(p[0]) - 1] = float(p[2])
            except ValueError:
                break
        elif q:
            break
    n = mol.GetNumAtoms()
    return np.array([q[i] for i in range(n)]) if len(q) == n else None


def _mlx_charges(mol, method):
    r = am1_bcc_charges_from_rdkit_mol(
        Chem.Mol(mol), am1_method=method, add_hs=False,
        validate_bcc_coverage=False, require_scf_convergence=False,
    )
    return np.asarray(r.am1_charges)


# common-element molecules (all four methods parameterise these)
_COMMON = {
    "water": "O", "methanol": "CO", "acetamide": "CC(N)=O", "fluoromethane": "CF",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O", "imidazole": "c1c[nH]cn1", "thiophene": "c1ccsc1",
    "dimethylphosphate": "COP(=O)(O)OC", "chloroform": "ClC(Cl)Cl", "pyridine": "c1ccncc1",
    "methanethiol": "CS", "bromoethane": "CCBr",
}
_TOL = 5e-4  # MOPAC prints 6 decimals; same-geometry charges agree to print precision


@unittest.skipUnless(_IMPORTS_OK, "mlxmolkit/rdkit not importable")
class TestElementCoverage(unittest.TestCase):
    """The CSV-augmentation fixes: element counts + sp-only PM6 removal."""

    def test_counts(self):
        self.assertEqual(len(METHOD_PARAMS["RM1"]), 10, "RM1 = full published set (Rocha 2006)")
        self.assertEqual(len(METHOD_PARAMS["AM1"]), 11, "AM1 base + Si from MOPAC CSV")
        self.assertEqual(len(METHOD_PARAMS["PM3"]), 25, "PM3 full main-group (sp-only)")
        self.assertEqual(len(METHOD_PARAMS["PM6"]), 40, "PM6 full main-group (d-orbital)")

    def test_pm6_covers_se_as_b(self):
        for z in (34, 33, 5):  # Se, As, B
            self.assertIn(z, METHOD_PARAMS["PM6"])

    def test_pm6_has_d_orbitals(self):
        d = [z for z, p in METHOD_PARAMS["PM6"].items() if getattr(p, "has_d", False)]
        for z in (15, 16, 17, 33, 35, 53):  # P, S, Cl, As, Br, I
            self.assertIn(z, d)

    def test_pm6_sp_only_removed(self):
        with self.assertRaises(ValueError):
            get_params("PM6_SP")

    def test_no_garbage_zero_param_elements(self):
        for meth in ("PM3", "PM6"):
            for z, p in METHOD_PARAMS[meth].items():
                self.assertGreater(abs(p.Uss), 1e-6, f"{meth} Z={z} has zero U_ss (garbage)")


@unittest.skipUnless(
    _IMPORTS_OK and _HAVE_MOPAC and _HAVE_BCC,
    f"needs OpenMOPAC at {MOPAC_BIN} and the AM1-BCC table "
    f"(mopac={_HAVE_MOPAC}, bcc_table={_HAVE_BCC})",
)
class TestMopacParity(unittest.TestCase):
    """mlxmolkit charges must equal OpenMOPAC at identical geometry."""

    @classmethod
    def setUpClass(cls):
        cls.wd = tempfile.mkdtemp(prefix="mopac_parity_")
        cls.mols = {n: _prep(s) for n, s in _COMMON.items()}

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.wd, ignore_errors=True)

    def _assert_method(self, method):
        worst = 0.0
        for name, m in self.mols.items():
            if m is None:
                continue
            qm = _mopac_charges(m, method, self.wd)
            qx = _mlx_charges(m, method)
            self.assertIsNotNone(qm, f"MOPAC produced no charges for {name}/{method}")
            d = float(np.abs(qm - qx).max())
            worst = max(worst, d)
            self.assertLess(d, _TOL, f"{method} {name}: max|Δ|={d:.5f} vs MOPAC")
        self.assertLess(worst, _TOL)

    def test_am1(self):
        self._assert_method("AM1")

    def test_pm3(self):
        self._assert_method("PM3")

    def test_pm6(self):
        self._assert_method("PM6")

    def test_rm1(self):
        self._assert_method("RM1")

    def _atom_delta(self, smiles, z, method="PM6"):
        m = _prep(smiles)
        self.assertIsNotNone(m)
        qm = _mopac_charges(m, method, self.wd)
        qx = _mlx_charges(m, method)
        self.assertIsNotNone(qm)
        i = next(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == z)
        return abs(float(qx[i] - qm[i]))

    def test_pm6_injected_sp_elements_exact(self):
        """Injected sp-only elements (Se, B) must match MOPAC exactly."""
        self.assertLess(self._atom_delta("C[Se]C", 34), _TOL, "Se")
        self.assertLess(self._atom_delta("OB(O)c1ccccc1", 5), _TOL, "B")

    def test_pm3_injected_se_as_exact(self):
        """PM3 (sp-only) Se + As match MOPAC exactly."""
        self.assertLess(self._atom_delta("C[Se]C", 34, "PM3"), _TOL, "PM3 Se")
        self.assertLess(self._atom_delta("C[As](C)C", 33, "PM3"), _TOL, "PM3 As")

    @unittest.expectedFailure
    def test_pm6_injected_d_elements_exact_KNOWN_GAP(self):
        """As/Si (PM6 d-orbital) do NOT yet match MOPAC — native d-two-center Fock is incomplete
        (scf.py _pm6d_via_pyseqm: missing dd_pp, dd_sp, exchange, d-nuclear-attraction). When that is
        completed this XFAIL becomes XPASS -> remove the decorator."""
        self.assertLess(self._atom_delta("C[As](=O)(O)O", 33), _TOL, "As")
        self.assertLess(self._atom_delta("C[Si](C)(C)C", 14), _TOL, "Si")


if __name__ == "__main__":
    unittest.main(verbosity=2)
