"""The numba w kernel must stay disabled until it agrees with the NumPy path.

`_w_withquaternion_kernel` returns wrong integrals for d-bearing pairs. It does
so silently — a wrong energy, not an exception — and it is not a JIT artefact:
it fails identically under `NUMBA_DISABLE_JIT=1` and with the numba cache
cleared, so it is a transcription error against the NumPy path rather than
fastmath reassociation.

The bug hid for a whole optimisation campaign because numba 0.60 cannot import
under numpy 2.4 ("Numba needs NumPy 2.0 or less"), so the fallback always ran
here. It surfaced on a second machine that happened to have a working numba, as
14 unexplained test failures: the batched Fock diverging from the sequential one
by 15-32 on CS/CCS/chlorobenzene/thioanisole/bromobenzene, and CSC coming out at
-786.50 against the correct -501.09.

numba is now in the `test` extra precisely so the JIT path is live in CI. That
makes it possible to flip `_W_KERNEL_ENABLED` back on and see a green suite on a
machine without numba, which is the failure mode this file exists to prevent.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from mlxmolkit.nddo.scf import nddo_energy  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# The NumPy path's own values at RDKit seed 42, which is what the whole suite
# and the MOPAC comparisons are built on. The kernel puts CSC hundreds of eV
# out — the original report had it at -786.50 where the NumPy path gives -501.
D_BEARING_REFERENCES = {
    "CSC": -500.9532912,
    "CSc1ccccc1": -1139.7806569,
}


def test_the_w_kernel_is_disabled():
    """A source-level assertion, because the flag is what actually gates it."""
    import mlxmolkit.nddo._pyseqm_port.two_elec_two_center_int_np as mod
    import inspect

    src = inspect.getsource(mod)
    assert "_W_KERNEL_ENABLED = False" in src, (
        "_w_withquaternion_kernel has been re-enabled. It returns wrong "
        "d-orbital integrals silently. Re-enable it only together with a test "
        "that pins the kernel's output against the NumPy path on d pairs."
    )


@pytest.mark.parametrize("smiles,reference", sorted(D_BEARING_REFERENCES.items()))
def test_d_bearing_energies_are_right_whether_or_not_numba_is_installed(
    smiles, reference
):
    """The end-to-end symptom, which is what a user would actually hit.

    This passes on a machine without numba no matter what the flag says, so it
    is a companion to the source assertion above rather than a replacement.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        pytest.skip(f"embedding failed for {smiles}")
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    coords = np.asarray(mol.GetConformer().GetPositions())

    result = nddo_energy(atoms, coords, method="PM6")

    assert result["converged"]
    assert result["energy_eV"] == pytest.approx(reference, abs=1e-4), (
        f"{smiles} PM6 energy {result['energy_eV']:.6f} vs {reference:.6f}. "
        f"A discrepancy of several hundred eV here is the numba w kernel."
    )


def test_the_numba_rotation_kernel_still_matches_the_python_path(monkeypatch):
    """The other numba kernel is live, so it needs its own pin.

    `_generate_rotation_matrix_kernel` is the one JIT path still enabled. If it
    ever drifts the way the w kernel did, a d-bearing energy is wrong and
    nothing else reports it — which is exactly how the w kernel survived.

    `GenerateRotationMatrix` picks the path internally, so the comparison is
    made by forcing the selector both ways on the same input.
    """
    jit = pytest.importorskip(
        "mlxmolkit.nddo._pyseqm_port._jit_kernels",
        reason="numba not installed; the kernel cannot run",
    )
    if not jit.is_numba_available():
        pytest.skip("numba not importable in this environment")

    from mlxmolkit.nddo._pyseqm_port import RotationMatrixD_np as rmd

    rng = np.random.default_rng(0)
    xij = rng.normal(size=(8, 3))
    xij = xij / np.linalg.norm(xij, axis=1, keepdims=True)

    with_jit = np.asarray(rmd.GenerateRotationMatrix(xij)).copy()

    monkeypatch.setattr(jit, "is_numba_available", lambda: False)
    without_jit = np.asarray(rmd.GenerateRotationMatrix(xij)).copy()

    assert with_jit.shape == without_jit.shape
    assert np.all(np.isfinite(with_jit)), "the JIT path produced non-finite entries"
    worst = float(np.max(np.abs(with_jit - without_jit)))
    assert worst < 1e-12, (
        f"the numba rotation kernel disagrees with the Python path by {worst:.3e}. "
        f"This is the same class of defect as the disabled w kernel: it changes "
        f"d-orbital energies and reports nothing."
    )
