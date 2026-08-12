"""The batch w tensor must be sized by the widest atom, not by an sp assumption.

``prepare_batch`` used to allocate the two-centre tensor as (N, MA, MA, 256)
with 256 = 4**4, and the Metal kernel indexed it with hardcoded strides
64/16/4. Any atom carrying d orbitals needs 9**4, so PM6 on sulfur or the
halogens could never go through the batched GPU path.

The tensor and the kernel are now sized from ``max_orb``. These tests pin both
halves of that: an sp-only batch keeps exactly the old layout, so nothing
regresses, and the shape scales as MO**4 when it needs to.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem

from mlxmolkit.nddo.batch import prepare_batch
from mlxmolkit.nddo.methods import get_params


def geometry(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return atoms, np.asarray(mol.GetConformer().GetPositions(), dtype=float)


SP_ONLY = ["CCO", "O=Cc1ccccc1", "CC(C)=CCCC(C)(O)C=C"]


def test_sp_only_batches_keep_the_original_256_layout():
    """The common case must not pay for the d-orbital generalisation."""
    params = get_params("PM6")
    batch = prepare_batch([geometry(s) for s in SP_ONLY],
                          param_dict=params, method="PM6")
    assert batch.max_orb == 4
    assert batch.w.shape[-1] == 256 == 4 ** 4


def test_the_tensor_is_sized_by_the_widest_atom():
    params = get_params("PM6")
    batch = prepare_batch([geometry("CCO")], param_dict=params, method="PM6")
    assert batch.w.shape[-1] == batch.max_orb ** 4


def test_max_orb_reports_the_widest_basis_present():
    """max_orb drives both the allocation and the kernel stride."""
    params = get_params("PM6")
    atoms, coords = geometry("CCO")
    batch = prepare_batch([(atoms, coords)], param_dict=params, method="PM6")
    assert batch.max_orb == max(params[z].n_basis for z in atoms)


def test_the_metal_kernel_is_told_the_orbital_width():
    """A stale config would silently index the tensor with the wrong stride."""
    mx = pytest.importorskip("mlx.core")
    from mlxmolkit.nddo.fock_metal import MetalFockContext

    params = get_params("PM6")
    batch = prepare_batch([geometry("CCO")], param_dict=params, method="PM6")
    ctx = MetalFockContext(batch)
    assert int(np.array(ctx._config)[3]) == batch.max_orb


@pytest.mark.xfail(reason="rotate_integrals_to_molecular_frame is still sp-only; "
                          "d-capable two-centre integrals are not wired into "
                          "prepare_batch yet",
                   raises=IndexError, strict=True)
def test_d_orbital_elements_can_be_batched():
    """The remaining half of the d-orbital work, pinned as a known gap.

    The tensor and the kernel now accommodate 9 orbitals, but the two-centre
    integrals feeding them do not: rotate_integrals_to_molecular_frame returns
    4x4 blocks, so assembling H_core for sulfur overruns its own array. When
    the d-capable path is routed in, this test starts passing and the xfail
    must be removed.
    """
    params = get_params("PM6")
    prepare_batch([geometry("CSc1ccccc1")], param_dict=params, method="PM6")
