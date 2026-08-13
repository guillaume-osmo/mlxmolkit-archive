"""Integration tests: NDDO SCF → COSMO cavity → sigma profile.

These drive the real SCF, so they are marked `slow`. Run just these with::

    pytest tests/test_cosmo_integration.py -m slow

The downstream goal is PM6_D → COSMO → sigma profile, so the PM6_D path gets
explicit coverage — including the currently-broken d-orbital case.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from mlxmolkit.cosmo.cavity import cosmo_surface, _mulliken_charges
from mlxmolkit.cosmo.ddcosmo import ddcosmo_surface, _switching_function
from mlxmolkit.cosmo.sigma import full_sigma_analysis


def _scf(smiles, method="RM1"):
    from mlxmolkit.nddo.pipeline import _smiles_to_3d
    from mlxmolkit.nddo.scf import nddo_energy
    result = _smiles_to_3d(smiles, seed=42)
    atoms, coords = result[0], result[1]   # 3-tuple since the conformer work
    result = nddo_energy(atoms, coords, method=method)
    if not result['converged']:
        pytest.skip(f"SCF did not converge for {smiles}/{method}")
    return atoms, coords, result


# ---------------------------------------------------------------------------
# Cavity / surface charges
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("smiles", ["O", "CCO", "c1ccccc1"])
def test_cosmo_surface_is_self_consistent(smiles):
    atoms, coords, scf = _scf(smiles)
    res = cosmo_surface(atoms, coords, scf['density'], n_points=194)

    assert res['n_seg'] == len(res['seg_pos']) == len(res['seg_area'])
    assert res['seg_normal'].shape == (res['n_seg'], 3)
    assert np.all(res['seg_area'] > 0)
    assert res['cavity_area'] == pytest.approx(res['seg_area'].sum())
    assert res['cavity_volume'] > 0

    # Neutral molecule -> the screening charge must cancel the solute charge.
    assert res['mulliken_charges'].sum() == pytest.approx(0.0, abs=1e-6)
    assert res['seg_charge'].sum() == pytest.approx(0.0, abs=5e-3)

    # sigma = q / A, by construction
    assert res['seg_sigma'] == pytest.approx(res['seg_charge'] / res['seg_area'])


@pytest.mark.slow
def test_cosmo_mulliken_matches_scf_charges_for_sp_only_method():
    """cosmo_surface re-derives Mulliken charges; for RM1 they must agree."""
    atoms, coords, scf = _scf("CCO", method="RM1")
    res = cosmo_surface(atoms, coords, scf['density'], n_points=110)
    assert res['mulliken_charges'] == pytest.approx(np.asarray(scf['charges']),
                                                    abs=1e-6)


@pytest.mark.slow
def test_cavity_area_grows_with_molecule_size():
    _, _, _ = _scf("O")
    areas = []
    for smi in ("O", "CCO", "CCCCCCO"):
        atoms, coords, scf = _scf(smi)
        res = cosmo_surface(atoms, coords, scf['density'], n_points=194)
        areas.append(res['cavity_area'])
    assert areas == sorted(areas), f"cavity areas not monotonic: {areas}"


# ---------------------------------------------------------------------------
# Sigma profiles
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sigma_profile_area_matches_cavity_area():
    atoms, coords, scf = _scf("CCO")
    res = cosmo_surface(atoms, coords, scf['density'], n_points=194)
    sig = full_sigma_analysis(res, atoms)
    assert sig['sigma_profile'].sum() == pytest.approx(res['cavity_area'],
                                                       rel=1e-9)
    assert sig['total_area'] == pytest.approx(res['cavity_area'])
    assert len(sig['sigma_profile']) == 301
    assert sig['sigma_moment_0'] == pytest.approx(res['cavity_area'])


@pytest.mark.slow
def test_water_sigma_profile_is_bimodal():
    """Water must show both a donor (negative σ) and acceptor (positive σ) lobe.

    The σ scale here is set by NDDO Mulliken point charges, which give
    |σ| <~ 0.005 e/Å² — measured 2.15x narrower than the DFT scale COSMO-RS
    was parameterised on. The threshold below is chosen for that scale.
    """
    atoms, coords, scf = _scf("O")
    res = cosmo_surface(atoms, coords, scf['density'], n_points=194)
    sig = full_sigma_analysis(res, atoms)
    grid, prof = sig['sigma_grid'], sig['sigma_profile']
    neg = prof[grid < -0.002].sum()
    pos = prof[grid > 0.002].sum()
    assert neg > 0, "no H-bond-donor surface found on water"
    assert pos > 0, "no H-bond-acceptor surface found on water"

    # The two lobes must sit on opposite sides of zero.
    centroid = (grid * prof).sum() / prof.sum()
    assert grid[prof.argmax()] != pytest.approx(centroid, abs=1e-9)


@pytest.mark.slow
def test_known_defect_nddo_sigma_never_reaches_the_hb_threshold():
    """HB_SIGMA_THRESH is calibrated for DFT σ and is unreachable from NDDO.

    `params.HB_SIGMA_THRESH` is 0.007686 e/Å², but NDDO Mulliken charges
    produce |σ| <~ 0.005 even for water — the most polar case there is. So
    `cosmors._compute_interaction_matrices` never writes a non-zero entry into
    A_hb and the hydrogen-bonding term is effectively dead on this charge
    source.

    The per-method MF_ALPHA_* values absorb the same scale mismatch for the
    misfit term: misfit goes as alpha*sigma^2, and MF_ALPHA_PM6/MF_ALPHA_DFT
    = 4.22 against the 2.15^2 = 4.6 the measured sigma deficit predicts. There
    is no per-method HB threshold to do the equivalent for hydrogen bonding.
    """
    from mlxmolkit.cosmo import params as P
    atoms, coords, scf = _scf("O")
    res = cosmo_surface(atoms, coords, scf['density'], n_points=194)
    sig = full_sigma_analysis(res, atoms)
    assert np.abs(sig['seg_sigma_av']).max() < P.HB_SIGMA_THRESH, (
        "σ now reaches the HB threshold — the HB term is live; update this test"
    )


@pytest.mark.slow
def test_hb_classification_on_water():
    atoms, coords, scf = _scf("O")
    res = cosmo_surface(atoms, coords, scf['density'], n_points=194)
    sig = full_sigma_analysis(res, atoms)
    hb = sig['seg_hb_type']
    assert set(np.unique(hb)) <= {0, 1, 2}
    assert (hb == 1).sum() > 0, "expected donor segments on the hydrogens"
    assert (hb == 2).sum() > 0, "expected acceptor segments on the oxygen"


# ---------------------------------------------------------------------------
# ddCOSMO
# ---------------------------------------------------------------------------

def test_switching_function_endpoints_and_monotonicity():
    eta = 0.2
    t = np.linspace(0.0, 1.5, 501)
    chi = _switching_function(t, eta)
    assert np.all((chi >= 0.0) & (chi <= 1.0))
    assert chi[t <= 1.0 - eta] == pytest.approx(1.0)
    assert chi[t >= 1.0] == pytest.approx(0.0)
    # Monotonically decreasing across the transition
    trans = chi[(t > 1.0 - eta) & (t < 1.0)]
    assert np.all(np.diff(trans) <= 1e-12)


@pytest.mark.slow
def test_ddcosmo_cavity_is_solver_independent():
    """The cavity is built before the solve, so it cannot depend on the solver."""
    atoms, coords, scf = _scf("CCO")
    kw = dict(n_points=194, epsilon=78.39)
    direct = ddcosmo_surface(atoms, coords, scf['density'], solver='direct', **kw)
    jacobi = ddcosmo_surface(atoms, coords, scf['density'], solver='jacobi', **kw)
    assert direct['n_seg'] == jacobi['n_seg']
    assert direct['cavity_area'] == pytest.approx(jacobi['cavity_area'])


@pytest.mark.slow
def test_known_defect_jacobi_solver_diverges_and_reports_nothing():
    """The Jacobi/DIIS path does not converge, and the caller ignores that.

    The COSMO matrix A has a 1/r off-diagonal block against a 1.07·√(4π/a)
    diagonal; for a real cavity it is nowhere near diagonally dominant. The
    spectral radius of D⁻¹O is ~33 here, so plain Jacobi diverges — DIIS only
    slows it down. At the 200 iterations `ddcosmo_charges` requests, the
    answer is already ~0.12 e per segment off; run it longer and it overflows.

    `ddcosmo_charges` unpacks `converged` from `_jacobi_diis_solve` and then
    never looks at it, so the wrong answer is returned silently.

    This matters because `solver='auto'` selects 'jacobi' whenever
    n_seg > 3000, which any medium-sized molecule exceeds at 194 points per
    atom. Only 'direct' and 'sh' are trustworthy today.
    """
    from scipy.spatial.distance import cdist
    from mlxmolkit.cosmo import params as P
    from mlxmolkit.cosmo.ddcosmo import build_ddcosmo_cavity, _jacobi_diis_solve
    from mlxmolkit.nddo.params import RM1_PARAMS

    atoms, coords, scf = _scf("CCO")
    mull = _mulliken_charges(atoms, scf['density'],
                             [RM1_PARAMS[z].n_basis for z in atoms])
    seg_pos, seg_area, _, _, seg_ui = build_ddcosmo_cavity(
        atoms, coords, n_points_per_atom=194)

    b = P.BOHR_TO_ANG
    spb, sab, cb = seg_pos / b, seg_area / b ** 2, coords / b
    dist = cdist(spb, spb)
    np.fill_diagonal(dist, 1.0)
    A = 1.0 / dist
    np.fill_diagonal(A, 1.07 * np.sqrt(4.0 * np.pi / np.maximum(sab, 1e-30)))
    rhs = -(mull[np.newaxis, :] / np.maximum(cdist(spb, cb), 1e-10)).sum(axis=1) * seg_ui

    # Jacobi's convergence condition is violated by a wide margin.
    D = np.diag(A)
    O = A.copy()
    np.fill_diagonal(O, 0.0)
    rho = np.abs(np.linalg.eigvals((O.T / D).T)).max()
    assert rho > 1.0, f"D^-1*O spectral radius {rho} < 1 — Jacobi would now converge"

    q_exact = np.linalg.solve(A, rhs)
    q_jac, n_iter, converged = _jacobi_diis_solve(A, rhs, max_iter=200, tol=1e-08)
    assert not converged
    assert n_iter == 200
    assert np.abs(q_jac - q_exact).max() > 1e-3

    # And it gets worse, not better, with more iterations.
    q_more, _, _ = _jacobi_diis_solve(A, rhs, max_iter=2000, tol=1e-10)
    assert np.abs(q_more - q_exact).max() > np.abs(q_jac - q_exact).max()


@pytest.mark.slow
def test_jacobi_falls_back_to_direct_instead_of_returning_garbage():
    """A diverged Jacobi run must warn and fall back, not return its iterate."""
    atoms, coords, scf = _scf("CCO")
    kw = dict(n_points=194, epsilon=78.39)
    direct = ddcosmo_surface(atoms, coords, scf['density'], solver='direct', **kw)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        jacobi = ddcosmo_surface(atoms, coords, scf['density'],
                                 solver='jacobi', **kw)
    assert jacobi['seg_charge'] == pytest.approx(direct['seg_charge'], abs=1e-9)


@pytest.mark.slow
def test_auto_solver_no_longer_routes_to_jacobi():
    """'auto' must give the same answer as 'direct' at any system size."""
    atoms, coords, scf = _scf("CCO")
    kw = dict(n_points=194, epsilon=78.39)
    auto = ddcosmo_surface(atoms, coords, scf['density'], solver='auto', **kw)
    direct = ddcosmo_surface(atoms, coords, scf['density'], solver='direct', **kw)
    assert auto['seg_charge'] == pytest.approx(direct['seg_charge'], abs=1e-9)


@pytest.mark.slow
def test_ddcosmo_exposure_weights_are_bounded():
    atoms, coords, scf = _scf("CCO")
    res = ddcosmo_surface(atoms, coords, scf['density'], n_points=194)
    ui = res['seg_ui']
    assert np.all((ui > 0.0) & (ui <= 1.0))
    # Segments below the 0.001 threshold are dropped by build_ddcosmo_cavity.
    assert ui.min() > 0.001


# ---------------------------------------------------------------------------
# PM6_D — the downstream target method
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pm6d_sp_only_molecule_matches_scf_charges():
    """With no d-orbital atoms, PM6_D and RM1 basis sizes agree, so this works."""
    atoms, coords, scf = _scf("CCO", method="PM6_D")
    res = cosmo_surface(atoms, coords, scf['density'], n_points=110)
    assert res['mulliken_charges'] == pytest.approx(np.asarray(scf['charges']),
                                                    abs=1e-6)


@pytest.mark.slow
def test_mulliken_charges_correct_when_given_pm6d_basis_sizes():
    """`_mulliken_charges` itself is correct — if the caller passes the right
    n_basis. This is what the recovered docstring instructs callers to do."""
    from mlxmolkit.nddo.methods import get_params
    atoms, coords, scf = _scf("CSC", method="PM6_D")     # dimethyl sulfide
    nb = [get_params("PM6_D")[z].n_basis for z in atoms]
    q = _mulliken_charges(atoms, scf['density'], nb)
    assert q.sum() == pytest.approx(0.0, abs=1e-6)
    assert q == pytest.approx(np.asarray(scf['charges']), abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("smiles", ["CSC", "CS", "CCS", "ClCCl", "CP(C)C"])
def test_pm6d_d_orbital_elements_give_correct_charges(smiles):
    """PM6_D gives P, S, Cl, Br and I nine orbitals where RM1 gives four.

    `cosmo_surface` used to build its basis sizes from RM1_PARAMS regardless
    of the method, which slid every atom past the first d-block element onto
    the wrong diagonal block of the density. Dimethyl sulfide came out at
    +4.14 e total instead of neutral.
    """
    atoms, coords, scf = _scf(smiles, method="PM6_D")
    res = cosmo_surface(atoms, coords, scf['density'], n_points=110,
                        method="PM6_D")
    q = res['mulliken_charges']
    assert q.sum() == pytest.approx(0.0, abs=1e-6)
    assert q == pytest.approx(np.asarray(scf['charges']), abs=1e-6)


@pytest.mark.slow
def test_wrong_method_basis_sizes_now_raise():
    """Passing a density that does not match the declared method must fail loudly."""
    atoms, coords, scf = _scf("CSC", method="PM6_D")
    with pytest.raises(ValueError, match="wrong method's basis sizes"):
        cosmo_surface(atoms, coords, scf['density'], n_points=110,
                      method="RM1")


@pytest.mark.slow
def test_ddcosmo_pm6d_d_orbitals_give_correct_charges():
    atoms, coords, scf = _scf("CSC", method="PM6_D")
    res = ddcosmo_surface(atoms, coords, scf['density'], n_points=110,
                          solver='direct', method="PM6_D")
    assert res['mulliken_charges'] == pytest.approx(np.asarray(scf['charges']),
                                                    abs=1e-6)


@pytest.mark.slow
def test_pipeline_threads_method_through_to_the_cavity():
    """smiles_to_cosmo must pass its method down, or sulfur breaks again."""
    from mlxmolkit.cosmo.pipeline import smiles_to_cosmo
    res = smiles_to_cosmo("CSC", method="PM6_D", n_surface_points=110)
    assert res is not None
    assert res['mulliken_charges'].sum() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cosmo_surface_batch_matches_the_single_molecule_path():
    from mlxmolkit.cosmo.cavity import cosmo_surface_batch
    smis = ["O", "CCO", "CSC"]
    mols, singles = [], []
    for smi in smis:
        atoms, coords, scf = _scf(smi, method="PM6_D")
        mols.append((atoms, coords, scf['density']))
        singles.append(cosmo_surface(atoms, coords, scf['density'],
                                     n_points=110, method="PM6_D"))

    batched = cosmo_surface_batch(mols, n_points=110, method="PM6_D")
    assert len(batched) == len(smis)
    for one, many in zip(singles, batched):
        assert many['n_seg'] == one['n_seg']
        assert many['seg_charge'] == pytest.approx(one['seg_charge'], abs=1e-12)
        assert many['mulliken_charges'].sum() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("smiles", ["O", "CCO"])
def test_batch_smiles_to_cosmo_matches_the_single_molecule_pipeline(smiles):
    """At index 0 the batch and single-molecule pipelines use the same seed.

    Compared one molecule at a time deliberately — see the positional-seed
    test below for why a longer list would not match.
    """
    from mlxmolkit.cosmo.batch import batch_smiles_to_cosmo
    from mlxmolkit.cosmo.pipeline import smiles_to_cosmo

    batched = batch_smiles_to_cosmo([smiles], method="RM1", n_surface_points=110)
    assert batched[0] is not None, "batch dropped the molecule"
    one = smiles_to_cosmo(smiles, method="RM1", n_surface_points=110)

    assert batched[0]['n_seg'] == one['n_seg']
    assert batched[0]['cavity_area'] == pytest.approx(one['cavity_area'], rel=1e-9)
    # The COSMO stage is deterministic, but the batch SCF converges to a
    # slightly different density than the single-molecule SCF (~2e-6 on the
    # density with use_metal=True), which carries into the profile.
    assert batched[0]['sigma_profile'] == pytest.approx(one['sigma_profile'],
                                                        rel=1e-3, abs=1e-4)


@pytest.mark.slow
def test_batch_output_is_order_dependent_via_the_embedding_seed():
    """`batch_smiles_to_cosmo` embeds with seed=42+i, so results follow i.

    The same SMILES gets a different ETKDG conformer depending only on where
    it sits in the list, so batch output is a function of (SMILES, position)
    rather than of SMILES alone, and batch agrees with the single-molecule
    pipeline only at index 0.

    The *size* of the difference is just ordinary conformer variation — see
    test_conformer_noise_floor_scales_with_flexibility for the measured
    spread — so this is a reproducibility property, not a correctness bug.
    It matters if you reorder inputs between runs and expect identical
    numbers, or diff batch against single-molecule output.

    Recovered verbatim (`LOAD_CONST 42; LOAD_FAST i; BINARY_OP +`). A fixed
    seed would make batch output order-independent.
    """
    from mlxmolkit.cosmo.batch import batch_smiles_to_cosmo

    areas = []
    for lst in (["CCO"], ["O", "CCO"], ["O", "O", "CCO"]):
        res = batch_smiles_to_cosmo(lst, method="RM1", n_surface_points=110)
        assert res[-1] is not None
        areas.append(res[-1]['cavity_area'])

    assert len(set(np.round(areas, 6))) > 1, (
        "batch is now order-independent — update this test"
    )


@pytest.mark.slow
def test_conformer_noise_floor_scales_with_flexibility():
    """One arbitrary conformer per molecule sets the precision floor.

    The pipeline embeds a single ETKDG conformer (n_confs=1) with no
    Boltzmann ensemble averaging, so every cavity area and sigma profile
    carries the spread of whichever conformer happened to be drawn. Measured
    over 20 seeds:

        water      0 rot. bonds   1.0% area spread
        ethanol    0              3.6%
        hexanol    4              6.1%
        decanal    8              8.0%

    Profile *shape* is much more stable than area — mean pairwise cosine
    between conformers' sigma profiles stays above 0.99.

    Two things follow. Any effect smaller than this floor cannot be resolved
    from single-conformer profiles on flexible molecules; and the floor is
    random, so it does not explain the systematic +25% area bias against the
    DFT reference set (see RECOVERY.md).
    """
    rigid, flexible = [], []
    for seed in range(42, 48):
        for smi, bucket in (("O", rigid), ("CCCCCCO", flexible)):
            _r = _smiles_to_3d_seeded(smi, seed)
            atoms, coords = _r[0], _r[1]   # 3-tuple since the conformer work
            scf = _energy(atoms, coords)
            if scf is None:
                continue
            res = cosmo_surface(atoms, coords, scf['density'], n_points=194)
            bucket.append(res['cavity_area'])

    def spread(xs):
        return (max(xs) - min(xs)) / np.mean(xs)

    assert spread(rigid) < 0.03, f"rigid molecule spread {spread(rigid):.3f}"
    assert spread(flexible) > spread(rigid), (
        "a 4-rotatable-bond molecule should vary more than water"
    )


def _smiles_to_3d_seeded(smiles, seed):
    from mlxmolkit.nddo.pipeline import _smiles_to_3d
    r = _smiles_to_3d(smiles, seed=seed)
    if r is None:
        pytest.skip(f"could not embed {smiles}")
    return r


def _energy(atoms, coords, method="RM1"):
    from mlxmolkit.nddo.scf import nddo_energy
    r = nddo_energy(atoms, coords, method=method)
    return r if r['converged'] else None


@pytest.mark.slow
@pytest.mark.parametrize("smiles", ["CSC", "CCS", "ClCCl"])
def test_batch_pm6d_handles_d_orbital_molecules(smiles):
    """PM6_D + S/P/Cl/Br/I used to raise IndexError from rm1.batch.

    prepare_batch stored the two-center w tensor as 4**4 per pair, so d-orbital
    elements did not fit and walked off the end of H. When this test was
    written those molecules were routed away to the *sequential* solver, and
    its float64 charges summed to ~1e-14 — hence the original abs=1e-6.

    They now go through the batch path proper, on Metal in float32, where the
    same sum is ~5e-6. That is accumulation, not error: the batch and
    sequential charges still agree to ~1e-5, which is what is actually asserted
    below. Tightening the sum back to 1e-6 would only be asserting float64.
    """
    from mlxmolkit.cosmo.batch import batch_smiles_to_cosmo
    from mlxmolkit.cosmo.pipeline import smiles_to_cosmo

    res = batch_smiles_to_cosmo([smiles], method="PM6_D", n_surface_points=110)
    assert res[0] is not None
    assert res[0]['mulliken_charges'].sum() == pytest.approx(0.0, abs=5e-5)

    # The real check: the batch answer is the sequential answer.
    seq = smiles_to_cosmo(smiles, method="PM6_D", n_surface_points=110)
    assert np.abs(np.asarray(res[0]['mulliken_charges'])
                  - np.asarray(seq['mulliken_charges'])).max() < 1e-4
    assert np.asarray(seq['mulliken_charges']).sum() == pytest.approx(0.0, abs=1e-9)

    one = smiles_to_cosmo(smiles, method="PM6_D", n_surface_points=110)
    assert res[0]['n_seg'] == one['n_seg']
    assert res[0]['sigma_profile'] == pytest.approx(one['sigma_profile'],
                                                    rel=1e-3, abs=1e-4)


@pytest.mark.slow
def test_batch_reports_cosmo_stage_failures_and_recovers():
    """A COSMO-stage error must warn, then fall back per molecule.

    It used to be swallowed by `except Exception: pass`, turning any failure
    into an all-None result indistinguishable from bad input.
    """
    import mlxmolkit.cosmo.cavity as cav
    from mlxmolkit.cosmo.batch import batch_smiles_to_cosmo

    original = cav.cosmo_surface_batch

    def boom(*args, **kwargs):
        raise RuntimeError("injected failure in the COSMO stage")

    cav.cosmo_surface_batch = boom
    try:
        with pytest.warns(RuntimeWarning, match="Batched COSMO failed"):
            res = batch_smiles_to_cosmo(["O", "CCO"], method="RM1",
                                        n_surface_points=110)
    finally:
        cav.cosmo_surface_batch = original

    # The per-molecule fallback still produces complete results.
    assert all(r is not None for r in res)
    # 5e-5, not 1e-6: the batch path is Metal float32 (see
    # test_batch_pm6d_handles_d_orbital_molecules for the measurement).
    assert all(r['mulliken_charges'].sum() == pytest.approx(0.0, abs=5e-5)
               for r in res)
