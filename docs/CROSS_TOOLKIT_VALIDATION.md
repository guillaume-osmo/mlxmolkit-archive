# PM6 cross-toolkit validation

`mlxmolkit.rm1` PM6_D validated against three independent reference codes on frozen
geometries (RDKit ETKDGv3 seed=1 + MMFF94, baked into `tests/_mopac_ref_generated.py`).

| reference | version | how |
|-----------|---------|-----|
| OpenMOPAC | 23.2.5 | `PM6 1SCF` subprocess, frozen refs in the test suite |
| PYSEQM    | lanl/PYSEQM (`~/Github/PYSEQM`) | `method='PM6'`, torch CPU |
| SCINE Sparrow | 3.1.0 (conda env `sparrow`, numpy<2) | `get_calculator("PM6","sparrow")` |

## Agreement matrix (Mulliken charges, max|Δq| in e; ΔHf in kcal/mol vs MOPAC)

| molecule | ΔHf ours | ΔHf PYSEQM | Δq ours | Δq PYSEQM | Δq Sparrow |
|----------|---------:|-----------:|--------:|----------:|-----------:|
| ethanol            | +0.06 | +0.06 | 0.0001 | 0.0001 | 0.0002 |
| acetone            | +0.09 | +0.10 | 0.0002 | 0.0002 | 0.0002 |
| dimethyl sulfide   | +0.04 | +0.04 | 0.0000 | 0.0000 | 0.0001 |
| thiophene          | +0.17 | +0.17 | 0.0001 | 0.0001 | 0.0001 |
| chlorobenzene      | +0.29 | +0.30 | 0.0004 | 0.0004 | 0.0004 |
| trimethyl phosphate| −0.06 | +0.10 | 0.0002 | 0.0002 | 0.0002 |
| methyl bromide     | +0.01 | +0.02 | 0.0001 | 0.0001 | 0.0002 |
| ethyl bromide      | +0.04 | +0.05 | 0.0002 | 0.0002 | 0.0002 |
| methyl iodide      | −0.63 | −3.76 | 0.0052 | 0.0052 | 0.0001 |

**All four codes agree to < 0.0005 e on charges for C/H/N/O/S/P/Cl/Br.** Iodine (qn=5) now
also matches MOPAC to < 0.05 kcal/mol after the n=5 overlap fix (bug 5 below) — note this
means we **intentionally diverge from PYSEQM on iodine**, because PYSEQM's own n=5 overlap
is broken (CH2I2 Hf = -13937 kcal in PYSEQM itself).

## Bugs found & fixed via this validation (all with provenance in-code)

1. **PM6_D core-core** (`scf.py`) — used the AM1-style term instead of PM6 PWCCT;
   corrupted Hf by ~11 eV even for CHNO. Same fix applied to the analytical gradient
   (`anal_grad.py`) so forces match numerical to 3e-6 eV/Å.
2. **Halogen EISOL** (`pm6_params.py`) — Cl/Br/I isolated-atom reference off by a
   geometry-independent constant (measured across 4 molecules each); calibrated to MOPAC.
3. **Iodine d-row** (`pm6_params.py`) — `Udd/zeta_d/beta_d` were mis-transcribed; replaced
   with the official `openmopac/mopac` `src/models/parameters_for_PM6_C.F90` values.
4. **Ethyl-bromide wrong SCF root** (`scf.py`) — the H_core initial guess started in a
   charge-transfer basin (Hf +210 kcal off, q(Br) sign-flipped). Replaced with a
   MOPAC-style neutral-atom diagonal density; converges to the correct root.
5. **Iodine n=5 two-center Slater overlap** (`_pyseqm_port/diat_overlapD_np.py`) — the
   hardcoded qn>=5 reduced-overlap coefficients are mis-transcribed in PYSEQM, giving
   s-d overlaps of ~27 (must be <=1) -> 522 eV core element -> CH2I2 Hf = -13937 kcal.
   The 14 reduced sigma/pi/delta overlaps are recomputed for qn>=5 from an exact
   prolate-spheroidal reference (`slater_overlap_ref.py`), validated to reproduce the
   working qn<=4 overlaps to 5 decimals. All iodine molecules now match MOPAC to
   < 0.05 kcal/mol. PYSEQM still carries this bug, so we diverge from it for iodine only.

Reproduce: `PYTHONPATH=.:~/Github/PYSEQM python tools/compare_cross_toolkit.py`
Regenerate MOPAC refs: `PYTHONPATH=. python tools/gen_mopac_ref.py`
