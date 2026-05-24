# Native MLX TETCI Port — Remaining 15% Work Plan

**State after 7 commits (5d70f22 → fe5f564):**

* sp-only TETCI fully ported + machine-eps validated (3 commits)
* d-orbital rotation matrix + RotateCore ported + EXACT match
* d-orbital H_core for YH case ported + working (charge conservation fixed)
* yh_fock_contribution helper in place but disabled (K-term over-corrects)
* d_two_center_fock uses legacy dd_ss monopole stopgap

**H2S native progression (PYSEQM delegation disabled):**
  Original broken:   q = [+3.77, +0.31, +0.21]  sum = +4.29
  After H_core fix:  q = [+0.31, -0.15, -0.16]  sum = 0  ✓ conservation
  With native Fock:  q = [-1.98, +0.99, +0.99]  sum = 0  (K over-corrects)
  MOPAC reference:   q = [-0.37, +0.19, +0.19]  sum = 0

## What's left — the K-term mismatch

The yh_fock_contribution exchange formula is mathematically identical
to PYSEQM's `_two_center` K computation for B = H (verified by tracing
K_ind_9 reduction in TETCILFDO YH path). But empirically the W matrix
contracted with P gives different result than PYSEQM's w tensor.

**Hypothesis**: my `yh_rotated_integral_matrix(pA, pB, coordA, coordB)`
produces a 9×9 W where W[μ, ν] = (μ_A ν_A | s_B s_B) in molecular
frame. PYSEQM stores the same data in a 45-element packed form `riYH`
that gets passed through `RotateCore`. The unpacking step in my code
(`for i in range(9): for j in range(i+1): W[i,j] = core_mol[INDX[i]+j]`)
may have a different convention than PYSEQM expects for the K
contraction.

**To fix**: compare element-by-element my 9×9 W vs the equivalent
extracted from PYSEQM's w tensor on H2S. Find the index where they
differ → that's the bug.

**Approach**:
1. Instrument PYSEQM to dump `w[YH=0, packed_μν, 0]` for all 45
   packed_μν values on the S-H1 pair of H2S
2. Compare to my `W[μ, ν]` flattened by INDX[i]+j
3. If they match exactly → the K formula application is wrong (not the
   W values); if they differ → the unpacking convention is wrong

If they match: the issue may be that PYSEQM also uses an n_orbital=9
basis for H (padded with zero), and the contraction sums over the
zero entries in a way that's nonzero with my non-padded basis.

## Original 270-LOC estimate

## What's done

Three sp-only TETCI files ported to NumPy with machine-eps validation
against PYSEQM:

| File | LOC | What | Validation |
|------|-----|------|-----------|
| `tetci_quaternion.py` | 148 | `rotate_with_quaternion()` | max diff vs PYSEQM = 4.4e-16 |
| `tetci_w.py` | 280 | `w_withquaternion()` (sp-only batched) | EXACT (0.00) on 3 cases |
| `tetci_local_frame.py` | 318 | `two_elec_two_center_int_local_frame()` | ~1.4e-14 on HH+XH+XX |

Together these reproduce the entire **sp-only** TETCI pipeline natively.

`d_two_center.py` + `fock_d.py` + `wigner_d.py` + `yy_integrals.py` +
`d_charge_sep.py` + `w_integrals.py` cover most of the d-orbital
infrastructure. The bug is just in how YH/YX d-orbital electron-nuclear
attraction is handled (or rather, NOT handled) by
`_build_core_hamiltonian`.

## Root cause of remaining H2S mismatch

`scf.py:_build_core_hamiltonian` produces a 4-block sp-only H_core for
d-orbital atoms; the d-orbital block (rows/cols 4-8) only gets `Udd`
on the diagonal — no electron-nuclear attraction contribution from
neighbour atoms.

PYSEQM's `TETCILFDO` (file `two_elec_two_center_int_local_frame_d_orbitals.py`)
computes 45 local-frame matrix elements `riYH[..., 0..44]` from 7
fundamental multipole integrals. After rotation to molecular frame
via the Wigner D-matrix, these populate the 5×5 d-block AND the 4×5
sp-d coupling block of `e2aD` (the electron-nuclear attraction for
atom A's electrons from atom B's nucleus, in 9-orbital basis).

Without these, the d-orbital diagonal stays artificially high (only
Udd, no extra attraction from neighbours), but the off-diagonal d↔sp
coupling via β_d * S_d_sp pulls electrons into d-orbitals anyway → the
SCF can't find the right equilibrium → pz collapses to 0, d-orbitals
over-fill.

## YH section of TETCILFDO — exact list of integrals to port

For atom pair (A=heavy w/ d, B=H), located at PYSEQM
`two_elec_two_center_int_local_frame_d_orbitals.py:92-470`:

**Step 1 — Compute 7 fundamental multipole integrals (line 337-358):**

```
rhoSS  = (rho0a + rho0b)²        # ss-ss
rhoSDD0 = (rho3a + rho0b)²        # d-monopole-ss
rhoSSP = (rho1a + rho0b)²        # sp-dipole-ss
rhoSPD = (rho4a + rho0b)²        # dp-dipole-ss
rhoSPP = (rho2a + rho0b)²        # sp-quadrupole-ss
rhoSSD = (rho5a + rho0b)²        # ds-quadrupole-ss
rhoSDD = (rho6a + rho0b)²        # dd-quadrupole-ss

qq  = ev/sqrt(r² + rhoSS)
DDq_Sq = ev/sqrt(r² + rhoSDD0)
DPUz_Sq = ev1/sqrt((r+dpa)² + rhoSPD) - ev1/sqrt((r-dpa)² + rhoSPD)
SPUz_Sq = ev1/sqrt((r+da)² + rhoSSP) - ev1/sqrt((r-da)² + rhoSSP)
DDQtzx_onehalfQtxy_Sq = ev2/sqrt((r-dda)² + rhoSDD) + ev2/sqrt((r+dda)² + rhoSDD) - ev1/sqrt(r² + dda² + rhoSDD)
DSQtzx_onehalfQtxy_Sq = ev2/sqrt((r-dsa)² + rhoSSD) + ev2/sqrt((r+dsa)² + rhoSSD) - ev1/sqrt(r² + dsa² + rhoSSD)
PPQtzx_onehalfQtxy_Sq = ev2/sqrt((r-qa)² + rhoSPP) + ev2/sqrt((r+qa)² + rhoSPP) - ev1/sqrt(r² + da² + rhoSPP)
```

The required atom-A parameters (rho0a, rho1a, rho2a, rho3a, rho4a,
rho5a, rho6a, da, qa, dpa, dsa, dda) are computed by
`d_charge_sep.compute_d_charge_separations()` plus
`two_center_integrals._compute_multipole_params()`. Both already
exist in mlxmolkit.

**Step 2 — Assemble 45 local-frame matrix elements (line 432-470).**
Each `riYH[..., k]` for k in 0..44 is a linear combination of the
7 above, with specific small coefficients (1.0, 1.154701, 0.666667,
1.333333, -1.333333, etc.). Pull the exact coefficients from PYSEQM
lines 432-470 — they map to specific (μ, ν) orbital pairs on atom A.

**Step 3 — Rotate the 9×9 matrix from local to molecular frame.**
The 4×4 sp block rotates with the 3×3 atomic rotation matrix R.
The 5×5 d block rotates with the Wigner D² matrix (already implemented
in `wigner_d.wigner_d_matrix(R)`). The 4×5 sp-d off-diagonal block
rotates with R outer Wigner-D — see PYSEQM `RotationMatrixD.py`
(355 LOC, can be ported in ~80 LOC of mlxmolkit-style code).

**Step 4 — Multiply by `-tore[ni]` and add to `H_core`** at the
appropriate (μ, ν) positions in `_build_core_hamiltonian`.

## Total estimated remaining work

- New file `tetci_yh_yx.py` (~150 LOC): steps 1-3 above
- New file `rotation_matrix_d.py` (~80 LOC): the 4×5 sp-d rotation
- `scf.py` `_build_core_hamiltonian` patch (~30 LOC): add the YH H_core
  contribution after the existing sp-only `e1b_ij` loop
- `_pm6d_via_pyseqm` removal (~10 LOC) after H2S validates against MOPAC

**Total: ~270 LOC of focused work + validation. Realistic 4-6 hour
session.**

Once that's done:
- mlxmolkit PM6_D becomes self-contained (no PYSEQM dep)
- MLX backend (Apple GPU) becomes viable since the code is now plain
  NumPy that can be migrated to mlx.core array-by-array
- Expected speedup over PYSEQM CPU: ~10× on Apple Silicon GPU

## Validation strategy

1. **H2S smoke test** (after each step): compare e1b, e2a, F matrices
   element-by-element against PYSEQM. Target: < 1e-6 per element.
2. **CH3SH** (small d-orb mol): SCF charge match to MOPAC < 0.01 e.
3. **NEEMP 8-mol cross-validation** (re-run `/tmp/compare_charges_large.py`):
   targets are MOPAC ≡ PYSEQM ≡ mlxmolkit-native at 0.0001 e.
4. **Timing on 50-mol set**: target ≥ 5 mol/s on CPU (matching PYSEQM CPU);
   ≥ 50 mol/s on MPS GPU after mlx.core port.

## Open-source provenance / citations

All ported code is BSD-3-Clause-derived from PYSEQM, retain attribution
in source headers (see `tetci_quaternion.py` / `tetci_w.py` /
`tetci_local_frame.py` for the established header pattern).

MOPAC binary (Apache 2.0) is the validation oracle — confirmed
agreement with PYSEQM to 0.011 e on H2S, 0.0001 e on 8 NEEMP mols.
