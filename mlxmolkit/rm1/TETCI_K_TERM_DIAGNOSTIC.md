# TETCI K-term: the precise remaining diff

Captured 2026-05-23 on H2S (S=0, H1=1, H2=2) at S=(0,0,0), H1=(1.336,0,0).

## PYSEQM rotate() output structure

`rotate()` returns `(w, e1b, e2a, riXH, ri)`:
- `w`: shape (n_pairs, 45, 45) — two-electron integrals
- `e1b`: shape (n_pairs, 9, 9) — V on atom j from nucleus i
- `e2a`: shape (n_pairs, 9, 9) — V on atom i from nucleus j

For H2S pair 0 (S, H1):
- `e1b[0, 0, 0] = -46.63` ≈ -Z_S * (s_H | s_H s_S) — H's s under S's nuke
- `e2a[0, 0, 0] = -7.77` ≈ -Z_H * (s_S | s_S s_H) — S's H_core from H

**`e2a` is the matrix I want for the S H_core block.**

## Storage convention divergence

PYSEQM's `e2a[0]` is **upper-triangular** (zeros below the diagonal),
while my `yh_e1b_contribution()` returns a fully symmetric matrix.
Eigendecomposition (`sym_eig_truncd` in PYSEQM, `np.linalg.eigh` in
mlxmolkit) handles the asymmetry by mirroring.

## Element-by-element diff (PYSEQM e2a[0] - my yh_e1b)

| Entry | mine    | PYSEQM   | diff   | notes |
|-------|---------|----------|--------|-------|
| [0,0] | -7.772  | -7.772   | 0      | diagonal exact |
| [1,1] | -8.155  | -8.155   | 0      | diagonal exact |
| [4,4] | -9.325  | -9.325   | 0      | diagonal exact |
| [0,1] | -1.966  | -2.626   | -0.660 | upper, mine misses ~0.66 |
| [1,0] | -1.966  |  0.000   | +1.966 | PYSEQM = 0 (lower) |
| [0,4] | +0.542  | +1.043   | +0.501 | upper, mine misses ~0.50 |
| [0,6] | -0.313  | -0.602   | -0.289 | similar |
| [4,1] | +1.285  |  0.000   | +1.285 | PYSEQM = 0 (lower) |
| [1,4] | +1.285  | +2.571   | +1.286 | upper, mine misses ~1.29 |

## Two findings

1. **Upper-triangular convention** — PYSEQM stores e2a with lower
   triangle = 0; my symmetric storage doubles the off-diagonal when
   the eigh symmetrizer mirrors. To match, either zero out my lower
   triangle before adding to H_core (and trust eigh to mirror) OR
   double my off-diagonal additions.

2. **Missing cross-mixing terms** — PYSEQM upper-triangle values are
   not simply 2× mine. E.g. mine[0,1]=-1.966, PYSEQM[0,1]=-2.626
   (extra -0.66). This extra contribution likely comes from the
   d-orbital → sp coupling injected via the K_ind_9 / RotateCore
   sp-and-d mixing that happens in PYSEQM's full TETCILFDO. The
   `riYH` slots I populate (10, 11, 14, 17, 20, 24, 27, 35, 44) cover
   only the (d|s) Coulomb-type integrals. The off-diagonal sp-d
   mixing from cross-quadrupole terms is captured in additional
   `coreYHLocal` slots (1, 3, 7, 10) that depend on `riXH` integrals
   pre-rotated differently.

## Deeper finding — multipole parameter mismatch (2026-05-23)

Beyond the upper-triangle storage and cross-mixing issues above, the
**root cause** of the K-term failure is that mlxmolkit's
`compute_d_charge_separations()` returns DIFFERENT values than what
PYSEQM passes to `TETCILFDO`. For sulfur:

| Param | mlxmolkit | PYSEQM     | match? |
|-------|-----------|------------|--------|
| dpa   | 0.49862   | 0.49862    | ✓      |
| dsa   | 0.96177   | 0.96177    | ✓      |
| dda   | 0.64321   | 0.64321    | ✓      |
| rho0a | 1.48359   | 1.48359    | ✓      |
| da    | **0.70209**   | **0.97545**    | ✗ ratio 0.72 |
| qa    | **0.66523**   | **0.90888**    | ✗ ratio 0.73 |
| rho1a | **0.53517**   | **0.62546**    | ✗      |
| rho2a | **0.81950**   | **1.01729**    | ✗      |
| rho3a | **0.62293**   | **0.44863**    | ✗ ratio 1.39 |
| rho4a | **0.66833**   | **1.89217**    | ✗ ratio 0.35 |
| rho5a | **0.53456**   | **3.23024**    | ✗ ratio 0.17 |
| rho6a | **0.54178**   | **0.48881**    | ✗      |

PYSEQM computes these via:
- `dp = AIJ52/sqrt(5)`
- `ds = sqrt(AIJ43*sqrt(1/15))*sqrt(2)` then `DS = POIJ(2, ds, dsAdditiveTerm)`
- `dorbdorb = sqrt(2*AIJ63/7)` then `DD = POIJ(2, dorbdorb, FG - (20/35)*FG1)`
- `DD0 = POIJ(0, 1.0, 0.2*(FG + 2*FG1 + 2*FG2))`
- `DP = POIJ(1, D, FG - 1.8*FG1)` where D = AIJ52/sqrt(5)

Where `AIJ52`, `AIJ43`, `AIJ63`, `dsAdditiveTerm`, `dpAdditiveTerm`,
`ddAdditiveTerm`, `dd0AdditiveTerm`, `dd4`, `dp3` come from runtime
lookup via `_pm6_d_param_from_key()` in
`/tmp/pyseqm/seqm/seqm_functions/two_elec_two_center_int.py:31`.

`POIJ` is PYSEQM's additive-term-rho function (see imports in same file).

**To fix**: port these (~150 LOC) into mlxmolkit's
`compute_d_charge_separations()` replacing the current simpler
formulas. Then re-validate H2S — once params match, the K-term
should match too because the W tensor will be identical to PYSEQM's.

## How to fix (revised — focused next session, ~3 hr)

1. In `tetci_yh.yh_rotated_integral_matrix()`, populate ALL nonzero
   `coreYHLocal[1..45]` slots, not just the 9 explicitly listed in
   PYSEQM lines 432-456. Specifically the sp-d cross-mixing slots
   that use riXH-derived terms (PYSEQM lines 481-493 show 10
   coreYHLocal entries; I currently only handle 6 d-specific ones).

2. After unpacking to 9×9, zero out the lower triangle to match
   PYSEQM's storage convention. Verify mlxmolkit's `_build_fock` /
   `_build_core_hamiltonian` use only the upper triangle (or rely on
   eigh to mirror).

3. Re-test on H2S — target S = -0.37 ± 0.01.

The diagnostic script that captured this is at `/tmp/` (regenerate
via the patched `rotate()` monkey-patch shown above).
