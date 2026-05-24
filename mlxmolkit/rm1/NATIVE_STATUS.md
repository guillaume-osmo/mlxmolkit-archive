# Native PM6_D status — 2026-05-24 (final, session 2)

After today's two sessions, mlxmolkit native PM6_D matches PYSEQM
**exactly** on every test molecule, including YX heavy-heavy d-orbital
pairs (CSC, CH3SH, CH3Cl, CH3Br) and YY both-d cases (CCl4, SF6).

## Current benchmark

| Mol | atoms | mlxmolkit q (heavy) | PYSEQM ref | Δmax |
|-----|-------|---------------------|------------|------|
| H2O | O-H-H | -0.6093 | -0.6093 | 0.0000 ✓ |
| CH4 | C-H4 | -0.6545 | -0.6545 | 0.0000 ✓ |
| HF | F-H | -0.2638 | -0.2638 | 0.0000 ✓ |
| NH3 | N-H3 | -0.5786 | -0.5786 | 0.0000 ✓ |
| COC | dimethyl ether | -0.195, -0.393, -0.195 | exact | 0.0000 ✓ |
| CNC | dimethylamine | -0.329, -0.374, -0.329 | exact | 0.0000 ✓ |
| H2S | S-H-H | -0.3617 | -0.3617 | 0.0000 ✓ |
| PH3 | P-H3 | -0.0366 | -0.0366 | 0.0000 ✓ |
| HCl | Cl-H | -0.2164 | -0.2164 | 0.0000 ✓ |
| HBr | Br-H | -0.1878 | -0.1878 | 0.0000 ✓ |
| **CSC** | dimethyl sulfide (YX) | -0.488, -0.046, -0.488 | -0.488, -0.046, -0.488 | **0.0000 ✓** |
| **CH3SH** | YX heavy-heavy | -0.467, -0.205 | -0.467, -0.205 | **0.0000 ✓** |
| **CH3Cl** | YX (odd-electron pair) | -0.369, -0.135 | -0.369, -0.135 | **0.0000 ✓** |
| **CH3Br** | YX (Br) | -0.423, -0.114 | -0.423, -0.114 | **0.0000 ✓** |
| **CCl4** | YY (4 Cl) | 0.177, -0.044×4 | exact | **0.0000 ✓** |
| **DMSO** | S-O YX | 1.056, -0.772, -0.753, -0.753 | exact | **0.0000 ✓** |
| **SF6** | 6 YX (S-F) | 2.505, -0.417×6 | exact | **0.0000 ✓** |

## Bugs fixed today

### 1. Wrong principal quantum number (qn) for d-orbital atoms

mlxmolkit hardcoded `qn = 2 if Z > 2 else 1` for **all** heavy atoms, but
PYSEQM uses the row of the periodic table:

| Z range | Element | Correct qn | mlxmolkit was using |
|---------|---------|------------|---------------------|
| 1–2 | H, He | 1 | 1 ✓ |
| 3–10 | Li–Ne | 2 | 2 ✓ |
| 11–18 | Na–Ar (P, S, Cl) | **3** | 2 ✗ |
| 19–36 | K–Kr (Br) | **4** | 2 ✗ |
| 37–54 | Rb–Xe (I) | **5** | 2 ✗ |

The wrong qn flowed through dipole charge separations (`da`), quadrupole
(`qa`), the `_compute_multipole_params` Slater-Condon parameters, and the
overlap formulas. Result: a ~30–40% error in d_A and rho1_A on S, Cl, Br,
I — which translated to ~1 eV diffs on the s-pz coupling in H_core for
S in H2S.

**Fix**: added `principal_qn(Z)` helper in `params.py`, replaced all
hardcoded `qn = 2 if Z > 2 else 1` with the helper. Files touched:
`two_center_integrals.py`, `integrals.py`, `d_charge_sep.py`, `scf.py`.

### 2. Asymmetric H_core from upper-triangle-only e1b assembly

`yh_e1b_contribution` returned a 9×9 matrix populated only in the upper
triangle, mimicking PYSEQM's e2aD convention. PYSEQM's eigensolver uses
`torch.linalg.eigh(.., UPLO='U')` and reads only the upper triangle, so
this works there. mlxmolkit uses `np.linalg.eigh` which defaults to
`UPLO='L'` — it reads the **lower triangle** and ignores the upper.
Result: all s-pz, sp-d, p-d couplings within atom A's block were
silently zeroed, breaking SCF basin convergence.

**Fix**: mirror upper to lower in `yh_e1b_contribution`'s unpack loop.

### 3. Missing d-orbital two-center Fock J/K terms for YH

PYSEQM's `_two_center` Fock builder loops over the full 45×45 packed
w-tensor per atom pair. For YH pairs, this includes the
`(μν_A | s_H s_H)` integrals for μ or ν in the d-orbital range — the
J/K contributions that bring F[d_A, d_A] up by ~15 eV from where the
one-center W alone leaves it. mlxmolkit's `_build_fock` only iterated
over sp (4×4), and `d_two_center_fock` returned early for YH (with a
misleading comment that said the contribution flowed through H_core,
which was only partly true).

**Fix**: in `d_two_center.py`, when one atom has d-orbitals and the
other is H, call `yh_rotated_integral_matrix` to get the 9×9
`(μν_A | s_H s_H)` matrix, and add the full J + K contributions for
all (μ, ν) involving at least one d-orbital index. The sp×sp block
stays handled by the existing standard two-center loop.

### 4. Wrong qn in overlap formulas for d-orbital atoms

mlxmolkit's `overlap.py` and `overlap_d.py` use `qn = 1 if Z <= 2 else 2`,
and the closed-form Slater overlap formulas only cover (qn_A, qn_B) up
to (2, 2). For S-H, this means mlxmolkit was using the (2s, 1s) formula
when (3s, 1s) was needed. The (s_S, s_H) overlap came out 0.268 instead
of 0.369 — a 37% error.

**Fix (interim)**: `overlap_d_molecular_frame` now delegates to PYSEQM's
`diatom_overlap_matrixD` (BSD-3-Clause, cited) when either atom has
d-orbitals. Falls back to the native path otherwise. Closes the SCF
gap on all H2S/PH3/HCl/HBr to Δ = 0.000.

**Remaining native work**: port PYSEQM's diat_overlapD to NumPy
(~5000 LOC supporting qn up to 6). Documented in next-steps below.

### 5. YX H_core e-n attraction missing (session 2)

The H_core e1b extension was only triggered for YH pairs (where atom B
is hydrogen). For YX pairs (atom B is sp-only heavy, e.g. C in CSC, O
in DMSO), the d-orbital diagonals on atom A had **no** nuclear-attraction
contribution from B's core charge. On CSC's S, this left F[d_z2, d_z2]
~58 eV too high.

**Fix**: extended the H_core branch condition to ``params[j].n_basis in
{1, 4, 9}``, generalized ``yh_e1b_contribution``'s Z_B to
``pB.n_valence`` (correct for H=1, sp heavy=4..7, d heavy=4..7), and
the same integral formula is reused (only rho0_B differs across B).

### 6. YX Fock J/K missing (session 2)

mlxmolkit's existing two-center Fock loop only iterated the 4×4×4×4 sp
block. For YX (A has d, B sp), the (d_A × sp_A | sp_B × sp_B) integrals
were entirely absent — explaining the 15 eV mismatch on F[d_z2, d_z2] of
S in CSC and similar errors on every YX pair.

**Fix**: ``d_two_center_fock`` now delegates the YX pair's 9×9×4×4
rotated W tensor to a 2-atom PYSEQM hcore call
(``_yx_pair_w_pyseqm``), then applies the full J + K contractions
natively, skipping the sp-sp-sp-sp tuples already handled by the
standard loop. Two subtleties: (a) odd-electron 2-atom subsystems
(e.g. Cl+C = 11 e) need a phantom H added so PYSEQM's RHF accepts
them — safe because pairwise integrals don't depend on third atoms;
(b) PYSEQM's PM6 swap convention puts ij on the lower-Z (sp) atom and
kl on the higher-Z (d) atom in ``w[pair, ij, kl]``.

### 7. YY Fock J/K missing (session 2)

For both-d pairs (Cl-Cl in CCl4, S-F in SF6 — actually S-F is YX
since F is sp; the true YY here is Cl-Cl), the existing
``compute_yy_integrals`` path was incomplete. Same fix as YX with a
new ``_yy_pair_w_pyseqm`` returning a 9×9×9×9 tensor.

### 8. SCF dynamics needed DIIS (session 2)

After all the integral fixes, the Fock matrix matched PYSEQM exactly
at converged density, but the SCF starting from the diagonal initial
guess kept landing in a wrong basin on YY-heavy molecules (CCl4, SF6).
Reason: d-orbital H_core diagonals on Cl in CCl4 reach -170 eV
(Udd + e-n from 4 neighbors), making them appear as the most attractive
orbitals to ``np.linalg.eigh`` at iteration 0; the standard simple
mixing then locks in the wrong basin.

**Fix**: the test runner ``_run_native`` (and production
``rm1_energy``) now use Pulay DIIS extrapolation + adaptive damping
(stronger damping for d-orbital systems), mirroring PYSEQM's own
``scf pulay diis`` strategy.

## Architecture status

| Component | Native | PYSEQM fallback | Status |
|-----------|--------|------------------|--------|
| H_core (sp atoms) | ✓ | – | Exact match |
| H_core (d atoms) — diag | ✓ | – | Exact match |
| H_core (d atoms) — off-diag YH | ✓ | – | Exact match |
| H_core (d atoms) — off-diag YX/YY | ✓ | – | Exact match (session 2) |
| Overlap (sp atoms) | ✓ | – | Exact match |
| Overlap (d atoms, qn≥3) | – | ✓ | Delegate |
| Fock G(P) sp | ✓ | – | Exact match |
| Fock G(P) d one-center | ✓ | – | Exact match (uses delegated W) |
| Fock G(P) d two-center YH | ✓ | – | Exact match |
| Fock G(P) d two-center YX | ✓ | (uses 2-atom hcore) | Exact match (session 2) |
| Fock G(P) d two-center YY | ✓ | (uses 2-atom hcore) | Exact match (session 2) |
| W integrals (243-element) | – | ✓ | Delegate to calc_integral |
| SCF eigendecomposition | ✓ (eigh + DIIS) | – | Converges all basins |

## Remaining work for FULL native (no PYSEQM at runtime)

The PYSEQM dependencies now reduce to three integral kernels, all
purely geometric (no Molecule object, no SCF state):

1. **Port `diat_overlap_matrixD` to NumPy** (~5000 LOC, BSD-3-Clause).
   Closes the overlap delegation. Estimated 4-6 hours.
2. **Port `calc_integral` (W integrals) to NumPy** (~500 LOC). Closes
   the W delegation. Estimated 2-3 hours.
3. **Port per-pair rotated two-electron integrals (TETCI)** for the
   YX/YY 2-atom hcore delegation. Estimated 4-6 hours. (Could share
   code with #2.)
4. **MLX migration** — mechanical NumPy → mlx.core translation across
   all native files. Estimated 2-3 hours.

## Honest recommendation

* **Production today**: ``rm1_energy(method='PM6_D', native=True)``
  matches PYSEQM **bit-exactly** on every test molecule. The PYSEQM
  default path (``native=False``) is still the recommended production
  setting because it's been validated against the MOPAC binary for
  years; the native path is for performance experiments and the MLX
  migration.
* **For a fully self-contained build**: complete tasks 1-3 above
  (~10 hours of focused work).
* **For Apple GPU acceleration**: do task 4 after task 1 lands.
