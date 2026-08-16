# COSMO-SAC: parameters, gaps, and a plan

Source: Bell, Mickoleit, Hsieh, Lin, Vrabec, Breitkopf, Jäger, *A Benchmark
Open-Source Implementation of COSMO-SAC*, J. Chem. Theory Comput. 2020, 16,
2635−2646, [10.1021/acs.jctc.9b01016](https://dx.doi.org/10.1021/acs.jctc.9b01016).

## Why this and not ADFCRS

ADFCRS-2018 needs an SCM licence — the download is 401 without one. This paper
ships **σ-profiles and parameters for 2261 species on Zenodo**
([10.5281/zenodo.3669311](https://doi.org/10.5281/zenodo.3669311)), openly, and
the models are fully specified in the text. It is the reference we can actually
verify against.

It also states our problem as its own central caveat:

> the parametrization and the results of COSMO models in general depend on the
> underlying method and software with which the sigma profiles are calculated.
> Hence, it is very important for the comparability and evaluation of these
> models to use exactly the same set of sigma profiles.

That sentence is the r = 0.16 we measured for PM6 against a DFT-calibrated
kernel. It is not a statement about PM6.

## The parameters

### COSMO-SAC 2002 (Lin & Sandler, as implemented by Mullins et al.)

| parameter | value | ours today | note |
|---|---|---|---|
| `a_eff` | **7.5 Å²** | 6.226 | standard segment area |
| `r_av` | **0.81764 Å** | 0.5 | averaging radius — Mullins', not Klamt's |
| `c_hb` | **85580 kcal Å⁴ mol⁻¹ e⁻²** | 27488747 J/mol | different units *and* model |
| `σ_hb` | **0.0084 e Å⁻²** | 0.007686 | HB threshold |
| `α'` | **16466.72 kcal Å⁴ mol⁻¹ e⁻²** | 1.0e8 (ddcosmo) | misfit |
| `q₀` | **79.53 Å²** | 47.999 (`COMB_SG_A_STD`) | normalised area |
| `r₀` | **66.69 Å³** | — (absent) | normalised volume |
| `z` | 10 | 10.0 | coordination number |
| σ grid | **51 bins, −0.025..+0.025, step 0.001 e/Å²** | 301 bins, ±0.150 | |

Two traps the paper documents explicitly:

* **α' in Mullins' own Table 1 is wrong.** The correct value comes from
  `0.3 · f_pol · a_eff^(3/2) / ε₀` with `ε₀ = 2.395e-4 (e² mol)/(kcal Å)`,
  `ε = 3.667`, `f_pol = (ε−1)/(ε+0.5)`, `a_eff = 7.5 Å²` — matching their
  Fortran, not their paper.
* **`l_i` in Lin & Sandler has misplaced parentheses**; the corrected form is
  the paper's eq 18.

### The averaging radius is not one number

The σ-averaging kernel carries an `f_decay` that exists only because Lin &
Sandler confused Å with bohr; their erratum sets it to `0.52918⁻² ≈ 3.57`.
Three conventions are in circulation:

| convention | `r_av` | `f_decay` |
|---|---|---|
| Klamt | 0.5 Å | 1 |
| Lin & Sandler | `(a_eff/π)^0.5` | 3.57 |
| **Mullins (VT)** | **0.81764 Å** | **1** |
| ours | 0.5 Å | 1 (no `f_decay` term at all) |

**Benchmarking against a Mullins/VT-derived database while averaging at
r_av = 0.5 Å compares differently-smoothed surfaces.** This has to be switched
per reference set, not fixed.

### COSMO-SAC 2010 (Hsieh et al.)

Changes on top of 2002:

* `a_eff` = **7.25 Å²**, and averaging by eq 4 (the `f_decay` form).
* Electrostatic parameter becomes temperature dependent:
  `c_ES(T) = A_ES + B_ES/T²`, with `A_ES = 6525.69 kcal Å⁴ mol⁻¹ e⁻²` and
  `B_ES = 1.4859e8 kcal Å⁴ K² mol⁻¹ e⁻²`.
* Hydrogen bonding splits by *pair type* rather than one constant:
  `c_OH-OH = 4013.78`, `c_OT-OT = 932.31`, `c_OH-OT = 3016.43`
  (kcal Å⁴ mol⁻¹ e⁻²), and zero otherwise — only for `σᵗ·σˢ < 0`.

### COSMO-SAC-dsp

2010 plus a dispersion term in the mixture interaction. Molecules are typed
`DSP_WATER`, `DSP_COOH`, `DSP_HB_ONLY_ACCEPTOR`, `DSP_HB_DONOR_ACCEPTOR`,
`DSP_NHB`. Aborts if an atom outside {C, H, O, N, F, Cl} is present — so it is
not usable for the sulfur-bearing fragrance molecules we care about.

## The structural gap: profiles split three ways

This is the part that is not a constant. COSMO-SAC 2010 splits each σ-profile
into **NHB / OH / OT**, where every atom is classified:

* **OH** — the oxygen or the hydrogen of an O–H pair
* **OT** — N, F, or an O not in an OH group
* **NHB** — everything else

with the segment routed by the *sign* of its averaged charge, and a Gaussian
hydrogen-bonding probability (Wang et al.)

    p_hb(σ) = 1 − exp(−σ² / 2σ₀²),  σ₀ = 0.007 e/Å²

so that `p = p_NHB + p_OH + p_OT` reconstructs the original profile exactly.

**We produce one profile plus a donor/acceptor flag** (`HB_DONOR_ELEMENTS =
{H}`, `HB_ACCEPTOR_ELEMENTS = {N, O, F, S}`). That is the 2002-era shape, not
2010's. Note also that COSMO-SAC's classes have **no sulfur** — ours includes
it as an acceptor, which is a deliberate divergence worth keeping and flagging.

## Plan

1. **Add a COSMO-SAC parameter set** beside the existing one, selected the way
   `charge_source` now selects the misfit prefactor. No behaviour change until
   asked for.
2. **Make `r_av` and `f_decay` arguments of the averaging**, not module
   constants. Without this no comparison against VT/Mullins profiles is valid.
3. **Implement the NHB/OH/OT split** in `sigma.py`, keeping the current
   single-profile path intact. Verify `p_NHB + p_OH + p_OT == p` exactly.
4. **Validate against the Zenodo σ-profiles before any activity coefficient.**
   The paper's own instruction: reprocess their COSMO files and agree on
   `p(σ)A` to **1e-14**. That isolates our σ machinery from our thermodynamics
   — if step 4 fails, nothing downstream means anything.
5. **Then** COSMO-SAC 2002 and 2010 activity coefficients, checked against the
   paper's validation set.
6. **Only then** feed PM6 charges in. At that point a disagreement is PM6's,
   which is the number we actually wanted.

Steps 1–4 are the ones that make the existing PM6 benchmark interpretable.
Step 6 is the question that started this.
