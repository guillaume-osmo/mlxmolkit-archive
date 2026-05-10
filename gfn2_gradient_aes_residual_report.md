# gfn2_gradient_analytical — open questions for the AES grad residual

## Goal

Produce an analytical gradient dE/dR for GFN2-xTB that matches the central-difference gradient on H2O to FD-floor. Currently FD-floor is achieved for GFN1 (8e-7 Ha/Å on H2O — gold standard) but GFN2 has a stubborn ~6.3e-3 Ha/Å residual concentrated in the H-H interaction direction. Production opt uses tblite's mature C path; this is a strategic-purity / pure-MLX item.

## Reference implementation (xtb, LGPL-3.0)

Repo: grimme-lab/xtb (https://github.com/grimme-lab/xtb)

- `src/scc_core.f90:50-150` — H0 build (`build_h0`) + `buildIsoAnisotropicH1` (Fock from H0 + V_es + F_aes)
- `src/scc_core.f90:686-722` — electro (energy = trace(P·H0)·evtoau + es from ies%getEnergy)
- `src/aespot.F90:441-664` — `aniso_electro` (`E_aes` ENERGY) and `aniso_electro_cpu`
- `src/aespot.F90:676-709` — `fockelectro` (`e_aes` returned via `e = 0.25 · sum(P · f)`)
- `src/aespot.F90:1070-1165` — `aniso_grad` (∂E_aes/∂R via `gab3`/`gab5`/`rij` with `q,dipm,qp` fixed)
- `src/xtb/hamiltonian_gpu.f90:778-1069` — `build_dSDQH0_gpu` (the band gradient: `stmp`/`dtmp`/`qtmp` formula)
- `src/intgrad.f90:703-728` — `shiftintg` (origin shift transformation for multipole derivatives)
- `src/scf_module.F90:660-790` — top-level GFN2 gradient assembly

## Convention key facts

xtb's stored `dpint(k,i,j)` and `qpint(k,i,j)` are at per-pair ket-atom origin (`build_dsdq_ints` called with `point=rj` at `hamiltonian_gpu.f90:942`). So:

```text
xtb_dpint = mlxmolkit_dpint - rj·S
```

per AO pair (μ on iat, ν on jat).

`vs(nat)`, `vd(3,nat)`, `vq(6,nat)` are per-atom AES potentials from `setvsdq`, derived such that:

```text
∂E_aes/∂q = vs
∂E_aes/∂dipm = vd
∂E_aes/∂qp = vq
```

at SCF convergence (variational definition).

mlxmolkit stores `dpint`/`qpint` at frame origin (=0). So mlxmolkit's `fockelectro` builds `F_aes` using frame-origin integrals; xtb's `buildIsoAnisotropicH1` uses ket-origin. Both yield the same eigenvalues (the difference shifts as a constant in F absorbed by the orbital occupation, AFAICT) — but the GRADIENT formulas differ.

## E_total (mlxmolkit GFN2)

```text
E_total = trace(P·H0) + ½ q·J·q + ⅓ q³·Γ + E_aes_aniso + E_rep + E_d4
```

with:

- `H0[μ,ν] = K_AB · ζ_ij · h_avg · Π · S[μ,ν]` for off-diag (`h_avg` has CN-shift via `selfE`)
- `J = Klopman-Ohno γ matrix` (shell-resolved)
- `E_aes_aniso` from `aniso_electro(q, dipm, qp, gab3, gab5)` — the energy-only form

During the SCF:

```text
F = H0 − ½(V·S + S·V) + F_aes
```

where `V_sh = J·q + q²·Γ_sh` and `F_aes` from `fockelectro` (which uses `e_aes = 0.25·sum(P·f)`).

Critical:

```text
aniso_electro(q,dipm,qp,gab3,gab5) ≠ 0.5·trace(P·F_aes)
```

in general. On H2O at convergence:

```text
aniso_electro = +1.4e-3 Ha
0.5·trace(P·F_aes) = −9.5e-3 Ha
```

They differ by sign and magnitude — so the SCF is not classically variational in the `(q,dipm,qp)` basis.

## My analytical formulation (Pulay-style)

For GFN1 (no AES) this gives FD-floor accuracy:

```text
dE/dR = trace(P·∂H0/∂R)              # HF-diag (CN chain) + HF-offdiag (full chain)
      + ½ q·∂J·q                      # Coulomb explicit
      − trace(W·∂S/∂R)                # Pulay (W = energy-weighted density)
      − trace(V_eff_diag · ∂S · P)    # Coulomb-overlap cross
      + ∂E_rep/∂R                     # closed-form
```

For GFN2 I extend with `V_eff = V_sh − vs` (so the `F_aes vs` piece is absorbed into the `V·∂S·P` cross), and add `aniso_grad`-equivalent / full-`E_aes` FD plus FD on `E_d4`:

```text
g_band_a = (g_pulay + g_hf_off + g_coul + g_vsop_cross + g_band_aes_vd_vq) * A2B + g_hf_diag
g_total = g_band_a + g_rep + g_aes + g_d4
```

The `g_band_aes_vd_vq` (analytical via origin-shifted multipole derivatives, xtb's `dtmp`/`qtmp` formula) is currently set to zero because adding it doubles the residual to ~3.6e-2 Ha/Å. The cancellation argument: in the substitution:

```text
E_total = 2Σε − V·z + ½q·J·q + ⅔q³Γ − 2·E_focke + E_aes_aniso
```

the `trace(P·∂F_aes/∂R)` from `d(2Σε)/dR` is exactly cancelled by `−2·dE_focke/dR`. So `band_aes` shouldn't appear — empirically confirmed.

## Empirical state (H2O, `conv_tol=1e-9`, `h=1e-3 Å`)

| Component | max\|·\| (Ha/Å) |
| --- | ---: |
| pulay | 4.71e−1 |
| hf_diag | 3.5e−4 |
| hf_offdiag | 6.63e−1 |
| coulomb | 2.96e−2 |
| vsop_cross | 6.32e−2 |
| band_aes_vd_vq | 0 (set) |
| repulsion | 1.97e−1 |
| aes (FD) | 4.16e−2 |
| dispersion (FD) | 7.87e−5 |
| Total analytical max | 1.99e−2 |
| Total FD max | 1.36e−2 |
| Residual (max\|Δ\|) | 6.3e−3 |

Residual is concentrated in H y-coordinates (perpendicular to the plane spanning O-H-H — the H atom lateral direction):

```text
residual on O.z:  −2.4e-4
residual on H1.y: −6.3e-3   ← dominant
residual on H1.z: +1.2e-4
residual on H2.y: +6.3e-3   ← mirror
residual on H2.z: +1.2e-4
```

Same FD diagnosis with the `V·∂S·P` cross using `V_eff = V_sh` (no `vs` subtraction): 2.0e-2 residual. With `V_eff = V_sh − vs`: 6.3e-3. So `vs` absorption helps but doesn't close the gap.

## What I tried, what failed

1. Add `band_aes_vd_vq` via shifted multipole derivatives (xtb's `dtmp`/`qtmp`). Off by ~5x in the wrong direction. Cancellation argument explains why this is correct as zero.

2. Switch AES FD to `aniso_grad`-equivalent (multipoles fixed, only `gab3`/`gab5`/`rij` vary). Worsens to 3.6e-2 — confirms the `dipm`/`qp` Mulliken chain DOES contribute to dE/dR (multipoles aren't variational; they're Mulliken-derived from P).

3. Add `q·∂J·z` term (which my algebraic derivation predicts via the `V·z` piece in `E_total = 2Σε − V·z + ...`). On H2O it's 0.187 Ha/Å — way larger than the residual. Adding/subtracting either makes things drastically worse.

GFN1 has the same algebraic structure (just no AES) and matches FD to 8e-7. So whatever cancellation makes `q·∂J·z` drop out for GFN1 should also work for GFN2.

## Open questions for ChatGPT

## Working Answers / Current Resolution

### A1

The `q·∂J·z` term is an artifact of differentiating the eigenvalue-sum rewrite inconsistently. In the direct Lagrangian / Pulay form, the fixed-density derivative of `½q·J·q` gives the explicit `+½q·∂J·q` term, while the derivative of the Mulliken population closure `q = z - pop(P,S)` is represented by the overlap-cross term using the Fock shift `V = Jq (+ q²Γ for GFN2)`. If the `V·z` term from:

```text
trace(P·H0) = 2Σε − V·z + V·q − ...
```

is differentiated separately without the matching `∂V` contribution already inside `d(2Σε)`, it creates a spurious `q·∂J·z`. The GFN1 FD-floor result is the empirical proof: the correct explicit kernel term is `+½q·∂J·q`, not `+q·∂J·z − ½q·∂J·q`.

### A2

The `−2·E_focke` term is not part of the original total energy. It appears only after rewriting `trace(P·H0)` through the eigenvalue sum, because the eigenvalue sum contains `trace(P·F_aes)`. In that rewritten algebra, `trace(P·∂F_aes)` cancels against `−2·dE_focke` at fixed AES potentials.

That cancellation does **not** remove the AES gradient. The remaining term is the derivative of the actual scalar energy `E_aes_aniso = aniso_electro(...)`. In the current mlxmolkit split, where `E_aes_aniso` is finite-differenced with frozen `P/qsh` while recomputing Mulliken `dipm/qp`, the explicit `band_aes_vd_vq` term should remain zero to avoid double-counting the `dpint/qpint` chain.

### A3

The H-y residual is not evidence for a missing physical H-H-only term. It is the symmetry channel most sensitive to AES origin/overlap bookkeeping in water: antisymmetric lateral H motion changes the two H-centered CAMM dipoles/quadrupoles with opposite signs while preserving much of the monopole geometry. The likely sources are therefore bookkeeping/split issues in AES multipole derivatives, not a missing quadrupole-quadrupole interaction in `aniso_electro`.

### A4

Both approaches are valid, but they are different gradient splits:

- xtb split: `aniso_grad` holds `q,dipm,qp` fixed and differentiates only the pair kernels/geometry; the missing Mulliken integral chain is supplied by the `stmp/dtmp/qtmp` band-gradient machinery, using `setdvsdq` and atom-origin multipole derivatives.
- current mlxmolkit split: finite-difference `E_aes_aniso` with frozen `P/qsh`, recomputing `dipm/qp` from perturbed `S/dpint/qpint`. This already includes the Mulliken dipole/quadrupole chain, so adding a separate `dtmp/qtmp` band term double-counts it.

One new code-level finding: xtb has a separate `setdvsdq` routine for nuclear gradients, distinct from `setvsdq` used for Fock construction. `setdvsdq` removes coordinate-origin shift terms because its potentials are meant to multiply atom-origin multipole derivative integrals.

### A5

The origin convention matters for analytical gradient bookkeeping. Frame-origin integrals can be used consistently with the current full-FD AES split. The ket-origin / atom-origin convention is needed when implementing xtb's fully analytical `stmp/dtmp/qtmp` split. The right near-term fix is not to globally change `multipole_integrals`, but to keep frame-origin storage and apply origin shifts only in the xtb-style analytical gradient path.

## Diagnostics From This Pass

- Replacing the current full-FD AES split with fixed-multipole AES plus a reconstructed `vd/vq·∂(D/Q)` band term did **not** close the residual in local tests.
- Using `setdvsdq` in the current overlap-cross path worsened the residual; `setdvsdq` belongs with the xtb-style atom-origin `dtmp/qtmp` path, not the current frame-origin full-FD path.
- Adding AES to the final post-SCF rediagonalization changed the numerical energy surface and worsened the analytical-vs-FD residual to ~2.6e-2 Ha/Å on the local H2O test. That change was reverted.
- The SCF convergence check has been strengthened to track `qsh`, `dipm`, and `qp`, matching xtb's GFN2 state-vector logic more closely.

### Q1 (highest priority)

Why does my algebraic Pulay-decomposition predict a `+q·∂J·z − ½q·∂J·q` contribution while empirically only `+½q·∂J·q` shows up (per GFN1's FD-floor match)?

Specifically: starting from:

```text
E_total = trace(P·H0) + ½q·J·q + ⅓q³Γ + ...
```

and substituting:

```text
trace(P·H0) = 2Σε − V·z + V·q − 2·E_focke
```

where:

```text
V_q = q·J·q + Σq³Γ
```

I get extra:

```text
+q·∂J·z − ½q·∂J·q
```

at fixed q. Should the `V·z` term cancel via Mulliken closure? Working out: is `V·z = q·J·z + Σ q²·Γ·z_atom_total` constant under fixed q if z is constant — but `∂(V·z)/∂R = q·∂J·z` is non-zero...

### Q2

For mlxmolkit's specific GFN2 SCF where `E_aes` uses `aniso_electro` (NOT `0.5·trace(P·F_aes)` from `fockelectro`), does my cancellation argument:

```text
dE_total/dR has trace(P·∂F_aes/∂R) − 2·dE_focke/dR = 0
```

actually hold, given that:

```text
E_aes_aniso ≠ 2·E_focke
```

Specifically my substitution wrote:

```text
E_total = ... − 2·E_focke + E_aes_aniso
```

but should it actually be just:

```text
+ E_aes_aniso
```

(no separate `−2·E_focke` term, since `E_focke` was never in `E_total` to begin with)?

### Q3

The 6.3e-3 residual is on H y-coords (perpendicular to molecular plane). What physical / mathematical pieces would be asymmetric in the H-H direction that I might be missing?

Candidates:

- (a) a missing AES quadrupole-quadrupole term I'm not capturing in my FD on `E_aes`
- (b) a Pulay-form correction specific to the d-shell / multipole-sector basis derivatives

### Q4

Looking at xtb's `aniso_grad` vs my full-`E_aes` FD: xtb holds `q`, `dipm`, `qp` FIXED and only varies `gab3`/`gab5`/`rij`. I FD with `q`, `qsh`, `P` fixed but recompute `dipm`/`qp` from perturbed `dpint`/`qpint` at each step. Which is correct for GFN2's variational structure?

xtb's choice presumably is correct (it gives FD-matching gradients for production) — but then how does xtb capture the `dipm`/`qp` chain through `dpint`/`qpint`? Is that what `dtmp`/`qtmp` (which I set to zero) is supposed to do?

### Q5

xtb's `build_dsdq_ints` is called with `point=rj` (line 942 of `hamiltonian_gpu.f90`), making xtb's stored `dpint`/`qpint` use ket-origin convention. mlxmolkit uses frame-origin throughout. For the `F_aes` Fock contribution, both conventions give the same SCF eigenvalues (the per-pair shift is absorbed). But for the gradient via `trace(P·∂F_aes/∂R)`, does the convention matter?

And is the right fix to:

- (a) change mlxmolkit's `multipole_integrals` to ket-origin
- (b) apply origin shift in `fockelectro`
- (c) something else?

## Reproduction setup

Python 3.11, NumPy, scipy

### Environment & reproduction

```text
# Python interpreter
/Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3   # Python 3.11.8

# Repo paths (importable via PYTHONPATH)
~/Github/mlxmolkit                  # main project — branch feature/xtb-gfn0
~/Github/mlx-addons/src             # MLX primitives (mlx_addons.linalg.gen_eigh etc.)
```

Versions in the `osmo` env:

```bash
/Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 -c "
import sys, numpy, scipy, mlx, tblite
print(f'Python:  {sys.version.split()[0]}')
print(f'NumPy:   {numpy.__version__}')
print(f'SciPy:   {scipy.__version__}')
print(f'MLX:     {mlx.__version__}')
print(f'tblite:  {tblite.__version__ if hasattr(tblite, \"__version__\") else \"(installed)\"}')
" 2>&1; /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 -c "
try:
    import dftd4
    print(f'dftd4:   {dftd4.__version__ if hasattr(dftd4, \"__version__\") else \"(installed)\"}')
except ImportError:
    print('dftd4:   not installed')
" 2>&1
```

Observed output:

```text
Traceback (most recent call last):
  File "<string>", line 6, in <module>
AttributeError: module 'mlx' has no attribute '__version__'
Python:  3.11.8
NumPy:   1.26.4
SciPy:   1.14.1
dftd4:   4.2.0
```

Exact versions via package metadata:

```bash
/Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 -c "
import importlib.metadata as md
for pkg in ['mlx', 'tblite', 'numpy', 'scipy', 'dftd4', 'rdkit']:
    try:
        print(f'{pkg:<10} {md.version(pkg)}')
    except Exception:
        print(f'{pkg:<10} not found')
"
```

Observed output:

```text
mlx        0.30.6
tblite     0.4.0
numpy      1.26.3
scipy      1.16.3
dftd4      4.2.0
rdkit      2025.9.4+osmordredv2
```

Summary:

```text
Python:  3.11.8
NumPy:   1.26.x       (note: setuptools metadata says 1.26.3, runtime says 1.26.4 — both work)
SciPy:   1.14 / 1.16  (system reports 1.16.3 from metadata; runtime 1.14.1)
MLX:     0.30.6       (Apple Metal backend; pinned)
tblite:  0.4.0        (Python bindings via conda-forge)
dftd4:   4.2.0        (used by simple-dftd4 wrapper for D4 reference)
RDKit:   2025.9.4+osmordredv2
Platform: Darwin 25.2.0 (Apple Silicon, M-series)
```

Reproduction snippet:

```bash
PYTHONPATH=~/Github/mlxmolkit:~/Github/mlx-addons/src \
~/miniconda3/envs/osmo/bin/python3 << 'EOF'
import numpy as np
from mlxmolkit.xtb.gradient_gfn2 import gfn2_gradient_analytical, numerical_gradient

atoms = [8, 1, 1]
coords = np.array([
    [0.0, 0.0,  0.117790],
    [0.0, 0.755453, -0.471160],
    [0.0,-0.755453, -0.471160],
])
res = gfn2_gradient_analytical(atoms, coords)
g_num = numerical_gradient(atoms, coords, h=1e-3)

print("Analytical (Ha/Å):"); print(res["gradient"])
print("\nFD (Ha/Å):");        print(g_num)
print(f"\nMax |Δ| = {np.max(np.abs(res['gradient']-g_num)):.3e} Ha/Å")
print(f"Max |g_FD| = {np.max(np.abs(g_num)):.3e} Ha/Å")

print("\nComponent magnitudes:")
for k, v in res["components"].items():
    print(f"  {k:>14}: {np.max(np.abs(v)):.3e}")
EOF
```

Expected output: total residual ~6.3e-3 Ha/Å, dominated by H y-coordinates. GFN1 (analogous structure, no AES) gives ~8e-7 Ha/Å — so the algebraic decomposition / Pulay structure is correct in form, but something in the AES sector for GFN2 contributes a small leftover that's invariant in the H-H direction.

Key source files for ChatGPT/Codex to inspect in `~/Github/mlxmolkit/mlxmolkit/xtb/`:

```text
gradient_gfn2.py             # Analytical assembler; detailed F_aes cancellation comments
gradient_gfn1.py             # GFN1 reference (FD-floor working)
gradient_pulay.py            # -trace(W·∂S/∂R) (FD-verified to 5e-7)
gradient_hf_offdiag_gfn2.py  # GFN2 H0 chain rule (K·ζ·h_avg·Π·S form)
gradient_coulomb.py          # ½q·∂J·q (FD-verified to 4e-13)
multipole_grad.py            # ∂dpint/∂r, ∂qpint/∂r + shift_multipole_grad
                             # (all FD-verified to ≤5e-12)
scf_gfn2.py                  # SCF loop (now optionally with alpb_solvent='water')
                             # — see line 374 for V_sh, line 387-393 for AES
hcore_gfn2.py                # H0 build with K·ζ·h_avg·Π·S form
aes.py                       # mmompop, setvsdq, fockelectro, aniso_electro
```

`gradient_gfn2.py::gfn2_gradient_analytical` is the best single function to paste for external review; it has the full assembler with comments on every term.

Each component is FD-verified independently:

- `gradient_pulay.pulay_gradient` — verified against `−∂trace(W·S)/∂R` to 5e-7
- `gradient_hf_offdiag_gfn2.hf_offdiag_gradient_gfn2` — chain rule on `K·ζ·h_avg·Π·S`, verified to 4.5e-9
- `gradient_coulomb.coulomb_gradient` — verified against `∂(½q·J·q)/∂R` to 4e-13
- `multipole_grad.multipole_gradient` — frame-origin `∂dpint`, `∂qpint`, verified to 5e-12
- `multipole_grad.shift_multipole_grad` — origin shift transform, verified to 4e-12

So individual analytical pieces are correct. The structural decomposition (how they sum) for GFN2 is what's off by 6.3e-3 Ha/Å on H2O.
