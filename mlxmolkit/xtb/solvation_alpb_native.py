# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Pure-MLX ALPB(water) implementation — scoping skeleton.

This module is the planned pure-MLX replacement for the tblite-backed
ALPB path in :mod:`solvation_alpb`. The full port is a transcription
of xtb's Fortran solvation stack at ``/private/tmp/xtb-src/src/solv/``:

    born.f90      (~110 lines) — analytical Born radii via integration
                                 over neighbor-list pair geometry.
    sasa.f90      (~210 lines) — numerical SASA via Lebedev quadrature.
    gbsa.f90      (~1160 lines) — Born matrix + HB correction +
                                  energy/gradient assembly.
    kernel.f90    (~760 lines) — pairwise GB kernel (R-dependence).
    model.f90     (~770 lines) — solvent-specific parameter table
                                 (water at minimum).
    lebedev.f90   (~5080 lines) — quadrature points; can be vendored
                                  as numpy data, no logic to port.
    ddvolume.f90  (~200 lines) — volume integration helpers.
    state.f90     (~90 lines)  — solvation state machine.

That's ~3300 lines of physics + 5080 lines of Lebedev data. Realistic
estimate: 3-5 days of careful porting + FD verification per piece.

API contract (matches :func:`solvation_alpb.gfn2_energy_alpb_water`):

    gfn2_alpb_water_native(atoms, coords_ang, charge=0)
        -> dict with energy_hartree (incl. ALPB), shell_charges, ...

Implementation phasing (when the time is invested):

    Phase A1 — Born radii (born.f90:compute_bornr)
              Pure-MLX neighbor-list pair sum. ~½ day.

    Phase A2 — ALPB Coulomb kernel (kernel.f90)
              Closed-form pairwise function with Born radii. ~½ day.

    Phase A3 — SASA (sasa.f90 + lebedev.f90 quadrature points)
              Vendor lebedev grids as ``.npz``; port numerical
              integration. ~1 day.

    Phase A4 — HB correction (gbsa.f90:compute_fhb)
              Per-pair HB scaling on SASA. ~½ day.

    Phase A5 — Energy + gradient assembly (gbsa.f90:getEnergy/addGradient)
              Combines Born + HB + SASA. Analytical gradient via
              ``addBornDeriv`` + ``addADetDeriv``. ~1 day.

    Phase A6 — ALPB(water) parameter table (model.f90)
              Vendor as ``.npz``: dielectric ε, vdW radii, freeEnergyShift,
              HB strength per element. ~½ day extraction.

    Phase A7 — Parity tests vs tblite ALPB(water)
              ≤ 0.1 kcal/mol on H2O / NH3 / CH4 / methanol / formamide /
              acetone / glycine. ~½ day.

Today (2026-05-09) the production COSMO-RS pipeline uses tblite's
ALPB via :func:`solvation_alpb.gfn2_alpb_water_optimize`. That works
correctly with analytical gradients. The pure-MLX port is strategic-
purity work, not a feature gap.

When you're ready to start: read ``born.f90`` first (it's the smallest
and fully analytical — gives you the data structures and parameter
shape). Then ``kernel.f90`` for the Coulomb form. SASA is the only
piece that needs Lebedev quadrature.
"""

# Intentionally empty — see docstring for the porting plan.
