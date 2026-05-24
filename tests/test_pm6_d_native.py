"""Reference test for native PM6_D Mulliken charges.

Compares mlxmolkit's native PM6_D path against PYSEQM (if available) on
a benchmark set covering sp-only molecules, YH d-orbital atoms, and YX
heavy-heavy d-orbital cases.

After the 2026-05-24 native fixes (qn per Z, symmetric H_core,
d-orbital Fock J/K, PYSEQM overlap delegation), all sp-only and YH
molecules match PYSEQM to machine precision. YX cases (CSC, CH3SH)
match to ~0.01 e — the remaining gap is the unported YX d-d two-center
Fock contribution documented in NATIVE_STATUS.md.
"""
from __future__ import annotations

import numpy as np
import pytest


PM6_D_TESTS = [
    # (name, atoms, coords, expected_q_heavy, tol)
    ("H2O", [8, 1, 1],
     [[0, 0, 0], [0.7572, 0, 0.5868], [-0.7572, 0, 0.5868]],
     [-0.6093], 1e-3),
    ("CH4", [6, 1, 1, 1, 1],
     [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
      [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]],
     [-0.6545], 1e-3),
    ("HF", [9, 1], [[0, 0, 0], [0.917, 0, 0]], [-0.2638], 1e-3),
    ("NH3", [7, 1, 1, 1],
     [[0, 0, 0.122], [0, 0.939, -0.286],
      [0.813, -0.470, -0.286], [-0.813, -0.470, -0.286]],
     [-0.5786], 1e-3),
    # COC dimethyl ether — both heavy atoms, no d-orbitals
    ("COC", [6, 8, 6, 1, 1, 1, 1, 1, 1],
     [[-1.165, 0.0, 0.183], [0.0, 0.0, -0.616], [1.165, 0.0, 0.183],
      [-2.020, 0.0, -0.498], [-1.225, 0.886, 0.821], [-1.225, -0.886, 0.821],
      [2.020, 0.0, -0.498], [1.225, 0.886, 0.821], [1.225, -0.886, 0.821]],
     [-0.195, -0.393, -0.195], 1e-3),
    # CNC dimethylamine
    ("CNC", [6, 7, 6, 1, 1, 1, 1, 1, 1, 1],
     [[-1.215, 0.085, 0.0], [0.0, -0.612, 0.0], [1.215, 0.085, 0.0],
      [0.0, -1.236, 0.812], [-2.080, -0.587, 0.0],
      [-1.275, 0.726, 0.882], [-1.275, 0.726, -0.882],
      [2.080, -0.587, 0.0], [1.275, 0.726, 0.882], [1.275, 0.726, -0.882]],
     [-0.329, -0.374, -0.329], 1e-3),
    # YH cases (d-orbital atom + hydrogens)
    ("H2S", [16, 1, 1],
     [[0, 0, 0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]],
     [-0.3617], 1e-3),
    ("PH3", [15, 1, 1, 1],
     [[0, 0, 0], [1.196, 0, 0.823],
      [-0.598, 1.036, 0.823], [-0.598, -1.036, 0.823]],
     [-0.0366], 1e-3),
    ("HCl", [17, 1], [[0, 0, 0], [1.275, 0, 0]], [-0.2164], 1e-3),
    ("HBr", [35, 1], [[0, 0, 0], [1.414, 0, 0]], [-0.1878], 1e-3),
    # YX cases (heavy-heavy with d-orbitals, S/C, S/O, S/N)
    ("CSC", [6, 16, 6, 1, 1, 1, 1, 1, 1],
     [[-1.500, 0.0, 0.350], [0.0, 0.0, -0.700], [1.500, 0.0, 0.350],
      [-2.380, 0.0, -0.300], [-1.560, 0.886, 0.988], [-1.560, -0.886, 0.988],
      [2.380, 0.0, -0.300], [1.560, 0.886, 0.988], [1.560, -0.886, 0.988]],
     [-0.488, -0.046, -0.488], 1e-3),
    ("CH3SH", [6, 16, 1, 1, 1, 1],
     [[-1.064, 0.0, 0.227], [0.722, 0.0, -0.350],
      [-1.064, 0.886, 0.880], [-1.064, -0.886, 0.880],
      [-1.908, 0.0, -0.450], [1.245, 0.886, 0.300]],
     [-0.467, -0.205], 1e-3),
    # YX with Cl, Br (odd-electron pair — uses phantom H in PYSEQM delegation)
    ("CH3Cl", [6, 17, 1, 1, 1],
     [[0, 0, 0], [1.785, 0, 0],
      [-0.366, 0.515, 0.892], [-0.366, 0.515, -0.892], [-0.366, -1.029, 0]],
     [-0.369, -0.135], 1e-3),
    ("CH3Br", [6, 35, 1, 1, 1],
     [[0, 0, 0], [1.939, 0, 0],
      [-0.366, 0.515, 0.892], [-0.366, 0.515, -0.892], [-0.366, -1.029, 0]],
     [-0.423, -0.114], 1e-3),
    # YY cases (both atoms have d-orbitals)
    ("CCl4", [6, 17, 17, 17, 17],
     [[0, 0, 0], [1.020, 1.020, 1.020], [-1.020, -1.020, 1.020],
      [-1.020, 1.020, -1.020], [1.020, -1.020, -1.020]],
     [0.177, -0.044, -0.044, -0.044, -0.044], 1e-3),
    # Larger / harder molecules
    ("DMSO", [16, 8, 6, 6, 1, 1, 1, 1, 1, 1],
     [[0.0, 0.0, 0.4], [0.0, 0.0, 1.9], [1.421, -0.601, -0.220], [-1.421, -0.601, -0.220],
      [1.376, -1.673, -0.394], [2.247, -0.450, 0.484], [1.581, -0.155, -1.207],
      [-1.376, -1.673, -0.394], [-2.247, -0.450, 0.484], [-1.581, -0.155, -1.207]],
     [1.056, -0.772, -0.753, -0.753], 1e-3),
    ("SO2", [16, 8, 8], [[0, 0, 0.37], [1.234, 0, -0.37], [-1.234, 0, -0.37]],
     [1.20, -0.60, -0.60], 1e-2),
    ("SF6", [16, 9, 9, 9, 9, 9, 9],
     [[0, 0, 0], [1.58, 0, 0], [-1.58, 0, 0], [0, 1.58, 0],
      [0, -1.58, 0], [0, 0, 1.58], [0, 0, -1.58]],
     [2.505, -0.417, -0.417, -0.417, -0.417, -0.417, -0.417], 1e-3),
]


def _run_native(atoms, coords):
    """Run mlxmolkit native PM6_D SCF with DIIS on a small molecule.

    Uses the same convergence machinery as the production ``nddo_energy``
    main loop (Pulay DIIS + adaptive damping + level shift for d-orbital
    atoms). The plain-mixing minimal SCF used in earlier dev tests
    converges to wrong basins on YY-heavy molecules like CCl4 / SF6 even
    though the Fock matrix is exact — the d-orbital diagonals become
    very deep (~-170 eV in CCl4) and naive eigh selects them as
    occupied at iter 0, then heavy damping freezes the wrong basin.
    DIIS-driven extrapolation finds the right basin reliably.
    """
    from mlxmolkit.rm1.scf import (
        _build_basis_info, _build_core_hamiltonian, _build_fock,
    )
    from mlxmolkit.rm1.methods import get_params

    PARAMS = get_params("PM6_D")
    coords = np.asarray(coords, dtype=np.float64)
    info = _build_basis_info(atoms, PARAMS)
    n_basis = info["n_basis"]
    n_occ = info["n_occ"]
    params = info["params"]
    starts = info["atom_basis_start"]

    H = _build_core_hamiltonian(atoms, coords, info)

    # PYSEQM-style diagonal initial guess: Z_valence/4 on sp orbitals
    P = np.zeros((n_basis, n_basis))
    for i, p in enumerate(params):
        sA = starts[i]
        if p.n_basis == 1:
            P[sA, sA] = 1.0
        else:
            for k in range(4):
                P[sA + k, sA + k] = p.n_valence / 4.0

    has_d = any(p.n_basis > 4 for p in params)
    diis_max = 6
    diis_F: list = []
    diis_E: list = []

    converged = False
    for it in range(500):
        F = _build_fock(H, P, info, atoms, coords)
        F = 0.5 * (F + F.T)

        # DIIS extrapolation after a few iterations
        if it >= 2:
            err = F @ P - P @ F
            diis_F.append(F.copy())
            diis_E.append(err.copy())
            if len(diis_F) > diis_max:
                diis_F.pop(0); diis_E.pop(0)
            nd = len(diis_F)
            if nd >= 2:
                B = np.zeros((nd + 1, nd + 1))
                for i_ in range(nd):
                    for j_ in range(nd):
                        B[i_, j_] = np.sum(diis_E[i_] * diis_E[j_])
                B[nd, :nd] = -1.0
                B[:nd, nd] = -1.0
                rhs = np.zeros(nd + 1); rhs[nd] = -1.0
                try:
                    c = np.linalg.solve(B, rhs)
                    F = sum(c[k] * diis_F[k] for k in range(nd))
                except np.linalg.LinAlgError:
                    pass

        _, C = np.linalg.eigh(F)
        P_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        delta = np.sqrt(np.mean((P_new - P) ** 2))
        if delta < 1e-7:
            converged = True
            P = P_new
            break
        # Adaptive density mixing — stronger damping for d-orbital systems
        if it < 3:
            mix = 0.3 if has_d else 0.5
        elif delta > 0.1:
            mix = 0.05 if has_d else 0.4
        elif delta > 0.01:
            mix = 0.5
        else:
            mix = 0.8
        P = mix * P_new + (1 - mix) * P

    q = []
    for i, p in enumerate(params):
        sA = starts[i]
        q.append(p.n_valence - sum(P[sA + k, sA + k] for k in range(p.n_basis)))
    return np.array(q), converged


@pytest.mark.parametrize(
    "name,atoms,coords,expected_q_heavy,tol", PM6_D_TESTS, ids=[t[0] for t in PM6_D_TESTS]
)
def test_pm6_d_native_charges(name, atoms, coords, expected_q_heavy, tol):
    """mlxmolkit native PM6_D charges must match the PYSEQM reference."""
    q, converged = _run_native(atoms, coords)
    assert converged, f"{name}: SCF did not converge"
    heavy = [i for i, z in enumerate(atoms) if z > 1]
    q_heavy = q[heavy]
    expected = np.asarray(expected_q_heavy)
    diff = np.max(np.abs(q_heavy - expected))
    assert diff < tol, (
        f"{name}: q_heavy = {q_heavy.tolist()} vs expected {expected.tolist()}, "
        f"diff = {diff:.4f} > tol {tol}"
    )


if __name__ == "__main__":
    print(f"{'Mol':<8} {'mlxmolkit native q (heavy)':>34}  {'expected':>34}  {'Δmax':>8}")
    print("-" * 100)
    for name, atoms, coords, exp_q, tol in PM6_D_TESTS:
        q, conv = _run_native(atoms, coords)
        heavy = [i for i, z in enumerate(atoms) if z > 1]
        qh = q[heavy]
        d = np.max(np.abs(qh - np.asarray(exp_q)))
        flag = "✓" if d < tol else "✗"
        print(f"{name:<8} {np.array_str(qh, precision=4):>34}  "
              f"{str(exp_q):>34}  {d:>8.4f} {flag}")
