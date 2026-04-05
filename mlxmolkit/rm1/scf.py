"""
SCF (Self-Consistent Field) solver for RM1.

Iterative procedure:
  1. Build Fock matrix F from density P and integrals
  2. Diagonalize F → eigenvalues, eigenvectors C
  3. Update density P = C_occ @ C_occ.T
  4. Check convergence |P_new - P_old|

For now: pure numpy implementation (Metal kernel in fock_metal.py later).
"""
from __future__ import annotations

import numpy as np
from .params import RM1_PARAMS, ElementParams, EV_TO_KCAL, ANG_TO_BOHR
from .integrals import (
    compute_one_center_integrals,
    compute_nuclear_repulsion,
    _additive_terms,
    _charge_separations,
    EV,
)


def _build_basis_info(atoms: list[int]):
    """Build basis function → atom mapping."""
    params = [RM1_PARAMS[z] for z in atoms]
    n_basis = sum(p.n_basis for p in params)
    n_elec = sum(p.n_valence for p in params)

    # basis_to_atom[mu] = atom index
    # basis_type[mu] = 0 (s), 1 (px), 2 (py), 3 (pz)
    basis_to_atom = []
    basis_type = []
    atom_basis_start = []
    for i, p in enumerate(params):
        atom_basis_start.append(len(basis_to_atom))
        for k in range(p.n_basis):
            basis_to_atom.append(i)
            basis_type.append(k)  # 0=s, 1=px, 2=py, 3=pz

    return {
        'params': params,
        'n_basis': n_basis,
        'n_elec': n_elec,
        'n_occ': n_elec // 2,  # closed-shell
        'basis_to_atom': np.array(basis_to_atom),
        'basis_type': np.array(basis_type),
        'atom_basis_start': atom_basis_start,
    }


def _overlap_ss(zA: float, zB: float, R_bohr: float, nA: int, nB: int) -> float:
    """Slater overlap integral between two s orbitals.

    Simplified for same principal quantum number.
    """
    if R_bohr < 1e-10:
        return 1.0 if abs(zA - zB) < 1e-10 else 0.0

    # Use Mulliken approximation for speed
    # S = exp(-0.5 * (zA + zB) * R) * polynomial
    rho = 0.5 * (zA + zB) * R_bohr
    if rho > 20:
        return 0.0

    # For (1s|1s): S = exp(-rho) * (1 + rho + rho²/3)
    if nA == 1 and nB == 1:
        return np.exp(-rho) * (1.0 + rho + rho ** 2 / 3.0)

    # For (2s|2s):
    if nA == 2 and nB == 2:
        p = zA * R_bohr / 2.0
        q = zB * R_bohr / 2.0
        if abs(zA - zB) < 1e-6:
            # Same exponent
            t = rho
            return np.exp(-t) * (1.0 + t + 2.0 * t ** 2 / 5.0 + t ** 3 / 15.0)
        else:
            # Different exponents — use numerical integration
            # Approximate with Mulliken
            SA = np.exp(-p) * (1.0 + p + p ** 2 / 3.0)
            SB = np.exp(-q) * (1.0 + q + q ** 2 / 3.0)
            return np.sqrt(SA * SB)

    return np.exp(-rho)  # fallback


def _build_core_hamiltonian(atoms, coords, info):
    """Build core Hamiltonian H_core."""
    n_basis = info['n_basis']
    params = info['params']
    b2a = info['basis_to_atom']
    btype = info['basis_type']
    starts = info['atom_basis_start']

    H = np.zeros((n_basis, n_basis))

    # Diagonal: one-electron one-center
    for mu in range(n_basis):
        i = b2a[mu]
        p = params[i]
        if btype[mu] == 0:
            H[mu, mu] = p.Uss
        else:
            H[mu, mu] = p.Upp

    # Off-diagonal: resonance integrals
    # H_μν = 0.5 * (beta_μ + beta_ν) * S_μν  (Wolfsberg-Helmholz)
    n_atoms = len(atoms)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pA = params[i]
            pB = params[j]
            R = np.linalg.norm(coords[i] - coords[j])
            R_bohr = R * ANG_TO_BOHR

            # Compute overlap and resonance for each basis pair
            for mu_off in range(pA.n_basis):
                mu = starts[i] + mu_off
                beta_mu = pA.beta_s if btype[mu] == 0 else pA.beta_p

                for nu_off in range(pB.n_basis):
                    nu = starts[j] + nu_off
                    beta_nu = pB.beta_s if btype[nu] == 0 else pB.beta_p

                    # Overlap (simplified — only s-s for now)
                    if btype[mu] == 0 and btype[nu] == 0:
                        qnA = 1 if pA.Z <= 2 else 2
                        qnB = 1 if pB.Z <= 2 else 2
                        S = _overlap_ss(pA.zeta_s, pB.zeta_s, R_bohr, qnA, qnB)
                    else:
                        # s-p and p-p overlaps: approximate
                        zeta_mu = pA.zeta_s if btype[mu] == 0 else pA.zeta_p
                        zeta_nu = pB.zeta_s if btype[nu] == 0 else pB.zeta_p
                        rho = 0.5 * (zeta_mu + zeta_nu) * R_bohr
                        S = np.exp(-rho) * (1.0 + rho) if rho < 20 else 0.0
                        # Directional factor for p orbitals
                        if btype[mu] > 0 or btype[nu] > 0:
                            dx = (coords[j] - coords[i]) / max(R, 1e-10)
                            # p orbital direction: 1=x, 2=y, 3=z
                            if btype[mu] > 0:
                                S *= dx[btype[mu] - 1]
                            if btype[nu] > 0:
                                S *= dx[btype[nu] - 1]

                    H[mu, nu] = 0.5 * (beta_mu + beta_nu) * S
                    H[nu, mu] = H[mu, nu]

    # Add electron-nuclear attraction (simplified)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            pB = params[j]
            R = np.linalg.norm(coords[i] - coords[j])
            R_bohr = R * ANG_TO_BOHR

            rho0A = 0.5 * EV / params[i].gss if params[i].gss > 0 else 0.0
            rho0B = 0.5 * EV / pB.gss if pB.gss > 0 else 0.0
            aee = (rho0A + rho0B) ** 2
            Vss = EV / np.sqrt(R_bohr ** 2 + aee)

            # Subtract nuclear attraction from diagonal
            mu0 = starts[i]
            for mu_off in range(params[i].n_basis):
                H[mu0 + mu_off, mu0 + mu_off] -= pB.n_valence * Vss

    return H


def _build_fock(H, P, info, atoms, coords):
    """Build Fock matrix F = H + G(P).

    G_μν = Σ_λσ P_λσ * [(μν|λσ) - 0.5*(μλ|νσ)]

    For NDDO: only one-center two-electron integrals contribute.
    """
    n_basis = info['n_basis']
    params = info['params']
    b2a = info['basis_to_atom']
    btype = info['basis_type']

    F = H.copy()

    # One-center contributions (NDDO: only same-atom integrals)
    for i, p in enumerate(params):
        mask = (b2a == i)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue

        # Extract sub-density for this atom
        P_sub = P[np.ix_(idx, idx)]

        # s orbital index
        s = idx[0]

        # Total electron density on this atom
        Pss = P[s, s]

        if p.n_basis == 1:
            # H atom: only (ss|ss)
            F[s, s] += Pss * p.gss * 0.5
        else:
            # Heavy atom: sp shell
            px, py, pz = idx[1], idx[2], idx[3]
            Ppp_total = P[px, px] + P[py, py] + P[pz, pz]

            # Coulomb on s orbital
            F[s, s] += Pss * p.gss * 0.5 + Ppp_total * (p.gsp - 0.5 * p.hsp)

            # Coulomb on p orbitals
            for k in range(1, 4):
                pk = idx[k]
                F[pk, pk] += Pss * (p.gsp - 0.5 * p.hsp) + \
                             P[pk, pk] * p.gpp * 0.5 + \
                             (Ppp_total - P[pk, pk]) * (p.gp2 - 0.5 * (p.gpp - p.gp2))

            # Exchange sp terms
            for k in range(1, 4):
                pk = idx[k]
                F[s, pk] += P[s, pk] * (2.0 * p.hsp - 0.5 * p.gsp)
                F[pk, s] = F[s, pk]

            # Exchange pp' terms
            for k in range(1, 4):
                for l in range(k + 1, 4):
                    pk, pl = idx[k], idx[l]
                    F[pk, pl] += P[pk, pl] * (0.5 * (p.gpp - p.gp2) - 0.5 * p.gp2)
                    F[pl, pk] = F[pk, pl]

    return F


def rm1_energy(
    atoms: list[int],
    coords: np.ndarray,
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """Compute RM1 single-point energy.

    Args:
        atoms: list of atomic numbers [1, 8, 1] for H2O
        coords: (N, 3) coordinates in Angstrom
        max_iter: max SCF iterations
        conv_tol: density matrix convergence threshold

    Returns:
        dict with 'energy_eV', 'energy_kcal', 'converged', 'n_iter'
    """
    coords = np.asarray(coords, dtype=np.float64)
    info = _build_basis_info(atoms)
    n_basis = info['n_basis']
    n_occ = info['n_occ']

    if verbose:
        print(f"RM1: {len(atoms)} atoms, {n_basis} basis functions, {info['n_elec']} electrons, {n_occ} occupied")

    # Core Hamiltonian
    H = _build_core_hamiltonian(atoms, coords, info)

    # Initial density: zero
    P = np.zeros((n_basis, n_basis))

    # SCF loop
    converged = False
    for iteration in range(max_iter):
        # Build Fock matrix
        F = _build_fock(H, P, info, atoms, coords)

        # Diagonalize
        eigenvalues, C = np.linalg.eigh(F)

        # Build new density matrix
        P_new = np.zeros((n_basis, n_basis))
        for k in range(n_occ):
            P_new += 2.0 * np.outer(C[:, k], C[:, k])

        # Density mixing (damping) for convergence
        mix = 0.5 if iteration < 5 else 0.3  # mix fraction of new density
        P_mixed = mix * P_new + (1.0 - mix) * P

        # Check convergence
        delta = np.sqrt(np.mean((P_new - P) ** 2))
        if verbose:
            E_elec = 0.5 * np.sum(P_mixed * (H + F))
            print(f"  iter {iteration:3d}: dP={delta:.2e}, E_elec={E_elec:.6f} eV")

        if delta < conv_tol:
            converged = True
            P = P_new  # use unmixed for final energy
            break

        P = P_mixed

    # Electronic energy
    E_elec = 0.5 * np.sum(P * (H + F))

    # Nuclear repulsion
    E_nuc = compute_nuclear_repulsion(atoms, coords)

    # Total energy
    E_total = E_elec + E_nuc

    # Heat of formation (approximate)
    E_isolated = sum(RM1_PARAMS[z].Uss * RM1_PARAMS[z].n_valence for z in atoms)
    E_hof_eV = E_total - E_isolated

    return {
        'energy_eV': E_total,
        'energy_kcal': E_total * EV_TO_KCAL,
        'electronic_eV': E_elec,
        'nuclear_eV': E_nuc,
        'heat_of_formation_eV': E_hof_eV,
        'heat_of_formation_kcal': E_hof_eV * EV_TO_KCAL,
        'converged': converged,
        'n_iter': iteration + 1,
        'eigenvalues': eigenvalues,
        'density': P,
        'n_basis': n_basis,
    }
