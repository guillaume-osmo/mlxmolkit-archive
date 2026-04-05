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
from .two_center_integrals import two_center_integrals


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

    Two contributions:
    1. One-center: G_μν (same atom) from Slater-Condon parameters
    2. Two-center: Coulomb/exchange between atoms via (ss|ss) integrals

    NDDO approximation: only (μμ|λλ)-type integrals survive.
    """
    n_basis = info['n_basis']
    n_atoms = len(atoms)
    params = info['params']
    b2a = info['basis_to_atom']
    btype = info['basis_type']
    starts = info['atom_basis_start']

    F = H.copy()

    # === One-center two-electron contributions ===
    for i, p in enumerate(params):
        mask = (b2a == i)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        s = idx[0]
        Pss = P[s, s]

        if p.n_basis == 1:
            F[s, s] += Pss * p.gss * 0.5
        else:
            px, py, pz = idx[1], idx[2], idx[3]
            Ppp_total = P[px, px] + P[py, py] + P[pz, pz]

            F[s, s] += Pss * p.gss * 0.5 + Ppp_total * (p.gsp - 0.5 * p.hsp)

            for k in range(1, 4):
                pk = idx[k]
                F[pk, pk] += Pss * (p.gsp - 0.5 * p.hsp) + \
                             P[pk, pk] * p.gpp * 0.5 + \
                             (Ppp_total - P[pk, pk]) * (p.gp2 - 0.5 * (p.gpp - p.gp2))

            for k in range(1, 4):
                pk = idx[k]
                F[s, pk] += P[s, pk] * (2.0 * p.hsp - 0.5 * p.gsp)
                F[pk, s] = F[s, pk]

            for k in range(1, 4):
                for l in range(k + 1, 4):
                    pk, pl = idx[k], idx[l]
                    F[pk, pl] += P[pk, pl] * (0.5 * (p.gpp - p.gp2) - 0.5 * p.gp2)
                    F[pl, pk] = F[pk, pl]

    # === Two-center contribution using full 22-integral set ===
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pA = params[i]
            pB = params[j]
            R = np.linalg.norm(coords[i] - coords[j])
            if R < 1e-10:
                continue

            ri, core, pair_type = two_center_integrals(pA, pB, R)

            sA = starts[i]
            sB = starts[j]

            if pair_type == 'HH':
                # (ss_A|ss_B)
                F[sA, sA] += P[sB, sB] * ri[0]
                F[sB, sB] += P[sA, sA] * ri[0]
                F[sA, sB] -= 0.5 * P[sA, sB] * ri[0]
                F[sB, sA] = F[sA, sB]

            elif pair_type == 'XH':
                # A=heavy, B=H: 4 integrals
                # ri[0]=(ss|ss), ri[1]=(sσ|ss), ri[2]=(σσ|ss), ri[3]=(ππ|ss)
                Pss_B = P[sB, sB]

                # Coulomb on A from B's density
                F[sA, sA] += Pss_B * ri[0]              # s_A feels ss
                if pA.n_basis > 1:
                    # Direction cosines for p orbitals
                    dx = (coords[j] - coords[i]) / R
                    # σ component (along bond): weighted combination of px,py,pz
                    for k in range(1, 4):
                        pk = sA + k
                        F[pk, pk] += Pss_B * (ri[2] * dx[k-1]**2 + ri[3] * (1.0 - dx[k-1]**2))
                    # s-σ cross term
                    for k in range(1, 4):
                        pk = sA + k
                        F[sA, pk] += Pss_B * ri[1] * dx[k-1]
                        F[pk, sA] = F[sA, pk]

                # Coulomb on B from A's density
                PA_total = sum(P[sA + k, sA + k] for k in range(pA.n_basis))
                F[sB, sB] += PA_total * ri[0]

                # Exchange s_A-s_B
                F[sA, sB] -= 0.5 * P[sA, sB] * ri[0]
                F[sB, sA] = F[sA, sB]

            elif pair_type == 'HX':
                # A=H, B=heavy — swap and handle
                ri_swap, core_swap, _ = two_center_integrals(pB, pA, R)
                Pss_A = P[sA, sA]

                F[sB, sB] += Pss_A * ri_swap[0]
                if pB.n_basis > 1:
                    dx = (coords[i] - coords[j]) / R
                    for k in range(1, 4):
                        pk = sB + k
                        F[pk, pk] += Pss_A * (ri_swap[2] * dx[k-1]**2 + ri_swap[3] * (1.0 - dx[k-1]**2))
                    for k in range(1, 4):
                        pk = sB + k
                        F[sB, pk] += Pss_A * ri_swap[1] * dx[k-1]
                        F[pk, sB] = F[sB, pk]

                PB_total = sum(P[sB + k, sB + k] for k in range(pB.n_basis))
                F[sA, sA] += PB_total * ri_swap[0]

                F[sA, sB] -= 0.5 * P[sA, sB] * ri_swap[0]
                F[sB, sA] = F[sA, sB]

            elif pair_type == 'XX':
                # Heavy-Heavy: full 22 integrals
                dx = (coords[j] - coords[i]) / R
                nA = pA.n_basis
                nB = pB.n_basis

                # Coulomb: F[μ_A,μ_A] += Σ_λ P[λ_B,λ_B] * (μ_A μ_A | λ_B λ_B)
                # For each orbital type on A and B, use appropriate integral
                for ma in range(nA):
                    mu = sA + ma
                    for lb in range(nB):
                        lam = sB + lb
                        # Map (type_mu, type_lam) to integral index
                        # Simplified: use (ss|ss) for s-s, (pp|ss)/(ss|pp) for s-p, (pp|pp) for p-p
                        if ma == 0 and lb == 0:
                            integral = ri[0]   # (SS|SS)
                        elif ma == 0 and lb > 0:
                            integral = ri[10] * dx[lb-1]**2 + ri[11] * (1.0 - dx[lb-1]**2)  # (SS|OO)/(SS|PP)
                        elif ma > 0 and lb == 0:
                            integral = ri[2] * dx[ma-1]**2 + ri[3] * (1.0 - dx[ma-1]**2)  # (OO|SS)/(PP|SS)
                        else:
                            # p-p: use (PP|PP) as average
                            integral = ri[18] * dx[ma-1]**2 * dx[lb-1]**2 \
                                     + ri[16] * (dx[ma-1]**2 * (1-dx[lb-1]**2) + (1-dx[ma-1]**2) * dx[lb-1]**2) / 2 \
                                     + ri[20] * (1-dx[ma-1]**2) * (1-dx[lb-1]**2)
                        F[mu, mu] += P[lam, lam] * integral

                # Same for B← A
                for mb in range(nB):
                    nu = sB + mb
                    for la in range(nA):
                        lam = sA + la
                        if mb == 0 and la == 0:
                            integral = ri[0]
                        elif mb == 0 and la > 0:
                            integral = ri[2] * dx[la-1]**2 + ri[3] * (1.0 - dx[la-1]**2)
                        elif mb > 0 and la == 0:
                            integral = ri[10] * dx[mb-1]**2 + ri[11] * (1.0 - dx[mb-1]**2)
                        else:
                            integral = ri[18] * dx[mb-1]**2 * dx[la-1]**2 \
                                     + ri[16] * (dx[mb-1]**2 * (1-dx[la-1]**2) + (1-dx[mb-1]**2) * dx[la-1]**2) / 2 \
                                     + ri[20] * (1-dx[mb-1]**2) * (1-dx[la-1]**2)
                        F[nu, nu] += P[lam, lam] * integral

                # Exchange: F[μ_A, λ_B] -= 0.5 * P[μ_A, λ_B] * (μ_A μ_A | λ_B λ_B)
                for ma in range(nA):
                    for lb in range(nB):
                        mu = sA + ma
                        lam = sB + lb
                        if ma == 0 and lb == 0:
                            exch = ri[0]
                        elif (ma == 0 and lb > 0) or (ma > 0 and lb == 0):
                            exch = ri[0] * 0.5  # approximate
                        else:
                            exch = ri[6]  # (SP|SP) for p-p exchange
                        F[mu, lam] -= 0.5 * P[mu, lam] * exch
                        F[lam, mu] = F[mu, lam]

    return F


def rm1_energy(
    atoms: list[int],
    coords: np.ndarray,
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    verbose: bool = False,
    use_metal: bool = False,
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

    # Precompute (ss|ss) integrals between all atom pairs (used in Fock build)
    n_atoms = len(atoms)
    params = info['params']
    ssss = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            R = np.linalg.norm(coords[i] - coords[j]) * ANG_TO_BOHR
            rho0A = 0.5 * EV / params[i].gss if params[i].gss > 0 else 0.0
            rho0B = 0.5 * EV / params[j].gss if params[j].gss > 0 else 0.0
            aee = (rho0A + rho0B) ** 2
            ssss[i, j] = EV / np.sqrt(R ** 2 + aee)

    # Core Hamiltonian
    H = _build_core_hamiltonian(atoms, coords, info)

    # Initial density: zero
    P = np.zeros((n_basis, n_basis))

    # Prepare Metal Fock kernel inputs (precompute once)
    if use_metal:
        from .fock_metal import build_fock_metal
        atom_params_metal = np.zeros((n_atoms, 5), dtype=np.float32)
        for i, p in enumerate(params):
            atom_params_metal[i] = [p.gss, p.gsp, p.gpp, p.gp2, p.hsp]
        atom_starts_metal = np.zeros(n_atoms + 1, dtype=np.int32)
        for i, p in enumerate(params):
            atom_starts_metal[i + 1] = atom_starts_metal[i] + p.n_basis

    def _fock(H, P):
        if use_metal:
            return build_fock_metal(
                H, P, atom_params_metal,
                info['basis_to_atom'], info['basis_type'],
                atom_starts_metal, ssss.astype(np.float32),
                n_basis, n_atoms,
            )
        return _build_fock(H, P, info, atoms, coords)

    # DIIS (Direct Inversion in the Iterative Subspace) storage
    diis_max = 6
    diis_F_list = []  # stored Fock matrices
    diis_e_list = []  # stored error vectors (FPS - SPF)

    # SCF loop
    converged = False
    for iteration in range(max_iter):
        # Build Fock matrix
        F = _fock(H, P)

        # DIIS extrapolation (after first few iterations)
        if iteration >= 2:
            # Error vector: e = F @ P - P @ F (commutator, should be zero at convergence)
            e = F @ P - P @ F
            diis_F_list.append(F.copy())
            diis_e_list.append(e.copy())

            # Keep only last diis_max entries
            if len(diis_F_list) > diis_max:
                diis_F_list.pop(0)
                diis_e_list.pop(0)

            nd = len(diis_F_list)
            if nd >= 2:
                # Build DIIS B matrix: B[i,j] = Tr(e_i @ e_j)
                B = np.zeros((nd + 1, nd + 1))
                for i in range(nd):
                    for j in range(nd):
                        B[i, j] = np.sum(diis_e_list[i] * diis_e_list[j])
                B[nd, :nd] = -1.0
                B[:nd, nd] = -1.0
                B[nd, nd] = 0.0

                rhs = np.zeros(nd + 1)
                rhs[nd] = -1.0

                try:
                    coeffs = np.linalg.solve(B, rhs)
                    F = sum(coeffs[i] * diis_F_list[i] for i in range(nd))
                except np.linalg.LinAlgError:
                    pass  # fall back to un-extrapolated F

        # Diagonalize
        eigenvalues, C = np.linalg.eigh(F)

        # Build new density matrix
        P_new = np.zeros((n_basis, n_basis))
        for k in range(n_occ):
            P_new += 2.0 * np.outer(C[:, k], C[:, k])

        # Check convergence
        delta = np.sqrt(np.mean((P_new - P) ** 2))
        if verbose:
            E_elec = 0.5 * np.sum(P_new * (H + F))
            print(f"  iter {iteration:3d}: dP={delta:.2e}, E_elec={E_elec:.6f} eV")

        if delta < conv_tol:
            converged = True
            P = P_new
            break

        # Density mixing (only before DIIS kicks in)
        if iteration < 2:
            mix = 0.5
            P = mix * P_new + (1.0 - mix) * P
        else:
            P = P_new

    # Final Fock with converged density
    F = _fock(H, P)

    # Electronic energy
    E_elec = 0.5 * np.sum(P * (H + F))

    # Nuclear repulsion
    E_nuc = compute_nuclear_repulsion(atoms, coords)

    # Total energy
    E_total = E_elec + E_nuc

    # Heat of formation:
    # ΔHf = E_total - Σ E_isol(atom) + Σ ΔHf_atom(experimental)
    E_isol_total = 0.0
    eheat_total = 0.0
    for z in atoms:
        p = RM1_PARAMS[z]
        # Compute isolated atom energy from one-center integrals
        nv = p.n_valence
        if p.n_basis == 1:
            eisol = p.Uss
        else:
            ns = min(nv, 2)
            np_el = nv - ns
            eisol = ns * p.Uss + np_el * p.Upp
            if ns == 2:
                eisol += p.gss
            eisol += ns * np_el * (p.gsp - p.hsp / 2.0)
            if np_el >= 2:
                n_pairs = np_el * (np_el - 1) / 2
                eisol += n_pairs * (p.gp2 + (p.gpp - p.gp2) / 3.0)
        E_isol_total += eisol
        eheat_total += p.eheat  # experimental ΔHf of atom (kcal/mol)

    E_binding_eV = E_total - E_isol_total
    E_hof_eV = E_binding_eV + eheat_total / EV_TO_KCAL

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
