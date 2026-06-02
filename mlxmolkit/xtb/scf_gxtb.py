# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Experimental native g-xTB single-point and gradient driver."""

from __future__ import annotations

import math

import numpy as np

from .dispersion_d4srev import d4srev_dispersion_gxtb
from .gxtb_acp import build_gxtb_acp_hamiltonian, gxtb_pacp_proxy_energy
from .gxtb_basis import ANG_TO_BOHR, build_gxtb_qvszp_basis
from .gxtb_reconstructed import gxtb_reconstructed_repulsion
from .hcore_gxtb import build_hcore_gxtb
from .params_gxtb import GXTB_PARAMS


EV_PER_HARTREE = 27.211386245988
KCAL_PER_HARTREE = 627.5094740631
GXTB_TB2_KEXP = 0.294621155
GXTB_TB3_K = 2.3
GXTB_TB3_KX = 0.2093327496
GXTB_TB3_REXP = 1.3
GXTB_TB1_KX = 1.0
GXTB_TB1_KDIS = 0.025
GXTB_TB1_KS = 0.666666666
GXTB_TB1_CN_EPS = 1.0e-12
GXTB_MFX_FR_SCALE = 0.15
GXTB_MFX_LR_SCALE = 0.85
GXTB_MFX_OMEGA = 0.20
GXTB_MFX_GEXP = 1.3826597204
GXTB_HALIDE_INCREMENT_CORRECTION = {
    # Oracle-calibrated additive shifts for the extracted release increments.
    # The correction is a constant per atom and therefore has zero gradient.
    9: -1.824363963678883,
    17: -0.432894263892963,
    35: -0.105914643649314,
}
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


def _coulomb_matrix(
    coords_bohr: np.ndarray,
    shell_atom: np.ndarray,
    shell_hardness: np.ndarray,
    k_exp: float = GXTB_TB2_KEXP,
) -> np.ndarray:
    """g-xTB second-order shell Coulomb kernel from SI Eq. 101."""

    n_sh = len(shell_atom)
    jmat = np.zeros((n_sh, n_sh), dtype=np.float64)
    for i in range(n_sh):
        ai = int(shell_atom[i])
        gi = float(max(shell_hardness[i], 1.0e-10))
        for j in range(i):
            aj = int(shell_atom[j])
            gj = float(max(shell_hardness[j], 1.0e-10))
            inv_avg = 0.5 * (1.0 / gi + 1.0 / gj)
            if ai == aj:
                value = 1.0 / inv_avg
            else:
                R = float(np.linalg.norm(coords_bohr[ai] - coords_bohr[aj]))
                value = 1.0 / (R + inv_avg * np.exp(-k_exp * R))
            jmat[i, j] = value
            jmat[j, i] = value
        jmat[i, i] = gi
    return jmat


def _halide_increment_correction(atomic_numbers: np.ndarray) -> float:
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    return float(sum(GXTB_HALIDE_INCREMENT_CORRECTION.get(int(Z), 0.0) for Z in atoms))


def _mulliken_shell_charges(P: np.ndarray, S: np.ndarray, bf_to_shell: np.ndarray, n_shell: int, z_ref: np.ndarray) -> np.ndarray:
    PS = P @ S
    pop = np.zeros(n_shell, dtype=np.float64)
    for mu in range(P.shape[0]):
        pop[int(bf_to_shell[mu])] += PS[mu, mu]
    return z_ref - pop


def _pulay_diis_numpy(F_hist: list[np.ndarray], e_hist: list[np.ndarray]) -> np.ndarray:
    nd = len(F_hist)
    if nd < 2:
        return F_hist[-1]
    B = np.zeros((nd, nd), dtype=np.float64)
    for i in range(nd):
        for j in range(i, nd):
            B[i, j] = float(np.sum(e_hist[i] * e_hist[j]))
            B[j, i] = B[i, j]
    A = np.zeros((nd + 1, nd + 1), dtype=np.float64)
    A[:nd, :nd] = B
    A[:nd, nd] = -1.0
    A[nd, :nd] = -1.0
    rhs = np.zeros(nd + 1, dtype=np.float64)
    rhs[nd] = -1.0
    try:
        coeffs = np.linalg.solve(A, rhs)[:nd]
    except np.linalg.LinAlgError:
        return F_hist[-1]
    out = np.zeros_like(F_hist[-1])
    for mat, coeff in zip(F_hist, coeffs):
        out += coeff * mat
    return out


def _solve_generalized(F: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.linalg import eigh

        return eigh(F, S)
    except Exception:
        s_eval, U = np.linalg.eigh(S)
        keep = s_eval > 1.0e-8
        X = U[:, keep] * (1.0 / np.sqrt(s_eval[keep]))[None, :]
        w, Cp = np.linalg.eigh(X.T @ F @ X)
        return w, X @ Cp


def _fock_from_shell_potential(
    H0: np.ndarray,
    S: np.ndarray,
    bf_to_shell: np.ndarray,
    V_sh: np.ndarray,
) -> np.ndarray:
    V_bf = V_sh[bf_to_shell]
    return H0 - 0.5 * (V_bf[:, None] + V_bf[None, :]) * S


def _generalized_hubbard_average(ua: float, ub: float, xi: float) -> float:
    """Generalized average from the g-xTB MFX SI Eq. 150."""

    if ua <= 0.0 or ub <= 0.0:
        return max(ua, ub, 1.0e-12)
    if abs(xi - 0.0) < 1.0e-14:
        return 0.5 * (ua + ub)
    if abs(xi - 1.0) < 1.0e-14:
        return math.sqrt(ua * ub)
    if abs(xi - 2.0) < 1.0e-14:
        return 2.0 / (1.0 / ua + 1.0 / ub)
    return (2.0 ** (xi - 1.0)) * ((ua * ub) ** (0.5 * xi)) / ((ua + ub) ** (xi - 1.0))


def _mfx_gamma_ao(
    atomic_numbers: np.ndarray,
    coords_ang: np.ndarray,
    basis,
    *,
    frscale: float = GXTB_MFX_FR_SCALE,
    lrscale: float = GXTB_MFX_LR_SCALE,
    omega: float = GXTB_MFX_OMEGA,
    gexp: float = GXTB_MFX_GEXP,
) -> np.ndarray:
    """Range-separated Mulliken Fock-exchange AO kernel.

    This follows the public ``mulliken-FX`` tblite matrix implementation and
    the scalar literals passed by the released g-xTB binary:
    ``frscale=0.15``, ``lrscale=0.85``, ``omega=0.2``, ``gexp=1.3826597204``.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    bf_to_shell = np.asarray(basis.bf_to_shell, dtype=np.int64)
    shell_atom = np.asarray(basis.shell_atom, dtype=np.int64)
    shell_l = np.asarray(basis.shell_l, dtype=np.int64)
    shell_local = _shell_local_indices(shell_atom)

    shell_u = np.zeros(shell_atom.size, dtype=np.float64)
    shell_xi = np.ones(shell_atom.size, dtype=np.float64)
    for ish, atom_idx0 in enumerate(shell_atom):
        atom_idx = int(atom_idx0)
        Z = int(atoms[atom_idx])
        local_shell = int(shell_local[ish])
        shell_u[ish] = float(GXTB_PARAMS["ps_fock_shell_hubbard"][Z - 1, local_shell])
        shell_xi[ish] = float(GXTB_PARAMS["ps_fock_avg_exp"][Z - 1, local_shell])
        if shell_u[ish] <= 0.0:
            shell_u[ish] = 1.0e-12

    n_basis = bf_to_shell.size
    gamma = np.zeros((n_basis, n_basis), dtype=np.float64)
    for mu in range(n_basis):
        ish = int(bf_to_shell[mu])
        atom_i = int(shell_atom[ish])
        for nu in range(mu + 1):
            jsh = int(bf_to_shell[nu])
            atom_j = int(shell_atom[jsh])
            rij = float(np.linalg.norm(coords_bohr[atom_i] - coords_bohr[atom_j]))
            xi = max(float(shell_xi[ish]), float(shell_xi[jsh]))
            favg = _generalized_hubbard_average(float(shell_u[ish]), float(shell_u[jsh]), xi)
            gam = favg * float(frscale)
            if rij < 1.0e-14:
                value = gam
            else:
                tmp = 1.0 / ((rij ** float(gexp) + gam ** (-float(gexp))) ** (1.0 / float(gexp)))
                value = (float(frscale) + float(lrscale) * math.erf(float(omega) * rij)) * tmp
            gamma[mu, nu] = value
            gamma[nu, mu] = value
    return gamma


def _mfx_fock_energy(P: np.ndarray, S: np.ndarray, gamma_ao: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``(E_MFX, F_MFX)`` using tblite's Mulliken-FX matrix factorization."""

    P_arr = np.asarray(P, dtype=np.float64)
    S_arr = np.asarray(S, dtype=np.float64)
    gamma = np.asarray(gamma_ao, dtype=np.float64)
    sp = S_arr @ P_arr
    prev = gamma * (0.5 * (sp @ S_arr))
    tmp = gamma * sp
    tmp = tmp + 0.5 * (S_arr @ (gamma * P_arr))
    prev = prev + 0.5 * (tmp @ S_arr)
    prev = -0.25 * (prev + prev.T)
    fock = 0.5 * prev
    fock = 0.5 * (fock + fock.T)
    energy = float(np.sum(P_arr * fock))
    return energy, fock


def _third_order_twobody(
    coords_bohr: np.ndarray,
    shell_atom: np.ndarray,
    shell_hardness: np.ndarray,
    shell_gamma: np.ndarray,
    qsh: np.ndarray,
    *,
    k3: float = GXTB_TB3_K,
    kx: float = GXTB_TB3_KX,
    rexp: float = GXTB_TB3_REXP,
) -> tuple[float, np.ndarray]:
    """Approximate g-xTB two-body third-order TB term from SI Eq. 129-133.

    The released binary constructs ``new_twobody_thirdorder`` rather than the
    classic onsite GFN2 term.  This function mirrors the SI algebra using the
    extracted shell Γ table and the visible binary scalar literals.
    """

    coords = np.asarray(coords_bohr, dtype=np.float64)
    atom = np.asarray(shell_atom, dtype=np.int64)
    hard = np.asarray(shell_hardness, dtype=np.float64)
    gamma = np.asarray(shell_gamma, dtype=np.float64)
    q = np.asarray(qsh, dtype=np.float64)
    n_shell = q.size
    if n_shell == 0:
        return 0.0, np.zeros(0, dtype=np.float64)

    n_atom = int(atom.max()) + 1
    qat = np.bincount(atom, weights=q, minlength=n_atom)
    shell_pot = np.zeros(n_shell, dtype=np.float64)
    atom_pot = np.zeros(n_atom, dtype=np.float64)
    energy = 0.0

    for i in range(n_shell):
        ai = int(atom[i])
        ui = max(float(hard[i]), 1.0e-12)
        qi = float(q[i])
        for j in range(n_shell):
            aj = int(atom[j])
            uj = max(float(hard[j]), 1.0e-12)
            qj = float(q[j])
            if ai == aj:
                tau_i = -0.5 / (ui * ui)
                tau_j = -0.5 / (uj * uj)
            else:
                rij = float(np.linalg.norm(coords[ai] - coords[aj]))
                uavg = 0.5 * (ui + uj)
                damp = math.exp(-float(kx) * uavg * (rij ** float(rexp)))
                common = float(k3) * damp
                tau_i = common * (
                    uavg * rij - 0.5 * float(kx) * (uavg * uavg) * (rij ** (float(rexp) + 1.0))
                )
                tau_j = tau_i

            ti = tau_i * float(gamma[i])
            tj = tau_j * float(gamma[j])
            pair_shift = qat[ai] * ti + qat[aj] * tj
            pref = (qi * qj) / 6.0
            energy += pref * pair_shift
            shell_pot[i] += (qj * pair_shift) / 6.0
            shell_pot[j] += (qi * pair_shift) / 6.0
            atom_pot[ai] += pref * ti
            atom_pot[aj] += pref * tj

    shell_pot += atom_pot[atom]
    return float(energy), shell_pot


def _shell_local_indices(shell_atom: np.ndarray) -> np.ndarray:
    local = np.zeros(shell_atom.size, dtype=np.intp)
    counts: dict[int, int] = {}
    for ish, atom_idx0 in enumerate(shell_atom):
        atom_idx = int(atom_idx0)
        local[ish] = counts.get(atom_idx, 0)
        counts[atom_idx] = int(local[ish]) + 1
    return local


def _first_order_onsite(
    atoms: np.ndarray,
    cn: np.ndarray,
    shell_atom: np.ndarray,
    qsh: np.ndarray,
    *,
    charge_sign: float = -1.0,
) -> tuple[float, np.ndarray]:
    """Binary-observed onsite first-order TB term and shell potential.

    The released g-xTB binary passes ``ps_tb1_ipea`` to the onsite-firstorder
    object and stores the discontinuity constants as ``kx=1.0``,
    ``kdis=0.025``, ``ks=2/3``.  Its switching is implemented as
    ``0.5 * (2 + kdis * (erf(kx*(q-ks)) + erf(kx*(q+ks))))``.

    ``mlxmolkit`` uses xTB's positive-deficiency Mulliken convention
    ``q = z_ref - population``.  The first-order module in the SI/binary acts
    on the opposite density-fluctuation sign, hence the default
    ``charge_sign=-1`` and the corresponding chain-rule sign on the returned
    potential.
    """

    n_shell = shell_atom.size
    potential_model = np.zeros(n_shell, dtype=np.float64)
    if n_shell == 0:
        return 0.0, potential_model

    q_model = charge_sign * np.asarray(qsh, dtype=np.float64)
    qat_model = np.bincount(shell_atom, weights=q_model, minlength=atoms.size)
    sqrt_cn = np.sqrt(np.asarray(cn, dtype=np.float64) + GXTB_TB1_CN_EPS) - 1.0e-6

    mu = np.zeros(n_shell, dtype=np.float64)
    shell_local = _shell_local_indices(shell_atom)
    for ish, atom_idx0 in enumerate(shell_atom):
        atom_idx = int(atom_idx0)
        Z = int(atoms[atom_idx])
        local_shell = int(shell_local[ish])
        mu0 = float(GXTB_PARAMS["ps_tb1_ipea"][Z - 1, local_shell])
        cn_scale = 1.0 + float(GXTB_PARAMS["pa_tb1_ipea_cn"][Z - 1]) * sqrt_cn[atom_idx]
        mu[ish] = mu0 * cn_scale

    energy = 0.0
    for atom_idx in range(atoms.size):
        mask = shell_atom == atom_idx
        if not np.any(mask):
            continue
        q_atom = float(qat_model[atom_idx])
        x_minus = GXTB_TB1_KX * (q_atom - GXTB_TB1_KS)
        x_plus = GXTB_TB1_KX * (q_atom + GXTB_TB1_KS)
        erf_sum = math.erf(x_minus) + math.erf(x_plus)
        f_switch = 1.0 + 0.5 * GXTB_TB1_KDIS * erf_sum
        df_switch = (
            0.5
            * GXTB_TB1_KDIS
            * GXTB_TB1_KX
            * _TWO_OVER_SQRT_PI
            * (math.exp(-(x_minus * x_minus)) + math.exp(-(x_plus * x_plus)))
        )

        moment = float(np.sum(mu[mask] * q_model[mask]))
        energy += f_switch * moment
        potential_model[mask] = mu[mask] * f_switch + df_switch * moment

    return energy, charge_sign * potential_model


def _first_order_offsite(
    atoms: np.ndarray,
    shell_atom: np.ndarray,
    jmat: np.ndarray,
    qsh: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Linear offsite/xvec term coupled through the second-order kernel.

    In ``add_coulomb`` the released g-xTB binary constructs an x-vector from
    ``ps_tb1_ipea - ps_tb1_zeffsh`` and passes it to
    ``new_effective_coulomb``.  The effective-coulomb object then supplies a
    charge-independent shell potential.  This clean-room reconstruction keeps
    the same linear form while reusing the current ``jmat`` kernel.
    """

    n_shell = shell_atom.size
    if n_shell == 0:
        return 0.0, np.zeros(0, dtype=np.float64)

    shell_local = _shell_local_indices(shell_atom)
    xvec = np.zeros(n_shell, dtype=np.float64)
    for ish, atom_idx0 in enumerate(shell_atom):
        atom_idx = int(atom_idx0)
        Z = int(atoms[atom_idx])
        local_shell = int(shell_local[ish])
        xvec[ish] = (
            float(GXTB_PARAMS["ps_tb1_ipea"][Z - 1, local_shell])
            - float(GXTB_PARAMS["ps_tb1_zeffsh"][Z - 1, local_shell])
        )

    offsite = shell_atom[:, None] != shell_atom[None, :]
    potential = (np.asarray(jmat, dtype=np.float64) * offsite) @ xvec
    energy = float(np.asarray(qsh, dtype=np.float64) @ potential)
    return energy, potential


def gxtb_energy(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    max_iter: int = 100,
    conv_tol: float = 1.0e-7,
    mix: float = 0.4,
    use_d4srev: bool = True,
    use_pacp: bool = True,
    use_acp_hamiltonian: bool = False,
    use_exchange: bool = False,
    use_mfx_exchange: bool = False,
    use_first_order: bool = False,
    use_first_order_offsite: bool = False,
    use_third_order: bool = False,
    use_twobody_third_order: bool = False,
    use_fourth_order: bool = False,
    use_diis: bool = True,
    use_halide_increment_correction: bool = True,
    scc_scale: float = 1.0,
    verbose: bool = False,
) -> dict[str, object]:
    """Compute an experimental native g-xTB single-point energy.

    This is a reconstruction scaffold: EEQ_BC, q-vSZP, overlap, H0, shell SCC,
    recovered repulsion, and a measurable exchange term are active.  D4Srev and
    p-ACP are explicit fallback/proxy components until their exact kernels are
    extracted.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    if coords.shape != (atoms.size, 3):
        raise ValueError("coords_ang must have shape (nat, 3)")

    basis = build_gxtb_qvszp_basis(atoms, coords, total_charge=float(charge))
    S = basis.S
    H_eht, shell_self = build_hcore_gxtb(atoms, coords, basis)
    H_acp = build_gxtb_acp_hamiltonian(atoms, coords, basis, enabled=use_acp_hamiltonian)
    H0 = H_eht + H_acp
    n_basis = S.shape[0]
    n_shell = basis.shell_atom.size
    coords_bohr = coords * ANG_TO_BOHR
    jmat = _coulomb_matrix(coords_bohr, basis.shell_atom, basis.shell_hardness)
    gamma_mfx = _mfx_gamma_ao(atoms, coords, basis) if use_mfx_exchange else None

    z_ref = basis.shell_zref
    n_elec_f = float(np.sum(z_ref)) - float(charge)
    n_elec = int(round(n_elec_f))
    if abs(n_elec - n_elec_f) > 1.0e-6 or n_elec % 2:
        raise NotImplementedError(f"only closed-shell integer electron counts are supported; got {n_elec_f}")
    n_occ = n_elec // 2

    z_atom_ref = np.bincount(basis.shell_atom, weights=z_ref, minlength=atoms.size)
    q_at_init = basis.eeqbc_charges
    qsh = np.where(
        z_atom_ref[basis.shell_atom] > 1.0e-12,
        q_at_init[basis.shell_atom] * z_ref / z_atom_ref[basis.shell_atom],
        0.0,
    )

    P = np.zeros((n_basis, n_basis), dtype=np.float64)
    F_hist: list[np.ndarray] = []
    e_hist: list[np.ndarray] = []
    converged = False
    n_iter = 0
    diis_warmup = 3
    diis_max = 6

    for it in range(max_iter):
        V_coul = jmat @ qsh
        E_first_iter, V_first = (
            _first_order_onsite(atoms, basis.cn, basis.shell_atom, qsh)
            if use_first_order
            else (0.0, 0.0)
        )
        if use_first_order and use_first_order_offsite:
            _, V_first_offsite = _first_order_offsite(atoms, basis.shell_atom, jmat, qsh)
            V_first = V_first + V_first_offsite
        V_third = qsh * qsh * basis.shell_third if use_third_order else 0.0
        _, V_third_twobody = (
            _third_order_twobody(coords_bohr, basis.shell_atom, basis.shell_hardness, basis.shell_third, qsh)
            if use_twobody_third_order
            else (0.0, 0.0)
        )
        V_fourth = qsh * qsh * qsh * basis.shell_fourth if use_fourth_order else 0.0
        V_exchange = basis.shell_exchange * qsh if use_exchange else 0.0
        V_sh = V_first + scc_scale * (V_coul + V_third + V_third_twobody + V_fourth + V_exchange)
        F = _fock_from_shell_potential(H0, S, basis.bf_to_shell, V_sh)
        if gamma_mfx is not None:
            _, F_mfx = _mfx_fock_energy(P, S, gamma_mfx)
            F = F + F_mfx

        if use_diis and it >= diis_warmup:
            e_diis = F @ P @ S - S @ P @ F
            F_hist.append(F)
            e_hist.append(e_diis)
            if len(F_hist) > diis_max:
                F_hist.pop(0)
                e_hist.pop(0)
            F_use = _pulay_diis_numpy(F_hist, e_hist)
        else:
            F_use = F

        eigvals, C = _solve_generalized(F_use, S)
        C_occ = C[:, :n_occ]
        P_new = 2.0 * (C_occ @ C_occ.T)
        qsh_new = _mulliken_shell_charges(P_new, S, basis.bf_to_shell, n_shell, z_ref)
        dq = float(np.max(np.abs(qsh_new - qsh)))
        if verbose:
            tag = f"DIIS hist={len(F_hist)}" if it >= diis_warmup else "linear"
            print(f"  g-xTB iter {it + 1:3d}: dq={dq:.3e} ({tag})")
        P = P_new
        if dq < conv_tol:
            qsh = qsh_new
            converged = True
            n_iter = it + 1
            break
        if it < diis_warmup:
            qsh = mix * qsh_new + (1.0 - mix) * qsh
        else:
            qsh = qsh_new
    if not converged:
        n_iter = max_iter

    V_coul = jmat @ qsh
    E_first, V_first = (
        _first_order_onsite(atoms, basis.cn, basis.shell_atom, qsh)
        if use_first_order
        else (0.0, 0.0)
    )
    E_first_offsite = 0.0
    if use_first_order and use_first_order_offsite:
        E_first_offsite, V_first_offsite = _first_order_offsite(atoms, basis.shell_atom, jmat, qsh)
        V_first = V_first + V_first_offsite
    V_third = qsh * qsh * basis.shell_third if use_third_order else 0.0
    _, V_third_twobody = (
        _third_order_twobody(coords_bohr, basis.shell_atom, basis.shell_hardness, basis.shell_third, qsh)
        if use_twobody_third_order
        else (0.0, 0.0)
    )
    V_fourth = qsh * qsh * qsh * basis.shell_fourth if use_fourth_order else 0.0
    V_exchange = basis.shell_exchange * qsh if use_exchange else 0.0
    V_sh = V_first + scc_scale * (V_coul + V_third + V_third_twobody + V_fourth + V_exchange)
    F = _fock_from_shell_potential(H0, S, basis.bf_to_shell, V_sh)
    if gamma_mfx is not None:
        _, F_mfx = _mfx_fock_energy(P, S, gamma_mfx)
        F = F + F_mfx
    eigvals, C = _solve_generalized(F, S)
    P = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
    qsh = _mulliken_shell_charges(P, S, basis.bf_to_shell, n_shell, z_ref)
    q_at = np.bincount(basis.shell_atom, weights=qsh, minlength=atoms.size)
    E_first, V_first = (
        _first_order_onsite(atoms, basis.cn, basis.shell_atom, qsh)
        if use_first_order
        else (0.0, 0.0)
    )
    E_first_offsite = 0.0
    if use_first_order and use_first_order_offsite:
        E_first_offsite, V_first_offsite = _first_order_offsite(atoms, basis.shell_atom, jmat, qsh)
        V_first = V_first + V_first_offsite
    E_first_total = E_first + E_first_offsite

    E_h0 = float(np.sum(P * H_eht))
    E_acp_h = float(np.sum(P * H_acp)) if use_acp_hamiltonian else 0.0
    E_coul = scc_scale * 0.5 * float(qsh @ (jmat @ qsh))
    E_third = (
        scc_scale * float(np.sum(qsh**3 * basis.shell_third) / 3.0)
        if use_third_order
        else 0.0
    )
    E_third_twobody = (
        scc_scale
        * _third_order_twobody(coords_bohr, basis.shell_atom, basis.shell_hardness, basis.shell_third, qsh)[0]
        if use_twobody_third_order
        else 0.0
    )
    E_fourth = (
        scc_scale * float(np.sum(qsh**4 * basis.shell_fourth) / 4.0)
        if use_fourth_order
        else 0.0
    )
    E_exchange = (
        scc_scale * 0.5 * float(np.sum(basis.shell_exchange * qsh * qsh))
        if use_exchange
        else 0.0
    )
    E_mfx, F_mfx = _mfx_fock_energy(P, S, gamma_mfx) if gamma_mfx is not None else (0.0, np.zeros_like(H0))
    rep = gxtb_reconstructed_repulsion(atoms, coords, descriptor=q_at, cn=basis.cn)
    E_rep = float(rep.energy)
    E_d4, d4_backend = d4srev_dispersion_gxtb(atoms, coords, enabled=use_d4srev)
    E_acp = gxtb_pacp_proxy_energy(atoms, coords, enabled=use_pacp)
    E_increment_raw = float(np.sum(GXTB_PARAMS["pa_increment"][atoms - 1]))
    E_increment_correction = _halide_increment_correction(atoms) if use_halide_increment_correction else 0.0
    E_increment = E_increment_raw + E_increment_correction

    E_total = (
        E_h0
        + E_first_total
        + E_coul
        + E_third
        + E_third_twobody
        + E_fourth
        + E_exchange
        + E_mfx
        + E_acp_h
        + E_rep
        + E_d4
        + E_acp
    )
    return {
        "energy_hartree": E_total,
        "energy_eV": E_total * EV_PER_HARTREE,
        "energy_kcal": E_total * KCAL_PER_HARTREE,
        "electronic_hartree": E_h0
        + E_first_total
        + E_coul
        + E_third
        + E_third_twobody
        + E_fourth
        + E_exchange
        + E_mfx
        + E_acp_h,
        "h0_hartree": E_h0,
        "acp_hamiltonian_hartree": E_acp_h,
        "first_order_hartree": E_first_total,
        "first_order_onsite_hartree": E_first,
        "first_order_offsite_hartree": E_first_offsite,
        "coulomb_hartree": E_coul,
        "third_order_hartree": E_third,
        "third_order_twobody_hartree": E_third_twobody,
        "fourth_order_hartree": E_fourth,
        "exchange_hartree": E_exchange,
        "mfx_exchange_hartree": E_mfx,
        "repulsion_hartree": E_rep,
        "dispersion_hartree": E_d4,
        "d4srev_backend": d4_backend,
        "pacp_hartree": E_acp,
        "raw_increment_hartree": E_increment_raw,
        "halide_increment_correction_hartree": E_increment_correction,
        "increment_hartree": E_increment,
        "energy_plus_increment_hartree": E_total + E_increment,
        "converged": converged,
        "n_iter": n_iter,
        "n_basis": n_basis,
        "n_shell": n_shell,
        "n_elec": n_elec,
        "n_occ": n_occ,
        "method": "g-xTB-reconstructed",
        "basis": basis,
        "H0": H0,
        "H_eht": H_eht,
        "H_acp": H_acp,
        "S": S,
        "F": F,
        "F_mfx": F_mfx,
        "density": P,
        "eigenvalues": eigvals,
        "shell_charges": qsh,
        "atom_charges": q_at,
        "coordination_number": basis.cn,
        "eeqbc_charges": basis.eeqbc_charges,
        "jmat": jmat,
        "shell_selfenergy": shell_self,
        "repulsion": rep,
        "exactness": {
            "eeqbc": "binary-formula",
            "qvszp": "binary tables, active pa_nshell shells",
            "h0": "binary parameter scaffold without anisotropic H0",
            "first_order": "binary-observed onsite firstorder plus xvec offsite scaffold"
            if use_first_order
            else "disabled",
            "scc": "shell-charge scaffold",
            "third_order": "enabled" if use_third_order else "decoded tables exposed; disabled by default",
            "third_order_twobody": "SI Eq. 129 two-body third-order scaffold"
            if use_twobody_third_order
            else "disabled",
            "exchange": "diagonal shell proxy from recovered tables" if use_exchange else "diagonal shell proxy disabled",
            "mfx_exchange": "SI Eq. 153 range-separated Mulliken Fock exchange"
            if use_mfx_exchange
            else "disabled",
            "p_acp": "pair-energy proxy",
            "acp_hamiltonian": "SI Eq. 78 reduced projector Hamiltonian"
            if use_acp_hamiltonian
            else "disabled",
            "d4srev": d4_backend,
            "halide_increment_correction": "oracle-calibrated additive shift"
            if use_halide_increment_correction
            else "disabled",
            "gradient": "central finite difference",
        },
    }


def gxtb_gradient_numerical(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    h: float = 1.0e-3,
    **energy_kwargs,
) -> np.ndarray:
    """Central-difference gradient of :func:`gxtb_energy` in Hartree/Angstrom."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    grad = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for k in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[i, k] += h
            minus[i, k] -= h
            ep = float(gxtb_energy(atoms, plus, **energy_kwargs)["energy_hartree"])
            em = float(gxtb_energy(atoms, minus, **energy_kwargs)["energy_hartree"])
            grad[i, k] = (ep - em) / (2.0 * h)
    return grad


def gxtb_energy_gradient(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    gradient_h: float = 1.0e-3,
    **energy_kwargs,
) -> dict[str, object]:
    """Return energy and central-difference gradient for the reconstructed path."""

    res = gxtb_energy(atomic_numbers, coords_ang, **energy_kwargs)
    res["gradient"] = gxtb_gradient_numerical(
        atomic_numbers,
        coords_ang,
        h=gradient_h,
        **energy_kwargs,
    )
    return res
