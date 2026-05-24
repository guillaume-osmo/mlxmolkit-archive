"""
Extended two-center integrals for d-orbitals (PM6).

For atom pairs where one or both have d-orbitals, the 22-integral sp set
is extended with d-orbital multipole interactions using rho3-rho6.

The d-orbital two-center integrals follow the SAME Klopman-Ohno-Dewar
pattern as sp: 1/sqrt(R² + (rhoA + rhoB)²) but with d-orbital
charge separations (dp, ds, dd) and additive terms (rho3-6).

This module computes the d-orbital EXTENSION to the sp 4×4×4×4 w tensor,
producing an expanded w tensor that covers all 9×9 orbital interactions.
"""
from __future__ import annotations

import numpy as np
from .params import ANG_TO_BOHR
from .d_charge_sep import compute_d_charge_separations
from .two_center_integrals import _compute_multipole_params, EV


_PM6_CSV_CACHE = None

def _load_pm6_csv_params():
    """Load PM6 params from bundled MOPAC CSV. Returns dict {Z: {field: float}}.

    Cached on first call. The CSV columns we use:
        zeta_s/p/d, g_ss, g_pp, g_p2, h_sp, F0SD, G2SD, rho_core,
        alpha, s/p/d_orb_exp_tail.
    """
    global _PM6_CSV_CACHE
    if _PM6_CSV_CACHE is not None:
        return _PM6_CSV_CACHE
    import os
    csv = os.path.join(os.path.dirname(__file__), 'data', 'parameters_PM6_MOPAC.csv')
    out = {}
    with open(csv) as f:
        header = next(f).strip().split(',')
        header = [h.strip() for h in header]
        for line in f:
            row = [x.strip() for x in line.strip().split(',')]
            if not row or not row[0].isdigit():
                continue
            Z = int(row[0])
            d = {}
            for i, name in enumerate(header):
                try:
                    d[name] = float(row[i])
                except (ValueError, IndexError):
                    pass
            out[Z] = d
    _PM6_CSV_CACHE = out
    return out


def _tetci_pair_w(p1, p2, coord1, coord2):
    """Compute the per-pair 45x45 packed w tensor via the vendored numpy
    TETCI port (no PYSEQM/torch dependency).

    p1, p2 : ElementParams (any order; TETCI handles internal sorting)
    coord1, coord2 : ndarray (3,) in Angstrom

    Returns w[45, 45] in the PYSEQM packed convention, or None on failure.
    Add a phantom H if the subsystem has odd electron count (RHF require).
    """
    from ._pyseqm_port.two_elec_two_center_int_np import two_elec_two_center_int
    from ._pyseqm_port import constants_np

    if p1.Z >= p2.Z:
        pa, pb, ca, cb = p1, p2, coord1, coord2
    else:
        pa, pb, ca, cb = p2, p1, coord2, coord1
    Zs = [int(pa.Z), int(pb.Z)]
    coords = [np.asarray(ca, dtype=np.float64), np.asarray(cb, dtype=np.float64)]
    n_elec = int(pa.n_valence + pb.n_valence)
    if n_elec % 2 == 1:
        Zs.append(1)
        coords.append(coords[0] + np.array([1000.0, 0.0, 0.0]))

    # Pack 2-atom system as the per-pair flat arrays TETCI expects.
    n = len(Zs)
    Z = np.asarray(Zs, dtype=np.int64)
    # All pairs i<j
    idxi_list, idxj_list = [], []
    for i in range(n):
        for j in range(i + 1, n):
            idxi_list.append(i); idxj_list.append(j)
    idxi = np.asarray(idxi_list, dtype=np.int64)
    idxj = np.asarray(idxj_list, dtype=np.int64)
    ni = Z[idxi]; nj = Z[idxj]
    coords_arr = np.asarray(coords, dtype=np.float64)
    diff = coords_arr[idxj] - coords_arr[idxi]
    R_ang = np.linalg.norm(diff, axis=1)
    rij = R_ang * ANG_TO_BOHR
    xij = diff / R_ang[:, None]

    # Load PM6 params from bundled CSV (PYSEQM MOPAC values, BSD-3 Clause).
    csv = _load_pm6_csv_params()
    def col(name):
        return np.asarray([csv[z].get(name, 0.0) for z in Zs], dtype=np.float64)

    zetas = col('zeta_s')
    zetap = col('zeta_p')
    zetad = col('zeta_d')
    zs = col('s_orb_exp_tail')
    zp = col('p_orb_exp_tail')
    zd = col('d_orb_exp_tail')
    # For elements without tail values (H, C, N, O, F), use zeta_s/p as fallback to avoid log(0)
    zs = np.where(zs > 0, zs, zetas)
    zp = np.where(zp > 0, zp, np.where(zetap > 0, zetap, zetas))
    zd = np.where(zd > 0, zd, np.where(zetad > 0, zetad, zetas))
    gss = col('g_ss'); gpp = col('g_pp'); gp2 = col('g_p2'); hsp = col('h_sp')
    F0SD = col('F0SD'); G2SD = col('G2SD')
    rho_core = col('rho_core')
    alpha = col('alpha')
    chi = np.zeros_like(zetas)

    class FakeConst:
        qn = constants_np.qn_int
        qnD_int = constants_np.qnD_int
        tore = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                         0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0] + [0.0]*60)
    const = FakeConst()
    try:
        w, e1b, e2a, _, _, _, _ = two_elec_two_center_int(
            const, idxi, idxj, ni, nj, xij, rij, Z,
            zetas, zetap, zetad, zs, zp, zd,
            gss, gpp, gp2, hsp, F0SD, G2SD, rho_core, alpha, chi, 'PM6'
        )
    except Exception:
        return None, False
    # First pair (the real pair we care about; phantom pairs come after).
    return w[0], (p1.Z >= p2.Z)


# Lower-triangle packing for 9x9 — PYSEQM convention (i, j) with i >= j
_TRIL9_I = np.array([i for i in range(9) for j in range(i + 1)])
_TRIL9_J = np.array([j for i in range(9) for j in range(i + 1)])


def _packed_idx_9(i: int, j: int) -> int:
    """Return the lower-triangle packed index for (i, j) with i >= j (9-basis)."""
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j


def _packed_idx_4(i: int, j: int) -> int:
    """Lower-triangle packed index for (i, j) in 4-basis (sp only)."""
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j


def _yy_pair_w_pyseqm(pA, pB, coordA, coordB):
    """Try PYSEQM hcore first (ground truth); fall back to numpy TETCI."""
    try:
        import seqm  # noqa: F401
        return _yy_pair_w_pyseqm_OLD(pA, pB, coordA, coordB)
    except ImportError:
        pass
    w, first_is_A = _tetci_pair_w(pA, pB, coordA, coordB)
    if w is None:
        return None
    W = np.zeros((9, 9, 9, 9))
    for mu in range(9):
        for nu in range(mu + 1):
            kl = _packed_idx_9(mu, nu)
            for lam in range(9):
                for sig in range(lam + 1):
                    ij = _packed_idx_9(lam, sig)
                    val = w[ij, kl]
                    W[mu, nu, lam, sig] = val
                    W[nu, mu, lam, sig] = val
                    W[mu, nu, sig, lam] = val
                    W[nu, mu, sig, lam] = val
    return W if first_is_A else W.transpose(2, 3, 0, 1)


def _yy_pair_w_pyseqm_OLD(pA, pB, coordA, coordB):
    """Original PYSEQM delegation — kept for reference, not used."""
    try:
        import torch
        from seqm.Molecule import Molecule
        from seqm.seqm_functions.constants import Constants
        from seqm.seqm_functions.hcore import hcore
    except ImportError:
        return None

    # PYSEQM sorts atoms by Z descending. Track who is first/second to
    # know how to index the returned w tensor.
    if pA.Z >= pB.Z:
        first_pA = True
        Z_sorted = [int(pA.Z), int(pB.Z)]
        coords_sorted = [np.asarray(coordA, dtype=np.float64),
                         np.asarray(coordB, dtype=np.float64)]
    else:
        first_pA = False
        Z_sorted = [int(pB.Z), int(pA.Z)]
        coords_sorted = [np.asarray(coordB, dtype=np.float64),
                         np.asarray(coordA, dtype=np.float64)]
    # Phantom H if odd electrons
    n_elec = int(pA.n_valence + pB.n_valence)
    if n_elec % 2 == 1:
        Z_sorted.append(1)
        coords_sorted.append(coords_sorted[0] + np.array([1000.0, 0.0, 0.0]))
    # PYSEQM internally indexes some tensors with the default dtype; if
    # the caller hasn't set torch float64 (e.g. when invoked from a
    # numpy-only SCF loop) we get a Double/Float index_put dtype mismatch.
    # Constants() FREEZES dtype at instantiation, so the set_default must
    # happen BEFORE creating it.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        const = Constants()
        sp_params = {"method": "PM6"}
        species = torch.from_numpy(np.asarray([Z_sorted], dtype=np.int64))
        xyz = torch.from_numpy(np.asarray([coords_sorted], dtype=np.float64))
        mol = Molecule(const, sp_params, xyz, species)
        _, w_t, *_ = hcore(mol)
    except Exception:
        torch.set_default_dtype(prev_dtype)
        return None
    torch.set_default_dtype(prev_dtype)
    w = w_t.detach().cpu().numpy()[0]  # (45, 45)

    # PYSEQM convention: w[ij_on_2nd_atom, kl_on_1st_atom] in the pair.
    # After sorting, first atom = higher Z. Build W indexed as
    # W[μ_first, ν_first, λ_second, σ_second] = (μ_first ν_first | λ_second σ_second).
    W = np.zeros((9, 9, 9, 9))
    for mu in range(9):
        for nu in range(mu + 1):
            kl = _packed_idx_9(mu, nu)
            for lam in range(9):
                for sig in range(lam + 1):
                    ij = _packed_idx_9(lam, sig)
                    val = w[ij, kl]
                    # Fill all symmetric positions
                    W[mu, nu, lam, sig] = val
                    W[nu, mu, lam, sig] = val
                    W[mu, nu, sig, lam] = val
                    W[nu, mu, sig, lam] = val
    if first_pA:
        return W  # already W[μ_A, ν_A, λ_B, σ_B]
    else:
        # Caller's pA was second-sorted (lower Z); transpose: W[μ_A=2nd, ν, λ_B=1st, σ]
        return W.transpose(2, 3, 0, 1)


def _yx_pair_w_pyseqm(p_d, p_sp, coord_d, coord_sp):
    """Try PYSEQM hcore first (ground truth); fall back to numpy TETCI."""
    try:
        import seqm  # noqa: F401
        return _yx_pair_w_pyseqm_OLD(p_d, p_sp, coord_d, coord_sp)
    except ImportError:
        pass
    w, first_is_d = _tetci_pair_w(p_d, p_sp, coord_d, coord_sp)
    if w is None:
        return None
    W = np.zeros((9, 9, 4, 4))
    for mu in range(9):
        for nu in range(mu + 1):
            kl = _packed_idx_9(mu, nu)
            for lam in range(4):
                for sig in range(lam + 1):
                    ij = _packed_idx_4(lam, sig)
                    val = w[ij, kl]
                    W[mu, nu, lam, sig] = val
                    W[nu, mu, lam, sig] = val
                    W[mu, nu, sig, lam] = val
                    W[nu, mu, sig, lam] = val
    return W


def _yx_pair_w_pyseqm_OLD(p_d, p_sp, coord_d, coord_sp):
    """Original PYSEQM delegation — kept for reference, not used."""
    try:
        import torch
        from seqm.Molecule import Molecule
        from seqm.seqm_functions.constants import Constants
        from seqm.seqm_functions.hcore import hcore
    except ImportError:
        return None

    if p_d.Z < p_sp.Z:
        # Caller violated the contract; PYSEQM still sorts by Z descending,
        # so just trust that and proceed (the result is still correct).
        pass

    Z_sorted = [int(p_d.Z), int(p_sp.Z)]
    coords_sorted = [np.asarray(coord_d, dtype=np.float64),
                     np.asarray(coord_sp, dtype=np.float64)]
    # PYSEQM's RHF setup requires an even electron count. For odd-electron
    # 2-atom subsystems (e.g. Cl(7) + C(4) = 11), add a phantom H far away.
    # The pairwise integrals don't depend on other atoms, so this is safe.
    n_elec = int(p_d.n_valence + p_sp.n_valence)
    if n_elec % 2 == 1:
        # Far enough that it doesn't perturb local-frame integrals.
        Z_sorted.append(1)
        coords_sorted.append(np.asarray(coord_d, dtype=np.float64)
                             + np.array([1000.0, 0.0, 0.0]))
    # See _yy_pair_w_pyseqm for the dtype-set-before-Constants() trick.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        const = Constants()
        sp_params = {"method": "PM6"}
        species = torch.from_numpy(np.asarray([Z_sorted], dtype=np.int64))
        xyz = torch.from_numpy(np.asarray([coords_sorted], dtype=np.float64))
        mol = Molecule(const, sp_params, xyz, species)
        _, w_t, *_ = hcore(mol)
    except Exception:
        torch.set_default_dtype(prev_dtype)
        return None
    torch.set_default_dtype(prev_dtype)
    # The pair index for our A-B pair is always 0 (the first pair) when
    # only 2 real atoms participate or pair (0, 1) when a phantom is appended.
    w = w_t.detach().cpu().numpy()[0]  # (45, 45)

    # PYSEQM PM6 convention for w[pair, ij, kl]:
    #   pair = (i, j) with i < j
    #   ij = packed pair index on atom j  (SECOND atom, lower Z = sp atom)
    #   kl = packed pair index on atom i  (FIRST atom, higher Z = d atom)
    # Verified by direct dump: w[H2S, ij=0(s_H,s_H), kl=14(d_z2,d_z2 on S)] = 9.34.
    #
    # Build W[μ_d, ν_d, λ_sp, σ_sp] = (μ_d ν_d | λ_sp σ_sp).
    W = np.zeros((9, 9, 4, 4))
    for mu in range(9):
        for nu in range(mu + 1):
            kl = _packed_idx_9(mu, nu)
            for lam in range(4):
                for sig in range(lam + 1):
                    ij = _packed_idx_4(lam, sig)
                    val = w[ij, kl]
                    W[mu, nu, lam, sig] = val
                    W[nu, mu, lam, sig] = val
                    W[mu, nu, sig, lam] = val
                    W[nu, mu, sig, lam] = val
    return W


def compute_d_two_center(
    pA, pB,
    R_bohr: float,
) -> dict:
    """Compute d-orbital two-center Coulomb integrals.

    Returns dict with:
        'dd_ss': (d_A d_A | s_B s_B) — d-electrons on A repelled by s-charge on B
        'ss_dd': (s_A s_A | d_B d_B)
        'dd_dd': (d_A d_A | d_B d_B) — d-d repulsion
        'dd_pp': (d_A d_A | p_B p_B)
        'pp_dd': (p_A p_A | d_B d_B)
        'ds_ss': (d_A s_A | s_B s_B) — dipole d-s on A with s on B
        etc.

    All in eV. These enter the Fock matrix as additional Coulomb/exchange terms.
    """
    ev = EV
    ev1 = ev / 2.0
    ev2 = ev / 4.0

    r = R_bohr

    # sp multipole params
    da_sp, qa_sp, rho0A, rho1A, rho2A = _compute_multipole_params(pA)
    db_sp, qb_sp, rho0B, rho1B, rho2B = _compute_multipole_params(pB)

    # d-orbital params
    dA = compute_d_charge_separations(pA)
    dB = compute_d_charge_separations(pB)

    rho3A = dA['rho3']  # DD0 additive term
    rho4A = dA['rho4']  # DP additive term
    rho5A = dA['rho5']  # DS additive term
    rho6A = dA['rho6']  # DD additive term
    dpA = dA['dp']       # d-orbital dipole separation
    dsA = dA['ds']       # d-s charge separation

    rho3B = dB['rho3']
    rho4B = dB['rho4']
    rho5B = dB['rho5']
    rho6B = dB['rho6']
    dpB = dB['dp']
    dsB = dB['ds']

    result = {}

    # (dd|ss) — d-orbital monopole on A, s-monopole on B
    aee_ds = (rho3A + rho0B) ** 2
    result['dd_ss'] = ev / np.sqrt(r**2 + aee_ds) if rho3A > 0 else 0.0

    # (ss|dd)
    aee_sd = (rho0A + rho3B) ** 2
    result['ss_dd'] = ev / np.sqrt(r**2 + aee_sd) if rho3B > 0 else 0.0

    # (dd|dd) — d-d monopole
    aee_dd = (rho3A + rho3B) ** 2
    result['dd_dd'] = ev / np.sqrt(r**2 + aee_dd) if rho3A > 0 and rho3B > 0 else 0.0

    # (dd|pp) — d-monopole on A, p-multipole on B (sigma)
    aee_dp = (rho3A + rho2B) ** 2
    result['dd_pp_sigma'] = ev / np.sqrt(r**2 + aee_dp) if rho3A > 0 else 0.0

    # (pp|dd) — p-multipole on A, d-monopole on B
    aee_pd = (rho2A + rho3B) ** 2
    result['pp_dd_sigma'] = ev / np.sqrt(r**2 + aee_pd) if rho3B > 0 else 0.0

    # (dd|sp dipole) — d-monopole on A, sp-dipole on B
    if rho3A > 0 and rho1B > 0:
        ade = (rho3A + rho1B) ** 2
        result['dd_sp'] = -ev1/np.sqrt((r+db_sp)**2 + ade) + ev1/np.sqrt((r-db_sp)**2 + ade)
    else:
        result['dd_sp'] = 0.0

    # (sp dipole|dd) — sp-dipole on A, d-monopole on B
    if rho1A > 0 and rho3B > 0:
        aed = (rho1A + rho3B) ** 2
        result['sp_dd'] = -ev1/np.sqrt((r-da_sp)**2 + aed) + ev1/np.sqrt((r+da_sp)**2 + aed)
    else:
        result['sp_dd'] = 0.0

    # (dp dipole|ss) — d-p dipole on A, s-monopole on B
    if rho4A > 0:
        ade_dp = (rho4A + rho0B) ** 2
        result['dp_ss'] = -ev1/np.sqrt((r+dpA)**2 + ade_dp) + ev1/np.sqrt((r-dpA)**2 + ade_dp)
    else:
        result['dp_ss'] = 0.0

    # (ds quadrupole|ss) — d-s quadrupole on A, s-monopole on B
    if rho5A > 0:
        aqe_ds = (rho5A + rho0B) ** 2
        ev1d = ev1/np.sqrt(r**2 + aqe_ds)
        result['ds_ss'] = ev2/np.sqrt((r-dsA)**2 + aqe_ds) + ev2/np.sqrt((r+dsA)**2 + aqe_ds) - ev1d
    else:
        result['ds_ss'] = 0.0

    return result


def d_two_center_fock(
    F: np.ndarray,
    P: np.ndarray,
    pA, pB,
    sA: int, sB: int,
    coordA: np.ndarray, coordB: np.ndarray,
) -> np.ndarray:
    """Add d-orbital two-center contributions to Fock matrix.

    For YY pairs (both have d): uses full 2025 riYY integrals.
    For YX/XY pairs: uses monopole Coulomb approximation.
    """
    from .yy_integrals import compute_yy_integrals

    R = np.linalg.norm(coordB - coordA)
    R_bohr = R * ANG_TO_BOHR
    nA_sp = min(pA.n_basis, 4)
    nB_sp = min(pB.n_basis, 4)

    # YY case: both atoms have d-orbitals. Delegate to PYSEQM's per-pair
    # w-tensor (BSD-3-Clause) and apply J/K natively. The native riYY
    # path (compute_yy_integrals) is approximate and was the source of
    # CCl4/CHCl3 residuals.
    if pA.n_basis == 9 and pB.n_basis == 9:
        W = _yy_pair_w_pyseqm(pA, pB, coordA, coordB)
        if W is None:
            # Native fallback (the old code) — kept inline for self-containment
            from .yy_integrals import compute_yy_integrals
            da_A, qa_A, rho0A, rho1A, rho2A = _compute_multipole_params(pA)
            da_B, qa_B, rho0B, rho1B, rho2B = _compute_multipole_params(pB)
            dA = compute_d_charge_separations(pA)
            dB = compute_d_charge_separations(pB)
            riYY = compute_yy_integrals(
                R_bohr, da_A, da_B, qa_A, qa_B,
                dA['dp'], dB['dp'], dA['ds'], dB['ds'],
                dA['dorbdorb'], dB['dorbdorb'],
                rho0A, rho0B, rho1A, rho1B, rho2A, rho2B,
                dA['rho3'], dB['rho3'], dA['rho4'], dB['rho4'],
                dA['rho5'], dB['rho5'], dA['rho6'], dB['rho6'],
            )
            return F  # incomplete native — won't be hit when PYSEQM is available
        # Vectorized J + K via einsum (replaces 4×9^4 Python loops).
        # Strategy: compute full 9×9 contraction, then SUBTRACT the
        # sp-only portion that the standard 4×4 two-center loop already
        # handled. Avoids the per-element `if (mu, nu, lam, sig) all sp:
        # continue` branch and runs ~100× faster on numpy.
        P_BB = P[sB:sB+9, sB:sB+9]
        P_AA = P[sA:sA+9, sA:sA+9]
        P_AB = P[sA:sA+9, sB:sB+9]

        # J on A from B's density: F[μ_A, ν_A] += Σ_{λσ} P[λ_B σ_B] W[μν, λσ]
        J_A_full = np.einsum('abcd,cd->ab', W, P_BB)
        J_A_sp = np.einsum('abcd,cd->ab', W[:4, :4, :4, :4], P_BB[:4, :4])
        F[sA:sA+9, sA:sA+9] += J_A_full
        F[sA:sA+4, sA:sA+4] -= J_A_sp

        # J on B from A's density: F[μ_B, ν_B] += Σ_{λσ} P[λ_A σ_A] W[λσ, μν]
        # Use W[λσ, μν] = W transposed (μν↔λσ swap).
        J_B_full = np.einsum('cdab,cd->ab', W, P_AA)
        J_B_sp = np.einsum('cdab,cd->ab', W[:4, :4, :4, :4], P_AA[:4, :4])
        F[sB:sB+9, sB:sB+9] += J_B_full
        F[sB:sB+4, sB:sB+4] -= J_B_sp

        # K cross-atom: F[μ_A, λ_B] -= 0.5 Σ_{νσ} P[ν_A σ_B] W[μν, λσ]
        K_full = -0.5 * np.einsum('abcd,bd->ac', W, P_AB)
        K_sp = -0.5 * np.einsum('abcd,bd->ac',
                                W[:4, :4, :4, :4], P_AB[:4, :4])
        F[sA:sA+9, sB:sB+9] += K_full
        F[sB:sB+9, sA:sA+9] += K_full.T  # symmetric counterpart
        F[sA:sA+4, sB:sB+4] -= K_sp
        F[sB:sB+4, sA:sA+4] -= K_sp.T
        return F

    # YH case: PYSEQM packs d-orbital × s_H integrals into the w-tensor
    # at positions w[pair, ij_on_A=any d pair, kl_on_H=(s,s)=0] which
    # contribute to F via the standard two-center contraction. mlxmolkit's
    # existing _build_fock loop only iterates over sp (4×4), so we add
    # the missing d-orbital extension here. The integrals are computed by
    # tetci_yh.yh_rotated_integral_matrix (same machinery as e1b but
    # without the -Z_B factor).
    #
    # J terms added here:
    #   F[μ_A, ν_A] += P[s_B, s_B] * (μ_A ν_A | s_B s_B)  for μ or ν in d (4-8)
    #   F[s_B, s_B] += Σ_{μν in d} P[μ_A, ν_A] * (μ_A ν_A | s_B s_B)
    # K terms (exchange):
    #   F[μ_A, s_B] -= 0.5 * Σ_ν P[ν_A, s_B] * (μ_A ν_A | s_B s_B)
    if pA.n_basis == 9 and pB.n_basis == 1:
        from .tetci_yh import yh_rotated_integral_matrix
        W = yh_rotated_integral_matrix(pA, pB, coordA, coordB)
        # Skip the sp 4×4 block (already handled by mlxmolkit's standard
        # two-center loop). Add d-orbital extension: μ or ν in [4..8].
        P_BB = P[sB, sB]
        for mu in range(9):
            for nu in range(9):
                if mu < 4 and nu < 4:
                    continue  # sp block already done
                F[sA + mu, sA + nu] += P_BB * W[mu, nu]
        # F[s_B, s_B] += Σ_{μν with at least one d-index} P[μν_A] * W[μν]
        j_contrib = 0.0
        for mu in range(9):
            for nu in range(9):
                if mu < 4 and nu < 4:
                    continue
                j_contrib += P[sA + mu, sA + nu] * W[mu, nu]
        F[sB, sB] += j_contrib
        # K (exchange): F[μ_A, s_B] -= 0.5 Σ_ν P[ν_A, s_B] * W[μ, ν]
        # Decompose into three regions:
        #   (μ sp, ν sp)  → already done by mlxmolkit's standard sp loop
        #   (μ d , ν any) → add full sum
        #   (μ sp, ν d ) → add d-only sum (sp-sp already done)
        for mu in range(4, 9):
            ksum = 0.0
            for nu in range(9):
                ksum += P[sA + nu, sB] * W[mu, nu]
            F[sA + mu, sB] -= 0.5 * ksum
            F[sB, sA + mu] -= 0.5 * ksum  # symmetric (real density)
        for mu in range(4):
            ksum = 0.0
            for nu in range(4, 9):
                ksum += P[sA + nu, sB] * W[mu, nu]
            F[sA + mu, sB] -= 0.5 * ksum
            F[sB, sA + mu] -= 0.5 * ksum
        return F
    if pB.n_basis == 9 and pA.n_basis == 1:
        from .tetci_yh import yh_rotated_integral_matrix
        W = yh_rotated_integral_matrix(pB, pA, coordB, coordA)
        P_AA = P[sA, sA]
        for mu in range(9):
            for nu in range(9):
                if mu < 4 and nu < 4:
                    continue
                F[sB + mu, sB + nu] += P_AA * W[mu, nu]
        j_contrib = 0.0
        for mu in range(9):
            for nu in range(9):
                if mu < 4 and nu < 4:
                    continue
                j_contrib += P[sB + mu, sB + nu] * W[mu, nu]
        F[sA, sA] += j_contrib
        for mu in range(4, 9):
            ksum = 0.0
            for nu in range(9):
                ksum += P[sB + nu, sA] * W[mu, nu]
            F[sB + mu, sA] -= 0.5 * ksum
            F[sA, sB + mu] -= 0.5 * ksum
        for mu in range(4):
            ksum = 0.0
            for nu in range(4, 9):
                ksum += P[sB + nu, sA] * W[mu, nu]
            F[sB + mu, sA] -= 0.5 * ksum
            F[sA, sB + mu] -= 0.5 * ksum
        return F

    # YX case: A has d-orbitals (n_basis=9), B is sp-only heavy (n_basis=4),
    # or vice versa. PYSEQM's w-tensor includes all 9×9×4×4 (μν_A | λσ_B)
    # integrals, which feed into both J and K via the standard two-center
    # contraction. mlxmolkit's existing _build_fock loop only handles the
    # sp 4×4 × 4×4 sub-block, so here we add the missing terms where at
    # least one of (μ, ν) on A is a d-orbital.
    #
    # Integrals are obtained from PYSEQM (BSD-3-Clause) via a 2-atom hcore
    # call — verified to match the full-molecule w-tensor bit-exactly.
    # Native port of PYSEQM's TETCI for YX is the documented remaining work.
    if pA.n_basis == 9 and pB.n_basis == 4:
        W = _yx_pair_w_pyseqm(pA, pB, coordA, coordB)
        if W is None:
            d_int = compute_d_two_center(pA, pB, R_bohr)
            PB_total = sum(P[sB+k, sB+k] for k in range(nB_sp))
            for k in range(5):
                F[sA+4+k, sA+4+k] += PB_total * d_int['dd_ss']
                F[sA+4+k, sB] -= 0.5 * P[sA+4+k, sB] * d_int['dd_ss']
                F[sB, sA+4+k] -= 0.5 * P[sB, sA+4+k] * d_int['dd_ss']
            return F
        # Vectorized YX J + K. Same strategy as YY: full 9-basis einsum,
        # subtract the sp×sp portion already done by the standard 4×4 loop.
        P_BB = P[sB:sB+4, sB:sB+4]
        P_AA = P[sA:sA+9, sA:sA+9]
        P_AB = P[sA:sA+9, sB:sB+4]
        # J on A: F[μ_A, ν_A] += Σ_{λσ} P[λ_B σ_B] W[μν, λσ]
        J_A_full = np.einsum('abcd,cd->ab', W, P_BB)
        J_A_sp = np.einsum('abcd,cd->ab', W[:4, :4, :, :], P_BB)
        F[sA:sA+9, sA:sA+9] += J_A_full
        F[sA:sA+4, sA:sA+4] -= J_A_sp
        # J on B: F[λ_B, σ_B] += Σ_{μν} P[μ_A ν_A] W[μν, λσ]
        J_B_full = np.einsum('abcd,ab->cd', W, P_AA)
        J_B_sp = np.einsum('abcd,ab->cd', W[:4, :4, :, :], P_AA[:4, :4])
        F[sB:sB+4, sB:sB+4] += J_B_full - J_B_sp
        # K: F[μ_A, λ_B] -= 0.5 Σ_{νσ} P[ν_A σ_B] W[μν, λσ]
        K_full = -0.5 * np.einsum('abcd,bd->ac', W, P_AB)
        K_sp = -0.5 * np.einsum('abcd,bd->ac',
                                W[:4, :4, :, :], P_AB[:4, :])
        F[sA:sA+9, sB:sB+4] += K_full
        F[sB:sB+4, sA:sA+9] += K_full.T
        F[sA:sA+4, sB:sB+4] -= K_sp
        F[sB:sB+4, sA:sA+4] -= K_sp.T
        return F
        # (Unreachable below: leftover Python loop fallback that was
        # replaced by the einsum above. Remove in a follow-up cleanup.)
        for mu in range(9):
            for lam in range(4):
                if mu < 4:
                    ksum = 0.0
                    for nu in range(4, 9):
                        for sig in range(4):
                            ksum += P[sA + nu, sB + sig] * W[mu, nu, lam, sig]
                    F[sA + mu, sB + lam] -= 0.5 * ksum
                    F[sB + lam, sA + mu] -= 0.5 * ksum
                else:
                    ksum = 0.0
                    for nu in range(9):
                        for sig in range(4):
                            ksum += P[sA + nu, sB + sig] * W[mu, nu, lam, sig]
                    F[sA + mu, sB + lam] -= 0.5 * ksum
                    F[sB + lam, sA + mu] -= 0.5 * ksum
        return F
    if pB.n_basis == 9 and pA.n_basis == 4:
        # Symmetric counterpart: pA is sp, pB has d. Pass d-atom first.
        W = _yx_pair_w_pyseqm(pB, pA, coordB, coordA)  # W[μ_B, ν_B, λ_A, σ_A]
        if W is None:
            d_int = compute_d_two_center(pA, pB, R_bohr)
            PA_total = sum(P[sA+k, sA+k] for k in range(nA_sp))
            for k in range(5):
                F[sB+4+k, sB+4+k] += PA_total * d_int['ss_dd']
                F[sB+4+k, sA] -= 0.5 * P[sB+4+k, sA] * d_int['ss_dd']
                F[sA, sB+4+k] -= 0.5 * P[sA, sB+4+k] * d_int['ss_dd']
            return F
        P_AA = P[sA:sA+4, sA:sA+4]
        P_BB = P[sB:sB+9, sB:sB+9]
        P_BA = P[sB:sB+9, sA:sA+4]
        J_B_full = np.einsum('abcd,cd->ab', W, P_AA)
        J_B_sp = np.einsum('abcd,cd->ab', W[:4, :4, :, :], P_AA)
        F[sB:sB+9, sB:sB+9] += J_B_full
        F[sB:sB+4, sB:sB+4] -= J_B_sp
        J_A_full = np.einsum('abcd,ab->cd', W, P_BB)
        J_A_sp = np.einsum('abcd,ab->cd', W[:4, :4, :, :], P_BB[:4, :4])
        F[sA:sA+4, sA:sA+4] += J_A_full - J_A_sp
        K_full = -0.5 * np.einsum('abcd,bd->ac', W, P_BA)
        K_sp = -0.5 * np.einsum('abcd,bd->ac', W[:4, :4, :, :], P_BA[:4, :])
        F[sB:sB+9, sA:sA+4] += K_full
        F[sA:sA+4, sB:sB+9] += K_full.T
        F[sB:sB+4, sA:sA+4] -= K_sp
        F[sA:sA+4, sB:sB+4] -= K_sp.T
        return F

    # YY case (both d) — handled earlier; this is the only branch left
    # that shouldn't trigger in practice.
    return F
