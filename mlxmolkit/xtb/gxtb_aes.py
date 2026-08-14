# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""g-xTB anisotropic electrostatics (AES), reconstructed from the release binary.

The g-xTB AES is the module ``tblite_coulomb_multipole_gxtb`` (a
``damped_multipole`` whose label is "erf_damped_anisotropic_electrostatics").
Decompilation (data/gxtb_decompiled/) established that the *interaction tensor*
(``get_multipole_matrix`` / ``get_energy_aes`` / ``get_potential``) is the GENERIC
``tblite_coulomb_multipole_type`` shared with GFN2 — i.e. the same charge-dipole
(1/R^3) + charge-quadrupole/dipole-dipole (1/R^5) structure already implemented in
:mod:`mlxmolkit.xtb.aes` (``aniso_electro`` / ``setvsdq`` / ``fockelectro``).

g-xTB differs from GFN2 only in:
  * the pair damping (``get_damping_pair``): erf switching on ``R - mrad_pair``
    instead of GFN2's polynomial.  Verified form (binary 0x73b550..0x73b570):
        damp_k = mag_k * 0.5 * (1 + erf((R_ij - mrad_pair_ij) * scale_k))
    channels (mag, scale): (0.3405910191, 0.5), (0.1691310614, 1.0),
                           (0.074034339, 1.0), (-0.02, 1.0)
  * the multipole radius:  mrad_pair[i,j] = vdw_pair(Zi,Zj) * avg(rvdw_scale_i, rvdw_scale_j)
    (averager = the geometric/general one at binary 0x73b480)
  * a CT/polarization kernel scaled by ``pa_aes_dip_scale`` (get_kernel_*).

NB the precise channel->term mapping and the CT kernel are still being calibrated
against the --gxtb oracle; ``GXTB_AES_*`` constants below are the binary-exact
values, the *wiring* is the part under validation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from .aes import aniso_electro, setvsdq, fockelectro, mmompop
from .gxtb_basis import GXTBQVSZPBasis, ANG_TO_BOHR
from .multipole_integrals import multipole_matrices
from .mctc_vdwrad import mctc_vdw_pair_matrix_bohr
from .params_gxtb import GXTB_PARAMS

# Binary-exact AES damping channels (libxtb __const 0x73b550..0x73b570).
GXTB_AES_DAMP_MAG = np.array([0.3405910191, 0.1691310614, 0.074034339, -0.02])
GXTB_AES_DAMP_SCALE = np.array([0.5, 1.0, 1.0, 1.0])


def _general_average(gi, gj, xi: float = 1.0):
    """Generalized Hubbard-style average (xi=1 -> geometric), elementwise."""
    gi = np.asarray(gi, dtype=np.float64)
    gj = np.asarray(gj, dtype=np.float64)
    return (2.0 / (gi + gj)) ** (xi - 1.0) * (gi * gj) ** (xi / 2.0)


_MP_CACHE = {}


def qvszp_multipoles(basis: GXTBQVSZPBasis):
    """Dipole/quadrupole AO integrals over the q-vSZP basis, in the SAO basis.

    Returns ``(S, dpint, qpint)`` with dpint (3, nao, nao), qpint (6, nao, nao)
    in xtb (xx,yy,zz,xy,xz,yz) order, transformed CAO->SAO. Integrals are
    geometry-only -> cached by basis identity (avoids O(nao^2) rebuild every SCF iter).
    """
    key = id(basis)
    cached = _MP_CACHE.get(key)
    if cached is not None and cached[0] is basis:   # verify identity (id can be reused)
        return cached[1]
    if len(_MP_CACHE) > 8:
        _MP_CACHE.clear()
    S_cao, dp_cao, qp_cao = multipole_matrices(basis.cao_basis)
    T = np.asarray(basis.T_cao_to_sao, dtype=np.float64)
    S = T @ S_cao @ T.T
    dp = np.stack([T @ dp_cao[k] @ T.T for k in range(3)], axis=0)
    qp = np.stack([T @ qp_cao[k] @ T.T for k in range(6)], axis=0)
    _MP_CACHE[id(basis)] = (basis, (S, dp, qp))
    return S, dp, qp


def gxtb_mrad_pair(atoms: np.ndarray) -> np.ndarray:
    """mrad_pair[i,j] = vdw_pair(Zi,Zj) * avg(rvdw_scale_i, rvdw_scale_j)  (Bohr)."""
    atoms = np.asarray(atoms, dtype=np.intp)
    vdw = mctc_vdw_pair_matrix_bohr(atoms)              # (nat, nat)
    rs = np.asarray(GXTB_PARAMS["pa_rvdw_scale"], dtype=np.float64)[atoms - 1]
    # binary-exact: new_average enum 0 = ARITHMETIC mean (libxtb 0x73b480), not
    # geometric.  (Charge-neutral vs the geometric mean — 0.0017 Bohr delta — but
    # this is the faithful form per the decompile.)
    avg = 0.5 * (rs[:, None] + rs[None, :])             # (nat, nat)
    return vdw * avg


def gxtb_aes_gab(coords_bohr: np.ndarray, mrad: np.ndarray, channel3: int = 0, channel5: int = 1):
    """g-xTB erf-damped gab3 (1/R^3) and gab5 (1/R^5).

    damp_k = mag_k * 0.5 * (1 + erf((R - mrad_pair) * scale_k)); gab3 = damp/R^3, gab5 = damp/R^5.
    channel3/channel5 select which of the 4 binary channels drive the dip (R^-3) and quad/dip-dip
    (R^-5) terms (under oracle calibration).
    """
    n = coords_bohr.shape[0]
    diff = coords_bohr[:, None, :] - coords_bohr[None, :, :]
    R = np.sqrt(np.sum(diff * diff, axis=-1))
    eye = np.eye(n, dtype=bool)
    Rsafe = np.where(eye, 1.0, R)
    arg = R - mrad
    def damp(k):
        return GXTB_AES_DAMP_MAG[k] * 0.5 * (1.0 + erf(arg * GXTB_AES_DAMP_SCALE[k]))
    gab3 = np.where(eye, 0.0, damp(channel3) / Rsafe ** 3)
    gab5 = np.where(eye, 0.0, damp(channel5) / Rsafe ** 5)
    return gab3, gab5


def gxtb_aes_fock(P: np.ndarray, basis: GXTBQVSZPBasis, atoms: np.ndarray,
                  coords_ang: np.ndarray, *, channel3: int = 0, channel5: int = 1):
    """AES Fock contribution F_aes (nao,nao) + energy e_aes (Hartree).

    Builds q-vSZP multipole integrals, Mulliken atomic multipoles from P, the
    g-xTB erf-damped interaction, and the AES potentials, then the Fock term.
    """
    atoms = np.asarray(atoms, dtype=np.intp)
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    S, dpint, qpint = qvszp_multipoles(basis)
    aoat = np.array([bf.atom_idx for bf in basis.sao_basis], dtype=np.int64)

    # Mulliken atomic charges + cumulative atomic dipoles/quadrupoles.
    PS = P @ S
    pop = np.zeros(len(atoms))
    for mu in range(P.shape[0]):
        pop[aoat[mu]] += PS[mu, mu]
    zref = np.bincount(basis.shell_atom, weights=basis.shell_zref, minlength=len(atoms))
    q = zref - pop
    dipm, qp = mmompop(P, S, dpint, qpint, aoat.tolist(), coords_bohr)

    # dipole scaling (pa_aes_dip_scale per atom) on the atomic dipoles.
    dipscale = np.asarray(GXTB_PARAMS["pa_aes_dip_scale"], dtype=np.float64)[atoms - 1]
    dipm = dipm * dipscale[None, :]

    mrad = gxtb_mrad_pair(atoms)
    gab3, gab5 = gxtb_aes_gab(coords_bohr, mrad, channel3, channel5)
    e_aes, _ = aniso_electro(atoms.tolist(), coords_bohr, q, dipm, qp, gab3, gab5)
    vs, vd, vq = setvsdq(atoms.tolist(), coords_bohr, q, dipm, qp, gab3, gab5)
    F_aes, _ = fockelectro(P, S, dpint, qpint, aoat.tolist(), vs, vd, vq)
    return F_aes, e_aes


# --- one-center (onsite) exchange: onecxints over same-atom shell pairs ---
import numpy as _np
_ONEC = _np.load(__import__("os").path.join(__import__("os").path.dirname(__file__),
                 "..", "..", "data", "gxtb_onecxints_extracted.npz"))
ONECX_TBL = _ONEC["onecxints"]   # (103,10)
ONECX_LIDX = _ONEC["lidx"]       # (4,4)


def gxtb_onsite_gamma(basis, atoms):
    """Same-atom one-center exchange gamma_onsite[mu,nu] = onecxints[Z, lidx[lmu,lnu]].

    Returns an (nao,nao) matrix nonzero only for AO pairs on the same atom; fed
    through the same S.P.S Fock as the Mulliken exchange.
    """
    atoms = _np.asarray(atoms, dtype=_np.intp)
    bts = _np.asarray(basis.bf_to_shell)
    sa = _np.asarray(basis.shell_atom)
    sl = _np.asarray(basis.shell_l)
    n = bts.size
    g = _np.zeros((n, n))
    for mu in range(n):
        ish = int(bts[mu]); ai = int(sa[ish]); li = int(sl[ish])
        Z = int(atoms[ai])
        for nu in range(n):
            jsh = int(bts[nu]); aj = int(sa[jsh]); lj = int(sl[jsh])
            if ai != aj:
                continue
            pack = int(ONECX_LIDX[li, lj])
            g[mu, nu] = float(ONECX_TBL[Z - 1, pack - 1])
    return g


def gxtb_aniso_h0(basis, atoms, coords_ang, *, kexp: float = 1.5):
    """Anisotropic-H0 dipole-field correction (tblite_xtb_h0 get_anisotropy).

    field[i] = sum_j aniso[i,j] * 0.5*(1+erf(-kexp*(R_ij - mrad_pair_ij))) * (r_j - r_i)/R_ij
    H0_aniso[mu,nu] += 0.5*(field[atom_mu]+field[atom_nu]) . dpint[:,mu,nu]
    aniso[i,j] = avg(pa_h0_dip_scale_i, pa_h0_dip_scale_j).  Uses the AES dpint.
    """
    atoms = np.asarray(atoms, dtype=np.intp)
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    nat = len(atoms)
    _, dpint, _ = qvszp_multipoles(basis)
    aoat = np.array([bf.atom_idx for bf in basis.sao_basis], dtype=np.int64)
    dipscale = np.asarray(GXTB_PARAMS["pa_h0_dip_scale"], dtype=np.float64)[atoms - 1]
    # pa_h0_dip_scale has mixed signs -> arithmetic average (geometric -> NaN).
    aniso = 0.5 * (dipscale[:, None] + dipscale[None, :])             # (nat,nat)
    mrad = gxtb_mrad_pair(atoms)
    diff = coords_bohr[None, :, :] - coords_bohr[:, None, :]           # r_j - r_i, (i,j,3)
    R = np.sqrt(np.sum(diff * diff, axis=-1))
    eye = np.eye(nat, dtype=bool)
    Rsafe = np.where(eye, 1.0, R)
    damp = 0.5 * (1.0 + erf(-kexp * (R - mrad)))
    w = np.where(eye, 0.0, aniso * damp / Rsafe)[:, :, None]           # (i,j,1)
    field = np.sum(w * diff, axis=1)                                   # (nat,3)
    f_ao = field[aoat]                                                 # (nao,3)
    H = np.zeros(dpint.shape[1:], dtype=np.float64)
    for k in range(3):
        H += 0.5 * (f_ao[:, k][:, None] + f_ao[:, k][None, :]) * dpint[k]
    return H


_ONSITE_BASE_CACHE = {}


def _onsite_base(basis, atoms):
    """Geometry/element-only onecx base[mu,nu] for same-atom AO pairs (cached)."""
    key = id(basis)
    c = _ONSITE_BASE_CACHE.get(key)
    if c is not None and c[0] is basis:             # verify identity (id can be reused)
        return c[1], c[2]
    if len(_ONSITE_BASE_CACHE) > 8:
        _ONSITE_BASE_CACHE.clear()
    atoms = _np.asarray(atoms, dtype=_np.intp)
    bts = _np.asarray(basis.bf_to_shell)
    aoat = _np.asarray(basis.shell_atom)[bts]          # atom per AO
    aol = _np.asarray(basis.shell_l)[bts]              # l per AO
    Zao = atoms[aoat]
    n = bts.size
    base = _np.zeros((n, n))
    same = aoat[:, None] == aoat[None, :]
    pack = ONECX_LIDX[aol[:, None], aol[None, :]] - 1   # (n,n) packed index
    val = ONECX_TBL[Zao[:, None] - 1, pack]             # (n,n) onecx per AO-pair (uses mu's Z)
    base = _np.where(same, val, 0.0)
    _ONSITE_BASE_CACHE[key] = (basis, base, bts)
    return base, bts


def gxtb_onsite_gamma_density(P, S, basis, atoms):
    """Density-dependent one-center exchange gamma (exact get_gons form), vectorized.

    K_onsite[mu,nu] = (1 - occ_i*occ_j) * onecx[li,lj,Z]  (same-atom AO pairs)
    occ = shell Mulliken occupation fraction. Base (onecx) cached; only the
    occupation factor recomputed each SCF iteration.
    """
    # The EXACT get_gons integrand IS cracked (lldb FP-trace): the K-matrix is
    #   gamma[mu,mu] = -0.5*frscale*kq_mu * sum_l onecx[lmu,ll]*pop_l   (validated vs
    #   lldb Ne gamma_ss=-0.105 vs -0.107), gamma[mu,nu] off-diag analogous;
    #   frscale=0.15, kq=pg_fock_kq, pop=(P@S) diag.  BUT this is a ONE-CENTER
    #   K-matrix and must be folded by `onsite_fx_symv` (a per-atom matrix-product,
    #   keeping it on-atom).  Folding it through the Mulliken S.P.S sandwich
    #   (_mfx_fock_energy, the only fold wired here) SPREADS it across bonds ->
    #   nails water (0.047->0.024) but over-corrects multi-bonded C/N -> net 0.0588.
    #   So until onsite_fx_symv is traced, the empirically-better (1-occ^2) form
    #   below (with the sandwich) is the best available (0.0478).
    base, bts = _onsite_base(basis, atoms)
    nsh = _np.asarray(basis.shell_atom).size
    diag = _np.einsum("ij,ji->i", P, S)                # (P@S) diagonal per AO
    pop_sh = _np.bincount(bts, weights=diag, minlength=nsh)
    ndeg = _np.bincount(bts, minlength=nsh).astype(_np.float64)
    occ = pop_sh / _np.maximum(2.0 * ndeg, 1e-12)
    occ_ao = occ[bts]
    return (1.0 - occ_ao[:, None] * occ_ao[None, :]) * base


def gxtb_onsite_potential(P, S, basis, atoms, frscale=0.15):
    """EXACT onsite one-center exchange — the ``onsite_fx_symv`` fold (lldb-cracked).

    The released ``onsite_fx_symv`` kernel does NOT build a density-folded matrix
    (the old ``gxtb_onsite_gamma_density`` SPS path); it produces a per-AO
    *anti-binding shell potential*

        V_ao[mu] = frscale * sum_{nu on same atom} onecx[Z_mu, l_mu, l_nu] * (P@S)[nu,nu]

    i.e. ``V = frscale * (onecx_base @ pop_ao)`` (a symmetric matrix-vector — the
    "symv").  Verified vs the binary (Ne: V_s=0.1899 / V_p=0.1110, formula
    0.1904 / 0.1114, residual ~0.4% from the charge factor below).  The potential
    is POSITIVE (anti-binding): it cancels the Mulliken-K over-binding.

    The 0.4% refinement is the get_gons charge modulation
    ``(1 - 0.5*(kq_a*q_a + kq_b*q_b))`` with ``kq=pg_fock_kq=[1.1,0.55,0.275,0.1375]``
    applied per shell-pair; pass ``kq``/``qsh`` to enable it.
    """
    base, bts = _onsite_base(basis, atoms)             # AO-level onecx (same-atom)
    diag = _np.einsum("ij,ji->i", P, S)                # pop_ao = (P@S) diagonal
    return float(frscale) * (base @ diag)


def gxtb_onsite_fock_exact(P, S, basis, atoms, frscale=0.15, mapping=(2, 0, 1)):
    """FULLY reverse-engineered onsite-exchange Fock (onsite_fx_symv + get_kfock).

    Three symv channels, each V_k = frscale * onecx @ diag(D_k) where D_k is one of
    the three density forms {P, S@P, S@P@S} (lldb-confirmed: channel pops match
    diag(P)/diag(SP)/diag(SPS)).  get_kfock folds them (disasm-confirmed prefactors):

        M = 0.25*OS(V[map0]) + 0.5*OS(V[map1]) + 0.25*diag(V[map2])
        F_onsite = -0.125*(M + M^T)        # net -0.125 off-diag, -0.25 diag

    where OS(V)[j,i] = V[i]*S[j,i] is the overlap-sandwich (daxpy column form).
    ``mapping`` selects which density form feeds (sandwich-0.25, sandwich-0.5,
    diag-0.25); default (SPS, P, SP) per the call-order + matrix.f90 correspondence.
    """
    base, bts = _onsite_base(basis, atoms)             # AO onecx (same-atom)
    SP = S @ P
    Dlist = [P, SP, SP @ S]                             # index 0=P, 1=SP, 2=SPS
    fr = float(frscale)
    V = [fr * (base @ _np.diag(Dlist[k])) for k in range(3)]
    Va, Vb, Vc = V[mapping[0]], V[mapping[1]], V[mapping[2]]
    M = 0.25 * (S * Va[None, :]) + 0.5 * (S * Vb[None, :]) + _np.diag(0.25 * Vc)
    return -0.125 * (M + M.T)


def gxtb_onsite_potential_q(P, S, basis, atoms, qsh, frscale=0.15):
    """Onsite potential WITH the get_gons charge modulation (full dVar64 integrand).

        V_a = frscale * sum_b (1 - 0.5*(kq[l_a]*q_a + kq[l_b]*q_b)) * onecx[Z,l_a,l_b] * pop_b

    kq = pg_fock_kq = [1.1,0.55,0.275,0.1375] (per angular momentum); q = shell
    Mulliken charge.  The charge factor is element/charge dependent, which is why
    the plain (charge-free) potential showed molecule-dependent sign error.
    """
    base, bts = _onsite_base(basis, atoms)
    diag = _np.einsum("ij,ji->i", P, S)
    kq = _np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=_np.float64)
    aol = _np.asarray(basis.shell_l)[bts]              # l per AO
    q_ao = _np.asarray(qsh, dtype=_np.float64)[bts]    # shell charge per AO
    kqq = kq[aol] * q_ao                               # kq_l * q_shell per AO
    factor = 1.0 - 0.5 * (kqq[:, None] + kqq[None, :])  # (n,n) per AO-pair
    return float(frscale) * ((factor * base) @ diag)


def gxtb_twobody_thirdorder(qsh, basis, atoms, coords_ang, *, k3=2.3, kx=1.3, rexp=0.2093327496):
    """Two-body 3rd-order (coulomb_thirdorder_twobody), density-dependent.

    Binary-exact algebra (libxtb get_taumat_0d__omp_fn_0 + get_energy/get_potential,
    mode==0). Per shell i on atom A with angular momentum l:

      eta_base_i = ps_tb2_shell_hubbard[Z,l] * pa_hubbard_parameter[Z]   (NO cn slope)
      eta_eff_i  = eta_base_i * (1 + pa_tb2_hubbard_cn[Z]*(sqrt(cn_A + 1e-12) - 1e-6))
      gamma_ij   = harmonic(eta_eff_i, eta_eff_j) = 2/(1/ei + 1/ej)   (new_average enum 2)
      off-site (A_i != A_j):  tau = k3*x*(1 - 0.5*kx*x)*exp(-kx*x),  x = R_ij/gamma_ij
      on-site  (A_i == A_j, INCLUDING the diagonal):  tau = -rexp*gamma_ij^2

    The three two-body scalars are stored consecutively in the binary param block
    as (2.3, 0.2093327496, 1.3).  The decoded get_taumat_0d._omp_fn.0 reads them as:
      off-site prefactor A = *param_3 = 2.3            -> k3
      on-site  prefactor C = *param_4 = 0.2093327496   -> rexp   (A2-validated vs Ne)
      off-site exp decay B = *param_5 = 1.3            -> kx
    The earlier port had kx=0.2093327496 / rexp=1.3 (i.e. it swapped the on-site
    and off-site-exp constants), which blew the term up (E3 ~ -127 Ha, MAE 1.86).
      g3d_i = pa_tb3_hubbard_derivs[Z] * pg_tb3_kshell[l]   (== basis.shell_third)

    E3 = sum_i g3d_i * q_i^2 * (tau@q)_i ;  V3 = 2*g3d*q*(tau@q) + tau@(g3d*q^2)
    (tau is symmetric, so tau@ and tau.T@ coincide.) Returns (E3, V3[nsh]).
    """
    atoms = _np.asarray(atoms, dtype=_np.intp)
    cb = _np.asarray(coords_ang, dtype=_np.float64) * ANG_TO_BOHR
    sa = _np.asarray(basis.shell_atom)
    sl = _np.asarray(basis.shell_l)
    cn = _np.asarray(basis.cn, dtype=_np.float64)
    q = _np.asarray(qsh, dtype=_np.float64)
    Z = atoms[sa]

    tb2sh = _np.asarray(GXTB_PARAMS["ps_tb2_shell_hubbard"], dtype=_np.float64)
    hubp = _np.asarray(GXTB_PARAMS["pa_hubbard_parameter"], dtype=_np.float64)
    cnsc = _np.asarray(GXTB_PARAMS["pa_tb2_hubbard_cn"], dtype=_np.float64)
    eta_base = tb2sh[Z - 1, sl] * hubp[Z - 1]                          # NO cn slope
    eta_eff = eta_base * (1.0 + cnsc[Z - 1] * (_np.sqrt(cn[sa] + 1e-12) - 1e-6))
    g3d = _np.asarray(basis.shell_third, dtype=_np.float64)            # pa_tb3 * pg_tb3_kshell

    ee_i = eta_eff[:, None]; ee_j = eta_eff[None, :]
    gam = 2.0 / (1.0 / ee_i + 1.0 / ee_j)                              # harmonic average
    R = _np.linalg.norm(cb[sa][:, None, :] - cb[sa][None, :, :], axis=-1)
    same = sa[:, None] == sa[None, :]
    x = R / gam
    tau_off = k3 * x * (1.0 - 0.5 * kx * x) * _np.exp(-kx * x)
    tau_on = -rexp * gam * gam                                         # rexp=0.2093327496; incl. diagonal (R=0)
    tau = _np.where(same, tau_on, tau_off)

    tq = tau @ q
    E3 = float(_np.sum(g3d * q * q * tq))
    V3 = 2.0 * g3d * q * tq + tau @ (g3d * q * q)
    return E3, V3


def gxtb_bocorr_gamma(basis, atoms, coords_ang, *, k=1.0):
    """Bond-order-correction exchange gamma (get_gbocorr): distance-switched per-atom-pair.

    bocorr[mu,nu] = 0.5*(1 + erf(-(R_ij - r0_ij)*k / crad_ij)) * cscale_ij   (i != j)
    r0 = vdw_pair*avg(rvdw_scale) (= mrad); crad = avg(pa_fock_crad); cscale = avg(pa_fock_cscale).
    Geometry-only; added to the exchange gamma (S.P.S Fock).
    """
    atoms = _np.asarray(atoms, dtype=_np.intp)
    cb = _np.asarray(coords_ang, dtype=_np.float64) * ANG_TO_BOHR
    bts = _np.asarray(basis.bf_to_shell); sa = _np.asarray(basis.shell_atom)
    crad = _np.asarray(GXTB_PARAMS["pa_fock_crad"], dtype=_np.float64)[atoms - 1]
    cscale = _np.asarray(GXTB_PARAMS["pa_fock_cscale"], dtype=_np.float64)[atoms - 1]
    mrad = gxtb_mrad_pair(atoms)                                  # r0 per atom pair
    nat = len(atoms)
    R = _np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    crad_ij = 0.5 * (crad[:, None] + crad[None, :])
    cscale_ij = 0.5 * (cscale[:, None] + cscale[None, :])
    arg = -(R - mrad) * k / _np.maximum(crad_ij, 1e-12)
    boc_at = 0.5 * (1.0 + erf(arg)) * cscale_ij                   # (nat,nat)
    _np.fill_diagonal(boc_at, 0.0)
    # expand atom-pair -> AO-pair
    n = bts.size
    aoat = sa[bts]
    g = boc_at[_np.ix_(aoat, aoat)]
    return g
