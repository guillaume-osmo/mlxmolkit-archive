"""
SCF (Self-Consistent Field) solver for RM1.

Iterative procedure:
  1. Build Fock matrix F from density P and integrals
  2. Diagonalize F → eigenvalues, eigenvectors C
  3. Update density P = C_occ @ C_occ.T
  4. Check convergence |P_new - P_old|

Two SCF entry points:

- :func:`nddo_energy_batch` — original numpy-loop implementation. Each
  molecule diagonalizes / DIISes on the CPU in a Python loop; the Fock
  build is the only batched-on-Metal step.
- :func:`rm1_energy_batch_mlx` — all-MLX batched SCF: Fock, optional
  ``mlx_addons.linalg.batched_eigh`` eigensolve, DIIS solve (Metal LU), and
  density update all run on ``mx.array``s without per-iteration host
  round-trips. Same physics, same accepted parameters, batched-first.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import os
from .params import RM1_PARAMS, ElementParams, EV_TO_KCAL, ANG_TO_BOHR
from .methods import get_params, METHOD_PARAMS
from .integrals import (
    compute_one_center_integrals,
    compute_nuclear_repulsion,
    _additive_terms,
    _charge_separations,
    EV,
)
from .two_center_integrals import two_center_integrals, _compute_multipole_params, EV
from .rotation import rotate_integrals_to_molecular_frame
from .overlap import overlap_molecular_frame
from .overlap_d import overlap_d_molecular_frame


_BATCHED_EIGH_LOOKED_UP = False
_BATCHED_EIGH_FN = None
_BATCHED_EIGH_MAX_N = None


def _get_addons_batched_eigh():
    """Return mlx-addons batched_eigh when available, otherwise None."""

    global _BATCHED_EIGH_LOOKED_UP, _BATCHED_EIGH_FN
    if os.environ.get("MLXMOLKIT_DISABLE_ADDONS_EIGH", "").lower() in {"1", "true", "yes"}:
        return None
    if not _BATCHED_EIGH_LOOKED_UP:
        try:
            from mlx_addons.linalg import batched_eigh
        except (ImportError, AttributeError):
            batched_eigh = None
        _BATCHED_EIGH_FN = batched_eigh
        _BATCHED_EIGH_LOOKED_UP = True
    return _BATCHED_EIGH_FN


def _addons_batched_eigh_max_n() -> int:
    """Return the largest matrix size handled by the mlx-addons GPU eigh."""

    global _BATCHED_EIGH_MAX_N
    if _BATCHED_EIGH_MAX_N is None:
        try:
            from mlx_addons.linalg import JACOBI_MAX_N
        except (ImportError, AttributeError):
            JACOBI_MAX_N = 32
        _BATCHED_EIGH_MAX_N = int(JACOBI_MAX_N)
    return _BATCHED_EIGH_MAX_N


def _batched_eigh_backend(matrix_size: int | None = None) -> str:
    if _get_addons_batched_eigh() is not None:
        if matrix_size is None:
            return "mlx_addons.linalg.batched_eigh"
        max_n = _addons_batched_eigh_max_n()
        if int(matrix_size) <= max_n:
            return f"mlx_addons.linalg.batched_eigh(gpu_jacobi,n<={max_n})"
        return f"mlx_addons.linalg.batched_eigh(cpu_fallback,n>{max_n})"
    return "mx.linalg.eigh(cpu)"


def _batched_symmetric_eigh(matrix: mx.array) -> tuple[mx.array, mx.array]:
    """Batched symmetric eigensolver with an mlx-addons GPU hook."""

    batched_eigh = _get_addons_batched_eigh()
    if batched_eigh is not None:
        return batched_eigh(matrix)
    return mx.linalg.eigh(matrix, stream=mx.cpu)


# Degree-5 matrix-sign schedule (Polar Express). Source:
# autoresearch-mlx/muon_and_beyond_mlx.py `_POLAR_EXPRESS_COEFFS`. For a
# symmetric matrix the polar factor equals the matrix sign function, so the
# same schedule drives every eigenvalue of (F - mu I)/s to +-1, giving the
# aufbau density P = I - sign(F - mu I) in pure batched matmuls.
_POLAR_EXPRESS_ABC = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


def _power_spectral_bound(A: mx.array, iters: int = 14) -> mx.array:
    """Per-batch upper bound on ||A||_2 for symmetric A — sync-free.

    Power iteration on A^2 with a deterministic start vector, padded by 5%
    at the call site. Padded (zero) rows contribute zero eigenvalues and do
    not disturb the bound.
    """
    n = A.shape[-1]
    v = mx.ones((n, 1), dtype=A.dtype) + 0.01 * mx.arange(n, dtype=A.dtype)[:, None]
    A2 = A @ A
    eps = mx.array(1e-30, dtype=A.dtype)
    for _ in range(iters):
        v = A2 @ v
        v = v / mx.maximum(mx.sqrt(mx.sum(v * v, axis=-2, keepdims=True)), eps)
    Av = A @ v
    return mx.sqrt(mx.sum(Av * Av, axis=(-2, -1)))


def _sign_density(
    F_sym: mx.array,
    mu: mx.array,
    block_mask: mx.array,
    active_diag: mx.array,
    pad_unit: mx.array,
    eye_MB: mx.array,
    polish: int = 2,
) -> mx.array:
    """Aufbau density P = I - sign(F - mu I) via fixed polynomial iteration.

    Mathematically equivalent to the eigh density 2 * C_occ @ C_occ.T
    whenever mu lies in the HOMO-LUMO gap (sign only depends on which side
    of mu each eigenvalue falls). All batched matmuls — no eigensolver, no
    host sync, no n>32 CPU fallback. Padded diagonals are pinned at +1 so
    they land on the virtual side; padded off-diagonals are hard-masked.

    The caller MUST validate trace(P)/2 == n_occ afterwards: if mu drifts
    out of the gap the trace is off by an integer (>= 1), which makes the
    failure detectable by a 0.25 threshold.
    """
    A = F_sym * block_mask - mu[:, None, None] * active_diag
    s = mx.maximum(_power_spectral_bound(A), mx.array(1.0e-3, dtype=A.dtype))
    X = A / (1.05 * s)[:, None, None] + pad_unit
    # One Newton-Schulz contraction first: tolerant to |lambda| slightly > 1,
    # so a marginally low power-iteration bound cannot blow up the schedule.
    X = 1.5 * X - 0.5 * (X @ (X @ X))
    for a, b, c in _POLAR_EXPRESS_ABC + [_POLAR_EXPRESS_ABC[-1]] * polish:
        X2 = X @ X
        X4 = X2 @ X2
        X = X @ (a * eye_MB + b * X2 + c * X4)
    P = (active_diag - X) * block_mask
    return 0.5 * (P + mx.transpose(P, (0, 2, 1)))


def _assert_finite_mx(matrix: mx.array, *, label: str) -> None:
    """Optional host-side finite check for debugging hard eigensolver crashes."""

    mx.eval(matrix)
    matrix_np = np.array(matrix)
    if not np.all(np.isfinite(matrix_np)):
        bad = np.where(~np.isfinite(matrix_np).all(axis=(-2, -1)))[0]
        raise RuntimeError(
            f"Non-finite {label} for batch indices {bad}; "
            f"max abs entry = {np.nanmax(np.abs(matrix_np)):.3e}"
        )


def _basis_bucket_limit(n_basis: int) -> int:
    """Compact basis-size bucket for batched SCF padding."""

    n_basis = int(n_basis)
    gpu_max = _addons_batched_eigh_max_n()
    if n_basis <= gpu_max:
        return gpu_max
    step = 16 if n_basis <= 128 else 32
    return ((n_basis + step - 1) // step) * step


def _bucket_molecule_indices_by_basis(
    molecules: list[tuple[list[int], np.ndarray]],
    param_dict: dict[int, ElementParams],
) -> list[list[int]]:
    """Group molecules so one large basis does not over-pad the whole batch."""

    buckets: dict[int, list[int]] = {}
    for index, molecule in enumerate(molecules):
        atoms = molecule[0]
        n_basis = sum(param_dict[z].n_basis for z in atoms)
        buckets.setdefault(_basis_bucket_limit(n_basis), []).append(index)
    return [buckets[key] for key in sorted(buckets)]


def _charged_electron_count(atoms: list[int], params: list[ElementParams], molecular_charge: float = 0.0) -> int:
    n_elec_float = float(sum(p.n_valence for p in params)) - float(molecular_charge)
    n_elec = int(round(n_elec_float))
    if not np.isclose(n_elec_float, n_elec, atol=1.0e-6):
        raise ValueError(f"molecular charge must produce an integer electron count: {n_elec_float}")
    if n_elec < 0:
        raise ValueError("molecular charge produces a negative electron count")
    if n_elec % 2 != 0:
        raise ValueError(
            "only closed-shell NDDO calculations are currently supported; "
            f"atoms={atoms} and molecular_charge={molecular_charge} give {n_elec} electrons"
        )
    return n_elec


def _build_basis_info(atoms: list[int], param_dict=None, molecular_charge: float = 0.0):
    """Build basis function → atom mapping."""
    if param_dict is None:
        param_dict = RM1_PARAMS
    params = [param_dict[z] for z in atoms]
    n_basis = sum(p.n_basis for p in params)
    n_elec = _charged_electron_count(atoms, params, molecular_charge)

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

    # Diagonal: one-electron one-center (Uss, Upp, Udd)
    for mu in range(n_basis):
        i = b2a[mu]
        p = params[i]
        if btype[mu] == 0:
            H[mu, mu] = p.Uss
        elif btype[mu] <= 3:
            H[mu, mu] = p.Upp
        else:
            # d-orbital (types 4-8)
            H[mu, mu] = p.Udd

    # Off-diagonal: resonance integrals using proper Slater overlap
    # H_μν = 0.5 * (beta_μ + beta_ν) * S_μν  (Wolfsberg-Helmholz)
    n_atoms = len(atoms)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pA = params[i]
            pB = params[j]

            # Overlap — uses d-orbital overlap for PM6, sp overlap otherwise
            nA = pA.n_basis
            nB = pB.n_basis

            if nA > 4 or nB > 4:
                # Full d-orbital overlap (proper Slater integrals)
                S_ij = overlap_d_molecular_frame(pA, pB, coords[i], coords[j])
            else:
                S_ij = overlap_molecular_frame(pA, pB, coords[i], coords[j])

            for mu_off in range(nA):
                mu = starts[i] + mu_off
                if btype[mu] == 0:
                    beta_mu = pA.beta_s
                elif btype[mu] <= 3:
                    beta_mu = pA.beta_p
                else:
                    beta_mu = pA.beta_d

                for nu_off in range(nB):
                    nu = starts[j] + nu_off
                    if btype[nu] == 0:
                        beta_nu = pB.beta_s
                    elif btype[nu] <= 3:
                        beta_nu = pB.beta_p
                    else:
                        beta_nu = pB.beta_d

                    H[mu, nu] = 0.5 * (beta_mu + beta_nu) * S_ij[mu_off, nu_off]
                    H[nu, mu] = H[mu, nu]

    # Electron-nuclear attraction using properly rotated integrals
    # Note: rotate_integrals works on sp (4×4) block only
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            _, e1b_ij, _ = rotate_integrals_to_molecular_frame(
                params[i], params[j], coords[i], coords[j],
            )
            # e1b is (min(nA,4), min(nA,4)) — only sp block
            nA_sp = min(params[i].n_basis, 4)
            for mu_a in range(nA_sp):
                for nu_a in range(nA_sp):
                    H[starts[i] + mu_a, starts[i] + nu_a] += e1b_ij[mu_a, nu_a]
            # d-orbital electron-nuclear attraction. Triggered whenever A
            # has d-orbitals — the (μν_A | s_B s_B) integral for e-n
            # attraction only depends on B through Z_B (= pB.n_valence)
            # and rho0_B (from gss_B), so the same code path handles YH
            # (B is H), YX (B is sp heavy) and YY (B has d-orbitals too).
            # Uses the proper PYSEQM-equivalent 45-element local-frame
            # integral + RotateCore Wigner-D rotation in
            # tetci_yh.yh_e1b_contribution. This REPLACES the sp-only
            # contribution above with the full 9×9 H_core block for atom A.
            if params[i].n_basis == 9 and params[j].n_basis in (1, 4, 9):
                from .tetci_yh import yh_e1b_contribution
                e1b_full = yh_e1b_contribution(
                    params[i], params[j], coords[i], coords[j]
                )
                # Subtract the sp contribution we already added (avoid double-count),
                # then add the full 9×9 block.
                sA = starts[i]
                for mu in range(nA_sp):
                    for nu in range(nA_sp):
                        H[sA + mu, sA + nu] -= e1b_ij[mu, nu]
                for mu in range(9):
                    for nu in range(9):
                        H[sA + mu, sA + nu] += e1b_full[mu, nu]

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

        if p.n_basis == 9 and getattr(p, 'has_d', False) and len(idx) >= 9:
            # PM6 d-orbital atom: sp formulas for 4×4 block + W for d cross-terms
            px, py, pz = idx[1], idx[2], idx[3]
            Pss = P[s, s]
            Ppp_total = P[px, px] + P[py, py] + P[pz, pz]

            # Standard sp one-center (same as non-d atoms)
            F[s, s] += Pss * p.gss * 0.5 + Ppp_total * (p.gsp - 0.5 * p.hsp)
            sp_fac_1 = p.gsp - 0.5 * p.hsp
            sp_fac_2 = 1.5 * p.hsp - 0.5 * p.gsp
            pp_fac_d = 1.25 * p.gp2 - 0.25 * p.gpp
            pp_fac_off = 0.75 * p.gpp - 1.25 * p.gp2

            for k in range(1, 4):
                pk = idx[k]
                F[pk, pk] += (Pss * sp_fac_1
                              + P[pk, pk] * p.gpp * 0.5
                              + (Ppp_total - P[pk, pk]) * pp_fac_d)
            for k in range(1, 4):
                pk = idx[k]
                F[s, pk] += P[s, pk] * sp_fac_2
                F[pk, s] += P[pk, s] * sp_fac_2
            for k in range(1, 4):
                for l in range(k + 1, 4):
                    pk, pl = idx[k], idx[l]
                    F[pk, pl] += P[pk, pl] * pp_fac_off
                    F[pl, pk] += P[pl, pk] * pp_fac_off

            # d-orbital cross terms via W integrals (pure NumPy).
            # Native compute_w_integrals matches PYSEQM's calc_integral to
            # machine precision (~1e-14) after the IntRep/IntRf2 table fix.
            # Uses TAIL exponents (PM6_TAIL_EXPONENTS) — the PYSEQM
            # convention for d-orbital atoms. No PyTorch dependency.
            from .fock_d import fock_d_one_center, TRIL_I, TRIL_J
            from .tetci_multipole_pyseqm import PM6_TAIL_EXPONENTS
            from .w_integrals import compute_w_integrals
            from .params import principal_qn
            qn_sp = principal_qn(p.Z)
            if p.Z in PM6_TAIL_EXPONENTS:
                zs_t, zp_t, zd_t = PM6_TAIL_EXPONENTS[p.Z]
            else:
                zs_t, zp_t, zd_t = p.zeta_s, p.zeta_p, p.zeta_d
            W = compute_w_integrals(
                zs_t, zp_t, zd_t, qn_sp, qn_sp,
                getattr(p, 'F0SD', 0.0), getattr(p, 'G2SD', 0.0),
            )
            # W integrals: ADDITIVE d-orbital contribution on top of sp formulas
            # W encodes s-d, p-d, and d-d cross-terms (NOT sp-sp replacement)
            F = fock_d_one_center(F, P, W, starts[i], n_basis=9)

        elif p.n_basis == 1:
            F[s, s] += Pss * p.gss * 0.5
        else:
            px, py, pz = idx[1], idx[2], idx[3]
            Ppp_total = P[px, px] + P[py, py] + P[pz, pz]

            F[s, s] += Pss * p.gss * 0.5 + Ppp_total * (p.gsp - 0.5 * p.hsp)

            sp_fac_1 = p.gsp - 0.5 * p.hsp
            sp_fac_2 = 1.5 * p.hsp - 0.5 * p.gsp
            pp_fac_d = 1.25 * p.gp2 - 0.25 * p.gpp
            pp_fac_off = 0.75 * p.gpp - 1.25 * p.gp2

            for k in range(1, 4):
                pk = idx[k]
                F[pk, pk] += (Pss * sp_fac_1
                              + P[pk, pk] * p.gpp * 0.5
                              + (Ppp_total - P[pk, pk]) * pp_fac_d)

            for k in range(1, 4):
                pk = idx[k]
                F[s, pk] += P[s, pk] * sp_fac_2
                F[pk, s] += P[pk, s] * sp_fac_2

            for k in range(1, 4):
                for l in range(k + 1, 4):
                    pk, pl = idx[k], idx[l]
                    F[pk, pl] += P[pk, pl] * pp_fac_off
                    F[pl, pk] += P[pl, pk] * pp_fac_off

    # === Two-center contribution (full 10x10 w tensor) ===
    from .two_center_d import two_center_w_10x10

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pA = params[i]
            pB = params[j]
            sA = starts[i]
            sB = starts[j]

            # 4x4 w tensor for sp block (verified against PYSEQM)
            w, e1b_ij, e2a_ij = rotate_integrals_to_molecular_frame(
                pA, pB, coords[i], coords[j],
            )
            nA_sp = min(pA.n_basis, 4)
            nB_sp = min(pB.n_basis, 4)
            # Vectorized sp two-center via einsum (replaces 4×4×4×4 loop):
            #   F[μν_A] += Σ P[λσ_B] w[μν, λσ]   (J on A)
            #   F[λσ_B] += Σ P[μν_A] w[μν, λσ]   (J on B)
            #   F[μλ_AB] -= 0.5 Σ P[νσ_AB] w[μν, λσ]  (K)
            w_sp = w[:nA_sp, :nA_sp, :nB_sp, :nB_sp]
            P_AA = P[sA:sA + nA_sp, sA:sA + nA_sp]
            P_BB = P[sB:sB + nB_sp, sB:sB + nB_sp]
            P_AB = P[sA:sA + nA_sp, sB:sB + nB_sp]
            F[sA:sA + nA_sp, sA:sA + nA_sp] += np.einsum('abcd,cd->ab', w_sp, P_BB)
            F[sB:sB + nB_sp, sB:sB + nB_sp] += np.einsum('abcd,ab->cd', w_sp, P_AA)
            K_sp = -0.5 * np.einsum('abcd,bd->ac', w_sp, P_AB)
            F[sA:sA + nA_sp, sB:sB + nB_sp] += K_sp
            F[sB:sB + nB_sp, sA:sA + nA_sp] += K_sp.T

            # d-orbital two-center: proper rho3-6 based multipole integrals
            if pA.n_basis == 9 or pB.n_basis == 9:
                from .d_two_center import d_two_center_fock
                F = d_two_center_fock(F, P, pA, pB, sA, sB, coords[i], coords[j])

    return F



_D_ORBITAL_ELEMENTS = frozenset({15, 16, 17, 35, 53})  # P, S, Cl, Br, I


def rm1_energy_batch_pm6d(
    molecules: list[tuple[list[int], np.ndarray]],
    backend: str = "pyseqm",
) -> list[dict]:
    """Batched PM6_D charges via PYSEQM (much faster than per-mol calls).

    PYSEQM accepts ``species`` of shape ``(n_mol, n_atom_max)`` and runs a
    single SCF over the whole batch. With batches of 64-128 mols this gives
    50+ mol/s on CPU vs ~1.5 mol/s single-call. Atoms shorter than the
    batch maximum get padded with zero (treated as ghost / dummy by PYSEQM).

    Parameters
    ----------
    molecules : list of (atoms, coords)
        Each entry is ``(atomic_numbers: list[int], coords: ndarray[N, 3])``.
        Sizes can differ — they're padded inside.
    backend : str
        Currently only ``"pyseqm"`` is supported.

    Returns
    -------
    list of dict, one per molecule, each containing:
      - ``q``: atomic Mulliken charges
      - ``density``: 9*n × 9*n density matrix in PYSEQM layout
      - ``converged``: SCF convergence flag
      - ``backend``: which engine was used
    """
    if backend != "pyseqm":
        raise NotImplementedError("Only PYSEQM backend supported for batched PM6_D")

    try:
        import torch
        from seqm.ElectronicStructure import Electronic_Structure
        from seqm.Molecule import Molecule
        from seqm.seqm_functions.constants import Constants
    except ImportError as exc:
        raise ImportError(
            "Batched PM6_D requires PYSEQM on PYTHONPATH "
            "(github.com/lanl/PYSEQM)."
        ) from exc

    n_mol = len(molecules)
    if n_mol == 0:
        return []

    # PYSEQM requires species sorted non-increasing by Z within each mol
    perms = []  # original-order index for each mol
    sym_padded = []
    coords_padded = []
    n_atom_max = max(len(atoms) for atoms, _ in molecules)
    for atoms, coords in molecules:
        atoms_arr = np.asarray(atoms)
        order = np.argsort(-atoms_arr)
        perms.append(np.argsort(order))
        sym_sorted = atoms_arr[order]
        coords_sorted = np.asarray(coords, dtype=np.float64)[order]
        # Pad to n_atom_max with z=0 atoms (PYSEQM treats z=0 as dummy)
        pad_n = n_atom_max - len(atoms)
        if pad_n > 0:
            sym_sorted = np.concatenate([sym_sorted, np.zeros(pad_n, dtype=int)])
            coords_sorted = np.vstack(
                [coords_sorted, np.tile([100.0 + 10 * np.arange(pad_n)[:, None], np.zeros((pad_n, 2))], 1)
                 if False else np.column_stack([
                     100.0 + 10 * np.arange(pad_n), np.zeros(pad_n), np.zeros(pad_n)
                 ])]
            )  # Place dummies far away to avoid spurious interactions
        sym_padded.append(sym_sorted)
        coords_padded.append(coords_sorted)

    species_t = torch.as_tensor(np.stack(sym_padded), dtype=torch.int64)
    xyz_t = torch.as_tensor(np.stack(coords_padded), dtype=torch.float64)

    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dev = torch.device("cpu")
        const = Constants().to(dev)
        seqm_params = {
            "method": "PM6",
            "scf_eps": 1.0e-8,
            "scf_converger": [2],
            "scf_backward": 0,
        }
        species_t = species_t.to(dev)
        xyz_t = xyz_t.to(dev)
        mol = Molecule(const, seqm_params, xyz_t, species_t).to(dev)
        es = Electronic_Structure(seqm_params).to(dev)
        es(mol)
        q_all = mol.q.detach().cpu().numpy()  # (n_mol, n_atom_max)
        dm_all = mol.dm.detach().cpu().numpy()  # (n_mol, 9*n_atom_max, 9*n_atom_max)
        nconv = getattr(es, "notconverged", None)
        if nconv is not None:
            converged_flags = (~nconv).cpu().numpy().tolist() if hasattr(nconv, "cpu") else [not bool(nconv.any())] * n_mol
        else:
            converged_flags = [True] * n_mol
    finally:
        torch.set_default_dtype(prev_dt)

    results = []
    for i, ((atoms, _), perm) in enumerate(zip(molecules, perms)):
        n_real = len(atoms)
        q_padded = q_all[i, :n_real]
        q_orig_order = q_padded[perm]  # un-permute back to caller atom order
        # For density: extract the real-atom slice (first n_real*9 rows/cols)
        # in PYSEQM-sorted order, then un-permute the 9-orbital blocks.
        dm_sorted = dm_all[i, :9 * n_real, :9 * n_real]
        # Build the un-permutation block-by-block
        dm_orig = np.zeros_like(dm_sorted)
        for a_p, a_c in enumerate(np.argsort(perm)):
            for b_p, b_c in enumerate(np.argsort(perm)):
                dm_orig[
                    9 * a_c : 9 * (a_c + 1), 9 * b_c : 9 * (b_c + 1)
                ] = dm_sorted[
                    9 * a_p : 9 * (a_p + 1), 9 * b_p : 9 * (b_p + 1)
                ]
        results.append({
            "q": q_orig_order,
            "density": dm_orig,
            "converged": converged_flags[i],
            "backend": "pyseqm_batched",
        })
    return results


def _pm6d_via_pyseqm(atoms: list[int], coords: np.ndarray) -> dict:
    """Delegate PM6_D SCF to PYSEQM backend.

    mlxmolkit's native PM6_D had structural bugs (incomplete d-orbital
    two-center Fock in d_two_center.py — only `dd_ss` monopole, missing
    `dd_pp`, `dd_sp`, exchange + proper d-orbital nuclear attraction).
    PYSEQM's PM6 is validated against MOPAC binary to 0.011 e on H2S.

    Until the native MLX port is complete, route PM6_D through PYSEQM.
    Output is reformatted into mlxmolkit's dict shape with `density`
    matching the caller's basis layout (1 orbital per H, 4 per CHNO,
    9 per P/S/Cl/Br/I), so :func:`_mulliken_charges` works unchanged.

    Citation: PYSEQM (Zhou et al.), BSD-3-Clause, github.com/lanl/PYSEQM.
    """
    try:
        import torch  # noqa: F401
        from seqm.ElectronicStructure import Electronic_Structure
        from seqm.Molecule import Molecule
        from seqm.seqm_functions.constants import Constants
    except ImportError as exc:
        raise ImportError(
            "PM6_D delegation requires PYSEQM on PYTHONPATH "
            "(github.com/lanl/PYSEQM). Set PYTHONPATH or pip install."
        ) from exc

    n = len(atoms)
    # PYSEQM requires atoms sorted non-increasing by Z; permute, run, un-permute
    order = np.argsort(-np.asarray(atoms))
    inv_order = np.argsort(order)
    syms_sorted = [int(atoms[i]) for i in order]
    coords_sorted = np.asarray(coords)[order]

    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dev = torch.device("cpu")
        const = Constants().to(dev)
        seqm_params = {
            "method": "PM6",
            "scf_eps": 1.0e-8,
            "scf_converger": [2],
            "scf_backward": 0,
        }
        species = torch.as_tensor([syms_sorted], dtype=torch.int64, device=dev)
        xyz = torch.as_tensor(np.asarray([coords_sorted]), dtype=torch.float64, device=dev)
        mol = Molecule(const, seqm_params, xyz, species).to(dev)
        es = Electronic_Structure(seqm_params).to(dev)
        es(mol)
        # PYSEQM density: shape (1, 9*n, 9*n) for PM6 — 9 orbitals per atom
        dm_pyseqm = mol.dm.detach().cpu().numpy()[0]  # (9n, 9n)
        hof_ev = float(mol.Hf.detach().cpu().numpy()[0])
        nconv = es.notconverged if hasattr(es, "notconverged") else None
        converged = (nconv is None) or (not bool(nconv.any().item()))
    finally:
        torch.set_default_dtype(prev_dt)

    # Re-map PYSEQM's 9*n density into mlxmolkit's variable-width layout.
    # PYSEQM packs 9 orbitals per atom regardless; we strip unused slots so
    # the caller's _mulliken_charges (which expects n_basis_per layout) works.
    PARAMS = get_params("PM6_D")
    # Build basis-size list in the CALLER's (un-permuted) atom order.
    n_basis_per = [PARAMS[z].n_basis for z in atoms]
    total_basis = sum(n_basis_per)
    # Map (caller_atom_idx, orb_offset) → flat caller-basis index
    caller_offsets = np.cumsum([0] + n_basis_per).tolist()
    # PYSEQM order: order[k] = caller_atom_idx of PYSEQM slot k
    P = np.zeros((total_basis, total_basis), dtype=np.float64)
    for a_p, a_c in enumerate(order):
        nb_c = n_basis_per[a_c]
        for b_p, b_c in enumerate(order):
            nb_b = n_basis_per[b_c]
            # PYSEQM block: rows 9*a_p .. 9*a_p+nb_c, cols 9*b_p .. 9*b_p+nb_b
            src = dm_pyseqm[
                9 * a_p : 9 * a_p + nb_c,
                9 * b_p : 9 * b_p + nb_b,
            ]
            P[
                caller_offsets[a_c] : caller_offsets[a_c] + nb_c,
                caller_offsets[b_c] : caller_offsets[b_c] + nb_b,
            ] = src

    return {
        "density": P,
        "converged": converged,
        "n_iter": -1,  # PYSEQM doesn't expose; signal external
        "heat_of_formation_eV": hof_ev,
        "energy_eV": hof_ev,
        "energy_kcal": hof_ev * 23.060547830619,
        "backend": "pyseqm",
    }


def nddo_energy(
    atoms: list[int],
    coords: np.ndarray,
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    verbose: bool = False,
    use_metal: bool = False,
    method: str = 'RM1',
    native: bool = False,
    molecular_charge: float = 0.0,
) -> dict:
    """Compute NDDO semi-empirical single-point energy.

    Args:
        atoms: list of atomic numbers [1, 8, 1] for H2O
        coords: (N, 3) coordinates in Angstrom
        max_iter: max SCF iterations
        conv_tol: density matrix convergence threshold
        method: 'RM1', 'AM1', 'AM1_STAR', 'PM6_SP', 'PM6_D'
        molecular_charge: net molecular charge used to set the closed-shell
            electron count.
        native: For ``method='PM6_D'``, force the native mlxmolkit path
            instead of delegating to PYSEQM. The native path matches
            PYSEQM to machine precision on the test set after the
            2026-05-24 fixes (qn per Z, symmetric H_core, YH/YX/YY d-orb
            Fock J/K + e-n attraction, PYSEQM-delegated diatomic overlap
            for qn≥3). PYSEQM is still imported as a thin BSD-3-Clause
            library for the W-integral, overlap, and per-pair w-tensor
            computations; a fully self-contained build is documented in
            ``NATIVE_STATUS.md``.

    Returns:
        dict with 'energy_eV', 'energy_kcal', 'converged', 'n_iter', 'density'

    Note:
        For ``method='PM6_D'`` by default the d-orbital SCF is delegated
        to PYSEQM (validated against the MOPAC binary). Set
        ``native=True`` to run the mlxmolkit path instead.
    """
    # PM6_D is bit-exact via the vendored numpy PYSEQM port
    # (mlxmolkit/rm1/_pyseqm_port/), so the native path is always used.
    # `native` kw is kept for back-compat but is now a no-op.
    _ = native

    PARAMS = get_params(method)
    coords = np.asarray(coords, dtype=np.float64)
    info = _build_basis_info(atoms, PARAMS, molecular_charge=molecular_charge)
    n_basis = info['n_basis']
    n_occ = info['n_occ']

    if verbose:
        print(f"{method}: {len(atoms)} atoms, {n_basis} basis functions, {info['n_elec']} electrons, {n_occ} occupied")

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

    # Initial density: MOPAC-style neutral-atom diagonal guess. Each atom keeps its own
    # valence electrons, spread uniformly over its s/p orbitals (d starts empty for
    # main group). Unlike an H_core-diagonalization guess, every atom starts NEUTRAL —
    # an H_core guess can start deep in a charge-transfer basin and the SCF then
    # converges to a wrong (higher) root: EtBr did exactly that (q(Br)=+0.22 instead of
    # MOPAC's -0.16, heat of formation +210 kcal/mol off) while MeBr was fine.
    starts_per_atom = info['atom_basis_start']
    P = np.zeros((n_basis, n_basis))
    for i, p in enumerate(params):
        sA = starts_per_atom[i]
        n_sp = min(p.n_basis, 4)
        for k in range(n_sp):
            P[sA + k, sA + k] = p.n_valence / n_sp

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

        # Level shifting for d-orbital convergence
        # Add shift to virtual orbitals to prevent oscillation
        has_d = any(p.n_basis > 4 for p in params)
        _delta = delta if iteration > 0 else 1.0
        if has_d and iteration < 100 and _delta > 0.001:
            level_shift = 5.0  # stronger level shift for d-orbital atoms
        else:
            level_shift = 0.0

        # Diagonalize
        eigenvalues, C = np.linalg.eigh(F)

        # Apply level shift to virtual orbitals (helps d-orbital convergence)
        if level_shift > 0 and n_occ < n_basis:
            for k in range(n_occ, n_basis):
                eigenvalues[k] += level_shift
            # Rebuild F with shifted eigenvalues
            F_shifted = C @ np.diag(eigenvalues) @ C.T
            eigenvalues, C = np.linalg.eigh(F_shifted)

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

        # Adaptive density mixing — always mix, stronger damping for d-orbitals
        has_d = any(p.n_basis > 4 for p in params)
        if iteration < 3:
            mix = 0.3 if has_d else 0.5
        elif delta > 0.1:
            mix = 0.05 if has_d else 0.4  # heavy damping for d-orbital atoms
        elif delta > 0.01:
            mix = 0.5
        else:
            mix = 0.8  # light damping near convergence
        P = mix * P_new + (1.0 - mix) * P

    # Final Fock with converged density
    F = _fock(H, P)

    # Electronic energy
    E_elec = 0.5 * np.sum(P * (H + F))

    # Nuclear repulsion — ALL PM6 variants use the PM6 PWCCT core-core; others AM1-style.
    # (PM6_D previously fell through to the AM1-style term, corrupting the heat-of-formation
    #  by ~11 eV even for CHNO molecules with no d-orbitals — the electronic energy/density
    #  were already correct, only this core-core term was wrong.)
    if method in ('PM6', 'PM6_SP', 'PM6_D'):
        from .pwcct import pm6_nuclear_repulsion
        E_nuc = pm6_nuclear_repulsion(atoms, coords, PARAMS)
    else:
        E_nuc = compute_nuclear_repulsion(atoms, coords, param_dict=PARAMS)

    # Total energy
    E_total = E_elec + E_nuc

    # Heat of formation:
    # ΔHf = E_total - Σ Eisol(atom) + Σ eheat(atom)
    # Eisol computed using PYSEQM/MOPAC coefficients (in params.py)
    E_isol_total = sum(PARAMS[z].eisol for z in atoms)
    eheat_total = sum(PARAMS[z].eheat for z in atoms)

    E_binding_eV = E_total - E_isol_total
    E_hof_eV = E_binding_eV + eheat_total / EV_TO_KCAL

    # Mulliken partial charges (NDDO/ZDO): q_A = Z_valence(A) - Σ_{μ∈A} P_μμ.
    # Returned by default so the toolkit is drop-in usable like OpenMOPAC (which prints
    # net atomic charges). Sums to ~0 for neutral molecules by construction.
    charges = np.empty(len(atoms))
    _mu = 0
    for _i, _p in enumerate(params):
        _pop = 0.0
        for _k in range(_p.n_basis):
            _pop += P[_mu, _mu]
            _mu += 1
        charges[_i] = _p.n_valence - _pop

    return {
        'energy_eV': E_total,
        'energy_kcal': E_total * EV_TO_KCAL,
        'electronic_eV': E_elec,
        'nuclear_eV': E_nuc,
        'heat_of_formation_eV': E_hof_eV,
        'heat_of_formation_kcal': E_hof_eV * EV_TO_KCAL,
        'charges': charges,
        'converged': converged,
        'n_iter': iteration + 1,
        'eigenvalues': eigenvalues,
        'density': P,
        'n_basis': n_basis,
        'method': method,
        'molecular_charge': float(molecular_charge),
        'n_electrons': int(info['n_elec']),
    }


def nddo_energy_batch(
    molecules: list[tuple[list[int], np.ndarray]],
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    use_metal: bool = True,
    verbose: bool = False,
    method: str = 'RM1',
    molecular_charges: list[float] | None = None,
    density_solver: str = 'auto',
) -> list[dict]:
    """Compute NDDO energies for N molecules simultaneously.

    Args:
        molecules: list of (atoms, coords) tuples
        max_iter: max SCF iterations
        conv_tol: density matrix convergence threshold
        use_metal: True for Metal GPU Fock build, False for CPU
        verbose: print convergence info
        method: 'RM1', 'AM1', or 'AM1_STAR'
        molecular_charges: optional net charge per molecule.
        density_solver: 'eigh' (per-iteration eigensolve), 'sign'
            (Fermi-shifted matrix-sign density, matmul-only middle
            iterations), or 'auto' (sign when the padded basis exceeds the
            GPU Jacobi limit, eigh otherwise).

    Returns:
        list of result dicts (same format as nddo_energy)
    """
    from .batch import prepare_batch
    from .fock_metal import build_fock_batch_metal, build_fock_batch_cpu, MetalFockContext

    PARAMS = get_params(method)

    N = len(molecules)
    if N == 0:
        return []

    if use_metal:
        try:
            buckets = _bucket_molecule_indices_by_basis(molecules, PARAMS)
            if len(buckets) > 1:
                results_by_index: list[dict | None] = [None] * N
                for bucket_index, indices in enumerate(buckets):
                    bucket_molecules = [molecules[index] for index in indices]
                    bucket_charges = (
                        None
                        if molecular_charges is None
                        else [molecular_charges[index] for index in indices]
                    )
                    bucket_results = rm1_energy_batch_mlx(
                        bucket_molecules,
                        max_iter=max_iter,
                        conv_tol=conv_tol,
                        verbose=verbose,
                        method=method,
                        molecular_charges=bucket_charges,
                        density_solver=density_solver,
                    )
                    bucket_max_basis = max(int(result["n_basis"]) for result in bucket_results)
                    for original_index, result in zip(indices, bucket_results):
                        result["batch_bucket_index"] = bucket_index
                        result["batch_bucket_count"] = len(buckets)
                        result["batch_bucket_size"] = len(indices)
                        result["batch_bucket_max_basis"] = bucket_max_basis
                        results_by_index[original_index] = result
                return [result for result in results_by_index if result is not None]
            return rm1_energy_batch_mlx(
                molecules,
                max_iter=max_iter,
                conv_tol=conv_tol,
                verbose=verbose,
                method=method,
                molecular_charges=molecular_charges,
                density_solver=density_solver,
            )
        except (ImportError, RuntimeError) as exc:
            if verbose:
                print(f"  all-MLX batch SCF unavailable; falling back to legacy batch path: {exc}")

    # Pre-compute all integrals (CPU, done once)
    batch = prepare_batch(molecules, param_dict=PARAMS, molecular_charges=molecular_charges)
    MB = batch.max_basis

    # Initial density: MOPAC-style neutral-atom diagonal guess (matches the
    # single-molecule path; an H_core guess can converge to a wrong root).
    batch.P = np.zeros((N, MB, MB), dtype=np.float64)
    for mol_idx in range(N):
        start = 0
        for z in batch.atoms_list[mol_idx]:
            p_z = PARAMS[z]
            n_sp = min(p_z.n_basis, 4)
            for k in range(n_sp):
                batch.P[mol_idx, start + k, start + k] = p_z.n_valence / n_sp
            start += p_z.n_basis

    # DIIS state per molecule
    diis_size = 6
    diis_F_hist = [[] for _ in range(N)]
    diis_E_hist = [[] for _ in range(N)]

    # Convergence tracking
    converged_arr = np.zeros(N, dtype=bool)
    n_iter_arr = np.zeros(N, dtype=np.int32)
    P_prev = batch.P.copy()

    # Build Fock function — pre-allocate GPU context for Metal
    if use_metal:
        try:
            metal_ctx = MetalFockContext(batch)
            def build_fock(b):
                return metal_ctx.build_fock(b.P)
        except RuntimeError:
            # Fallback to CPU if Metal not available
            build_fock = build_fock_batch_cpu
            use_metal = False
    else:
        build_fock = build_fock_batch_cpu

    for iteration in range(max_iter):
        # Build Fock matrices for all molecules
        F_all = build_fock(batch)

        # Per-molecule: diagonalize, update density, check convergence
        for mol_idx in range(N):
            if converged_arr[mol_idx]:
                continue

            nb = batch.n_basis_arr[mol_idx]
            nocc = batch.n_occ_arr[mol_idx]
            F = F_all[mol_idx, :nb, :nb]
            P = batch.P[mol_idx, :nb, :nb]
            H = batch.H_core[mol_idx, :nb, :nb]

            # Symmetrize F
            F = 0.5 * (F + F.T)

            # DIIS extrapolation
            if iteration >= 2:
                # Error matrix: FPS - SPF (with S=I in NDDO)
                E_diis = F @ P - P @ F
                diis_F_hist[mol_idx].append(F.copy())
                diis_E_hist[mol_idx].append(E_diis)

                if len(diis_F_hist[mol_idx]) > diis_size:
                    diis_F_hist[mol_idx].pop(0)
                    diis_E_hist[mol_idx].pop(0)

                nd = len(diis_F_hist[mol_idx])
                if nd >= 2:
                    B = np.zeros((nd + 1, nd + 1))
                    for ii in range(nd):
                        for jj in range(nd):
                            B[ii, jj] = np.sum(diis_E_hist[mol_idx][ii] *
                                              diis_E_hist[mol_idx][jj])
                    B[:nd, nd] = -1.0
                    B[nd, :nd] = -1.0
                    rhs = np.zeros(nd + 1)
                    rhs[nd] = -1.0

                    try:
                        c = np.linalg.solve(B, rhs)
                        F = sum(c[ii] * diis_F_hist[mol_idx][ii] for ii in range(nd))
                    except np.linalg.LinAlgError:
                        pass

            # Diagonalize (guard NaN)
            if not np.all(np.isfinite(F)):
                converged_arr[mol_idx] = False
                continue
            try:
                eigvals, C = np.linalg.eigh(F)
            except np.linalg.LinAlgError:
                converged_arr[mol_idx] = False
                continue
            P_new = 2.0 * C[:, :nocc] @ C[:, :nocc].T

            # Check convergence
            dP = np.max(np.abs(P_new - P))
            if dP < conv_tol:
                converged_arr[mol_idx] = True
                n_iter_arr[mol_idx] = iteration + 1

            # Update density
            if iteration < 2:
                P_mixed = 0.5 * P_new + 0.5 * P
            else:
                P_mixed = P_new

            batch.P[mol_idx, :nb, :nb] = P_mixed

        if verbose and (iteration % 5 == 0 or np.all(converged_arr)):
            n_conv = np.sum(converged_arr)
            print(f"  SCF iter {iteration+1}: {n_conv}/{N} converged")

        if np.all(converged_arr):
            break

    # Mark unconverged
    for mol_idx in range(N):
        if not converged_arr[mol_idx]:
            n_iter_arr[mol_idx] = max_iter

    # Final Fock with converged density
    F_final = build_fock(batch)
    results = []

    for mol_idx in range(N):
        nb = batch.n_basis_arr[mol_idx]
        atoms = batch.atoms_list[mol_idx]
        P = batch.P[mol_idx, :nb, :nb]
        F = F_final[mol_idx, :nb, :nb]
        H = batch.H_core[mol_idx, :nb, :nb]

        # Symmetrize
        F = 0.5 * (F + F.T)

        # Electronic energy
        E_elec = 0.5 * np.sum(P * (H + F))
        E_nuc = batch.E_nuc[mol_idx]
        E_total = E_elec + E_nuc

        # Heat of formation
        E_isol = sum(PARAMS[z].eisol for z in atoms)
        eheat = sum(PARAMS[z].eheat for z in atoms)
        E_binding = E_total - E_isol
        E_hof = E_binding + eheat / EV_TO_KCAL

        # Eigenvalues
        eigvals, _ = np.linalg.eigh(F)

        charges = np.empty(len(atoms))
        mu = 0
        for atom_index, atomic_number in enumerate(atoms):
            atom_params = PARAMS[atomic_number]
            population = 0.0
            for _ in range(atom_params.n_basis):
                population += P[mu, mu]
                mu += 1
            charges[atom_index] = atom_params.n_valence - population

        results.append({
            'energy_eV': E_total,
            'energy_kcal': E_total * EV_TO_KCAL,
            'electronic_eV': E_elec,
            'nuclear_eV': E_nuc,
            'heat_of_formation_eV': E_hof,
            'heat_of_formation_kcal': E_hof * EV_TO_KCAL,
            'converged': bool(converged_arr[mol_idx]),
            'n_iter': int(n_iter_arr[mol_idx]),
            'eigenvalues': eigvals,
            'charges': charges,
            'density': P,
            'n_basis': int(nb),
            'molecular_charge': float(batch.molecular_charges[mol_idx]),
            'n_electrons': int(2 * batch.n_occ_arr[mol_idx]),
        })

    return results


def rm1_energy_batch_mlx(
    molecules: list[tuple[list[int], np.ndarray]],
    max_iter: int = 100,
    conv_tol: float = 1e-6,
    verbose: bool = False,
    method: str = 'RM1',
    molecular_charges: list[float] | None = None,
    density_solver: str = 'auto',
) -> list[dict]:
    """All-MLX batched NDDO SCF.

    Same numerical recipe as :func:`nddo_energy_batch` (NDDO Fock build,
    DIIS-extrapolated F, eigh-based density), but every per-iteration
    array stays an ``mx.array`` from start to finish:

    - Fock build:   :meth:`MetalFockContext.build_fock_mlx` (no host copy)
    - eigh:         ``mlx_addons.linalg.batched_eigh`` when installed, else
                    ``mx.linalg.eigh`` on the CPU stream.
    - DIIS solve:   :func:`mlx_addons.linalg.solve_lu` (Metal LU) on the
                    augmented Pulay system, batched per molecule
    - Density:      ``2 * C_occ @ C_occ.T`` via masked batched matmul

    Padded rows/cols (``i >= n_basis[b]``) are augmented with a large
    diagonal (``1e10``) before eigh so spurious zero-eigenvalues from
    padding cannot enter the occupied set; the corresponding eigenvectors
    are masked out in the density build.

    Parameters mirror :func:`nddo_energy_batch`. ``use_metal`` is implicit
    (always Metal Fock build — the CPU Fock path doesn't fit the all-MLX
    contract).

    Args:
        molecules: list of ``(atoms, coords)`` tuples.
        max_iter: maximum SCF iterations.
        conv_tol: per-molecule density-RMS convergence threshold.
        verbose: print convergence info every 5 iters.
        method: RM1 / AM1 / AM1_STAR / RM1_STAR.

    Returns:
        list of result dicts (same shape as :func:`nddo_energy_batch`).
    """
    from .batch import prepare_batch
    from .fock_metal import MetalFockContext
    from mlx_addons.linalg import solve_lu

    PARAMS = get_params(method)
    N = len(molecules)
    if N == 0:
        return []

    batch = prepare_batch(molecules, param_dict=PARAMS, molecular_charges=molecular_charges)
    MB = batch.max_basis

    # Initial density: MOPAC-style neutral-atom diagonal guess — same as the
    # single-molecule path. An H_core-eigendecomposition guess can start deep
    # in a charge-transfer basin and converge the SCF to a wrong (higher)
    # root; on this batched float32 path it did exactly that (e.g. aspirin
    # off by ~4.7 e in a Mulliken charge vs the float64 CPU reference, and
    # 190+ iteration counts for naphthalene/dodecane).
    P_np = np.zeros((N, MB, MB), dtype=np.float32)
    for mol_idx in range(N):
        start = 0
        for z in batch.atoms_list[mol_idx]:
            p_z = PARAMS[z]
            n_sp = min(p_z.n_basis, 4)
            for k in range(n_sp):
                P_np[mol_idx, start + k, start + k] = p_z.n_valence / n_sp
            start += p_z.n_basis
    batch.P = P_np.astype(np.float64)            # keep numpy copy for compatibility
    P = mx.array(P_np)                            # (N, MB, MB) float32

    # Static MLX-resident inputs.
    metal_ctx = MetalFockContext(batch)
    H_mx = mx.array(batch.H_core.astype(np.float32))                      # (N, MB, MB)
    n_basis_mx = mx.array(batch.n_basis_arr.astype(np.int32))             # (N,)
    n_occ_mx = mx.array(batch.n_occ_arr.astype(np.int32))                 # (N,)
    arange_MB = mx.arange(MB, dtype=mx.int32)

    # Padding mask: 1.0 on padded diagonals, 0.0 elsewhere.
    basis_pad_mask = (arange_MB[None, :] >= n_basis_mx[:, None]).astype(mx.float32)  # (N, MB)
    eye_MB = mx.eye(MB, dtype=mx.float32)
    pad_diag = basis_pad_mask[:, :, None] * eye_MB[None, :, :]            # (N, MB, MB)
    pad_diag = pad_diag * mx.array(1e10, dtype=mx.float32)

    # Occupied-orbital mask: 1.0 on first n_occ columns per molecule.
    occ_mask = (arange_MB[None, :] < n_occ_mx[:, None]).astype(mx.float32)  # (N, MB)

    # DIIS state: per-molecule history of (F, e), each (N, MB, MB).
    diis_max = 6
    F_hist: list[mx.array] = []
    e_hist: list[mx.array] = []

    converged_mask = mx.zeros((N,), dtype=mx.bool_)
    n_iter_arr = np.full(N, max_iter, dtype=np.int32)
    check_finite = os.environ.get("MLXMOLKIT_SCF_CHECK_FINITE", "").lower() in {"1", "true", "yes"}
    eigh_backend = _batched_eigh_backend(MB)

    # Active-block mask (used by the sign solver and the final energy sum).
    basis_active_mask = (arange_MB[None, :] < n_basis_mx[:, None]).astype(mx.float32)  # (N, MB)
    block_mask = basis_active_mask[:, :, None] * basis_active_mask[:, None, :]          # (N, MB, MB)

    # Resolve the density solver. 'sign' replaces the per-iteration eigensolve
    # with the Fermi-shifted matrix-sign projector (matmul-only, sync-free);
    # eigh remains for warmup, periodic mu refresh, and the final pass.
    env_solver = os.environ.get("MLXMOLKIT_DENSITY_SOLVER", "").lower()
    if env_solver in {"sign", "eigh"}:
        density_solver = env_solver
    if density_solver not in {"auto", "sign", "eigh"}:
        raise ValueError(f"unknown density_solver: {density_solver!r}")
    if density_solver == "auto":
        density_solver = "sign" if MB > _addons_batched_eigh_max_n() else "eigh"
    use_sign = density_solver == "sign"
    if verbose:
        print(f"  SCF eigensolver: {eigh_backend} (density_solver={density_solver})")

    if use_sign:
        active_diag = basis_active_mask[:, :, None] * eye_MB[None, :, :]   # diag 1 on active
        pad_unit = basis_pad_mask[:, :, None] * eye_MB[None, :, :]         # diag 1 on padded
        mu = mx.zeros((N,), dtype=mx.float32)
        n_occ_f32 = n_occ_mx.astype(mx.float32)
        sign_warmup = 4          # eigh iterations before the first sign step
        sign_refresh = 8         # eigh + mu refresh every this many iterations
        sign_sync_every = 4      # host convergence sync every this many iterations
        sign_settle_dp = 0.02    # enter sign mode only once max dP drops below this
        sign_settled = False     # frozen-mu projection is only safe after level
                                 # ordering stabilizes; until then every iteration
                                 # uses the exact aufbau eigh
        force_eigh_next = False
        converged_np = np.zeros(N, dtype=bool)
        pending: list[tuple[int, mx.array, mx.array | None, mx.array]] = []

    F = None
    eigvals_final = None
    for iteration in range(max_iter):
        # 1. Fock build (Metal, all in MLX).
        F_raw = metal_ctx.build_fock_mlx(P)                                # (N, MB, MB)
        # Symmetrize against tiny kernel asymmetries.
        F_sym = 0.5 * (F_raw + mx.transpose(F_raw, (0, 2, 1)))

        # 2. DIIS extrapolation (after warmup).
        F_for_eigh = F_sym
        if iteration >= 2:
            e = F_sym @ P - P @ F_sym                                      # (N, MB, MB)
            F_hist.append(F_sym)
            e_hist.append(e)
            if len(F_hist) > diis_max:
                F_hist.pop(0)
                e_hist.pop(0)
            if len(F_hist) >= 2:
                F_for_eigh = _pulay_diis_extrap(F_hist, e_hist, solve_lu)

        # 3. Density update. Either a batched eigh (padding diagonals
        # augmented so spurious zero eigenvalues stay above the occupied
        # set), or — in sign mode between mu refreshes — the matmul-only
        # spectral projector P = I - sign(F - mu I).
        use_eigh_iter = (
            not use_sign
            or not sign_settled
            or iteration < sign_warmup
            or (iteration - sign_warmup) % sign_refresh == 0
            or force_eigh_next
        )
        if use_sign:
            force_eigh_next = False
        tr_err = None
        if use_eigh_iter:
            F_eigh_input = F_for_eigh + pad_diag
            if check_finite:
                _assert_finite_mx(F_eigh_input, label=f"F_eigh_input at iteration {iteration}")
            eigvals, C = _batched_symmetric_eigh(F_eigh_input)              # (N, MB), (N, MB, MB)

            # 4. Density: P = 2 * (C * occ_mask[:, None, :]) @ C^T.
            C_occ = C * occ_mask[:, None, :]                               # zero out unoccupied cols
            P_new = 2.0 * (C_occ @ mx.transpose(C, (0, 2, 1)))

            if use_sign:
                # Refresh the per-molecule chemical potential mu = mid-gap.
                # Eigenvalues are ascending; padded levels sit at ~1e10, so
                # a full shell (n_occ == n_basis) is detected and handled.
                homo_idx = mx.maximum(n_occ_mx - 1, 0)
                lumo_idx = mx.minimum(n_occ_mx, MB - 1)
                homo = mx.take_along_axis(eigvals, homo_idx[:, None], axis=1)[:, 0]
                lumo = mx.take_along_axis(eigvals, lumo_idx[:, None], axis=1)[:, 0]
                mu = mx.where(lumo > 1.0e6, homo + 1.0, 0.5 * (homo + lumo))
        else:
            P_new = _sign_density(F_for_eigh, mu, block_mask, active_diag, pad_unit, eye_MB)
            # Aufbau guard: if mu drifted out of the gap the trace is off by
            # an integer — gate convergence on it and trigger an eigh refresh.
            tr_err = mx.abs(0.5 * mx.trace(P_new, axis1=-2, axis2=-1) - n_occ_f32)

        # 5. Per-mol convergence: max-abs density change.
        dP = mx.max(mx.abs(P_new - P), axis=(-2, -1))                       # (N,)
        new_conv = dP < mx.array(conv_tol, dtype=dP.dtype)                  # (N,)
        if tr_err is not None:
            new_conv = new_conv & (tr_err < 0.25)

        if not use_sign:
            # Mark first-time-converged molecules with the current iteration count.
            first_conv = new_conv & (~converged_mask)
            if mx.any(first_conv).item():
                first_conv_np = np.array(first_conv)
                for idx in np.where(first_conv_np)[0]:
                    n_iter_arr[idx] = iteration + 1
            converged_mask = converged_mask | new_conv

            # 6. Density mixing for first 2 iterations.
            if iteration < 2:
                P = 0.5 * P_new + 0.5 * P
            else:
                P = P_new

            if verbose and (iteration % 5 == 0):
                n_conv = int(mx.sum(converged_mask.astype(mx.int32)).item())
                max_dp = float(mx.max(dP).item())
                print(f"  SCF iter {iteration + 1}: {n_conv}/{N} converged, max dP = {max_dp:.2e}")

            if bool(mx.all(converged_mask).item()):
                break

            F = F_for_eigh
            eigvals_final = eigvals
        else:
            # Chunked host sync: queue lazy convergence/trace arrays and
            # evaluate every sign_sync_every iterations, so the GPU pipeline
            # is not stalled by per-iteration .item() round-trips.
            pending.append((iteration, new_conv, tr_err, mx.max(dP)))
            if iteration < 2:
                P = 0.5 * P_new + 0.5 * P
            else:
                P = P_new

            if len(pending) >= sign_sync_every or iteration == max_iter - 1:
                eval_targets: list[mx.array] = [P]
                for _, conv_lazy, tr_lazy, dp_lazy in pending:
                    eval_targets.append(conv_lazy)
                    eval_targets.append(dp_lazy)
                    if tr_lazy is not None:
                        eval_targets.append(tr_lazy)
                mx.eval(*eval_targets)
                trace_violated = False
                chunk_max_dp = 0.0
                for it_idx, conv_lazy, tr_lazy, dp_lazy in pending:
                    conv_np = np.array(conv_lazy)
                    newly = conv_np & (~converged_np)
                    n_iter_arr[newly] = it_idx + 1
                    converged_np |= conv_np
                    chunk_max_dp = max(chunk_max_dp, float(dp_lazy.item()))
                    if tr_lazy is not None and bool(np.any(np.array(tr_lazy) >= 0.25)):
                        trace_violated = True
                pending.clear()
                if trace_violated:
                    # mu left the gap for at least one molecule: drop the
                    # (possibly poisoned) DIIS history and re-bracket mu.
                    F_hist.clear()
                    e_hist.clear()
                    force_eigh_next = True
                    sign_settled = False
                elif not sign_settled and chunk_max_dp < sign_settle_dp:
                    sign_settled = True
                if verbose:
                    print(
                        f"  SCF iter {iteration + 1}: {int(converged_np.sum())}/{N} converged "
                        f"(sign{', settled' if sign_settled else ', warming'}, max dP {chunk_max_dp:.2e})"
                    )
                if bool(converged_np.all()):
                    break

    # Final Fock with converged density (already an mx.array).
    F_final_raw = metal_ctx.build_fock_mlx(P)
    F_final = 0.5 * (F_final_raw + mx.transpose(F_final_raw, (0, 2, 1)))

    # Per-molecule energy via masked sum (only the active n_basis × n_basis
    # block; basis_active_mask / block_mask were built before the SCF loop).
    E_elec_mx = 0.5 * mx.sum(block_mask * P * (H_mx + F_final), axis=(-2, -1))          # (N,)
    mx.eval(E_elec_mx, F_final, P, eigvals_final if eigvals_final is not None else mx.zeros((1,)))

    # Eigenvalues for the final F (one more eigh on the converged Fock).
    F_final_eigh_input = F_final + pad_diag
    if check_finite:
        _assert_finite_mx(F_final_eigh_input, label="final F_eigh_input")
    eigvals_padded = _batched_symmetric_eigh(F_final_eigh_input)[0]         # (N, MB)
    mx.eval(eigvals_padded)
    eigvals_np = np.array(eigvals_padded)

    P_np_out = np.array(P)
    F_np_out = np.array(F_final)
    E_elec_np = np.array(E_elec_mx)
    if not use_sign:
        converged_np = np.array(converged_mask)

    results = []
    for mol_idx in range(N):
        nb = int(batch.n_basis_arr[mol_idx])
        nocc = int(batch.n_occ_arr[mol_idx])
        atoms = batch.atoms_list[mol_idx]
        E_elec = float(E_elec_np[mol_idx])
        E_nuc = float(batch.E_nuc[mol_idx])
        E_total = E_elec + E_nuc
        E_isol = sum(PARAMS[z].eisol for z in atoms)
        eheat = sum(PARAMS[z].eheat for z in atoms)
        E_binding = E_total - E_isol
        E_hof = E_binding + eheat / EV_TO_KCAL
        # Drop the spurious huge eigenvalues from padding.
        eigvals_active = eigvals_np[mol_idx, :nb]
        charges = np.empty(len(atoms))
        mu = 0
        density = P_np_out[mol_idx, :nb, :nb]
        for atom_index, atomic_number in enumerate(atoms):
            atom_params = PARAMS[atomic_number]
            population = 0.0
            for _ in range(atom_params.n_basis):
                population += density[mu, mu]
                mu += 1
            charges[atom_index] = atom_params.n_valence - population
        results.append({
            'energy_eV': E_total,
            'energy_kcal': E_total * EV_TO_KCAL,
            'electronic_eV': E_elec,
            'nuclear_eV': E_nuc,
            'heat_of_formation_eV': E_hof,
            'heat_of_formation_kcal': E_hof * EV_TO_KCAL,
            'converged': bool(converged_np[mol_idx]),
            'n_iter': int(n_iter_arr[mol_idx]),
            'eigenvalues': eigvals_active,
            'charges': charges,
            'density': density,
            'n_basis': nb,
            'molecular_charge': float(batch.molecular_charges[mol_idx]),
            'n_electrons': int(2 * batch.n_occ_arr[mol_idx]),
            'eigh_backend': eigh_backend,
            'density_solver': density_solver,
        })

    if use_sign:
        # Belt-and-suspenders: any converged molecule whose final density
        # fails the aufbau trace / idempotency invariants is recomputed on
        # the reference eigh path. The per-iteration trace gate makes this
        # rare; this guarantees delivered results meet the eigh standard.
        bad_indices = []
        for mol_idx, result in enumerate(results):
            if not result['converged']:
                continue
            density = result['density']
            tr_err_final = abs(0.5 * float(np.trace(density)) - float(batch.n_occ_arr[mol_idx]))
            idem_final = float(np.max(np.abs(0.5 * density @ density - density)))
            if tr_err_final > 0.05 or idem_final > 1.0e-3:
                bad_indices.append(mol_idx)
        if bad_indices:
            if verbose:
                print(f"  sign-density validation: re-running {len(bad_indices)} molecule(s) via eigh")
            redo_charges = (
                None
                if molecular_charges is None
                else [molecular_charges[i] for i in bad_indices]
            )
            redo = rm1_energy_batch_mlx(
                [molecules[i] for i in bad_indices],
                max_iter=max_iter,
                conv_tol=conv_tol,
                verbose=verbose,
                method=method,
                molecular_charges=redo_charges,
                density_solver='eigh',
            )
            for original_index, redo_result in zip(bad_indices, redo):
                redo_result['density_solver'] = 'eigh(sign_fallback)'
                results[original_index] = redo_result

    return results


def _pulay_diis_extrap(F_hist, e_hist, solve_lu) -> mx.array:
    """Inline Pulay DIIS using mlx_addons solve_lu on the augmented system.

    Replaces ``mlx_addons.solvers.pulay_diis`` for the SCF inner loop only
    because here we already know the leading-batch dim is non-empty and the
    history length ``nd >= 2``.
    """
    nd = len(F_hist)
    leading = F_hist[0].shape[:-2]   # (N,)
    dtype = F_hist[0].dtype
    F_stack = mx.stack(F_hist, axis=0)             # (nd, N, MB, MB)
    e_stack = mx.stack(e_hist, axis=0)             # (nd, N, MB, MB)
    B = mx.einsum("i...nm,j...nm->...ij", e_stack, e_stack)                 # (N, nd, nd)
    minus_col = -mx.ones((*leading, nd, 1), dtype=dtype)
    minus_row = -mx.ones((*leading, 1, nd), dtype=dtype)
    zero_corner = mx.zeros((*leading, 1, 1), dtype=dtype)
    A = mx.concatenate(
        [mx.concatenate([B, minus_col], axis=-1),
         mx.concatenate([minus_row, zero_corner], axis=-1)],
        axis=-2,
    )                                              # (N, nd+1, nd+1)
    rhs = mx.concatenate(
        [mx.zeros((*leading, nd, 1), dtype=dtype),
         -mx.ones((*leading, 1, 1), dtype=dtype)],
        axis=-2,
    )                                              # (N, nd+1, 1)
    coeffs_full = solve_lu(A, rhs)                 # (N, nd+1, 1)
    coeffs = coeffs_full[..., :nd, 0]              # (N, nd)
    # F_extrap[..., n, m] = sum_i c_i F_stack[i, ..., n, m]
    return mx.einsum("...i,i...nm->...nm", coeffs, F_stack)


# --- Back-compat aliases (the function was named rm1_energy before this PR
# expanded it to all 8 NDDO methods; keep the old names so external callers
# don't break). ---
rm1_energy = nddo_energy
rm1_energy_batch = nddo_energy_batch
