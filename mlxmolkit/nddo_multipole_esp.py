"""True NDDO electrostatic potential from the SCF density matrix.

The existing ``pm6_esp_resp_charge_labels`` builds its grid ESP from the
*Mulliken monopoles* (``scf["charges"]``) and then fits RESP to that
point-charge field -- which is circular: it merely recovers Mulliken.

This module computes the **physical** electrostatic potential of the NDDO
wavefunction by condensing each atom's density block into Dewar-Thiel
atom-centred multipoles (monopole + sp-hybrid dipole + pp quadrupole) and
evaluating their classical, undamped, far-field potential on the grid. That
ESP -- fed to ``fit_resp_charges_mlx`` -- yields genuine ESP/RESP charges for
any element the SCF covers.  With ``method="PM6"`` the NDDO SCF reaches the
whole main group (Se, As, Si, B ...), i.e. exactly the atoms that AM1-BCC /
ABCG2 / Espaloma cannot parameterise.

References
----------
* B. H. Besler, K. M. Merz, P. A. Kollman, *J. Comput. Chem.* **11**, 431 (1990)
  -- ESP charges from semiempirical densities via atomic multipoles.
* M. J. S. Dewar, W. Thiel, *Theor. Chim. Acta* **46**, 89 (1977)
  -- the sp/pp multipole model and the DD/QQ charge separations (``da``/``qa``).

The atom-centred multipoles are the same quantities MOPAC prints; the molecular
dipole reassembled from them (``molecular_dipole_debye``) is the built-in
correctness check.
"""
from __future__ import annotations

import numpy as np

try:  # GPU path; pure-numpy fallback keeps the module importable everywhere.
    import mlx.core as mx

    _HAS_MLX = True
except Exception:  # pragma: no cover - environment without mlx
    _HAS_MLX = False

BOHR = 0.52917721067  # Angstrom per Bohr
AU_TO_DEBYE = 2.541746  # e*Bohr -> Debye


def atomic_multipoles_from_density(atoms, params, density):
    """Condense the NDDO AO density into per-atom monopole + dipole.

    Parameters
    ----------
    atoms : sequence[int]
        Atomic numbers (only used for length / sanity; the population comes
        from ``params`` + ``density``).
    params : sequence
        The per-atom ``ElementParams`` list returned alongside the SCF (same
        order as the AO blocks; each atom owns ``p.n_basis`` consecutive AOs in
        the order s, px, py, pz, [d...]).
    density : np.ndarray
        The converged AO density matrix ``P`` (``scf["density"]``).

    Returns
    -------
    q : np.ndarray, shape (N,)
        Net atomic charge  q_A = Z_valence(A) - sum_{mu in A} P_mu_mu   (= Mulliken).
    dip : np.ndarray, shape (N, 3)
        Atomic (electronic) dipole in atomic units (e*Bohr), molecular frame,
        from the s-p coherence:  d_A,i = -2 * da_A * P(s_A, p_i,A).
    quad : np.ndarray, shape (N, 3, 3)
        Traceless Cartesian atomic quadrupole (e*Bohr^2) from the p-p block:
        Q_A,ij = qa_A^2 * (3 P(p_i,p_j) - delta_ij * Tr_pp).  (Overall sign/scale
        for the ESP is applied at evaluation via ``quad_scale``, calibrated to QM.)
    """
    from .rm1.integrals import _charge_separations

    P = np.asarray(density, dtype=np.float64)
    N = len(params)
    q = np.zeros(N, dtype=np.float64)
    dip = np.zeros((N, 3), dtype=np.float64)
    quad = np.zeros((N, 3, 3), dtype=np.float64)
    mu = 0
    for i, p in enumerate(params):
        nb = int(p.n_basis)
        pop = float(np.trace(P[mu:mu + nb, mu:mu + nb]))
        q[i] = float(p.n_valence) - pop
        if nb >= 4:
            da, qa = _charge_separations(p)
            # <s|r|p_i> = da along axis i (Bohr); P symmetric so the s-p block
            # contributes 2 * P(s,p_i); electrons carry -1 -> electronic dipole.
            dip[i, 0] = -2.0 * da * P[mu, mu + 1]
            dip[i, 1] = -2.0 * da * P[mu, mu + 2]
            dip[i, 2] = -2.0 * da * P[mu, mu + 3]
            # p-p block -> traceless Cartesian quadrupole (qa = quadrupole sep).
            pblk = P[mu + 1:mu + 4, mu + 1:mu + 4]
            trpp = float(np.trace(pblk))
            quad[i] = qa * qa * (3.0 * pblk - np.eye(3) * trpp)
        mu += nb
    return q, dip, quad


def molecular_dipole_debye(coords_ang, q, dip):
    """Total molecular dipole (Debye) reassembled from the atom multipoles.

    This is the correctness check: it must match the QM/MOPAC molecular dipole.
    For a neutral molecule the monopole term is origin-independent.
    """
    R = np.asarray(coords_ang, dtype=np.float64) / BOHR  # Bohr
    q = np.asarray(q, dtype=np.float64)
    dip = np.asarray(dip, dtype=np.float64)
    D_au = (q[:, None] * R).sum(0) + dip.sum(0)  # e*Bohr
    return float(np.linalg.norm(D_au) * AU_TO_DEBYE)


def nddo_esp_on_grid(coords_ang, q, dip, grid_ang, quad=None, quad_scale=1.0):
    """Classical monopole+dipole(+quadrupole) ESP (Hartree/e) at the grid points.

    V(r_g) = sum_A [ q_A/|d| + d_A . d/|d|^3 + (quad_scale/2) d^T Q_A d / |d|^5 ]
    with d = r_g - R_A.  The quadrupole term is added only if ``quad`` is given;
    ``quad_scale`` carries the one convention-ambiguous sign/scale, pinned to QM
    by ``calibrate_quad_scale``.

    Coordinates in Angstrom in, atomic-unit potential out (consistent with the
    other mlxmolkit ESP routines, which work in a.u. internally).
    """
    R = np.asarray(coords_ang, dtype=np.float64) / BOHR
    G = np.asarray(grid_ang, dtype=np.float64) / BOHR
    q = np.asarray(q, dtype=np.float64)
    dip = np.asarray(dip, dtype=np.float64)
    use_quad = quad is not None and quad_scale != 0.0

    if _HAS_MLX:
        Rm = mx.array(R)
        Gm = mx.array(G)
        qm = mx.array(q)
        dm = mx.array(dip)
        d = Gm[:, None, :] - Rm[None, :, :]            # [Ng, N, 3]
        r2 = (d * d).sum(-1)
        inv = mx.rsqrt(r2)                              # 1/r
        inv3 = inv * inv * inv                          # 1/r^3
        mono = (qm[None, :] * inv).sum(-1)
        dipv = ((dm[None, :, :] * d).sum(-1) * inv3).sum(-1)
        out = mono + dipv
        if use_quad:
            Qm = mx.array(np.asarray(quad, dtype=np.float64))            # [N,3,3]
            dQd = (d[:, :, :, None] * Qm[None] * d[:, :, None, :]).sum((-1, -2))
            out = out + 0.5 * quad_scale * (dQd * inv3 * inv * inv)      # /r^5
        mx.eval(out)
        return np.asarray(out)

    d = G[:, None, :] - R[None, :, :]
    r2 = (d * d).sum(-1)
    inv = 1.0 / np.sqrt(r2)
    mono = (q[None, :] * inv).sum(-1)
    dipv = ((dip[None, :, :] * d).sum(-1) * inv ** 3).sum(-1)
    out = mono + dipv
    if use_quad:
        Q = np.asarray(quad, dtype=np.float64)
        dQd = np.einsum("gni,nij,gnj->gn", d, Q, d)
        out = out + 0.5 * quad_scale * dQd * inv ** 5
    return out


def rwresp_restraint_factor(atoms, coords_ang, *, shell_factors=(1.4, 1.6, 1.8, 2.0),
                            point_density=1.0, ref_point_density=1.0):
    """Restraint reweighting factor f_rwt for reweighted RESP.

    Tripathy, Palos, Merz, Paesani & Goetz, *J. Chem. Inf. Model.* **66**, 3173
    (2026), eqs 8-9.  Standard RESP holds the restraint strength ``a`` fixed
    while the data term of the normal matrix, ``A_ii = sum_k r_ik^-2``, grows
    with the number of grid points.  On a dense grid the restraint is swamped
    and the RESP charges drift toward the (orientation-robust but unrestrained)
    ESP charges.  rwRESP multiplies the restraint contribution to ``A_ii`` by

        f_rwt = <N_grid>_{S_grid} / <N_grid>_{ref}                       (eq 9)

    so the restraint keeps a constant *relative* weight at any grid density: the
    charges stay close to standard RESP at the reference spacing yet inherit the
    dense grid's orientation-independence.  Rule of thumb f_rwt ~ 1/S_grid^2;
    the paper tabulates 1.0, 1.82, 4.17, 17.0, 431.08 for S_grid = 1.0, 0.75,
    0.5, 0.25, 0.05 A (Table 1).

    Computed here exactly, per molecule, as the ratio of the actual (overlap-
    pruned) Connolly grid-point counts, so it adapts to molecular shape.
    ``ref_point_density`` is the density at which the restraint was calibrated --
    1.0 for the AmberTools default ``restraint_a = 5e-4``.  Then pass
    ``fit_resp_charges_mlx(..., restraint_a=5e-4 * f_rwt)``.
    """
    from .esp_resp import connolly_surface_grid
    A = np.asarray(atoms, dtype=np.int64)
    C = np.asarray(coords_ang, dtype=np.float64)
    n_cur = len(connolly_surface_grid(A, C, shell_factors=shell_factors, point_density=point_density))
    n_ref = len(connolly_surface_grid(A, C, shell_factors=shell_factors, point_density=ref_point_density))
    return n_cur / max(n_ref, 1)


def fit_rwresp_charges(atom_coords, grid_coords, esp_values, *, atoms,
                       total_charge=0, f_rwt=1.0, restraint_a=5.0e-4,
                       restraint_b=0.1, equivalence_groups=None, **kw):
    """Reweighted-RESP fit: standard RESP with the restraint scaled by ``f_rwt``.

    Thin wrapper over ``fit_resp_charges_mlx`` that applies the rwRESP restraint
    reweighting (``restraint_a -> restraint_a * f_rwt``).  Get ``f_rwt`` from
    ``rwresp_restraint_factor``.  ``f_rwt = 1`` reduces exactly to standard RESP.
    """
    from .esp_resp import fit_resp_charges_mlx
    return fit_resp_charges_mlx(
        atom_coords, grid_coords, esp_values,
        total_charge=total_charge, atoms=np.asarray(atoms),
        equivalence_groups=equivalence_groups,
        restraint_a=restraint_a * float(f_rwt), restraint_b=restraint_b, **kw)


def nddo_density_esp(atoms, coords_ang, *, method="PM6", total_charge=0,
                     shell_factors=(1.4, 1.6, 1.8, 2.0), point_density=1.0,
                     quad_scale=0.0, return_multipoles=False):
    """Run the NDDO SCF and return (grid_coords, true_esp[, q, dip, quad]).

    Convenience wrapper: SCF -> density -> atomic multipoles -> Connolly grid ->
    true ESP.  The returned (grid, esp) drop straight into
    ``fit_resp_charges_mlx`` / ``fit_esp_resp_charges_mlx``.  Pass
    ``quad_scale`` (from ``calibrate_quad_scale``) to include the quadrupole
    term; default 0.0 keeps the well-validated monopole+dipole ESP.
    """
    from .rm1 import nddo_energy
    from .rm1.methods import get_params
    from .esp_resp import connolly_surface_grid

    atom_list = [int(z) for z in atoms]
    coords = np.asarray(coords_ang, dtype=np.float64)

    scf = nddo_energy(atom_list, coords, method=method, molecular_charge=total_charge)
    if not scf.get("converged", False):
        raise RuntimeError(f"{method} SCF did not converge")

    pdict = get_params(method)
    params = [pdict[z] for z in atom_list]
    q, dip, quad = atomic_multipoles_from_density(atom_list, params, scf["density"])

    grid = connolly_surface_grid(
        np.asarray(atom_list, dtype=np.int64), coords,
        shell_factors=shell_factors, point_density=point_density,
    )
    esp = nddo_esp_on_grid(coords, q, dip, grid, quad=quad, quad_scale=quad_scale)
    if return_multipoles:
        return grid, esp, q, dip, quad
    return grid, esp
