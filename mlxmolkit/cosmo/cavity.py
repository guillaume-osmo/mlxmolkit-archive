"""
COSMO (Conductor-like Screening Model) cavity and surface charge solver.

Pipeline:
1. Build molecular cavity from scaled VdW radii
2. Tesselate using Lebedev quadrature
3. Remove buried surface points
4. Compute electrostatic potential from RM1 Mulliken charges
5. Solve COSMO linear system: q = -f(ε) · A⁻¹ · Φ
6. Output: segment positions, areas, charges, sigma = q/A

Reference: Klamt, A. J. Phys. Chem. 1995, 99, 2224-2235.
"""
from __future__ import annotations

import numpy as np
from .params import VDW_RADII, CAVITY_SCALING, EPSILON_WATER, BOHR_TO_ANG
from .lebedev import get_lebedev_grid


def _mulliken_charges(atoms: list[int], density: np.ndarray, n_basis_per_atom: list[int]) -> np.ndarray:
    """Extract Mulliken partial charges from NDDO density matrix.

    In NDDO (S=I), Mulliken charges = Z_valence - sum(diagonal(P)) per atom.

    Uses the **provided** ``n_basis_per_atom`` to slice the density —
    previously this argument was ignored in favour of ``RM1_PARAMS``
    (sp-only), which silently mis-indexed PM6_D density matrices
    (S with 9 orbitals was sliced as 4, then H1's slot picked up S's
    d_xy density). For PM6_D, callers should pass
    ``[get_params('PM6_D')[z].n_basis for z in atoms]``.

    ``n_valence`` is element-dependent (not method-dependent) so we
    still look it up from RM1_PARAMS as the authoritative source.
    """
    from ..rm1.params import RM1_PARAMS
    n_atoms = len(atoms)
    if len(n_basis_per_atom) != n_atoms:
        raise ValueError(
            f"n_basis_per_atom length {len(n_basis_per_atom)} != n_atoms {n_atoms}"
        )
    charges = np.zeros(n_atoms)
    idx = 0
    for i, z in enumerate(atoms):
        nb = n_basis_per_atom[i]
        q = 0.0
        for k in range(nb):
            q += density[idx + k, idx + k]
        charges[i] = RM1_PARAMS[z].n_valence - q
        idx += nb
    return charges


def build_cavity(
    atoms: list[int],
    coords: np.ndarray,
    n_points_per_atom: int = 194,
    scaling: float = CAVITY_SCALING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build COSMO molecular cavity surface.

    Args:
        atoms: atomic numbers
        coords: (n_atoms, 3) in Angstrom
        n_points_per_atom: Lebedev grid size per atom
        scaling: VdW radius scaling factor

    Returns:
        seg_pos: (n_seg, 3) segment positions in Angstrom
        seg_area: (n_seg,) segment areas in Angstrom²
        seg_normal: (n_seg, 3) outward normal vectors
        seg_atom: (n_seg,) atom index for each segment
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)

    # Get Lebedev grid on unit sphere
    sphere_pts, sphere_weights = get_lebedev_grid(n_points_per_atom)
    n_pts = len(sphere_pts)

    # Build all surface points
    all_pos = []
    all_area = []
    all_normal = []
    all_atom_idx = []

    for i in range(n_atoms):
        z = atoms[i]
        if z not in VDW_RADII:
            raise ValueError(f"No VdW radius for Z={z}")
        r = VDW_RADII[z] * scaling

        # Place Lebedev points on scaled VdW sphere
        pts = coords[i] + r * sphere_pts  # (n_pts, 3)
        normals = sphere_pts.copy()  # outward normals = unit vectors from center

        # Areas: weight * r² (sphere surface area element)
        areas = sphere_weights * r * r  # weight already has 4π factor

        # Check burial: remove points inside OTHER atoms' VdW spheres
        mask = np.ones(n_pts, dtype=bool)
        for j in range(n_atoms):
            if j == i:
                continue
            rj = VDW_RADII[atoms[j]] * scaling
            dists = np.linalg.norm(pts - coords[j], axis=1)
            mask &= (dists > rj * 0.99)  # 0.99 tolerance for numerical stability

        # Keep only exposed points
        all_pos.append(pts[mask])
        all_area.append(areas[mask])
        all_normal.append(normals[mask])
        all_atom_idx.append(np.full(np.sum(mask), i, dtype=np.int32))

    seg_pos = np.vstack(all_pos)
    seg_area = np.concatenate(all_area)
    seg_normal = np.vstack(all_normal)
    seg_atom = np.concatenate(all_atom_idx)

    return seg_pos, seg_area, seg_normal, seg_atom


def compute_cosmo_charges(
    atoms: list[int],
    coords: np.ndarray,
    mulliken_charges: np.ndarray,
    seg_pos: np.ndarray,
    seg_area: np.ndarray,
    epsilon: float = EPSILON_WATER,
) -> np.ndarray:
    """Solve COSMO equation for surface screening charges.

    The COSMO equation: A · q = -Φ
    where:
      A[i,j] = 1/|r_i - r_j|  for i ≠ j
      A[i,i] = 1.07 · √(4π/a_i)  (self-interaction, Klamt convention)
      Φ[i] = electrostatic potential at segment i from molecular charges

    The surface charges screen the molecular potential on a conductor surface.
    For real solvents: q_eff = f(ε) · q where f(ε) = (ε-1)/(ε+0.5)

    Args:
        atoms: atomic numbers
        coords: (n_atoms, 3) in Angstrom
        mulliken_charges: (n_atoms,) partial charges in elementary charge units
        seg_pos: (n_seg, 3) segment positions in Angstrom
        seg_area: (n_seg,) segment areas in Angstrom²
        epsilon: dielectric constant of solvent

    Returns:
        seg_charge: (n_seg,) surface charges in elementary charge units
    """
    n_seg = len(seg_pos)
    n_atoms = len(atoms)

    # Convert to Bohr for Coulomb integrals
    seg_pos_bohr = seg_pos / BOHR_TO_ANG
    seg_area_bohr = seg_area / (BOHR_TO_ANG ** 2)
    coords_bohr = coords / BOHR_TO_ANG

    # A matrix: vectorized pairwise 1/|r_i - r_j|
    diff = seg_pos_bohr[:, np.newaxis, :] - seg_pos_bohr[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    np.fill_diagonal(dist, 1.0)  # avoid divide by zero
    A = 1.0 / dist
    # Diagonal: Klamt self-interaction
    np.fill_diagonal(A, 1.07 * np.sqrt(4.0 * np.pi / seg_area_bohr))

    # Electrostatic potential: vectorized Φ(r_i) = Σ_A q_A / |r_i - R_A|
    diff_ac = seg_pos_bohr[:, np.newaxis, :] - coords_bohr[np.newaxis, :, :]  # (n_seg, n_atoms, 3)
    dist_ac = np.sqrt(np.sum(diff_ac * diff_ac, axis=2))  # (n_seg, n_atoms)
    dist_ac = np.maximum(dist_ac, 1e-10)
    Phi = (mulliken_charges[np.newaxis, :] / dist_ac).sum(axis=1)  # (n_seg,)

    # Solve: A · q = -Φ
    q = np.linalg.solve(A, -Phi)

    # Dielectric scaling: q_eff = f(ε) · q
    f_eps = (epsilon - 1.0) / (epsilon + 0.5)
    return f_eps * q


def cosmo_surface(
    atoms: list[int],
    coords: np.ndarray,
    density: np.ndarray,
    n_points: int = 194,
    epsilon: float = EPSILON_WATER,
) -> dict:
    """Complete COSMO surface calculation from RM1 results.

    Args:
        atoms: atomic numbers
        coords: (n_atoms, 3) in Angstrom
        density: (n_basis, n_basis) RM1 density matrix
        n_points: Lebedev points per atom
        epsilon: dielectric constant

    Returns:
        dict with seg_pos, seg_area, seg_charge, seg_sigma, etc.
    """
    coords = np.asarray(coords, dtype=np.float64)

    seg_pos, seg_area, seg_normal, seg_atom = build_cavity(
        atoms, coords, n_points_per_atom=n_points
    )

    n_basis_per = []
    from ..rm1.params import RM1_PARAMS
    for z in atoms:
        n_basis_per.append(RM1_PARAMS[z].n_basis)

    mulliken = _mulliken_charges(atoms, density, n_basis_per)

    seg_charge = compute_cosmo_charges(
        atoms, coords, mulliken, seg_pos, seg_area, epsilon=epsilon
    )

    seg_sigma = seg_charge / seg_area
    cavity_area = np.sum(seg_area)
    r_dot_n = np.sum((seg_pos - np.mean(coords, axis=0)) * seg_normal, axis=1)
    cavity_volume = np.abs(np.sum(r_dot_n * seg_area) / 3.0)

    return {
        'seg_pos': seg_pos,
        'seg_area': seg_area,
        'seg_charge': seg_charge,
        'seg_sigma': seg_sigma,
        'seg_normal': seg_normal,
        'seg_atom': seg_atom,
        'mulliken_charges': mulliken,
        'cavity_area': cavity_area,
        'cavity_volume': cavity_volume,
        'n_seg': len(seg_pos),
    }


def cosmo_surface_batch(
    molecules: list[tuple[list[int], np.ndarray, np.ndarray]],
    n_points: int = 194,
    epsilon: float = EPSILON_WATER,
    use_metal: bool = True,
) -> list[dict]:
    """Batch COSMO surface for N molecules.

    Metal GPU path: builds A matrices on GPU, solves on CPU (numpy batch).
    CPU path: sequential vectorized numpy.

    Args:
        molecules: list of (atoms, coords, density) tuples
        n_points: Lebedev points per atom
        epsilon: dielectric constant
        use_metal: try Metal GPU for matrix assembly

    Returns:
        list of COSMO result dicts
    """
    from ..rm1.params import RM1_PARAMS

    N = len(molecules)
    f_eps = (epsilon - 1.0) / (epsilon + 0.5)

    # Phase 1: Build all cavities
    cavities = []
    mulliken_list = []
    for atoms, coords, density in molecules:
        coords = np.asarray(coords, dtype=np.float64)
        seg_pos, seg_area, seg_normal, seg_atom = build_cavity(
            atoms, coords, n_points_per_atom=n_points
        )
        n_basis_per = [RM1_PARAMS[z].n_basis for z in atoms]
        mulliken = _mulliken_charges(atoms, density, n_basis_per)
        cavities.append((seg_pos, seg_area, seg_normal, seg_atom))
        mulliken_list.append(mulliken)

    # Phase 2: Solve COSMO — Metal GPU matrix build + numpy solve
    seg_charges = [None] * N

    if use_metal and N > 1:
        try:
            from .cosmo_metal import cosmo_solve_metal

            seg_pos_bohr_list = [c[0] / BOHR_TO_ANG for c in cavities]
            seg_area_bohr_list = [c[1] / (BOHR_TO_ANG ** 2) for c in cavities]
            coords_bohr_list = [np.asarray(molecules[k][1]) / BOHR_TO_ANG for k in range(N)]

            seg_charges = cosmo_solve_metal(
                seg_pos_bohr_list, seg_area_bohr_list,
                coords_bohr_list, mulliken_list, epsilon=epsilon,
            )
            use_metal = True  # success
        except Exception:
            use_metal = False

    if not use_metal or N == 1:
        # Sequential CPU fallback
        seg_charges = []
        for k in range(N):
            atoms, coords, _ = molecules[k]
            coords = np.asarray(coords, dtype=np.float64)
            seg_pos, seg_area, _, _ = cavities[k]
            seg_charges.append(compute_cosmo_charges(
                atoms, coords, mulliken_list[k], seg_pos, seg_area, epsilon=epsilon
            ))

    # Phase 3: Assemble results
    results = []
    for k in range(N):
        atoms, coords, _ = molecules[k]
        coords = np.asarray(coords, dtype=np.float64)
        seg_pos, seg_area, seg_normal, seg_atom = cavities[k]

        seg_sigma = seg_charges[k] / seg_area
        cavity_area = np.sum(seg_area)
        r_dot_n = np.sum((seg_pos - np.mean(coords, axis=0)) * seg_normal, axis=1)
        cavity_volume = np.abs(np.sum(r_dot_n * seg_area) / 3.0)

        results.append({
            'seg_pos': seg_pos, 'seg_area': seg_area,
            'seg_charge': seg_charges[k], 'seg_sigma': seg_sigma,
            'seg_normal': seg_normal, 'seg_atom': seg_atom,
            'mulliken_charges': mulliken_list[k],
            'cavity_area': cavity_area, 'cavity_volume': cavity_volume,
            'n_seg': len(seg_pos),
        })

    return results
