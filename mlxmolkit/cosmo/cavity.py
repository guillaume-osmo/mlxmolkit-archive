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
    from ..nddo.params import RM1_PARAMS
    n_atoms = len(atoms)
    if len(n_basis_per_atom) != n_atoms:
        raise ValueError(
            f"n_basis_per_atom length {len(n_basis_per_atom)} != n_atoms {n_atoms}"
        )
    # Catch a basis-size mismatch loudly. Passing sp-only sizes for a density
    # built with d orbitals used to slide every atom past the first d-block
    # element onto the wrong diagonal, silently returning nonsense.
    n_basis_total = sum(n_basis_per_atom)
    if n_basis_total != density.shape[0]:
        raise ValueError(
            f"n_basis_per_atom sums to {n_basis_total} but density is "
            f"{density.shape[0]}x{density.shape[0]} — wrong method's basis sizes?"
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

    # Unit-sphere tesselation, reused for every atom
    sphere_pts, sphere_weights = get_lebedev_grid(n_points_per_atom)
    n_pts = len(sphere_pts)

    # Accumulate the exposed points of each atomic sphere
    all_pos = []
    all_area = []
    all_normal = []
    all_atom_idx = []

    for i in range(n_atoms):
        z = atoms[i]
        if z not in VDW_RADII:
            raise ValueError(f"No VdW radius for Z={z}")
        r = VDW_RADII[z] * scaling

        # Scale and translate the unit sphere onto this atom
        pts = coords[i] + r * sphere_pts
        normals = sphere_pts.copy()

        # Lebedev weights sum to 4π, so w·r² gives the segment area
        areas = sphere_weights * r * r

        # Drop points buried inside a neighbouring sphere
        mask = np.ones(n_pts, dtype=bool)
        for j in range(n_atoms):
            if j == i:
                continue
            rj = VDW_RADII[atoms[j]] * scaling
            dists = np.linalg.norm(pts - coords[j], axis=1)
            mask &= dists > rj * 0.99

        # Keep only the solvent-exposed segments
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

    # Atomic units throughout the electrostatics
    seg_pos_bohr = seg_pos / BOHR_TO_ANG
    seg_area_bohr = seg_area / BOHR_TO_ANG ** 2
    coords_bohr = coords / BOHR_TO_ANG

    # Segment-segment Coulomb matrix
    diff = seg_pos_bohr[:, np.newaxis, :] - seg_pos_bohr[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    A = 1.0 / dist

    np.fill_diagonal(A, 1.07 * np.sqrt(4.0 * np.pi / seg_area_bohr))

    # Potential at each segment from the atomic point charges
    diff_ac = seg_pos_bohr[:, np.newaxis, :] - coords_bohr[np.newaxis, :, :]
    dist_ac = np.sqrt(np.sum(diff_ac * diff_ac, axis=2))
    dist_ac = np.maximum(dist_ac, 1e-10)
    Phi = (mulliken_charges[np.newaxis, :] / dist_ac).sum(axis=1)

    # Conductor limit
    q = np.linalg.solve(A, -Phi)

    # Scale from the conductor to a finite dielectric
    f_eps = (epsilon - 1.0) / (epsilon + 0.5)
    return f_eps * q


def cosmo_surface(
    atoms: list[int],
    coords: np.ndarray,
    density: np.ndarray,
    n_points: int = 194,
    epsilon: float = EPSILON_WATER,
    method: str = 'RM1',
) -> dict:
    """Complete COSMO surface calculation from RM1 results.

    Args:
        atoms: atomic numbers
        coords: (n_atoms, 3) in Angstrom
        density: (n_basis, n_basis) RM1 density matrix
        n_points: Lebedev points per atom
        epsilon: dielectric constant
        method: NDDO method the density came from. Must match, because the
            basis size per element is method-dependent — PM6_D gives P, S,
            Cl, Br and I nine orbitals where the sp-only methods give four.

    Returns:
        dict with seg_pos, seg_area, seg_charge, seg_sigma, etc.
    """
    coords = np.asarray(coords, dtype=np.float64)

    seg_pos, seg_area, seg_normal, seg_atom = build_cavity(
        atoms, coords, n_points_per_atom=n_points
    )

    from ..nddo.methods import get_params
    method_params = get_params(method)
    n_basis_per = [method_params[z].n_basis for z in atoms]

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
    method: str = 'RM1',
) -> list[dict]:
    """Batch COSMO surface for N molecules.

    Metal GPU path: builds A matrices on GPU, solves on CPU (numpy batch).
    CPU path: sequential vectorized numpy.

    Args:
        molecules: list of (atoms, coords, density) tuples
        n_points: Lebedev points per atom
        epsilon: dielectric constant
        use_metal: try Metal GPU for matrix assembly
        method: NDDO method the densities came from (see cosmo_surface)

    Returns:
        list of COSMO result dicts
    """
    from ..nddo.methods import get_params
    method_params = get_params(method)

    N = len(molecules)
    f_eps = (epsilon - 1.0) / (epsilon + 0.5)

    # Cavities and charges are cheap; build them all up front
    cavities = []
    mulliken_list = []
    for atoms, coords, density in molecules:
        coords = np.asarray(coords, dtype=np.float64)
        seg_pos, seg_area, seg_normal, seg_atom = build_cavity(
            atoms, coords, n_points_per_atom=n_points
        )
        n_basis_per = [method_params[z].n_basis for z in atoms]
        mulliken = _mulliken_charges(atoms, density, n_basis_per)
        cavities.append((seg_pos, seg_area, seg_normal, seg_atom))
        mulliken_list.append(mulliken)

    # Solve the COSMO systems, on GPU if available
    seg_charges = [None] * N

    if use_metal and N > 1:
        try:
            from .cosmo_metal import cosmo_solve_metal

            seg_pos_bohr_list = [c[0] / BOHR_TO_ANG for c in cavities]
            seg_area_bohr_list = [c[1] / BOHR_TO_ANG ** 2 for c in cavities]
            coords_bohr_list = [np.asarray(molecules[k][1]) / BOHR_TO_ANG for k in range(N)]

            seg_charges = cosmo_solve_metal(
                seg_pos_bohr_list, seg_area_bohr_list,
                coords_bohr_list, mulliken_list, epsilon=epsilon
            )
            use_metal = True
        except Exception:
            use_metal = False

    if not use_metal or N == 1:
        # CPU fallback: one dense solve per molecule
        seg_charges = []
        for k in range(N):
            atoms, coords, _ = molecules[k]
            coords = np.asarray(coords, dtype=np.float64)
            seg_pos, seg_area, _, _ = cavities[k]
            seg_charges.append(compute_cosmo_charges(
                atoms, coords, mulliken_list[k], seg_pos, seg_area, epsilon=epsilon
            ))

    # Assemble per-molecule result dicts
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
