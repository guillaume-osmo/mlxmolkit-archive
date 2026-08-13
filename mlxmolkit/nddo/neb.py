"""Nudged elastic band and barrier estimation on top of the NDDO SCF.

Why this exists: ranking tautomers by their *energy* answers the wrong question. Two
forms are one substance when the barrier between them is low, not when their energies
are close. alpha- and beta-ionone sit 3 kcal/mol apart yet are separately isolable and
separately sold, because interconverting them means moving an H from one ring carbon to
another -- kinetically locked. keto <-> enol moves the same H onto oxygen and is fast.
Only a barrier tells those two situations apart, so this module computes barriers.

NDDO is parameterized against gas-phase heats of formation, so every barrier here is a
vacuum barrier with no solvation model to switch off. That is the right ensemble for a
molecule acting in air after evaporation; it is the wrong one for aqueous chemistry,
where a solvent proton relay lowers H-transfer barriers substantially.

Method: climbing-image NEB with the Henkelman-Jonsson improved tangent, optimized with
FIRE. Endpoints are held fixed. Barriers come back in kcal/mol.

    from mlxmolkit.nddo import neb
    res = neb.tautomer_barrier(
        "CC(=O)C=CC1C(C)=CCCC1(C)C",   # alpha-ionone
        "CC(=O)C=CC1=C(C)CCCC1(C)C",   # beta-ionone
        method="PM6",
    )
    res["barrier_forward_kcal"]   # -> climb out of the first well

Caveat on gradients: ``analytical=True`` uses the frozen-density gradient, which is ~6x
faster but approximate. It is fine for screening barriers into "labile" vs "locked"
buckets; pass ``analytical=False`` for a numerical gradient when a barrier height is
being quoted rather than thresholded.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .gradient import nddo_gradient

EV_TO_KCAL = 23.060541945329334


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigid-body aligns ``mobile`` onto ``target`` (both (n, 3)).

    NEB measures distances between images, so any leftover translation or rotation
    between the endpoints would be charged to the band as spurious path length.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mc = mobile - mobile.mean(axis=0)
    tc = target - target.mean(axis=0)
    v, _, wt = np.linalg.svd(mc.T @ tc)
    # reflection guard: keep a proper rotation
    d = np.sign(np.linalg.det(v @ wt))
    correction = np.diag([1.0, 1.0, d])
    rotation = v @ correction @ wt
    return mc @ rotation + target.mean(axis=0)


def interpolate_path(start: np.ndarray, end: np.ndarray, n_images: int) -> list[np.ndarray]:
    """Linear interpolation between aligned endpoints, inclusive of both.

    Linear interpolation is only valid when the two endpoints share a conformer, so that
    the path is a single atom moving a short way. Two geometries embedded *independently*
    from SMILES differ in ring pucker and torsions as well, and interpolating between them
    linearly drives atoms through each other -- which shows up as a barrier of 10^6 kcal/mol
    or more. Use ``min_interatomic_distance`` to catch that, and build the second endpoint
    from the first (moving only the transferred H) rather than embedding it separately.
    """
    if n_images < 3:
        raise ValueError("n_images must be at least 3 (two endpoints plus one interior image)")
    end_aligned = kabsch_align(end, start)
    fractions = np.linspace(0.0, 1.0, n_images)
    return [(1.0 - f) * start + f * end_aligned for f in fractions]


def min_interatomic_distance(coords: np.ndarray) -> float:
    """Smallest distance between any two atoms, in Angstrom."""
    coords = np.asarray(coords, dtype=np.float64)
    delta = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def check_band_geometry(images, min_distance: float = 0.7) -> Optional[str]:
    """Returns a description of the first physically impossible image, else None.

    0.7 A is below any real bond length, so anything under it is atoms overlapping rather
    than a chemically meaningful structure.
    """
    for i, image in enumerate(images):
        d = min_interatomic_distance(image)
        if d < min_distance:
            return (f"image {i} of {len(images)} has two atoms {d:.2f} A apart "
                    f"(< {min_distance} A): the interpolated path passes atoms through each "
                    f"other, so any barrier from it is meaningless. Build the second endpoint "
                    f"from the first instead of embedding it independently.")
    return None


# --------------------------------------------------------------------------------------
# tautomer atom mapping
# --------------------------------------------------------------------------------------
def _heavy_skeleton(mol):
    """Builds a heavy-atom-only single-bonded graph, plus the map back to real indices.

    The H atoms must be excluded: their per-atom counts are precisely what differs
    between two tautomers, so a graph that includes them is never isomorphic.
    """
    from rdkit import Chem

    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    back = {orig: new for new, orig in enumerate(heavy)}
    sk = Chem.RWMol()
    for orig in heavy:
        sk.AddAtom(Chem.Atom(mol.GetAtomWithIdx(orig).GetAtomicNum()))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in back and j in back:
            sk.AddBond(back[i], back[j], Chem.BondType.SINGLE)
    out = sk.GetMol()
    Chem.GetSymmSSSR(out)
    out.UpdatePropertyCache(strict=False)
    return out, heavy


def map_tautomer_atoms(mol_a, mol_b, max_matches: int = 64) -> Optional[list[int]]:
    """Maps atoms of ``mol_b`` onto ``mol_a``, for two tautomers with explicit Hs.

    NEB needs atom-for-atom correspondence, but RDKit orders the atoms of two tautomer
    SMILES independently. Tautomers share an identical *heavy-atom graph* -- only bond
    orders and H counts differ -- so heavy atoms are matched on a single-bonded,
    H-stripped skeleton. Hydrogens are then matched by the heavy atom they hang off; the
    donor's surplus H is paired with the acceptor's.

    A symmetric skeleton (gem-dimethyls, for instance) admits several valid matches. All
    of them are chemically equivalent, but they differ in how far atoms have to travel,
    so the one giving the lowest aligned RMSD is chosen to keep the band short.

    Returns a list ``perm`` with ``perm[i]`` giving the index in ``mol_b`` that
    corresponds to atom ``i`` of ``mol_a``, or None if the skeletons do not match.
    """
    sk_a, heavy_a = _heavy_skeleton(mol_a)
    sk_b, heavy_b = _heavy_skeleton(mol_b)
    if len(heavy_a) != len(heavy_b):
        return None

    matches = sk_b.GetSubstructMatches(sk_a, useChirality=False, uniquify=False,
                                       maxMatches=max_matches)
    if not matches:
        return None

    pos_a = mol_a.GetConformer().GetPositions() if mol_a.GetNumConformers() else None
    pos_b = mol_b.GetConformer().GetPositions() if mol_b.GetNumConformers() else None

    best, best_rmsd = None, float("inf")
    for match in matches:
        candidate = _complete_h_mapping(mol_a, mol_b, heavy_a, heavy_b, match)
        if candidate is None:
            continue
        if pos_a is None or pos_b is None:
            return candidate
        coords_b = np.array([pos_b[candidate[i]] for i in range(len(candidate))])
        aligned = kabsch_align(coords_b, pos_a)
        rmsd = float(np.sqrt(np.mean(np.sum((aligned - pos_a) ** 2, axis=1))))
        if rmsd < best_rmsd:
            best, best_rmsd = candidate, rmsd
    return best


def _complete_h_mapping(mol_a, mol_b, heavy_a, heavy_b, match) -> Optional[list[int]]:
    """Extends a heavy-atom match to the hydrogens."""
    perm: list[Optional[int]] = [None] * mol_a.GetNumAtoms()
    for new_i, orig_i in enumerate(heavy_a):
        perm[orig_i] = heavy_b[match[new_i]]

    # Hydrogens: pair them up per heavy atom, then pair the leftovers with each other.
    def attached_hs(mol, idx):
        return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]

    used_b = {j for j in perm if j is not None}
    leftover_a: list[int] = []
    leftover_b: list[int] = []
    for i in heavy_a:
        j = perm[i]
        hs_a = [h for h in attached_hs(mol_a, i) if perm[h] is None]
        hs_b = [h for h in attached_hs(mol_b, j) if h not in used_b]
        for h_a, h_b in zip(hs_a, hs_b):
            perm[h_a] = h_b
            used_b.add(h_b)
        leftover_a.extend(hs_a[len(hs_b):])
        leftover_b.extend(hs_b[len(hs_a):])

    # The mobile proton: surplus H on the donor in A, surplus site in B.
    leftover_b.extend(
        h for h in range(mol_b.GetNumAtoms())
        if mol_b.GetAtomWithIdx(h).GetAtomicNum() == 1 and h not in used_b
    )
    for h_a, h_b in zip(leftover_a, dict.fromkeys(leftover_b)):
        perm[h_a] = h_b

    if any(p is None for p in perm):
        return None
    return [int(p) for p in perm]


# --------------------------------------------------------------------------------------
# NEB
# --------------------------------------------------------------------------------------
def _improved_tangent(images, energies, i):
    """Henkelman-Jonsson upwind tangent; avoids the kinks a central difference gives."""
    tau_plus = images[i + 1] - images[i]
    tau_minus = images[i] - images[i - 1]
    v_prev, v_here, v_next = energies[i - 1], energies[i], energies[i + 1]

    if v_next > v_here > v_prev:
        tau = tau_plus
    elif v_next < v_here < v_prev:
        tau = tau_minus
    else:
        d_max = max(abs(v_next - v_here), abs(v_prev - v_here))
        d_min = min(abs(v_next - v_here), abs(v_prev - v_here))
        if v_next > v_prev:
            tau = tau_plus * d_max + tau_minus * d_min
        else:
            tau = tau_plus * d_min + tau_minus * d_max

    norm = np.linalg.norm(tau)
    return tau / norm if norm > 1e-12 else tau


def neb_run(
    atoms: list[int],
    images: list[np.ndarray],
    method: str = "RM1",
    k_spring: float = 5.0,
    max_iter: int = 200,
    force_tol: float = 0.05,
    climbing: bool = True,
    climb_after: int = 10,
    analytical: bool = True,
    dt: float = 0.15,
    verbose: bool = False,
) -> dict:
    """Relaxes a band of images; endpoints stay fixed.

    Args:
        atoms: atomic numbers, shared by every image.
        images: initial band, ``images[0]`` and ``images[-1]`` are the endpoints.
        k_spring: spring constant in eV/A^2 along the band.
        force_tol: convergence on the max NEB force component, eV/A.
        climbing: drive the top image to the saddle instead of letting springs hold it.
        climb_after: iterations to relax before switching the top image to climbing, so
            the band has a sane shape before one image starts chasing the saddle.
        dt: FIRE timestep. Lower it if the band oscillates.

    Returns a dict with the relaxed band, per-image energies (eV and kcal/mol), the
    barrier in each direction, and the index of the highest image.
    """
    images = [np.asarray(im, dtype=np.float64).copy() for im in images]
    n_img = len(images)
    if n_img < 3:
        raise ValueError("need at least 3 images")

    velocity = [np.zeros_like(im) for im in images]
    alpha, alpha_start, f_inc, f_dec, alpha_dec, n_min = 0.1, 0.1, 1.1, 0.5, 0.99, 5
    steps_since_negative = 0
    dt_now = dt

    energies = np.zeros(n_img)
    gradients: list[Optional[np.ndarray]] = [None] * n_img
    converged = False
    max_force = float("inf")
    iteration = 0

    for iteration in range(max_iter):
        for i in range(n_img):
            # endpoints never move, so their energy/gradient is computed once
            if i in (0, n_img - 1) and gradients[i] is not None:
                continue
            e, g = nddo_gradient(atoms, images[i], method=method, analytical=analytical)
            energies[i] = e
            gradients[i] = g

        top = int(np.argmax(energies[1:-1])) + 1
        use_climb = climbing and iteration >= climb_after

        forces = [np.zeros_like(im) for im in images]
        for i in range(1, n_img - 1):
            tau = _improved_tangent(images, energies, i)
            grad = gradients[i]
            g_par = float(np.sum(grad * tau))

            if use_climb and i == top:
                # invert the parallel component: this image climbs to the saddle
                forces[i] = -grad + 2.0 * g_par * tau
            else:
                f_perp = -grad + g_par * tau
                d_next = np.linalg.norm(images[i + 1] - images[i])
                d_prev = np.linalg.norm(images[i] - images[i - 1])
                forces[i] = f_perp + k_spring * (d_next - d_prev) * tau

        max_force = max(float(np.abs(forces[i]).max()) for i in range(1, n_img - 1))
        if verbose and (iteration % 10 == 0 or max_force < force_tol):
            barrier = (energies.max() - energies[0]) * EV_TO_KCAL
            print(f"  neb {iteration:4d}: Fmax={max_force:.4f} eV/A  top={top}  Ea={barrier:.2f} kcal/mol")
        if max_force < force_tol:
            converged = True
            break

        # FIRE: mix velocity toward the force while descending, reset on any climb
        power = sum(float(np.sum(forces[i] * velocity[i])) for i in range(1, n_img - 1))
        if power > 0:
            steps_since_negative += 1
            if steps_since_negative > n_min:
                dt_now = min(dt_now * f_inc, 10.0 * dt)
                alpha *= alpha_dec
        else:
            steps_since_negative = 0
            dt_now *= f_dec
            alpha = alpha_start
            for i in range(1, n_img - 1):
                velocity[i][:] = 0.0

        for i in range(1, n_img - 1):
            f_norm = np.linalg.norm(forces[i])
            v_norm = np.linalg.norm(velocity[i])
            if f_norm > 1e-12:
                velocity[i] = (1.0 - alpha) * velocity[i] + alpha * forces[i] / f_norm * v_norm
            velocity[i] = velocity[i] + dt_now * forces[i]
            step = dt_now * velocity[i]
            # cap the step so a bad gradient cannot fling an image across the molecule
            step_max = np.abs(step).max()
            if step_max > 0.2:
                step *= 0.2 / step_max
            images[i] = images[i] + step
            gradients[i] = None

    top = int(np.argmax(energies))
    e_kcal = (energies - energies[0]) * EV_TO_KCAL
    return {
        "images": images,
        "energies_eV": energies,
        "energies_kcal": e_kcal,
        "barrier_forward_kcal": float((energies.max() - energies[0]) * EV_TO_KCAL),
        "barrier_reverse_kcal": float((energies.max() - energies[-1]) * EV_TO_KCAL),
        "delta_e_kcal": float((energies[-1] - energies[0]) * EV_TO_KCAL),
        "ts_index": top,
        "ts_coords": images[top],
        "neb_converged": converged,
        "n_iter": iteration + 1,
        "max_force": max_force,
        "method": method,
    }


def estimate_barrier(
    atoms: list[int],
    coords_start: np.ndarray,
    coords_end: np.ndarray,
    method: str = "RM1",
    n_images: int = 9,
    optimize_endpoints: bool = True,
    **neb_kwargs,
) -> dict:
    """Optimizes both endpoints, interpolates a band between them, and relaxes it."""
    from .gradient import nddo_optimize

    start = np.asarray(coords_start, dtype=np.float64)
    end = np.asarray(coords_end, dtype=np.float64)
    if optimize_endpoints:
        start = nddo_optimize(atoms, start, method=method)["coords"]
        end = nddo_optimize(atoms, end, method=method)["coords"]

    band = interpolate_path(start, end, n_images)
    result = neb_run(atoms, band, method=method, **neb_kwargs)
    result["coords_start"] = start
    result["coords_end"] = end
    return result


def lst_barrier(
    atoms: list[int],
    coords_start: np.ndarray,
    coords_end: np.ndarray,
    method: str = "RM1",
    n_points: int = 21,
    optimize_endpoints: bool = False,
) -> dict:
    """Energy-only linear-synchronous-transit scan: a cheap **upper bound** on the barrier.

    Use this to screen, and NEB only on the edges that matter. The reason is a large cost
    asymmetry in this package: a single SCF on a 34-atom molecule takes ~0.6 s, while the
    frozen-density "analytical" gradient takes ~23 s -- roughly 41x the SCF. A relaxed NEB
    therefore costs hundreds of gradient calls, whereas a 21-point scan costs 21 energies.

    Because the path is not relaxed perpendicular to itself, the maximum found here sits
    above the true saddle -- typically by tens of percent. That is the right error
    direction for a screen: a scan barrier that is already low is genuinely low.
    """
    from .gradient import nddo_optimize
    from .scf import nddo_energy

    start = np.asarray(coords_start, dtype=np.float64)
    end = np.asarray(coords_end, dtype=np.float64)
    if optimize_endpoints:
        start = nddo_optimize(atoms, start, method=method)["coords"]
        end = nddo_optimize(atoms, end, method=method)["coords"]

    band = interpolate_path(start, end, n_points)
    problem = check_band_geometry(band)
    if problem is not None:
        raise ValueError(f"interpolated band is not physical: {problem}")

    energies = np.array([nddo_energy(atoms, c, method=method)["energy_eV"] for c in band])
    top = int(np.argmax(energies))
    return {
        "images": band,
        "energies_eV": energies,
        "energies_kcal": (energies - energies[0]) * EV_TO_KCAL,
        "barrier_forward_kcal": float((energies.max() - energies[0]) * EV_TO_KCAL),
        "barrier_reverse_kcal": float((energies.max() - energies[-1]) * EV_TO_KCAL),
        "delta_e_kcal": float((energies[-1] - energies[0]) * EV_TO_KCAL),
        "ts_index": top,
        "ts_coords": band[top],
        "is_upper_bound": True,
        "method": method,
    }


def _prepare_tautomer_pair(smiles_a: str, smiles_b: str, seed: int = 42):
    """Embeds both tautomers and puts ``mol_b`` into ``mol_a``'s atom order.

    Returns ``(atoms, coords_a, coords_b, perm)``. Raises ValueError when the two inputs
    are not a tautomer pair (different formula, or heavy-atom skeletons that do not match).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def embed(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"could not embed: {smiles}")
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    mol_a, mol_b = embed(smiles_a), embed(smiles_b)
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        raise ValueError("not a tautomer pair: different atom counts")

    perm = map_tautomer_atoms(mol_a, mol_b)
    if perm is None:
        raise ValueError("not a tautomer pair: heavy-atom skeletons do not match")

    atoms = [a.GetAtomicNum() for a in mol_a.GetAtoms()]
    atoms_b = [mol_b.GetAtomWithIdx(perm[i]).GetAtomicNum() for i in range(len(perm))]
    if atoms != atoms_b:
        raise ValueError("not a tautomer pair: element sequence differs after mapping")

    conf_a = mol_a.GetConformer().GetPositions()
    conf_b = mol_b.GetConformer().GetPositions()
    coords_b = np.array([conf_b[perm[i]] for i in range(len(perm))])
    return atoms, conf_a, coords_b, perm


def tautomer_scan(
    smiles_a: str,
    smiles_b: str,
    method: str = "PM6",
    n_points: int = 21,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Cheap screening barrier between two tautomers given as SMILES (see lst_barrier)."""
    atoms, coords_a, coords_b, perm = _prepare_tautomer_pair(smiles_a, smiles_b, seed=seed)
    result = lst_barrier(atoms, coords_a, coords_b, method=method, n_points=n_points, **kwargs)
    result.update(smiles_a=smiles_a, smiles_b=smiles_b, atom_map=perm)
    return result


def tautomer_barrier(
    smiles_a: str,
    smiles_b: str,
    method: str = "PM6",
    n_images: int = 9,
    seed: int = 42,
    **neb_kwargs,
) -> dict:
    """Gas-phase NEB barrier between two tautomers given as SMILES.

    Accurate but expensive -- see ``lst_barrier`` for the cost asymmetry and prefer
    ``tautomer_scan`` to screen before spending a relaxation here.

    Raises ValueError if the two inputs are not a tautomer pair (different formula, or
    heavy-atom skeletons that do not match).
    """
    atoms, coords_a, coords_b, perm = _prepare_tautomer_pair(smiles_a, smiles_b, seed=seed)
    result = estimate_barrier(atoms, coords_a, coords_b, method=method, n_images=n_images, **neb_kwargs)
    result.update(smiles_a=smiles_a, smiles_b=smiles_b, atom_map=perm)
    return result
