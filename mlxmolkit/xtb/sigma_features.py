from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .cosmo_sigma import (
    CosmoSegments,
    OPENCOSMORS25A_PARAMS,
    klamt_average_sigmas,
    sigma_potential,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


BOHR_TO_ANG = 0.5291772108
DEFAULT_SIGMA_GRID_E_PER_A2 = np.round(np.arange(-0.030, 0.0301, 0.001), 6)


def _regular_grid_edges(grid: np.ndarray) -> tuple[np.ndarray, float]:
    centers = np.asarray(grid, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("sigma_grid_e_per_A2 must be a 1D array with at least 2 points")
    step = np.diff(centers)
    if not np.allclose(step, step[0], atol=1.0e-10, rtol=1.0e-7):
        raise ValueError("sigma_grid_e_per_A2 must be evenly spaced")
    dx = float(step[0])
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[0] = centers[0] - 0.5 * dx
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[-1] = centers[-1] + 0.5 * dx
    return edges, dx


def _bin_signal_on_grid(
    values: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    *,
    kernel_sigma: float | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    centers = np.asarray(grid, dtype=np.float64)
    if values.size == 0:
        return np.zeros_like(centers, dtype=np.float64)

    if kernel_sigma is None or kernel_sigma <= 0.0:
        edges, _ = _regular_grid_edges(centers)
        hist, _ = np.histogram(values, bins=edges, weights=weights)
        return hist.astype(np.float64, copy=False)

    diff = values[:, None] - centers[None, :]
    kernel = np.exp(-0.5 * (diff / float(kernel_sigma)) ** 2)
    return np.sum(weights[:, None] * kernel, axis=0)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    return float(np.sum(weights * values) / total)


def _weighted_var(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    return float(np.sum(weights * (values - mean) ** 2) / total)


def _geometry_invariants(rel_xyz_bohr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rel_xyz_ang = np.asarray(rel_xyz_bohr, dtype=np.float64) * BOHR_TO_ANG
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if rel_xyz_ang.size == 0 or total <= 0.0:
        return np.zeros(7, dtype=np.float64)

    w = weights / total
    centroid = np.sum(w[:, None] * rel_xyz_ang, axis=0)
    r = np.linalg.norm(rel_xyz_ang, axis=1)
    mean_r = float(np.sum(w * r))
    rms_r = float(np.sqrt(np.sum(w * r * r)))
    std_r = float(np.sqrt(max(np.sum(w * (r - mean_r) ** 2), 0.0)))
    centered = rel_xyz_ang - centroid[None, :]
    cov = centered.T @ (centered * w[:, None])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    return np.array(
        [
            float(np.linalg.norm(centroid)),
            mean_r,
            rms_r,
            std_r,
            float(eigvals[0]),
            float(eigvals[1]),
            float(eigvals[2]),
        ],
        dtype=np.float64,
    )


def _coerce_explicit_h_mol(mol: Mol, expected_z: list[int]) -> Mol:
    from rdkit import Chem

    expected = list(int(z) for z in expected_z)
    candidates: list[Mol] = [mol]
    try:
        candidates.append(Chem.AddHs(Chem.Mol(mol)))
    except Exception:
        pass

    for cand in candidates:
        z = [int(atom.GetAtomicNum()) for atom in cand.GetAtoms()]
        if z == expected:
            return cand

    raise ValueError(
        "mol atom ordering does not match the COSMO atoms; pass an explicit-H "
        "RDKit molecule with the same atom order as the sigma source"
    )


def _atom_chem_features(mol: Mol) -> tuple[np.ndarray, list[str]]:
    features: list[list[float]] = []
    for atom in mol.GetAtoms():
        ring_info = atom.GetOwningMol().GetRingInfo()
        features.append(
            [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetTotalValence()),
                float(atom.GetFormalCharge()),
                float(atom.GetTotalNumHs()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                float(ring_info.NumAtomRings(atom.GetIdx())),
                float(int(atom.GetHybridization())),
                float(atom.GetNumRadicalElectrons()),
            ]
        )
    names = [
        "atomic_number",
        "degree",
        "total_valence",
        "formal_charge",
        "total_num_h",
        "is_aromatic",
        "is_in_ring",
        "ring_count",
        "hybridization_code",
        "radical_electrons",
    ]
    return np.asarray(features, dtype=np.float64), names


def _build_bond_graph(mol: Mol, atom_coords_ang: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    senders: list[int] = []
    receivers: list[int] = []
    attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        dist = float(np.linalg.norm(atom_coords_ang[i] - atom_coords_ang[j]))
        feat = [
            dist,
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsAromatic()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
        senders.extend([i, j])
        receivers.extend([j, i])
        attrs.extend([feat, feat])
    edge_index = np.asarray([senders, receivers], dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float64) if attrs else np.zeros((0, 5), dtype=np.float64)
    names = ["distance_ang", "bond_order", "is_aromatic", "is_conjugated", "is_in_ring"]
    return edge_index, edge_attr, names


def _build_angle_graph(
    mol: Mol,
    atom_coords_ang: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    neighbors: dict[int, list[int]] = {int(atom.GetIdx()): [] for atom in mol.GetAtoms()}
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        neighbors[i].append(j)
        neighbors[j].append(i)

    triplets: list[list[int]] = []
    attrs: list[list[float]] = []
    for j, nbrs in neighbors.items():
        nbrs = sorted(set(nbrs))
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i = nbrs[a]
                k = nbrs[b]
                v1 = atom_coords_ang[i] - atom_coords_ang[j]
                v2 = atom_coords_ang[k] - atom_coords_ang[j]
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 <= 1.0e-12 or n2 <= 1.0e-12:
                    angle = 0.0
                else:
                    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    angle = float(np.arccos(c))
                triplets.append([i, j, k])
                attrs.append([angle, float(np.cos(angle)), float(np.sin(angle))])

    if not triplets:
        return None, None, ["angle_rad", "cos_angle", "sin_angle"]
    angle_index = np.asarray(triplets, dtype=np.int64).T
    angle_attr = np.asarray(attrs, dtype=np.float64)
    return angle_index, angle_attr, ["angle_rad", "cos_angle", "sin_angle"]


def _extract_atom_targets(gxtb_res: dict[str, object], n_atoms: int) -> dict[str, np.ndarray]:
    atom_charge = np.asarray(gxtb_res.get("atom_charges", np.zeros(n_atoms)), dtype=np.float64)
    if atom_charge.shape != (n_atoms,):
        atom_charge = np.zeros(n_atoms, dtype=np.float64)

    coord = np.asarray(gxtb_res.get("coordination_number", np.zeros(n_atoms)), dtype=np.float64)
    if coord.shape != (n_atoms,):
        coord = np.zeros(n_atoms, dtype=np.float64)

    eeqbc = np.asarray(gxtb_res.get("eeqbc_charges", np.zeros(n_atoms)), dtype=np.float64)
    if eeqbc.shape != (n_atoms,):
        eeqbc = np.zeros(n_atoms, dtype=np.float64)

    shell_sum = atom_charge.copy()
    shell_abs_sum = np.zeros(n_atoms, dtype=np.float64)
    shell_l2 = np.zeros(n_atoms, dtype=np.float64)
    n_shells = np.zeros(n_atoms, dtype=np.float64)

    shell_charges = np.asarray(gxtb_res.get("shell_charges", np.zeros(0)), dtype=np.float64)
    basis = gxtb_res.get("basis")
    shell_atom = getattr(basis, "shell_atom", None)
    if shell_atom is not None:
        shell_atom = np.asarray(shell_atom, dtype=np.int64)
        if shell_charges.shape == shell_atom.shape:
            shell_abs_sum = np.bincount(
                shell_atom, weights=np.abs(shell_charges), minlength=n_atoms
            ).astype(np.float64, copy=False)
            shell_l2 = np.sqrt(
                np.bincount(shell_atom, weights=shell_charges * shell_charges, minlength=n_atoms)
            ).astype(np.float64, copy=False)
            n_shells = np.bincount(shell_atom, minlength=n_atoms).astype(np.float64, copy=False)

    return {
        "atom_charges": atom_charge,
        "shell_charge_sum": shell_sum,
        "shell_charge_abs_sum": shell_abs_sum,
        "shell_charge_l2": shell_l2,
        "coordination_number": coord,
        "eeqbc_charges": eeqbc,
        "n_shells": n_shells,
    }


def _extract_mol_scalar_targets(gxtb_res: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in gxtb_res.items():
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            if key.endswith(("_hartree", "_eV", "_kcal")) or "alpb" in key.lower():
                out[key] = float(value)
    return out


def sigma_feature_tensors(
    cosmo: CosmoSegments,
    gxtb_res: dict[str, object],
    mol: Mol,
    *,
    sigma_grid_e_per_A2: np.ndarray | None = None,
    T: float = 298.15,
    params: dict | None = None,
    include_angles: bool = False,
    kernel_sigma: float | None = None,
) -> dict[str, object]:
    """Build atom-local sigma/potential tensors plus graph structure.

    The exporter keeps the molecular sigma objects and the atom-local matrices
    side-by-side:

    * ``v_pot``: molecular sigma-potential on the 25a grid
    * ``v_profile``: molecular primary sigma-profile on the same grid
    * ``X_atom``: rich atom-local representation
    * ``edge_index`` / ``edge_attr``: bond graph
    * ``angle_index`` / ``angle_attr``: optional i-j-k angle triples

    ``mol`` must match the atom order of ``cosmo``; if it lacks explicit
    hydrogens we attempt ``Chem.AddHs`` automatically.
    """

    sigma_grid = (
        np.asarray(sigma_grid_e_per_A2, dtype=np.float64)
        if sigma_grid_e_per_A2 is not None
        else DEFAULT_SIGMA_GRID_E_PER_A2.copy()
    )
    _regular_grid_edges(sigma_grid)

    mol_h = _coerce_explicit_h_mol(mol, cosmo.atom_z)
    n_atoms = len(cosmo.atom_z)
    atom_coords_bohr = np.asarray(cosmo.atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.shape != (n_atoms, 3):
        raise ValueError("cosmo.atom_coords_bohr has inconsistent shape")
    atom_coords_ang = atom_coords_bohr * BOHR_TO_ANG

    sigma_primary_seg = klamt_average_sigmas(cosmo, variant="ocrs25a_primary")
    sigma_corr_seg = klamt_average_sigmas(cosmo, variant="ocrs25a_corr")
    sigma_raw_seg = np.asarray(cosmo.segments_sigma, dtype=np.float64)
    seg_area = np.asarray(cosmo.segments_area, dtype=np.float64)
    seg_pot = np.asarray(cosmo.segments_potential, dtype=np.float64)
    seg_atom = np.asarray(cosmo.segments_atom, dtype=np.int64)
    seg_xyz_bohr = np.asarray(cosmo.segments_xyz_bohr, dtype=np.float64)

    v_profile = _bin_signal_on_grid(
        sigma_primary_seg,
        seg_area,
        sigma_grid,
        kernel_sigma=kernel_sigma,
    )
    v_profile_corr = _bin_signal_on_grid(
        sigma_corr_seg,
        seg_area,
        sigma_grid,
        kernel_sigma=kernel_sigma,
    )
    sigma_grid_out, v_pot = sigma_potential(
        cosmo,
        sigma_grid_e_per_A2=sigma_grid,
        T=T,
        params=params,
    )
    sigma_grid_out = np.asarray(sigma_grid_out, dtype=np.float64)
    if not np.allclose(sigma_grid_out, sigma_grid, atol=1.0e-12):
        raise RuntimeError("sigma_potential returned an unexpected sigma grid")

    chem_feat, chem_names = _atom_chem_features(mol_h)
    atom_targets = _extract_atom_targets(gxtb_res, n_atoms)

    primary_bins = np.zeros((n_atoms, sigma_grid.size), dtype=np.float64)
    corr_bins = np.zeros((n_atoms, sigma_grid.size), dtype=np.float64)
    potential_bins = np.zeros((n_atoms, sigma_grid.size), dtype=np.float64)
    local_scalars = np.zeros((n_atoms, 13), dtype=np.float64)
    geom_feat = np.zeros((n_atoms, 7), dtype=np.float64)

    for i in range(n_atoms):
        mask = seg_atom == (i + 1)
        sigma_i = sigma_primary_seg[mask]
        sigma_corr_i = sigma_corr_seg[mask]
        area_i = seg_area[mask]
        pot_i = seg_pot[mask]
        xyz_i = seg_xyz_bohr[mask] - atom_coords_bohr[i]

        primary_bins[i] = _bin_signal_on_grid(sigma_i, area_i, sigma_grid, kernel_sigma=kernel_sigma)
        corr_bins[i] = _bin_signal_on_grid(sigma_corr_i, area_i, sigma_grid, kernel_sigma=kernel_sigma)
        potential_bins[i] = _bin_signal_on_grid(
            sigma_i,
            area_i * pot_i,
            sigma_grid,
            kernel_sigma=kernel_sigma,
        )

        area_total = float(np.sum(area_i))
        mean_sigma = _weighted_mean(sigma_i, area_i)
        mean_abs_sigma = _weighted_mean(np.abs(sigma_i), area_i)
        var_sigma = _weighted_var(sigma_i, area_i, mean_sigma)
        mean_pot = _weighted_mean(pot_i, area_i)
        pos_frac = float(np.sum(area_i[sigma_i > 0.0]) / area_total) if area_total > 0.0 else 0.0
        neg_frac = float(np.sum(area_i[sigma_i < 0.0]) / area_total) if area_total > 0.0 else 0.0
        local_scalars[i] = np.array(
            [
                area_total,
                mean_sigma,
                mean_abs_sigma,
                var_sigma,
                pos_frac,
                neg_frac,
                mean_pot,
                float(atom_targets["atom_charges"][i]),
                float(atom_targets["shell_charge_sum"][i]),
                float(atom_targets["shell_charge_abs_sum"][i]),
                float(atom_targets["shell_charge_l2"][i]),
                float(atom_targets["coordination_number"][i]),
                float(atom_targets["eeqbc_charges"][i]),
            ],
            dtype=np.float64,
        )
        geom_feat[i] = _geometry_invariants(xyz_i, area_i)

    local_scalar_names = [
        "surface_area",
        "mean_sigma_primary",
        "mean_abs_sigma_primary",
        "var_sigma_primary",
        "positive_area_fraction",
        "negative_area_fraction",
        "mean_segment_potential",
        "atom_charge",
        "shell_charge_sum",
        "shell_charge_abs_sum",
        "shell_charge_l2",
        "coordination_number",
        "eeqbc_charge",
    ]
    geom_names = [
        "segment_centroid_norm_ang",
        "segment_mean_radius_ang",
        "segment_rms_radius_ang",
        "segment_std_radius_ang",
        "segment_cov_eig1",
        "segment_cov_eig2",
        "segment_cov_eig3",
    ]

    parts = [primary_bins, corr_bins, potential_bins, local_scalars, geom_feat, chem_feat]
    X_atom = np.concatenate(parts, axis=1)

    feature_slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, width in (
        ("sigma_primary_bins", primary_bins.shape[1]),
        ("sigma_corr_bins", corr_bins.shape[1]),
        ("sigma_potential_bins", potential_bins.shape[1]),
        ("local_scalars", local_scalars.shape[1]),
        ("geometry_invariants", geom_feat.shape[1]),
        ("chemical_features", chem_feat.shape[1]),
    ):
        feature_slices[name] = (cursor, cursor + width)
        cursor += width

    atom_feature_names = (
        [f"sigma_primary_bin_{k:02d}" for k in range(primary_bins.shape[1])]
        + [f"sigma_corr_bin_{k:02d}" for k in range(corr_bins.shape[1])]
        + [f"sigma_potential_bin_{k:02d}" for k in range(potential_bins.shape[1])]
        + local_scalar_names
        + geom_names
        + chem_names
    )

    edge_index, edge_attr, edge_feature_names = _build_bond_graph(mol_h, atom_coords_ang)
    angle_index, angle_attr, angle_feature_names = _build_angle_graph(mol_h, atom_coords_ang)
    if not include_angles:
        angle_index = None
        angle_attr = None

    y_atom = {
        **atom_targets,
        "mean_segment_potential": local_scalars[:, 6].copy(),
        "surface_area": local_scalars[:, 0].copy(),
        "profile_primary": primary_bins.copy(),
        "profile_corr": corr_bins.copy(),
        "profile_potential": potential_bins.copy(),
        "sigma_raw_mean": np.array(
            [_weighted_mean(sigma_raw_seg[seg_atom == (i + 1)], seg_area[seg_atom == (i + 1)]) for i in range(n_atoms)],
            dtype=np.float64,
        ),
    }
    y_mol = {
        "sigma_potential": v_pot.copy(),
        "sigma_profile": v_profile.copy(),
        "sigma_profile_corr": v_profile_corr.copy(),
        **_extract_mol_scalar_targets(gxtb_res),
    }

    params_out = dict(OPENCOSMORS25A_PARAMS)
    if params is not None:
        params_out.update(params)

    return {
        "sigma_grid_e_per_A2": sigma_grid.copy(),
        "v_pot": v_pot.copy(),
        "v_profile": v_profile.copy(),
        "X_atom": X_atom,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "angle_index": angle_index,
        "angle_attr": angle_attr,
        "y_atom": y_atom,
        "y_mol": y_mol,
        "meta": {
            "method": gxtb_res.get("method", "unknown"),
            "temperature_K": float(T),
            "kernel_sigma": None if kernel_sigma is None else float(kernel_sigma),
            "feature_slices": feature_slices,
            "atom_feature_names": atom_feature_names,
            "edge_feature_names": edge_feature_names,
            "angle_feature_names": angle_feature_names,
            "sigma_params": params_out,
        },
    }
