# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""COSMO σ-profile pipeline for openCOSMO-RS.

g-xTB has no implicit-solvation support (verified 2026-05-11 against the
binary at ``/tmp/gxtb-v2-macos/bin/xtb``), so the hybrid path is:

1. ``xtb --gxtb --opt``   — g-xTB geometry optimization (best 2026 quality).
2. ``xtb --gfn 2 --tmcosmo SOLVENT`` — GFN2 single point on the optimized
   geometry; writes a TURBOMOLE-format ``xtb.cosmo`` file.
3. The ``xtb.cosmo`` file is consumed directly by openCOSMO-RS.

The TURBOMOLE ``.cosmo`` segment block columns are:

    n  atom  x  y  z  charge  area  charge/area  potential

with position in Bohr and area in Å²; ``charge/area`` is σ in e/Å².
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_DEFAULT_XTB = Path("/tmp/gxtb-v2-macos/bin/xtb")
_ELEMENTS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe"
).split()


def _xtb_env(xtb_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    libdir = str(xtb_path.parent.parent / "lib")
    bindir = str(xtb_path.parent)
    env["DYLD_LIBRARY_PATH"] = f"{libdir}:{bindir}:{env.get('DYLD_LIBRARY_PATH', '')}"
    return env


def _write_xyz(path: Path, atoms: list[int] | np.ndarray, coords_ang: np.ndarray) -> None:
    atoms = np.asarray(atoms, dtype=int)
    coords = np.asarray(coords_ang, dtype=np.float64)
    with path.open("w") as f:
        f.write(f"{atoms.size}\n\n")
        for z, (x, y, zc) in zip(atoms, coords):
            f.write(f"{_ELEMENTS[int(z)]:<3s} {x: .10f} {y: .10f} {zc: .10f}\n")


def _read_xyz(path: Path) -> tuple[list[int], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms: list[int] = []
    coords = np.zeros((n, 3), dtype=np.float64)
    for i, line in enumerate(lines[2 : 2 + n]):
        sym, x, y, z = line.split()[:4]
        atoms.append(_ELEMENTS.index(sym))
        coords[i] = [float(x), float(y), float(z)]
    return atoms, coords


def _run_xtb(cwd: Path, args: list[str], *, xtb_path: Path, timeout: float = 600.0) -> str:
    cmd = [str(xtb_path), *args]
    proc = subprocess.run(
        cmd, cwd=cwd, env=_xtb_env(xtb_path),
        text=True, capture_output=True, check=False, timeout=timeout,
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"xtb failed (code {proc.returncode}):\n{log[-3000:]}")
    return log


@dataclass
class CosmoSegments:
    """Parsed TURBOMOLE-format ``xtb.cosmo`` content."""

    epsilon: float
    fepsi: float
    area: float
    volume: float
    total_screening_charge: float
    total_energy_hartree: float
    dielectric_energy_hartree: float
    atom_radii: np.ndarray  # (n_atoms,) Å
    atom_coords_bohr: np.ndarray  # (n_atoms, 3) Bohr
    atom_z: list[int]
    segments_atom: np.ndarray  # (n_seg,) 1-based atom index
    segments_xyz_bohr: np.ndarray  # (n_seg, 3)
    segments_charge: np.ndarray  # (n_seg,) e
    segments_area: np.ndarray  # (n_seg,) Å²
    segments_sigma: np.ndarray  # (n_seg,) e/Å² = charge/area
    segments_potential: np.ndarray  # (n_seg,) Hartree·Å
    cosmo_text: str = field(repr=False)


_SECTION_RE = re.compile(r"^\$([a-zA-Z_]+)")
_NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def parse_xtb_cosmo(path: Path) -> CosmoSegments:
    """Parse the TM-format ``xtb.cosmo`` written by ``xtb --tmcosmo``."""

    text = Path(path).read_text()
    lines = text.splitlines()

    sections: dict[str, list[str]] = {}
    current = None
    for line in lines:
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    def kv(section: str, key: str) -> float:
        for ln in sections.get(section, []):
            if key in ln:
                return float(_NUMBER_RE.findall(ln)[-1])
        raise KeyError(f"{key} not found in ${section}")

    epsilon = kv("cosmo", "epsilon")
    fepsi = kv("cosmo_data", "fepsi")
    area = kv("cosmo_data", "area")
    volume = kv("cosmo_data", "volume")
    total_q = kv("screening_charge", "total")

    def energy(label_substr: str) -> float:
        for ln in sections.get("cosmo_energy", []):
            if label_substr in ln:
                return float(_NUMBER_RE.findall(ln)[-1])
        return float("nan")

    e_total = energy("Total energy [a.u.]")
    e_diel = energy("Dielectric energy [a.u.]")

    # $coord_rad: skip the first '#atom ...' header line, parse n rows
    atoms_z: list[int] = []
    radii: list[float] = []
    coords_at: list[list[float]] = []
    for ln in sections["coord_rad"]:
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split()
        # columns: idx  x  y  z  element  radius
        if len(parts) < 6:
            continue
        coords_at.append([float(parts[1]), float(parts[2]), float(parts[3])])
        atoms_z.append(_ELEMENTS.index(parts[4]))
        radii.append(float(parts[5]))

    # $segment_information: rows beginning with an integer in the first column
    seg_atom: list[int] = []
    seg_xyz: list[list[float]] = []
    seg_q: list[float] = []
    seg_area: list[float] = []
    seg_sigma: list[float] = []
    seg_pot: list[float] = []
    for ln in sections["segment_information"]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        try:
            n_idx = int(parts[0])
            atom_idx = int(parts[1])
        except ValueError:
            continue
        seg_atom.append(atom_idx)
        seg_xyz.append([float(parts[2]), float(parts[3]), float(parts[4])])
        seg_q.append(float(parts[5]))
        seg_area.append(float(parts[6]))
        seg_sigma.append(float(parts[7]))
        seg_pot.append(float(parts[8]) if len(parts) > 8 else float("nan"))

    return CosmoSegments(
        epsilon=epsilon,
        fepsi=fepsi,
        area=area,
        volume=volume,
        total_screening_charge=total_q,
        total_energy_hartree=e_total,
        dielectric_energy_hartree=e_diel,
        atom_radii=np.asarray(radii),
        atom_coords_bohr=np.asarray(coords_at),
        atom_z=atoms_z,
        segments_atom=np.asarray(seg_atom, dtype=np.intp),
        segments_xyz_bohr=np.asarray(seg_xyz),
        segments_charge=np.asarray(seg_q),
        segments_area=np.asarray(seg_area),
        segments_sigma=np.asarray(seg_sigma),
        segments_potential=np.asarray(seg_pot),
        cosmo_text=text,
    )


def gxtb_optimize_geometry(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    uhf: int = 0,
    xtb_path: Path = _DEFAULT_XTB,
    workdir: Path | None = None,
    keep_workdir: bool = False,
    acc: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Run ``xtb --gxtb --opt`` and return the optimized coords and energy.

    Returns ``(coords_ang_opt, energy_hartree)``.
    """

    workdir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="gxtb-opt-"))
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        xyz = workdir / "mol.xyz"
        _write_xyz(xyz, atoms, coords_ang)
        log = _run_xtb(
            workdir,
            [str(xyz.name), "--gxtb", "--opt", "--acc", str(acc),
             "--chrg", str(charge), "--uhf", str(uhf)],
            xtb_path=xtb_path,
        )
        opt_xyz = workdir / "xtbopt.xyz"
        if not opt_xyz.exists():
            raise RuntimeError(f"xtb --gxtb --opt produced no xtbopt.xyz:\n{log[-1500:]}")
        _, coords_opt = _read_xyz(opt_xyz)
        m = re.search(r"TOTAL ENERGY\s+([-+0-9.Ee]+)\s+Eh", log)
        energy = float(m.group(1)) if m else float("nan")
        return coords_opt, energy
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def gfn2_tmcosmo_singlepoint(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    solvent: str = "water",
    charge: int = 0,
    uhf: int = 0,
    xtb_path: Path = _DEFAULT_XTB,
    workdir: Path | None = None,
    keep_workdir: bool = False,
    acc: float = 0.1,
) -> CosmoSegments:
    """Run ``xtb --gfn 2 --tmcosmo SOLVENT`` and parse ``xtb.cosmo``."""

    workdir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="gfn2-cosmo-"))
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        xyz = workdir / "mol.xyz"
        _write_xyz(xyz, atoms, coords_ang)
        log = _run_xtb(
            workdir,
            [str(xyz.name), "--gfn", "2", "--tmcosmo", solvent,
             "--acc", str(acc), "--chrg", str(charge), "--uhf", str(uhf)],
            xtb_path=xtb_path,
        )
        cosmo_path = workdir / "xtb.cosmo"
        if not cosmo_path.exists():
            raise RuntimeError(f"xtb did not write xtb.cosmo:\n{log[-1500:]}")
        return parse_xtb_cosmo(cosmo_path)
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def hybrid_gxtb_gfn2_cosmo(
    atoms: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    solvent: str = "water",
    charge: int = 0,
    uhf: int = 0,
    xtb_path: Path = _DEFAULT_XTB,
    workdir: Path | None = None,
    keep_workdir: bool = False,
    acc: float = 0.1,
) -> dict[str, object]:
    """Hybrid g-xTB(opt) + GFN2(tmcosmo) pipeline.

    Returns a dict with ``coords_opt`` (Å), ``gxtb_energy_hartree`` and the
    parsed :class:`CosmoSegments` under ``cosmo``.
    """

    coords_opt, e_gxtb = gxtb_optimize_geometry(
        atoms, coords_ang,
        charge=charge, uhf=uhf, xtb_path=xtb_path,
        workdir=workdir / "step1_gxtb_opt" if workdir else None,
        keep_workdir=keep_workdir, acc=acc,
    )
    cosmo = gfn2_tmcosmo_singlepoint(
        atoms, coords_opt,
        solvent=solvent, charge=charge, uhf=uhf, xtb_path=xtb_path,
        workdir=workdir / "step2_gfn2_tmcosmo" if workdir else None,
        keep_workdir=keep_workdir, acc=acc,
    )
    return {
        "atoms": np.asarray(atoms, dtype=int),
        "coords_opt_ang": coords_opt,
        "gxtb_energy_hartree": e_gxtb,
        "cosmo": cosmo,
        "solvent": solvent,
    }


def hybrid_gxtb_gfn2_cosmo_from_smiles(
    smiles: str,
    *,
    solvent: str = "water",
    seed: int = 42,
    charge: int = 0,
    uhf: int = 0,
    xtb_path: Path = _DEFAULT_XTB,
    workdir: Path | None = None,
    keep_workdir: bool = False,
    acc: float = 0.1,
) -> dict[str, object]:
    """End-to-end: SMILES → RDKit embed → g-xTB opt → GFN2 tmcosmo → parsed segments."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles!r}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    conf = mol.GetConformer()
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    coords = np.asarray(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    out = hybrid_gxtb_gfn2_cosmo(
        atoms, coords, solvent=solvent, charge=charge, uhf=uhf,
        xtb_path=xtb_path, workdir=workdir, keep_workdir=keep_workdir, acc=acc,
    )
    out["smiles"] = smiles
    return out


def write_cosmo_file(cosmo: CosmoSegments, path: Path | str) -> None:
    """Write the raw TURBOMOLE-format ``.cosmo`` text back to disk."""

    Path(path).write_text(cosmo.cosmo_text)


def sigma_profile_histogram(
    cosmo: CosmoSegments,
    *,
    sigma_min: float = -0.025,
    sigma_max: float = 0.025,
    n_bins: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted σ-profile histogram p(σ).

    Returns ``(sigma_centers, p_area)`` where ``p_area`` is the surface area
    in Å² summed into each σ bin. This is the "raw" σ-profile before Klamt
    r-averaging; openCOSMO-RS applies its own averaging downstream.
    """

    edges = np.linspace(sigma_min, sigma_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.asarray(cosmo.segments_area, dtype=np.float64)
    sigmas = np.asarray(cosmo.segments_sigma, dtype=np.float64)
    p_area, _ = np.histogram(sigmas, bins=edges, weights=weights)
    return centers, p_area
