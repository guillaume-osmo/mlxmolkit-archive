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
                # xtb writes ``epsilon=Inf`` literally for --tmcosmo inf.
                if "inf" in ln.lower() and "=" in ln:
                    return float("inf")
                matches = _NUMBER_RE.findall(ln)
                if matches:
                    return float(matches[-1])
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


_BOHR_PER_ANG = 1.0 / 0.52917721092  # match opencosmorspy's constant
_ANG2_PER_BOHR2 = 0.52917721092 ** 2
_ANG3_PER_BOHR3 = 0.52917721092 ** 3


def write_cosmo_file(
    cosmo: CosmoSegments,
    path: Path | str,
    *,
    method_tag: str = "xtb;gfn2-tmcosmo",
    opencosmors_compat: bool = True,
) -> None:
    """Write the TURBOMOLE-format ``.cosmo`` text to disk.

    With ``opencosmors_compat=True`` (default) the output is patched so that
    openCOSMO-RS_py's ``SigmaProfileParser`` reads it directly:

      * The line under ``$info`` is rewritten to a ``program;method`` tag
        (xtb writes ``prog.: xtb``, the parser expects a ``;``-separated
        string).
      * The global ``area=`` and ``volume=`` fields in ``$cosmo_data`` are
        converted from Å²/Å³ (xtb's convention) to Bohr²/Bohr³ — the parser
        applies a ``Bohr→Å`` conversion factor to those fields. Per-segment
        ``seg_area`` and ``seg_pos`` already match the parser's expected
        units (Å² and Bohr respectively), so they are left untouched.

    Set ``opencosmors_compat=False`` to write the raw xtb-format text.
    """

    text = cosmo.cosmo_text
    if not opencosmors_compat:
        Path(path).write_text(text)
        return

    lines = text.splitlines(keepends=True)
    out = []
    rewrite_next = False
    in_cosmo_data = False
    for line in lines:
        if rewrite_next:
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            out.append(f"{indent}{method_tag}{newline}")
            rewrite_next = False
            continue

        stripped = line.strip()
        if stripped.startswith("$"):
            in_cosmo_data = stripped == "$cosmo_data"
            out.append(line)
            if stripped == "$info":
                rewrite_next = True
            continue

        if in_cosmo_data:
            if stripped.startswith("area"):
                val = float(stripped.split("=")[1])
                indent = line[: len(line) - len(line.lstrip())]
                newline = "\r\n" if line.endswith("\r\n") else "\n"
                out.append(f"{indent}area={val / _ANG2_PER_BOHR2:.15g}{newline}")
                continue
            if stripped.startswith("volume"):
                val = float(stripped.split("=")[1])
                indent = line[: len(line) - len(line.lstrip())]
                newline = "\r\n" if line.endswith("\r\n") else "\n"
                out.append(f"{indent}volume={val / _ANG3_PER_BOHR3:.15g}{newline}")
                continue

        out.append(line)

    Path(path).write_text("".join(out))


def sigma_profile_histogram(
    cosmo: CosmoSegments,
    *,
    sigma_min: float = -0.025,
    sigma_max: float = 0.025,
    n_bins: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw (un-averaged) area-weighted σ-profile histogram p(σ).

    Returns ``(sigma_centers, p_area)`` where ``p_area`` is the surface area
    in Å² summed into each σ bin. For the canonical Klamt-averaged COSMO-RS
    σ-profile use :func:`sigma_profile_klamt` instead.
    """

    edges = np.linspace(sigma_min, sigma_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.asarray(cosmo.segments_area, dtype=np.float64)
    sigmas = np.asarray(cosmo.segments_sigma, dtype=np.float64)
    p_area, _ = np.histogram(sigmas, bins=edges, weights=weights)
    return centers, p_area


_KLAMT_PARAMS = {
    # Mullins 2008 (IECR), used by NIST COSMOSAC v2:
    "mullins": (0.8176300195**2, 1.0),
    # Hsieh 2010 (Fluid Phase Equil.), used by COSMO-SAC-2010:
    "hsieh": (7.25 / np.pi, 3.57),
}


def klamt_average_sigmas(
    cosmo: CosmoSegments,
    *,
    variant: str = "mullins",
    bohr_to_ang: float = 0.5291772108,
) -> np.ndarray:
    """Klamt r-averaged segment σ values (e/Å²).

    For each segment i:

        σ̄_i = Σ_j w_ij · σ_j / Σ_j w_ij

        w_ij = (r_n² · r_av² / (r_n² + r_av²))
               · exp( -f_decay · d_ij² / (r_n² + r_av²) )

    where ``r_n²`` is the segment's effective radius squared (area / π),
    ``r_av²`` and ``f_decay`` are the variant constants, and ``d_ij`` is the
    pairwise segment-segment distance in Å.

    Variants:
      * ``'mullins'`` (default): ``r_av² = 0.8176²``, ``f_decay = 1.0``.
      * ``'hsieh'``: ``r_av² = 7.25/π``, ``f_decay = 3.57``.
    """

    if variant not in _KLAMT_PARAMS:
        raise ValueError(f"variant must be one of {sorted(_KLAMT_PARAMS)}; got {variant!r}")
    r_av2, f_decay = _KLAMT_PARAMS[variant]

    area = np.asarray(cosmo.segments_area, dtype=np.float64)
    sigma = np.asarray(cosmo.segments_sigma, dtype=np.float64)
    xyz_ang = np.asarray(cosmo.segments_xyz_bohr, dtype=np.float64) * bohr_to_ang

    rn2 = area / np.pi  # per-segment r² (Å²)
    diff = xyz_ang[:, None, :] - xyz_ang[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)

    # Both NIST COSMOSAC's to_sigma.py and openCOSMO-RS_py weight by the
    # contributing segment j's radius — denom[i, j] = r_n[j]² + r_av².
    denom = rn2[None, :] + r_av2
    pref = rn2[None, :] * r_av2 / denom
    w = pref * np.exp(-f_decay * d2 / denom)

    return (w @ sigma) / w.sum(axis=1)


def sigma_profile_klamt(
    cosmo: CosmoSegments,
    *,
    variant: str = "mullins",
    sigma_min: float = -0.025,
    sigma_max: float = 0.025,
    bin_width: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical Klamt-averaged σ-profile p(σ) used by COSMO-SAC / openCOSMO-RS.

    Returns ``(sigma_centers, p_area, sigma_avg)`` where ``sigma_avg`` is the
    per-segment averaged σ values (e/Å²) and ``p_area`` is the area-weighted
    histogram (Å²) on the standard ``[-0.025, 0.025]`` grid at ``0.001`` step.
    """

    sigma_avg = klamt_average_sigmas(cosmo, variant=variant)
    n_bins = int(round((sigma_max - sigma_min) / bin_width))
    edges = np.linspace(sigma_min, sigma_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.asarray(cosmo.segments_area, dtype=np.float64)
    p_area, _ = np.histogram(sigma_avg, bins=edges, weights=weights)
    return centers, p_area, sigma_avg
