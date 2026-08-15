"""OpenMOPAC's diskless C API, bound with ctypes.

MOPAC ships `libmopac.dylib` alongside the `mopac` executable, and its header
says why it exists:

    the other API functions run MOPAC without any disk access, which may be
    beneficial for high-throughput settings with file systems that perform
    poorly when large numbers of small files are being read and written
    simultaneously

Driving the executable means a process and a `.mop`, `.out`, `.arc` and `.den`
per molecule. Measured over 200 PM6 single points on an M4 Pro:

    executable, one process per molecule   3010 ms
    this module, one process                671 ms

so **78% of the executable's time was process startup and file I/O**, not
chemistry. The results are identical — heat of formation agrees to every digit
MOPAC prints.

Note `libmopac` threads internally. When running this across a process pool,
set OMP_NUM_THREADS=1 or the workers oversubscribe the machine and it gets
*slower*: 200 molecules over 14 processes took 2030 ms unpinned against 91 ms
pinned.

Usage:

    from tools.mopac_api import scf, relax
    result = scf([6, 6, 8, 1, 1, 1, 1, 1, 1], coords, model="PM6")
    result["heat_kcal"], result["charges"]
"""
from __future__ import annotations

import ctypes as C
import os
import shutil
from pathlib import Path

import numpy as np

# semiempirical model -> the integer `mopac_system.model` wants
MODELS = {"PM7": 0, "PM6-D3H4": 1, "PM6-ORG": 2, "PM6": 3, "AM1": 4, "RM1": 5}


def _find_libmopac() -> str:
    """Locate libmopac.

    MOPAC is installed in the conda *base* prefix rather than in the working
    environment, so neither `shutil.which` under the env nor an env-relative
    path finds it — and concluding "MOPAC is not installed" is exactly the
    wrong answer for a parity run that would then silently produce nothing.
    """
    override = os.environ.get("MOPAC_LIB")
    if override:
        return override

    candidates = []
    exe = shutil.which("mopac")
    if exe:                       # <prefix>/bin/mopac -> <prefix>/lib/libmopac.dylib
        candidates.append(str(Path(exe).resolve().parent.parent / "lib" / "libmopac.dylib"))
    candidates += [
        os.path.expanduser("~/miniconda3/lib/libmopac.dylib"),
        os.path.expanduser("~/miniconda3/envs/osmo/lib/libmopac.dylib"),
        "/opt/homebrew/lib/libmopac.dylib",
        "/usr/local/lib/libmopac.dylib",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "libmopac not found. Set MOPAC_LIB, or install MOPAC. Looked in: "
        + ", ".join(candidates)
    )


class _System(C.Structure):
    _fields_ = [
        ("natom", C.c_int), ("natom_move", C.c_int), ("charge", C.c_int),
        ("spin", C.c_int), ("model", C.c_int), ("epsilon", C.c_double),
        ("atom", C.POINTER(C.c_int)), ("coord", C.POINTER(C.c_double)),
        ("nlattice", C.c_int), ("nlattice_move", C.c_int),
        ("pressure", C.c_double), ("lattice", C.POINTER(C.c_double)),
        ("tolerance", C.c_double), ("max_time", C.c_int),
    ]


class _Properties(C.Structure):
    _fields_ = [
        ("heat", C.c_double), ("dipole", C.c_double * 3),
        ("charge", C.POINTER(C.c_double)),
        ("coord_update", C.POINTER(C.c_double)),
        ("coord_deriv", C.POINTER(C.c_double)),
        ("freq", C.POINTER(C.c_double)), ("disp", C.POINTER(C.c_double)),
        ("bond_index", C.POINTER(C.c_int)), ("bond_atom", C.POINTER(C.c_int)),
        ("bond_order", C.POINTER(C.c_double)),
        ("lattice_update", C.POINTER(C.c_double)),
        ("lattice_deriv", C.POINTER(C.c_double)),
        ("stress", C.c_double * 6), ("nerror", C.c_int),
        ("error_msg", C.POINTER(C.POINTER(C.c_char))),
    ]


class _State(C.Structure):
    _fields_ = [("mpack", C.c_int), ("uhf", C.c_int),
                ("pa", C.POINTER(C.c_double)), ("pb", C.POINTER(C.c_double))]


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        lib = C.CDLL(_find_libmopac())
        for name in ("mopac_scf", "mopac_relax"):
            fn = getattr(lib, name)
            fn.argtypes = [C.POINTER(_System), C.POINTER(_State),
                           C.POINTER(_Properties)]
            fn.restype = None
        lib.destroy_mopac_properties.argtypes = [C.POINTER(_Properties)]
        lib.destroy_mopac_state.argtypes = [C.POINTER(_State)]
        _LIB = lib
    return _LIB


class MopacError(RuntimeError):
    """MOPAC reported one or more errors for this system."""


def _run(kind, atoms, coords, model="PM6", charge=0, spin=0, epsilon=1.0,
         tolerance=1.0, max_time=3600):
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODELS)}")
    atoms = [int(z) for z in atoms]
    n = len(atoms)
    xyz = np.ascontiguousarray(coords, dtype=np.float64).reshape(n, 3)

    system = _System()
    system.natom = system.natom_move = n
    system.charge = int(charge)
    system.spin = int(spin)
    system.model = MODELS[model]
    system.epsilon = float(epsilon)
    # Keep references alive for the duration of the call: ctypes does not own
    # what these pointers point at, and a garbage-collected buffer here is a
    # use-after-free inside Fortran.
    atom_buf = (C.c_int * n)(*atoms)
    coord_buf = (C.c_double * (3 * n))(*xyz.ravel())
    system.atom = atom_buf
    system.coord = coord_buf
    system.nlattice = system.nlattice_move = 0
    system.pressure = 0.0
    system.lattice = None
    system.tolerance = float(tolerance)
    system.max_time = int(max_time)

    state = _State()          # mpack = 0 -> start from MOPAC's own atomic guess
    props = _Properties()
    lib = _lib()
    getattr(lib, kind)(C.byref(system), C.byref(state), C.byref(props))

    try:
        if props.nerror:
            messages = []
            for i in range(max(0, props.nerror)):
                messages.append(C.cast(props.error_msg[i], C.c_char_p)
                                .value.decode(errors="replace").strip())
            raise MopacError(f"{kind} failed: " + "; ".join(messages))

        out = {
            "heat_kcal": float(props.heat),
            "dipole": np.array(props.dipole[:3]),
            "charges": np.array([props.charge[i] for i in range(n)]),
        }
        if props.coord_deriv:
            out["gradient_kcal"] = np.array(
                [props.coord_deriv[i] for i in range(3 * n)]).reshape(n, 3)
        if props.coord_update:
            out["coords"] = np.array(
                [props.coord_update[i] for i in range(3 * n)]).reshape(n, 3)
        return out
    finally:
        lib.destroy_mopac_properties(C.byref(props))
        lib.destroy_mopac_state(C.byref(state))


def scf(atoms, coords, model="PM6", charge=0, **kwargs):
    """Single-point energy. Equivalent to the `1SCF` keyword."""
    return _run("mopac_scf", atoms, coords, model=model, charge=charge, **kwargs)


def relax(atoms, coords, model="PM6", charge=0, **kwargs):
    """Geometry optimisation. Returns the relaxed coordinates under "coords"."""
    return _run("mopac_relax", atoms, coords, model=model, charge=charge, **kwargs)


if __name__ == "__main__":
    # Ethanol at the geometry tools/gen_mopac_ref.py uses, as a smoke test.
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    result = scf([a.GetAtomicNum() for a in mol.GetAtoms()],
                 np.array([list(conf.GetAtomPosition(i))
                           for i in range(mol.GetNumAtoms())]))
    print(f"ethanol PM6 heat of formation = {result['heat_kcal']:.5f} kcal/mol")
    print(f"charges = {np.round(result['charges'], 4)}")
