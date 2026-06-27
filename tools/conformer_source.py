#!/usr/bin/env python
"""Single source-of-truth RDKit (CPU) conformer generation for the CHEESE pipeline.

The teacher ensembles, the ESP/RESP charge-label cache, and the LIT-PCBA
set-similarity eval must all embed 3D conformers the SAME way, or the train and
eval distributions silently drift apart. This module is that one place:

  * ONE ETKDG version (``ETKDG_VARIANT`` below) for every caller. Change it here
    and every cache + eval moves together -- there is no second definition to
    forget.
  * ONE knowledge-based-embed -> random-coordinates fallback policy for the hard
    molecules ETKDG cannot place from the bounds matrix alone.
  * ONE force-field cleanup policy (MMFF94 with a UFF fallback for molecules
    lacking MMFF parameters).

Scope / non-goals
-----------------
* This is the CPU/RDKit source of truth. Matching the MLX/GPU conformer
  generator bit-for-bit is deliberately OUT OF SCOPE -- see
  ``tools/compare_rdkit_mlx_conformers.py``. The consistency that matters is
  train-vs-eval, not RDKit-vs-GPU. If GPU conformers are ever adopted for
  deployment speed, route BOTH the teacher caches and the eval through the same
  generator (do not mix).
* Imports are restricted to ``rdkit`` + ``numpy`` on purpose. ``import
  mlxmolkit`` eagerly initialises MLX/Metal, which is unsafe to carry into a
  forked/spawned multiprocessing worker; ``tools`` is a namespace package (no
  ``__init__``) so importing ``tools.conformer_source`` never touches MLX. This
  keeps ``tools/build_litpcba_conformer_cache.py`` fork-safe.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# The single knob: every CHEESE-pipeline conformer comes from this ETKDG variant.
# ``ETKDGv3`` adds small-ring + macrocycle torsion knowledge on top of ETKDGv2.
ETKDG_VARIANT = "ETKDGv3"

# Shared defaults so callers agree without repeating magic numbers.
DEFAULT_MAX_EMBED_ATTEMPTS = 1000
DEFAULT_RANDOM_COORDS_SEED_OFFSET = 7919
DEFAULT_MMFF_VARIANT = "MMFF94"
DEFAULT_MAX_OPT_ITERS = 200


def make_etkdg_params(
    *,
    seed: int,
    prune_rms_thresh: float | None = None,
    max_attempts: int | None = DEFAULT_MAX_EMBED_ATTEMPTS,
    use_random_coords: bool = False,
):
    """Build the canonical ETKDG embedding parameters.

    This is the only place ``AllChem.ETKDGv3()`` is constructed in the pipeline.
    ``prune_rms_thresh`` is left at the RDKit default when ``None`` (it has no
    effect for single-conformer embeds, where only one conformer is produced).
    """

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if prune_rms_thresh is not None:
        params.pruneRmsThresh = float(prune_rms_thresh)
    if max_attempts is not None and hasattr(params, "maxAttempts"):
        params.maxAttempts = int(max_attempts)
    params.useRandomCoords = bool(use_random_coords)
    return params


def embed_conformers(
    mol,
    *,
    n_conformers: int = 1,
    min_conformers: int = 1,
    seed: int,
    prune_rms_thresh: float | None = None,
    max_attempts: int | None = DEFAULT_MAX_EMBED_ATTEMPTS,
    random_coords_fallback: bool = True,
    random_coords_seed_offset: int = DEFAULT_RANDOM_COORDS_SEED_OFFSET,
) -> list[int]:
    """Embed up to ``n_conformers`` conformers into ``mol`` and return their ids.

    Uses ``EmbedMultipleConfs`` for both the single- and multi-conformer cases;
    ``EmbedMultipleConfs(numConfs=1, params=p)`` is bit-identical to
    ``EmbedMolecule(mol, p)`` for the same seed, so the single-conformer charge
    and eval paths get exactly the geometry RDKit would have produced directly.

    When fewer than ``min_conformers`` are placed and ``random_coords_fallback``
    is set, all conformers are cleared and a second pass runs with random start
    coordinates, ``pruneRmsThresh = -1`` (keep everything), and an offset seed --
    the standard RDKit recipe for molecules ETKDG cannot place from the distance
    bounds alone.
    """

    params = make_etkdg_params(
        seed=seed,
        prune_rms_thresh=prune_rms_thresh,
        max_attempts=max_attempts,
        use_random_coords=False,
    )
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params))
    if random_coords_fallback and len(conf_ids) < int(min_conformers):
        mol.RemoveAllConformers()
        params = make_etkdg_params(
            seed=int(seed) + int(random_coords_seed_offset),
            prune_rms_thresh=-1.0,
            max_attempts=max_attempts,
            use_random_coords=True,
        )
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params))
    return conf_ids


def mmff_optimize_conformers(
    mol,
    conf_ids=None,
    *,
    optimize: bool = True,
    mmff_variant: str = DEFAULT_MMFF_VARIANT,
    max_iters: int = DEFAULT_MAX_OPT_ITERS,
    catch_errors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Force-field-optimize every conformer; return ``(energies, converged)``.

    MMFF94 is used when the molecule is fully parameterised, otherwise UFF. The
    returned arrays are aligned to ``conf_ids`` (or to ``mol``'s conformers when
    ``conf_ids`` is None): ``energies`` in kcal/mol (NaN where unavailable) and
    ``converged[i] == True`` when the optimizer reported status 0.

    ``catch_errors=True`` swallows optimizer exceptions and returns the NaN/False
    arrays (used by the graph-built fallback path); ``catch_errors=False`` lets
    them propagate (the SMILES path relies on this to trigger the graph fallback
    in the caller).
    """

    n = mol.GetNumConformers() if conf_ids is None else len(conf_ids)
    energies = np.full((n,), np.nan, dtype=np.float32)
    converged = np.zeros((n,), dtype=bool)
    if not optimize or n == 0:
        return energies, converged

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol, numThreads=0, maxIters=int(max_iters), mmffVariant=mmff_variant
            )
        else:
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol, numThreads=0, maxIters=int(max_iters)
            )
    except Exception:
        if catch_errors:
            return energies, converged
        raise

    for i, (status, energy) in enumerate(results[:n]):
        converged[i] = int(status) == 0
        energies[i] = float(energy)
    return energies, converged


def embed_molecule_3d(
    mol_or_smiles,
    *,
    add_hs: bool = True,
    optimize: bool = True,
    seed: int,
    mmff_variant: str = DEFAULT_MMFF_VARIANT,
    max_iters: int = DEFAULT_MAX_OPT_ITERS,
    prune_rms_thresh: float | None = None,
    max_attempts: int | None = DEFAULT_MAX_EMBED_ATTEMPTS,
):
    """Return a single-conformer, force-field-cleaned RDKit ``Mol`` (with Hs).

    Accepts a SMILES string or an existing ``Mol`` and is the one entry point
    shared by the charge-label cache and the LIT-PCBA eval. Raises ``ValueError``
    when RDKit cannot parse the SMILES or place a conformer; callers that prefer
    a ``None`` sentinel should catch it.
    """

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {mol_or_smiles!r}")
    else:
        mol = Chem.Mol(mol_or_smiles)

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)

    if mol.GetNumConformers() == 0:
        conf_ids = embed_conformers(
            mol,
            n_conformers=1,
            min_conformers=1,
            seed=seed,
            prune_rms_thresh=prune_rms_thresh,
            max_attempts=max_attempts,
        )
        if not conf_ids:
            raise ValueError("RDKit could not generate a 3D conformer")

    mmff_optimize_conformers(
        mol,
        optimize=optimize,
        mmff_variant=mmff_variant,
        max_iters=max_iters,
        catch_errors=True,
    )
    return mol
