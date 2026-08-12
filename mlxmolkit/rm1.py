"""Backwards-compatible alias for :mod:`mlxmolkit.nddo`.

The package was called ``rm1`` when it only implemented RM1. It now carries
AM1, AM1_STAR, PM3, PM6, PM6_D, RM1 and RM1_STAR, so it is named for the
theory they share (NDDO) rather than for one of them.

``import mlxmolkit.rm1`` keeps working, and so does every submodule path under
it. Both resolve to the *same* module objects as ``mlxmolkit.nddo`` — aliasing
rather than re-importing, because a second copy of e.g. ``scf`` would carry its
own module-level parameter tables and caches, and the two would drift.
"""
import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings

from mlxmolkit import nddo as _nddo

_OLD = __name__          # "mlxmolkit.rm1"
_NEW = _nddo.__name__    # "mlxmolkit.nddo"


class _AliasLoader(importlib.abc.Loader):
    """Resolves an ``mlxmolkit.rm1.X`` request to the live ``nddo.X`` module."""

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD):])

    def exec_module(self, module):
        """Already executed under its real name — nothing more to do."""


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_OLD + "."):
            return importlib.machinery.ModuleSpec(fullname, _AliasLoader())
        return None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn(
    "mlxmolkit.rm1 has been renamed to mlxmolkit.nddo; the old name will be "
    "removed in a future release",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[_OLD] = _nddo
