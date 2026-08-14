"""Every runtime data file must be matched by a package-data glob.

`mlxmolkit/rm1/` was renamed to `mlxmolkit/nddo/` in 6980e77, but the
package-data globs in pyproject.toml kept pointing at `rm1/data/*.csv`. They
then matched nothing, and the built wheel shipped 45 nddo modules with zero
parameter files — so `pip install` produced a package that could not load a
single PM6/AM1/PM3/RM1 parameter set.

Nothing caught it because the test suite runs from the source tree, where the
files are present whether or not packaging knows about them. These tests check
the packaging declaration itself, which is the only thing a wheel is built from.
"""
from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "mlxmolkit"

# Extensions that are loaded at runtime rather than imported. A .py file is
# picked up by package discovery; these are not.
DATA_SUFFIXES = {".csv", ".npz", ".json", ".metal", ".txt"}


def _package_data_globs() -> list[str]:
    with open(ROOT / "pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)
    globs = config["tool"]["setuptools"]["package-data"]["mlxmolkit"]
    assert globs, "no package-data declared for mlxmolkit"
    return globs


def _runtime_data_files() -> list[Path]:
    """Data files under mlxmolkit/, relative to the package directory."""
    return sorted(
        path.relative_to(PACKAGE)
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and path.suffix in DATA_SUFFIXES
        and "__pycache__" not in path.parts
    )


def test_every_data_file_is_covered_by_a_glob():
    globs = _package_data_globs()
    uncovered = [
        str(rel)
        for rel in _runtime_data_files()
        if not any(fnmatch.fnmatch(str(rel), pattern) for pattern in globs)
    ]
    assert not uncovered, (
        "these files ship in the source tree but no package-data glob matches "
        f"them, so they will be missing from the wheel: {uncovered}\n"
        f"declared globs: {globs}"
    )


def test_no_glob_matches_nothing():
    """A glob matching nothing is the signature of a renamed directory."""
    files = [str(rel) for rel in _runtime_data_files()]
    dead = [
        pattern
        for pattern in _package_data_globs()
        if not any(fnmatch.fnmatch(name, pattern) for name in files)
    ]
    # data/bcc/*.json is knowingly dead: mlxmolkit/data/ has no tracked files
    # and the AM1-BCC table is absent from every clone (issue #60). Keeping the
    # glob is correct — it is where the file belongs once it is vendored.
    dead = [pattern for pattern in dead if pattern != "data/bcc/*.json"]
    assert not dead, (
        f"package-data globs that match no file: {dead} — a directory was "
        "probably renamed without updating pyproject.toml"
    )


@pytest.mark.parametrize(
    "required",
    ["nddo/data/parameters_PM6_MOPAC.csv", "nddo/data/r0ab_d3.npz"],
)
def test_named_parameter_files_are_packaged(required):
    """Spot-check the two the NDDO methods cannot run without."""
    assert (PACKAGE / required).exists(), f"{required} missing from the tree"
    globs = _package_data_globs()
    assert any(fnmatch.fnmatch(required, pattern) for pattern in globs), (
        f"{required} is not covered by {globs}"
    )
