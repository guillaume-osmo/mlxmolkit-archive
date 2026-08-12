"""Minimal Constants helpers — periodic-table-row arrays used by the
vendored PYSEQM diat/TETCI ports.

Only ``qn_int`` (principal qn for valence shell) and ``qnD_int``
(principal qn for d-orbital shell) are used by diatom_overlap_matrixD
and the d-orbital TETCI routines.

Original PYSEQM Constants class wraps these in a torch.nn.Module, which
is irrelevant for the NumPy port. Tables reproduced verbatim from
``seqm/seqm_functions/constants.py`` (BSD-3 Clause).
"""
import numpy as np

# PYSEQM uses MOPAC7-vintage values for these constants
ev = 27.21
a0 = 0.529167
ev_kcalpmol = 23.061
overlap_cutoff = 40.0
charge_on_electron = 1.60217733e-19
speed_of_light = 2.99792458e8
to_debye = charge_on_electron * 1e-10 * speed_of_light / 1e-21
debye_to_AU = 0.393456


# Principal quantum number of the valence sp shell, indexed by Z.
# qn_int[Z] == row of periodic table for the sp shell.
qn_int = np.asarray([0,
    1,                                                                          1,
    2, 2,                                                                       2, 2, 2, 2, 2, 2,
    3, 3,                                                                       3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
], dtype=np.int64)


# Principal quantum number for the d shell, indexed by Z.
# qnD_int[Z] == 0 for atoms without d-orbitals; otherwise the d-shell row.
qnD_int = np.asarray([0,
    0,                                                                          0,
    0, 0,                                                                       0, 0, 0, 0, 0, 0,
    0, 0,                                                                       3, 3, 3, 3, 3, 0,
    0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0,
    0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0,
    0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 0,
], dtype=np.int64)


class Constants:
    """Minimal stand-in for the PYSEQM Constants class used by the
    vendored numpy ports — only exposes ``qn_int`` and ``qnD_int``."""
    qn_int = qn_int
    qnD_int = qnD_int

    def __init__(self, *args, **kwargs):
        pass
