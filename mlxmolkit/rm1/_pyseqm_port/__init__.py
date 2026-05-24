"""Vendored NumPy port of selected PYSEQM modules (BSD-3 Clause, LANL).

Source: https://github.com/lanl/PYSEQM

Currently vendored:
  - ``constants_np`` — minimal stand-in for PYSEQM's Constants class
    (just qn_int / qnD_int / physical constants).
  - ``diat_overlapD_np`` — mechanical torch->numpy port of
    ``seqm/seqm_functions/diat_overlapD.py``. Implements the full
    sp + d-orbital diatomic overlap for all PYSEQM-supported pairs
    (qn=1..6, including qn=4 Br and qn=5 I).

Pending (TETCI port — still requires PYSEQM fallback in d_two_center.py):
  - ``two_elec_two_center_int_local_frame_d_orbitals`` needs custom
    torch.autograd.Function ports + the RotationMatrixD chain.
"""
