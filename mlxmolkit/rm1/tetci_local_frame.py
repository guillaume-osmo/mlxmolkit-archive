"""Two-electron two-center integrals in the LOCAL (bond) frame.

Port of PYSEQM ``two_elec_two_center_int_local_frame`` (BSD-3-Clause,
github.com/lanl/PYSEQM, seqm/seqm_functions/two_elec_two_center_int_local_frame.py).
Step 3a in the TETCI port roadmap.

Computes the 22 unique NDDO two-electron integrals between two atoms
in the local (bond) frame where the z-axis is aligned with the
internuclear vector. These are then rotated to the molecular frame
by :func:`mlxmolkit.rm1.tetci_w.w_withquaternion` (sp-only path) or
by the d-orbital extension in the full ``rotate()`` (pending).

The 22 integrals follow MOPAC's convention (see ``repp.f`` /
``rotate.f``):

    (SS|SS) (SO|SS) (OO|SS) (PP|SS) (SS|OS) (SO|SO) (SP|SP) (OO|SO)
    (PP|SO) (PO|SP) (SS|OO) (SS|PP) (SO|OO) (SO|PP) (SP|OP) (OO|OO)
    (PP|OO) (OO|PP) (PP|PP) (PO|PO) (PP|P*P*) (P*P|P*P)

where P-SIGMA = O, P-PI = P/P*.

The 4 X-H integrals are subsets of the above for the H s-only basis;
the single H-H integral is just (SS|SS).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EV = 27.21  # PYSEQM constant (constants.py:4) — used in MOPAC7


def two_elec_two_center_int_local_frame(
    ni: NDArray[np.int64],
    nj: NDArray[np.int64],
    r0: NDArray[np.float64],
    tore: NDArray[np.float64],
    da0: NDArray[np.float64],
    db0: NDArray[np.float64],
    qa0: NDArray[np.float64],
    qb0: NDArray[np.float64],
    rho0a: NDArray[np.float64],
    rho0b: NDArray[np.float64],
    rho1a: NDArray[np.float64],
    rho1b: NDArray[np.float64],
    rho2a: NDArray[np.float64],
    rho2b: NDArray[np.float64],
    themethod: str = "PM6",
) -> tuple[
    NDArray[np.float64],  # riHH:  (n_HH,)
    NDArray[np.float64],  # riXH:  (n_XH, 4)
    NDArray[np.float64],  # ri:    (n_XX, 22)
    NDArray[np.float64],  # coreHH: (n_HH, 2)
    NDArray[np.float64],  # coreXH: (n_XH, 5)
    NDArray[np.float64],  # core:   (n_XX, 8)
]:
    """Compute local-frame two-electron + nuclear-attraction integrals.

    Parameters
    ----------
    ni, nj : (n_pairs,) int
        Atomic numbers, A and B.
    r0 : (n_pairs,) float
        Interatomic distance in Bohr.
    tore : (n_elem+1,) float
        Valence electron count indexed by atomic number.
    da0, db0 : (n_pairs,) float
        Dipole charge separations (s-p) for A and B.
    qa0, qb0 : (n_pairs,) float
        Quadrupole charge separations (p-p) for A and B (will be
        doubled internally — PYSEQM convention).
    rho0a/b, rho1a/b, rho2a/b : (n_pairs,) float
        Klopman-Ohno additive terms for s, sp-dipole, p-quadrupole.
    themethod : str
        Method name (kept for API compatibility, behaviour identical
        across NDDO methods in the local frame).

    Returns
    -------
    riHH, riXH, ri, coreHH, coreXH, core
        Mask-restricted arrays (no padding). Compose with HH/XH/XX
        masks downstream.
    """
    n_pairs = r0.shape[0]
    dtype = r0.dtype

    ev1 = EV / 2.0
    ev2 = EV / 4.0
    ev3 = EV / 8.0
    ev4 = EV / 16.0

    HH = (ni == 1) & (nj == 1)
    XH = (ni > 1) & (nj == 1)
    XX = (ni > 1) & (nj > 1)
    n_HH = int(HH.sum())
    n_XH = int(XH.sum())
    n_XX = int(XX.sum())

    # ── HH ──────────────────────────────────────────────────────────
    if n_HH > 0:
        riHH = EV / np.sqrt(r0[HH] ** 2 + (rho0a[HH] + rho0b[HH]) ** 2)
        coreHH = np.zeros((n_HH, 2), dtype=dtype)
        coreHH[:, 0] = tore[1] * riHH
        coreHH[:, 1] = tore[1] * riHH
    else:
        riHH = np.zeros(0, dtype=dtype)
        coreHH = np.zeros((0, 2), dtype=dtype)

    # ── XH (heavy + hydrogen) ───────────────────────────────────────
    if n_XH > 0:
        aeeXH = (rho0a[XH] + rho0b[XH]) ** 2
        rXH = r0[XH]
        daXH = da0[XH]
        qaXH = qa0[XH] * 2.0
        adeXH = (rho1a[XH] + rho0b[XH]) ** 2
        aqeXH = (rho2a[XH] + rho0b[XH]) ** 2
        ev1dsqr6XH = ev1 / np.sqrt(rXH ** 2 + aqeXH)
        riXH = np.zeros((n_XH, 4), dtype=dtype)
        eeXH = EV / np.sqrt(rXH ** 2 + aeeXH)
        riXH[:, 0] = eeXH
        riXH[:, 1] = (
            ev1 / np.sqrt((rXH + daXH) ** 2 + adeXH)
            - ev1 / np.sqrt((rXH - daXH) ** 2 + adeXH)
        )
        riXH[:, 2] = (
            eeXH
            + ev2 / np.sqrt((rXH + qaXH) ** 2 + aqeXH)
            + ev2 / np.sqrt((rXH - qaXH) ** 2 + aqeXH)
            - ev1dsqr6XH
        )
        riXH[:, 3] = eeXH + ev1 / np.sqrt(rXH ** 2 + qaXH ** 2 + aqeXH) - ev1dsqr6XH

        coreXH = np.zeros((n_XH, 5), dtype=dtype)
        coreXH[:, 0] = tore[1] * riXH[:, 0]
        coreXH[:, 1] = tore[1] * riXH[:, 1]
        coreXH[:, 2] = tore[1] * riXH[:, 2]
        coreXH[:, 3] = tore[1] * riXH[:, 3]
        coreXH[:, 4] = tore[ni[XH]] * riXH[:, 0]
    else:
        riXH = np.zeros((0, 4), dtype=dtype)
        coreXH = np.zeros((0, 5), dtype=dtype)

    # ── XX (heavy + heavy) — full 22 integrals ──────────────────────
    if n_XX > 0:
        r = r0[XX]
        da = da0[XX]
        db = db0[XX]
        qa = qa0[XX] * 2.0
        qb = qb0[XX] * 2.0
        qa1 = qa0[XX]
        qb1 = qb0[XX]
        ri = np.zeros((n_XX, 22), dtype=dtype)
        core = np.zeros((n_XX, 8), dtype=dtype)

        # Repeated terms
        aee = (rho0a[XX] + rho0b[XX]) ** 2
        ade = (rho1a[XX] + rho0b[XX]) ** 2
        aqe = (rho2a[XX] + rho0b[XX]) ** 2
        aed = (rho0a[XX] + rho1b[XX]) ** 2
        aeq = (rho0a[XX] + rho2b[XX]) ** 2
        axx = (rho1a[XX] + rho1b[XX]) ** 2
        adq = (rho1a[XX] + rho2b[XX]) ** 2
        aqd = (rho2a[XX] + rho1b[XX]) ** 2
        aqq = (rho2a[XX] + rho2b[XX]) ** 2

        ee = EV / np.sqrt(r ** 2 + aee)
        dze = -ev1 / np.sqrt((r + da) ** 2 + ade) + ev1 / np.sqrt((r - da) ** 2 + ade)
        ev1dsqr6 = ev1 / np.sqrt(r ** 2 + aqe)
        qzze = (
            ev2 / np.sqrt((r - qa) ** 2 + aqe)
            + ev2 / np.sqrt((r + qa) ** 2 + aqe)
            - ev1dsqr6
        )
        qxxe = ev1 / np.sqrt(r ** 2 + qa ** 2 + aqe) - ev1dsqr6
        edz = -ev1 / np.sqrt((r - db) ** 2 + aed) + ev1 / np.sqrt((r + db) ** 2 + aed)
        ev1dsqr12 = ev1 / np.sqrt(r ** 2 + aeq)
        eqzz = (
            ev2 / np.sqrt((r - qb) ** 2 + aeq)
            + ev2 / np.sqrt((r + qb) ** 2 + aeq)
            - ev1dsqr12
        )
        eqxx = ev1 / np.sqrt(r ** 2 + qb ** 2 + aeq) - ev1dsqr12

        ev2dsqr20 = ev2 / np.sqrt((r + da) ** 2 + adq)
        ev2dsqr22 = ev2 / np.sqrt((r - da) ** 2 + adq)
        ev2dsqr24 = ev2 / np.sqrt((r - db) ** 2 + aqd)
        ev2dsqr26 = ev2 / np.sqrt((r + db) ** 2 + aqd)
        ev2dsqr36 = ev2 / np.sqrt(r ** 2 + aqq)
        ev2dsqr39 = ev2 / np.sqrt(r ** 2 + qa ** 2 + aqq)
        ev2dsqr40 = ev2 / np.sqrt(r ** 2 + qb ** 2 + aqq)
        ev3dsqr42 = ev3 / np.sqrt((r - qb) ** 2 + aqq)
        ev3dsqr44 = ev3 / np.sqrt((r + qb) ** 2 + aqq)
        ev3dsqr46 = ev3 / np.sqrt((r + qa) ** 2 + aqq)
        ev3dsqr48 = ev3 / np.sqrt((r - qa) ** 2 + aqq)

        # All ri indices shifted by 1 to be 0-based
        ri[:, 0] = ee                                             # (SS|SS)
        ri[:, 1] = -dze                                            # (SO|SS)
        ri[:, 2] = ee + qzze                                       # (OO|SS)
        ri[:, 3] = ee + qxxe                                       # (PP|SS)
        ri[:, 4] = -edz                                            # (SS|OS)
        ri[:, 5] = (                                              # (SO|SO) = DZDZ
            ev2 / np.sqrt((r + da - db) ** 2 + axx)
            + ev2 / np.sqrt((r - da + db) ** 2 + axx)
            - ev2 / np.sqrt((r - da - db) ** 2 + axx)
            - ev2 / np.sqrt((r + da + db) ** 2 + axx)
        )
        ri[:, 6] = (                                              # (SP|SP) = DXDX
            ev1 / np.sqrt(r ** 2 + (da - db) ** 2 + axx)
            - ev1 / np.sqrt(r ** 2 + (da + db) ** 2 + axx)
        )
        ri[:, 7] = (                                              # (OO|SO) = -EDZ -QZZDZ
            -edz
            + ev3 / np.sqrt((r + qa - db) ** 2 + aqd)
            - ev3 / np.sqrt((r + qa + db) ** 2 + aqd)
            + ev3 / np.sqrt((r - qa - db) ** 2 + aqd)
            - ev3 / np.sqrt((r - qa + db) ** 2 + aqd)
            - ev2dsqr24
            + ev2dsqr26
        )
        ri[:, 8] = (                                              # (PP|SO) = -EDZ -QXXDZ
            -edz
            - ev2dsqr24
            + ev2 / np.sqrt((r - db) ** 2 + qa ** 2 + aqd)
            + ev2dsqr26
            - ev2 / np.sqrt((r + db) ** 2 + qa ** 2 + aqd)
        )
        ri[:, 9] = (                                              # (PO|SP) = -QXZDX
            ev2 / np.sqrt((qa1 - db) ** 2 + (r + qa1) ** 2 + aqd)
            - ev2 / np.sqrt((qa1 - db) ** 2 + (r - qa1) ** 2 + aqd)
            - ev2 / np.sqrt((qa1 + db) ** 2 + (r + qa1) ** 2 + aqd)
            + ev2 / np.sqrt((qa1 + db) ** 2 + (r - qa1) ** 2 + aqd)
        )
        ri[:, 10] = ee + eqzz                                      # (SS|OO)
        ri[:, 11] = ee + eqxx                                      # (SS|PP)
        ri[:, 12] = (                                              # (SO|OO) = -DZE -DZQZZ
            -dze
            + ev3 / np.sqrt((r + da - qb) ** 2 + adq)
            - ev3 / np.sqrt((r - da - qb) ** 2 + adq)
            + ev3 / np.sqrt((r + da + qb) ** 2 + adq)
            - ev3 / np.sqrt((r - da + qb) ** 2 + adq)
            + ev2dsqr22
            - ev2dsqr20
        )
        ri[:, 13] = (                                              # (SO|PP) = -DZE -DZQXX
            -dze
            - ev2dsqr20
            + ev2 / np.sqrt((r + da) ** 2 + qb ** 2 + adq)
            + ev2dsqr22
            - ev2 / np.sqrt((r - da) ** 2 + qb ** 2 + adq)
        )
        ri[:, 14] = (                                              # (SP|OP) = -DXQXZ
            ev2 / np.sqrt((da - qb1) ** 2 + (r - qb1) ** 2 + adq)
            - ev2 / np.sqrt((da - qb1) ** 2 + (r + qb1) ** 2 + adq)
            - ev2 / np.sqrt((da + qb1) ** 2 + (r - qb1) ** 2 + adq)
            + ev2 / np.sqrt((da + qb1) ** 2 + (r + qb1) ** 2 + adq)
        )
        ri[:, 15] = (                                              # (OO|OO)
            ee + eqzz + qzze
            + ev4 / np.sqrt((r + qa - qb) ** 2 + aqq)
            + ev4 / np.sqrt((r + qa + qb) ** 2 + aqq)
            + ev4 / np.sqrt((r - qa - qb) ** 2 + aqq)
            + ev4 / np.sqrt((r - qa + qb) ** 2 + aqq)
            - ev3dsqr48
            - ev3dsqr46
            - ev3dsqr42
            - ev3dsqr44
            + ev2dsqr36
        )
        ri[:, 16] = (                                              # (PP|OO)
            ee + eqzz + qxxe
            + ev3 / np.sqrt((r - qb) ** 2 + qa ** 2 + aqq)
            + ev3 / np.sqrt((r + qb) ** 2 + qa ** 2 + aqq)
            - ev3dsqr42 - ev3dsqr44 - ev2dsqr39 + ev2dsqr36
        )
        ri[:, 17] = (                                              # (OO|PP)
            ee + eqxx + qzze
            + ev3 / np.sqrt((r + qa) ** 2 + qb ** 2 + aqq)
            + ev3 / np.sqrt((r - qa) ** 2 + qb ** 2 + aqq)
            - ev3dsqr46 - ev3dsqr48 - ev2dsqr40 + ev2dsqr36
        )
        qxxqxx = (
            ev3 / np.sqrt(r ** 2 + (qa - qb) ** 2 + aqq)
            + ev3 / np.sqrt(r ** 2 + (qa + qb) ** 2 + aqq)
            - ev2dsqr39 - ev2dsqr40 + ev2dsqr36
        )
        ri[:, 18] = ee + eqxx + qxxe + qxxqxx                      # (PP|PP)
        ri[:, 19] = (                                              # (PO|PO) = QXZQXZ
            ev3 / np.sqrt((r + qa1 - qb1) ** 2 + (qa1 - qb1) ** 2 + aqq)
            - ev3 / np.sqrt((r + qa1 + qb1) ** 2 + (qa1 - qb1) ** 2 + aqq)
            - ev3 / np.sqrt((r - qa1 - qb1) ** 2 + (qa1 - qb1) ** 2 + aqq)
            + ev3 / np.sqrt((r - qa1 + qb1) ** 2 + (qa1 - qb1) ** 2 + aqq)
            - ev3 / np.sqrt((r + qa1 - qb1) ** 2 + (qa1 + qb1) ** 2 + aqq)
            + ev3 / np.sqrt((r + qa1 + qb1) ** 2 + (qa1 + qb1) ** 2 + aqq)
            + ev3 / np.sqrt((r - qa1 - qb1) ** 2 + (qa1 + qb1) ** 2 + aqq)
            - ev3 / np.sqrt((r - qa1 + qb1) ** 2 + (qa1 + qb1) ** 2 + aqq)
        )
        qxxqyy = (
            ev2 / np.sqrt(r ** 2 + qa ** 2 + qb ** 2 + aqq)
            - ev2dsqr39 - ev2dsqr40 + ev2dsqr36
        )
        ri[:, 20] = ee + eqxx + qxxe + qxxqyy                      # (PP|P*P*)
        ri[:, 21] = 0.5 * (qxxqxx - qxxqyy)                        # (P*P|P*P)

        core[:, 0] = tore[nj[XX]] * ri[:, 0]
        core[:, 1] = tore[nj[XX]] * ri[:, 1]
        core[:, 2] = tore[nj[XX]] * ri[:, 2]
        core[:, 3] = tore[nj[XX]] * ri[:, 3]
        core[:, 4] = tore[ni[XX]] * ri[:, 0]
        core[:, 5] = tore[ni[XX]] * ri[:, 4]
        core[:, 6] = tore[ni[XX]] * ri[:, 10]
        core[:, 7] = tore[ni[XX]] * ri[:, 11]
    else:
        ri = np.zeros((0, 22), dtype=dtype)
        core = np.zeros((0, 8), dtype=dtype)

    return riHH, riXH, ri, coreHH, coreXH, core
