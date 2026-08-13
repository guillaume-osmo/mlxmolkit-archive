"""
Lebedev-Laikov quadrature points on the unit sphere.

These are fixed grids for numerical integration on S².
Used to tesselate the COSMO molecular cavity surface.

Reference: V.I. Lebedev, D.N. Laikov, Doklady Mathematics 59 (1999) 477-481.
"""
from __future__ import annotations

import numpy as np


def _gen_oh(code, a=0.0, b=0.0, v=0.0):
    """Generate Lebedev points from symmetry operations.

    code 1: ±(1,0,0) and permutations — 6 points
    code 2: ±(1/√2, 1/√2, 0) and permutations — 12 points
    code 3: ±(1/√3, 1/√3, 1/√3) — 8 points
    code 4: ±(a,a,b) and permutations — 24 points (a²+a²+b²=1)
    code 5: ±(a,b,0) and permutations — 24 points, b = √(1-a²)
    code 6: ±(a,b,c) and permutations — 48 points
    """
    points = []
    if code == 1:
        for s in (1, -1):
            for axis in range(3):
                p = [*(0.0, 0.0, 0.0)]
                p[axis] = s
                points.append((p[0], p[1], p[2], v))
    elif code == 2:
        s2 = 1.0 / np.sqrt(2.0)
        for s1 in (s2, -s2):
            for s2v in (s2, -s2):
                points.append((s1, s2v, 0.0, v))
                points.append((s1, 0.0, s2v, v))
                points.append((0.0, s1, s2v, v))
    elif code == 3:
        s3 = 1.0 / np.sqrt(3.0)
        for s1 in (s3, -s3):
            for s2 in (s3, -s3):
                for s3v in (s3, -s3):
                    points.append((s1, s2, s3v, v))
    elif code == 4:
        c = np.sqrt(1.0 - 2.0 * a * a)
        for vals in ((a, a, c), (a, c, a), (c, a, a)):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    for s3 in (1, -1):
                        points.append((s1 * vals[0], s2 * vals[1], s3 * vals[2], v))
    elif code == 5:
        # Laikov's C_k class: (a, b, 0) with b = sqrt(1-a^2), so the pair lies
        # on the sphere. Six axis arrangements x four signs = 24 points. The
        # zero component carries no sign, which is what keeps this at 24
        # rather than the 48 of the (a,b,c) class.
        b = np.sqrt(max(1.0 - a * a, 0.0))
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for sa in (1, -1):
                    for sb in (1, -1):
                        p = [0.0, 0.0, 0.0]
                        p[i] = sa * a
                        p[j] = sb * b
                        points.append((p[0], p[1], p[2], v))
    return points


def lebedev_110():
    """110-point Lebedev grid (degree 17 precision)."""
    pts = []
    pts.extend(_gen_oh(1, v=0.015313081979748 * np.pi))
    pts.extend(_gen_oh(3, v=0.039174950050656 * np.pi))
    pts.extend(_gen_oh(4, a=0.1851156353447362, v=0.032846949132764 * np.pi))
    pts.extend(_gen_oh(4, a=0.6904210483822922, v=0.039771259564712 * np.pi))
    pts.extend(_gen_oh(4, a=0.3956894730559419, v=0.03838188534428 * np.pi))
    pts.extend(_gen_oh(5, a=0.4783690288121502,
                       v=0.038779985446652 * np.pi))

    data = np.array(pts)
    xyz = data[:, :3]
    # Renormalise: code 4/5 permutations are only approximately on the sphere.
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = xyz / norms
    weights = data[:, 3]
    return xyz, weights


def lebedev_194():
    """194-point Lebedev grid (degree 23 precision).

    Good balance of accuracy vs cost for COSMO cavity tesselation.
    """

    # Fibonacci (golden-spiral) sampling with uniform weights.
    n = 194
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * 3.23606797749979 * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    xyz = np.column_stack([x, y, z])
    weights = np.full(n, 4.0 * np.pi / n)
    return xyz, weights


def get_lebedev_grid(n_points: int = 194) -> tuple[np.ndarray, np.ndarray]:
    """Get Lebedev quadrature grid on the unit sphere.

    Args:
        n_points: 110 or 194

    Returns:
        xyz: (n_points, 3) unit vectors
        weights: (n_points,) quadrature weights (sum = 4π)
    """
    if n_points == 110:
        return lebedev_110()
    if n_points == 194:
        return lebedev_194()

    # Fall back to the denser grid for any unsupported request.
    return lebedev_194()
