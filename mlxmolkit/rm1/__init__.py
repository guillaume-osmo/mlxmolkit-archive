"""
RM1 (Recife Model 1) semi-empirical quantum chemistry on Apple Metal GPU.

A reparameterization of AM1 for H, C, N, O, P, S, F, Cl, Br, I.
GPU-accelerated via MLX/Metal for Fock matrix construction and SCF.

Reference:
  Rocha et al., J. Comput. Chem. 2006, 27, 1101-1111.
  Parameters from MOPAC (Apache 2.0): https://github.com/openmopac/mopac

Usage:
    from mlxmolkit.rm1 import rm1_energy
    energy = rm1_energy(atoms, coords)
"""

from .params import RM1_PARAMS
from .scf import rm1_energy
from .gradient import rm1_gradient, rm1_optimize
