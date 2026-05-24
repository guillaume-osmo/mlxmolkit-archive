"""Torch -> MLX shim for the vendored PYSEQM modules.

Vendored PYSEQM files do ``import torch as th`` and use torch APIs.
Most torch functions have a one-to-one MLX equivalent, so this shim
re-exports MLX under the torch namespace.

For tensor *methods* that MLX lacks (.clone, .clamp_min, .unsqueeze,
.nonzero, ...), the source-conversion pass in convert_torch_to_mlx.py
rewrites the call sites to the equivalent MLX function call. The shim
adds module-level helpers for those.

For ``torch.autograd.Function`` we provide a forward-only stub — SCF
doesn't need backward, and MLX has its own autograd via ``mx.grad`` if
gradients are ever required.
"""
from __future__ import annotations

import mlx.core as _mx
import numpy as _np

# --- module-level re-exports (the easy 95%) ---
def _ensure_mx(x):
    """Auto-promote numpy arrays / Python scalars to mx.array."""
    if isinstance(x, _mx.array):
        return x
    return _mx.array(_np.asarray(x))


abs = _mx.abs
exp = _mx.exp
sqrt = _mx.sqrt
where = lambda c, a, b: _mx.where(_ensure_mx(c), _ensure_mx(a), _ensure_mx(b))
maximum = _mx.maximum
minimum = _mx.minimum
zeros = _mx.zeros
ones = _mx.ones
zeros_like = lambda x: _mx.zeros_like(_ensure_mx(x))
ones_like = lambda x: _mx.ones_like(_ensure_mx(x))
stack = _mx.stack
diag = _mx.diag
arange = _mx.arange
clip = _mx.clip
einsum = _mx.einsum
expand_dims = _mx.expand_dims
swapaxes = _mx.swapaxes
tile = _mx.tile
concatenate = _mx.concatenate
cat = _mx.concatenate  # torch.cat alias
power = _mx.power
pow = _mx.power  # torch.pow alias
tensor = _mx.array
as_tensor = _mx.array

# dtypes
float16 = _mx.float16
float32 = _mx.float32
float64 = _mx.float64
int8 = _mx.int8
int16 = _mx.int16
int32 = _mx.int32
int64 = _mx.int64
bool_ = _mx.bool_

# Sub-namespace shims
class _linalg:
    norm = _mx.linalg.norm
    eigh = _mx.linalg.eigh
linalg = _linalg

# torch.norm(x, dim=N) -> mx.linalg.norm(x, axis=N).
def norm(x, dim=None, p=None, keepdim=False):
    if dim is None:
        return _mx.linalg.norm(x)
    return _mx.linalg.norm(x, axis=dim, keepdims=keepdim)


# Method substitutes (called by rewritten source).
def clone(x):
    return x + 0  # forces a copy in MLX's lazy graph

def clamp_min(x, vmin):
    return _mx.maximum(x, vmin)

def clamp_max(x, vmax):
    return _mx.minimum(x, vmax)

def clamp(x, vmin=None, vmax=None):
    if vmin is not None:
        x = _mx.maximum(x, vmin)
    if vmax is not None:
        x = _mx.minimum(x, vmax)
    return x

def unsqueeze(x, dim):
    return _mx.expand_dims(x, dim)

def nonzero(x, as_tuple=False):
    """Return indices of nonzero elements. Compatible with torch's signature."""
    idx = _np.asarray(x).nonzero()  # MLX nonzero via numpy fallback (small tensors)
    if as_tuple:
        return tuple(_mx.array(a) for a in idx)
    return _mx.array(_np.stack(idx, axis=-1))

def numel(x):
    return x.size

def is_tensor(x):
    return isinstance(x, _mx.array)


# torch.autograd.Function stub — forward-only.
class _AutogradFunction:
    """SCF doesn't need backward; calling .apply(*args) invokes forward(None, *args)."""
    @classmethod
    def apply(cls, *args, **kwargs):
        return cls.forward(None, *args, **kwargs)

class _Autograd:
    Function = _AutogradFunction

autograd = _Autograd


# torch.nn.Module compatibility
class _NN:
    Module = object  # vendored code only uses this as a base class for the bag-of-data Constants

nn = _NN
