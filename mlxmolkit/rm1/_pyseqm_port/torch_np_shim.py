"""Torch -> NumPy shim for vendored PYSEQM modules.

Replaces ``import torch as th`` with this module. NumPy supports
boolean indexing and in-place mutation natively, so the converted
PYSEQM code runs without semantic rewrites — only API substitutions
are needed.

For ``torch.autograd.Function`` we provide a forward-only stub
(SCF doesn't need backward gradients).

Tensor methods that NumPy lacks (.clamp_min, .unsqueeze, .nonzero,
.clone, .to, .detach, .numpy, .item) are provided as module-level
helpers; the source converter rewrites call sites accordingly.
"""
from __future__ import annotations

import numpy as _np

# --- direct re-exports ---
abs = _np.abs
exp = _np.exp
sqrt = _np.sqrt
where = _np.where
maximum = _np.maximum
minimum = _np.minimum
def _shape_arg(args, kwargs):
    """torch.zeros(a, b, dtype=...) and torch.zeros((a, b), dtype=...) both work;
    NumPy only accepts the tuple form. Detect and normalize."""
    if len(args) == 0:
        return (), kwargs
    if isinstance(args[0], (tuple, list)):
        return args, kwargs
    # All positional args before any non-int look like the shape
    if all(isinstance(a, (int, _np.integer)) for a in args):
        return ((tuple(int(a) for a in args),), kwargs)
    return args, kwargs

def zeros(*args, dtype=None, **kwargs):
    a, k = _shape_arg(args, {})
    if dtype is not None:
        k['dtype'] = dtype
    return _np.zeros(*a, **k)

def ones(*args, dtype=None, **kwargs):
    a, k = _shape_arg(args, {})
    if dtype is not None:
        k['dtype'] = dtype
    return _np.ones(*a, **k)

def empty(*args, dtype=None, **kwargs):
    a, k = _shape_arg(args, {})
    if dtype is not None:
        k['dtype'] = dtype
    return _np.empty(*a, **k)

zeros_like = _np.zeros_like
ones_like = _np.ones_like
def stack(tensors, dim=None, axis=None):
    if axis is None:
        axis = dim if dim is not None else 0
    return _np.stack(tensors, axis=axis)
diag = _np.diag
arange = _np.arange
clip = _np.clip
einsum = _np.einsum
expand_dims = _np.expand_dims
swapaxes = _np.swapaxes
tile = _np.tile
def concatenate(tensors, dim=None, axis=None):
    """Accept both torch `dim` and numpy `axis` kwargs."""
    if axis is None:
        axis = dim if dim is not None else 0
    return _np.concatenate(tensors, axis=axis)

cat = concatenate  # torch.cat alias
power = _np.power
pow = _np.power  # torch.pow alias

def tensor(x, dtype=None, device=None):
    """torch.tensor — drop device, honor dtype."""
    return _np.asarray(x, dtype=dtype)

def as_tensor(x, dtype=None, device=None):
    return _np.asarray(x, dtype=dtype)

def from_numpy(x):
    return _np.asarray(x)

# dtypes
float16 = _np.float16
float32 = _np.float32
float64 = _np.float64
int8 = _np.int8
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
bool_ = _np.bool_

# linalg sub-namespace
class _linalg:
    norm = staticmethod(_np.linalg.norm)
    eigh = staticmethod(_np.linalg.eigh)
linalg = _linalg

def norm(x, dim=None, axis=None, p=None, keepdim=False, keepdims=None):
    ax = axis if axis is not None else dim
    kd = keepdims if keepdims is not None else keepdim
    if ax is None:
        return _np.linalg.norm(x)
    return _np.linalg.norm(x, axis=ax, keepdims=kd)


# Method substitutes — called by rewritten source.
def clone(x):
    return _np.asarray(x).copy()

def clamp_min(x, vmin):
    return _np.maximum(x, vmin)

def clamp_max(x, vmax):
    return _np.minimum(x, vmax)

def clamp(x, vmin=None, vmax=None):
    if vmin is not None: x = _np.maximum(x, vmin)
    if vmax is not None: x = _np.minimum(x, vmax)
    return x

def unsqueeze(x, dim):
    return _np.expand_dims(x, dim)

def nonzero(x, as_tuple=False):
    idx = _np.nonzero(x)
    if as_tuple:
        return idx
    return _np.stack(idx, axis=-1)

def numel(x):
    return _np.asarray(x).size

def transpose(x, dim0, dim1):
    return _np.swapaxes(x, dim0, dim1)

def bmm(a, b):
    return _np.einsum('bij,bjk->bik', a, b)

def matmul(a, b):
    return _np.matmul(a, b)

def sum(x, dim=None, axis=None, keepdim=False, keepdims=None):
    ax = axis if axis is not None else dim
    kd = keepdims if keepdims is not None else keepdim
    return _np.sum(x, axis=ax, keepdims=kd)

def mean(x, dim=None, axis=None, keepdim=False, keepdims=None):
    ax = axis if axis is not None else dim
    kd = keepdims if keepdims is not None else keepdim
    return _np.mean(x, axis=ax, keepdims=kd)

def matmul_x(a, b):
    return _np.matmul(a, b)

def is_tensor(x):
    return isinstance(x, _np.ndarray)


class _NoOpCtx:
    """Stand-in for autograd ctx; ignores save_for_backward (no backward needed)."""
    def save_for_backward(self, *args, **kwargs): pass
    def __setattr__(self, name, value): pass  # ignore ctx.flag = x


# torch.autograd.Function — forward-only stub
class _AutogradFunction:
    @classmethod
    def apply(cls, *args, **kwargs):
        return cls.forward(_NoOpCtx(), *args, **kwargs)

class _Autograd:
    Function = _AutogradFunction

autograd = _Autograd

# torch.nn shim
class _NN:
    Module = object
nn = _NN

# Misc: get/set_default_dtype no-ops (numpy doesn't have global dtype)
def get_default_dtype():
    return _np.float64

def set_default_dtype(dt):
    pass

# torch.is_tensor — for code that branches on tensor type
def is_tensor(x):
    return isinstance(x, _np.ndarray)
