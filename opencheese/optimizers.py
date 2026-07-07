"""Optimizers used by openCHEESE training.

The MuonV2 implementation is adapted from the local
``lucnmr_predict.optimizers`` module on this workstation. It keeps the same
Polar Express/Jordan/Gram-NS update family and wraps it as Muon-for-2D plus
AdamW-for-everything-else, which is the practical recipe for transformer-like
models.
"""

from __future__ import annotations

from typing import Callable, Literal, Union

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_map_with_path


_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


def _neuron_wise_l2_norm(update: mx.array, eps: float = 1.0e-8) -> mx.array:
    row_norms = mx.sqrt(mx.sum(update * update, axis=1, keepdims=True) + eps)
    return update / row_norms


def _muon_plus_normalize(update: mx.array, mode: str, eps: float = 1.0e-8) -> mx.array:
    if mode == "none":
        return update
    if mode == "col":
        return update / (mx.sqrt(mx.sum(update * update, axis=0, keepdims=True)) + eps)
    if mode == "row":
        return update / (mx.sqrt(mx.sum(update * update, axis=1, keepdims=True)) + eps)
    if mode == "col_row":
        update = update / (mx.sqrt(mx.sum(update * update, axis=0, keepdims=True)) + eps)
        return update / (mx.sqrt(mx.sum(update * update, axis=1, keepdims=True)) + eps)
    if mode == "row_col":
        update = update / (mx.sqrt(mx.sum(update * update, axis=1, keepdims=True)) + eps)
        return update / (mx.sqrt(mx.sum(update * update, axis=0, keepdims=True)) + eps)
    raise ValueError(f"unknown MuonPlus mode: {mode!r}")


def _scaled_polar_coeffs(steps: int, safety: float) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for i, (a, b, c) in enumerate(_POLAR_EXPRESS_COEFFS):
        if i < len(_POLAR_EXPRESS_COEFFS) - 1:
            out.append((a / safety, b / (safety**3), c / (safety**5)))
        else:
            out.append((a, b, c))
    while len(out) < steps:
        out.append(out[-1])
    return out[:steps]


def polar_express(
    update: mx.array,
    *,
    steps: int = 5,
    safety: float = 1.01,
    eps: float = 1.0e-7,
) -> mx.array:
    """Polar Express approximation of the matrix polar factor."""

    if update.ndim != 2:
        raise ValueError(f"polar_express expects a 2D matrix, got {update.shape}")
    dtype = update.dtype
    x = update.astype(mx.float32)
    x = x / (mx.linalg.norm(x) * safety + eps)

    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    for a, b, c in _scaled_polar_coeffs(steps, safety):
        gram = x @ x.T
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x

    if transposed:
        x = x.T
    return x.astype(dtype)


def newton_schulz_jordan(update: mx.array, *, steps: int = 5, eps: float = 1.0e-7) -> mx.array:
    """Classic Muon Newton-Schulz/Jordan polynomial."""

    if update.ndim != 2:
        raise ValueError(f"newton_schulz_jordan expects a 2D matrix, got {update.shape}")
    dtype = update.dtype
    x = update.astype(mx.float32)
    x = x / (mx.linalg.norm(x) + eps)

    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x

    if transposed:
        x = x.T
    return x.astype(dtype)


def gram_newton_schulz(
    update: mx.array,
    *,
    steps: int = 5,
    safety: float = 1.01,
    eps: float = 1.0e-7,
    restart_at: tuple[int, ...] = (2,),
) -> mx.array:
    """Gram Newton-Schulz polar approximation for fat matrices."""

    if update.ndim != 2:
        raise ValueError(f"gram_newton_schulz expects a 2D matrix, got {update.shape}")
    dtype = update.dtype
    x = update.astype(mx.float32)
    x = x / (mx.linalg.norm(x) * safety + eps)

    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    n = x.shape[0]
    gram = x @ x.T
    q = mx.eye(n, dtype=x.dtype)
    for step, (a, b, c) in enumerate(_scaled_polar_coeffs(steps, safety)):
        if step in restart_at and step > 0:
            x = q @ x
            gram = x @ x.T
            q = mx.eye(n, dtype=x.dtype)
        gram2 = gram @ gram
        z = b * gram + c * gram2
        q = q @ z + a * q
        gram_z = gram @ z + a * gram
        gram = z @ gram_z + a * gram_z

    x = q @ x
    if transposed:
        x = x.T
    return x.astype(dtype)


def _apply_polar(
    update: mx.array,
    *,
    method: Literal["jordan", "polar_express", "gram_ns"],
    steps: int,
    safety: float,
) -> mx.array:
    if method == "polar_express":
        return polar_express(update, steps=steps, safety=safety)
    if method == "gram_ns":
        return gram_newton_schulz(update, steps=steps, safety=safety)
    if method == "jordan":
        return newton_schulz_jordan(update, steps=steps)
    raise ValueError(f"unknown MuonV2 polar method {method!r}")


class MuonV2(optim.Optimizer):
    """MuonV2 optimizer for 2D parameter tensors."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        *,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        polar_method: Literal["jordan", "polar_express", "gram_ns"] = "polar_express",
        polar_safety: float = 1.01,
        use_normuon: bool = False,
        normuon_eps: float = 1.0e-8,
        muonplus_mode: str = "none",
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self.polar_method = polar_method
        self.polar_safety = float(polar_safety)
        self.use_normuon = bool(use_normuon)
        self.normuon_eps = float(normuon_eps)
        self.muonplus_mode = str(muonplus_mode)

    def init_single(self, parameter: mx.array, state: dict):
        state["momentum"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        momentum = state["momentum"]
        momentum = self.momentum * momentum + gradient
        state["momentum"] = momentum

        update = gradient + self.momentum * momentum if self.nesterov else momentum
        if self.use_normuon:
            update = _neuron_wise_l2_norm(update, eps=self.normuon_eps)
        update = _apply_polar(
            update,
            method=self.polar_method,
            steps=self.ns_steps,
            safety=self.polar_safety,
        )
        if self.muonplus_mode != "none":
            update = _muon_plus_normalize(update, self.muonplus_mode)

        lr = self.learning_rate.astype(gradient.dtype)
        parameter = parameter - lr * update
        if self.weight_decay:
            parameter = parameter - lr * self.weight_decay * parameter
        return parameter


def _all_2d_filter(_path: str, value: mx.array) -> bool:
    return value.ndim == 2


def _opencheese_hidden_filter(path: str, value: mx.array) -> bool:
    if value.ndim != 2 or min(value.shape) < 16:
        return False
    return path.startswith("blocks.") or path.startswith("embedding_head.")


class MuonV2WOptimizer:
    """Path-aware MuonV2W optimizer for MLX module trees."""

    def __init__(
        self,
        *,
        muon_lr: float,
        adamw_lr: float,
        muon_momentum: float,
        muon_weight_decay: float,
        adamw_weight_decay: float,
        muon_ns_steps: int,
        muon_nesterov: bool,
        polar_method: Literal["jordan", "polar_express", "gram_ns"],
        polar_safety: float,
        use_normuon: bool,
        muonplus_mode: str,
        adamw_betas: tuple[float, float],
        adamw_eps: float,
        filter_mode: Literal["all_2d", "opencheese_hidden"],
    ):
        self.muon_lr = float(muon_lr)
        self.adamw_lr = float(adamw_lr)
        self.muon_momentum = float(muon_momentum)
        self.muon_weight_decay = float(muon_weight_decay)
        self.adamw_weight_decay = float(adamw_weight_decay)
        self.muon_ns_steps = int(muon_ns_steps)
        self.muon_nesterov = bool(muon_nesterov)
        self.polar_method = polar_method
        self.polar_safety = float(polar_safety)
        self.use_normuon = bool(use_normuon)
        self.muonplus_mode = str(muonplus_mode)
        self.adamw_betas = adamw_betas
        self.adamw_eps = float(adamw_eps)
        self.filter_mode = filter_mode
        self._initialized = False
        self.state = {"step": mx.array(0, mx.uint64), "params": None}

    def init(self, parameters: dict):
        self.state["params"] = tree_map(
            lambda p: {
                "momentum": mx.zeros_like(p),
                "adam_m": mx.zeros_like(p),
                "adam_v": mx.zeros_like(p),
            },
            parameters,
        )
        self._initialized = True

    def update(self, model, gradients: dict):
        parameters = model.trainable_parameters()
        if not self._initialized:
            self.init(parameters)
        self.state["step"] = self.state["step"] + 1
        updated = tree_map_with_path(self._apply_single, gradients, parameters, self.state["params"])
        model.update(updated)

    def _use_muon(self, path: str, parameter: mx.array) -> bool:
        if self.filter_mode == "all_2d":
            return _all_2d_filter(path, parameter)
        if self.filter_mode == "opencheese_hidden":
            return _opencheese_hidden_filter(path, parameter)
        raise ValueError(f"unknown MuonV2W filter mode {self.filter_mode!r}")

    def _apply_single(self, path: str, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        if self._use_muon(path, parameter):
            return self._muon_step(gradient, parameter, state)
        return self._adamw_step(gradient, parameter, state)

    def _muon_step(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        momentum = state["momentum"]
        momentum = self.muon_momentum * momentum + gradient
        state["momentum"] = momentum
        update = gradient + self.muon_momentum * momentum if self.muon_nesterov else momentum
        if self.use_normuon:
            update = _neuron_wise_l2_norm(update)
        update = _apply_polar(
            update,
            method=self.polar_method,
            steps=self.muon_ns_steps,
            safety=self.polar_safety,
        )
        if self.muonplus_mode != "none":
            update = _muon_plus_normalize(update, self.muonplus_mode)
        parameter = parameter - mx.array(self.muon_lr, dtype=gradient.dtype) * update
        if self.muon_weight_decay:
            parameter = parameter - mx.array(self.muon_lr * self.muon_weight_decay, dtype=gradient.dtype) * parameter
        return parameter

    def _adamw_step(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        b1, b2 = self.adamw_betas
        m = state["adam_m"]
        v = state["adam_v"]
        m = b1 * m + (1.0 - b1) * gradient
        v = b2 * v + (1.0 - b2) * mx.square(gradient)
        state["adam_m"] = m
        state["adam_v"] = v
        lr = mx.array(self.adamw_lr, dtype=gradient.dtype)
        parameter = parameter * (1.0 - lr * self.adamw_weight_decay)
        return parameter - lr * m / (mx.sqrt(v) + self.adamw_eps)


def MuonV2W(
    *,
    muon_lr: float = 0.002,
    adamw_lr: float = 5.0e-4,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adamw_weight_decay: float = 1.0e-4,
    muon_ns_steps: int = 5,
    muon_nesterov: bool = True,
    polar_method: Literal["jordan", "polar_express", "gram_ns"] = "polar_express",
    polar_safety: float = 1.01,
    use_normuon: bool = False,
    muonplus_mode: str = "none",
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1.0e-8,
    filter_mode: Literal["all_2d", "opencheese_hidden"] = "opencheese_hidden",
) -> MuonV2WOptimizer:
    """MuonV2 for selected 2D matrices plus AdamW for the rest."""

    return MuonV2WOptimizer(
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        muon_momentum=muon_momentum,
        muon_weight_decay=muon_weight_decay,
        adamw_weight_decay=adamw_weight_decay,
        muon_ns_steps=muon_ns_steps,
        muon_nesterov=muon_nesterov,
        polar_method=polar_method,
        polar_safety=polar_safety,
        use_normuon=use_normuon,
        muonplus_mode=muonplus_mode,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
        filter_mode=filter_mode,
    )


__all__ = [
    "MuonV2",
    "MuonV2W",
    "MuonV2WOptimizer",
    "polar_express",
    "newton_schulz_jordan",
    "gram_newton_schulz",
]
