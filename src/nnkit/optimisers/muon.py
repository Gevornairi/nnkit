"""Muon: momentum with orthogonalised updates for weight matrices (Jordan, 2024)."""

from collections.abc import Callable, Sequence

import numpy as np

from ..base.backend import Array, asarray, materialize, to_numpy, xp
from ..base.module import Module
from ..base.param import Param

# quintic Newton-Schulz coefficients: push every singular value of X towards ~1 fast
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: Array, steps: int = 5, eps: float = 1e-7) -> Array:
    """Approximately orthogonalise a matrix: U V^T of its SVD, using only matmuls.

    The result has g's singular vectors and all singular values near 1 (roughly
    0.7-1.2; exactness doesn't matter for the update). The iteration runs on the
    smaller Gram matrix, so tall matrices are transposed first.
    """
    a, b, c = NS_COEFFS
    tall = g.shape[0] > g.shape[1]
    x = g.T if tall else g
    x = x / (xp.sqrt((x * x).sum()) + eps)  # Frobenius norm: all singular values <= 1
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if tall else x


def split_muon_params(
    model: Module, exclude: Sequence[Param] = ()
) -> tuple[list[Param], list[Param]]:
    """Split a model's parameters into (Muon, AdamW) groups.

    2-D matrices go to Muon, except those in ``exclude`` (pass the embedding
    table, an untied output head and MoE routers); everything else, including
    all 1-D parameters, goes to AdamW.
    """
    skip = {id(p) for p in exclude}
    muon, adamw = [], []
    for p in model.parameters():
        (muon if p.value.ndim == 2 and id(p) not in skip else adamw).append(p)

    return muon, adamw


class Muon:
    """Momentum whose update is orthogonalised per weight matrix (Jordan, 2024; Liu et al., 2025).

        m = momentum * m + g
        u = g + momentum * m                (Nesterov; plain momentum uses u = m)
        value -= lr * (scale * newton_schulz(u) + weight_decay * value)

    Orthogonalising gives every direction of the update the same size, so
    directions the gradient rarely points along still learn. ``scale =
    0.2 * sqrt(max(rows, cols))`` matches the update RMS of AdamW (Liu et al.,
    2025), so AdamW's learning rate and weight decay carry over.

    Only for 2-D hidden weight matrices; embedding tables, the output head,
    routers and all 1-D parameters belong to AdamW (see ``split_muon_params``).
    No gradient clipping: the update's size doesn't depend on the gradient's.
    Same interface as ``AdamW``; ``schedule`` maps the 1-based step to the lr.
    """

    def __init__(
        self,
        params: list[Param],
        schedule: Callable[[int], float],
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        self.params = list(params)
        bad = [p.shape for p in self.params if p.value.ndim != 2]
        if bad:
            raise ValueError(f"Muon only takes 2-D weight matrices, got shapes {bad}")
        self.schedule = schedule
        self.momentum, self.nesterov = momentum, nesterov
        self.weight_decay, self.ns_steps = weight_decay, ns_steps
        self.step_count = 0
        self.m = [xp.zeros_like(p.value) for p in self.params]
        self.scales = [0.2 * max(p.shape) ** 0.5 for p in self.params]
        self.last_lr = 0.0

    def step(self) -> Array:
        """Update every parameter; returns the global grad norm (for logging) as a 0-d array."""
        self.step_count += 1
        grad_norm = xp.sqrt(sum((p.grad * p.grad).sum() for p in self.params))
        lr = self.schedule(self.step_count)
        self.last_lr = lr
        for i, p in enumerate(self.params):
            self.m[i] = self.momentum * self.m[i] + p.grad
            u = p.grad + self.momentum * self.m[i] if self.nesterov else self.m[i]
            update = self.scales[i] * newton_schulz(u, self.ns_steps)
            p.value = p.value - lr * (update + self.weight_decay * p.value)

        materialize([p.value for p in self.params], self.m, grad_norm)

        return grad_norm

    def zero_grad(self) -> None:
        """Reset every parameter's gradient to zeros."""

        for p in self.params:
            p.grad = xp.zeros_like(p.value)

    def state(self) -> dict[str, int | np.ndarray]:
        """Step count and momentum buffers, as NumPy arrays keyed ``m0, m1, ...``."""
        out: dict[str, int | np.ndarray] = {"step_count": self.step_count}
        for i in range(len(self.params)):
            out[f"m{i}"] = to_numpy(self.m[i])

        return out

    def load_state(self, state) -> None:
        """Restore what ``state`` saved; the parameter list must match."""
        self.step_count = int(state["step_count"])
        for i, p in enumerate(self.params):
            self.m[i] = asarray(state[f"m{i}"], p.value.dtype)
