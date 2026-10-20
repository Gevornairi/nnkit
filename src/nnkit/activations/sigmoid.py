"""Sigmoid and its backward pass."""

from ..base.backend import Array, xp


def sigmoid(z: Array) -> Array:
    """1 / (1 + e^-z), evaluated so that exp never overflows."""
    e = xp.exp(-xp.abs(z))
    return xp.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sigmoid_backward(s: Array, dy: Array) -> Array:
    """Gradient w.r.t. the sigmoid input, given its output ``s``: dy * s * (1 - s)."""
    return dy * s * (1.0 - s)
