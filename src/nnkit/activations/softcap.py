"""Logit soft-capping (Gemma 2) and its backward pass."""

from ..base.backend import Array, xp


def softcap(x: Array, cap: float) -> Array:
    """``cap * tanh(x / cap)``: identity for |x| << cap, saturates smoothly at ±cap."""
    return cap * xp.tanh(x / cap)


def softcap_backward(y: Array, dy: Array, cap: float) -> Array:
    """Gradient w.r.t. the input, given the output ``y``: dy * (1 - (y / cap)^2).

    ``tanh(x / cap) = y / cap``, so no cache is needed. The gradient goes to
    zero as a logit nears the cap, which is what stops logits growing.
    """
    t = y / cap
    return dy * (1.0 - t * t)
