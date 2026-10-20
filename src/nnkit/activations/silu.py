"""SiLU (swish) activation."""

from ..base.backend import Array
from ..base.module import Module
from .sigmoid import sigmoid


class SiLU(Module):
    """``silu(z) = z * sigmoid(z)``; caches z and sigmoid(z) for the backward pass."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: tuple[Array, Array] | None = None  # (z, sigmoid(z))

    def forward(self, z: Array) -> Array:
        s = sigmoid(z)
        self._cache = (z, s)

        return z * s

    def backward(self, dy: Array) -> Array:
        """silu'(z) = s * (1 + z * (1 - s)), with s = sigmoid(z)."""
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        z, s = self._cache
        return dy * s * (1 + z * (1 - s))
