"""Stochastic depth (drop path)."""

from ..base.backend import Array, keep_mask
from ..base.module import Module


class DropPath(Module):
    """Drop a whole residual branch for each sample with probability ``p`` (Huang et al., 2016).

    Applied to the branch before the residual add (``x + DropPath(f(x))``):
    each sample's branch output is zeroed or scaled by 1/(1-p), so the
    expected output is unchanged. Identity in eval mode or with p = 0.
    Shapes: (batch, ...) -> the same.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"drop path p must be in [0, 1), got {p}")
        self.p = p
        self._mask = None  # (batch, 1, ..., 1) scaled keep-mask, None if identity

    def forward(self, x: Array) -> Array:
        if not self.training or self.p == 0.0:
            self._mask = None

            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        self._mask = keep_mask(1.0 - self.p, shape) * (1.0 / (1.0 - self.p))

        return x * self._mask

    def backward(self, dy: Array) -> Array:
        """Same per-sample mask as the forward pass."""
        return dy if self._mask is None else dy * self._mask
