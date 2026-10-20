"""Parameter-free L2 normalisation."""

from ..base.backend import Array, xp
from ..base.module import Module


class L2Norm(Module):
    """``y = x / sqrt(sum(x^2) + eps)`` over the last axis.

    Projects vectors onto the unit hypersphere without learnable parameters,
    like ``torch.nn.functional.normalize`` (e.g. DINO's head, contrastive
    losses). For Llama 4's QK-Norm, which keeps norm ``sqrt(d)``, use
    ``RMSNorm(d, gain=False)``.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self._cache: tuple[Array, Array] | None = None  # (y, 1/norm)

    def forward(self, x: Array) -> Array:
        inv_norm = 1.0 / xp.sqrt((x * x).sum(axis=-1, keepdims=True) + self.eps)
        y = x * inv_norm
        self._cache = (y, inv_norm)

        return y

    def backward(self, dy: Array) -> Array:
        """dx = (dy - y * sum(dy * y)) / norm, projecting out the radial component."""
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        y, inv_norm = self._cache

        return inv_norm * (dy - y * (dy * y).sum(axis=-1, keepdims=True))
