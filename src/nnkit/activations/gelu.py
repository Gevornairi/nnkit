"""GELU activation (exact, erf-based)."""

import math

from ..base.backend import Array, erf, xp
from ..base.module import Module


class GELU(Module):
    """``gelu(z) = z * Phi(z)``, Phi the standard normal CDF; caches z and Phi(z).

    The exact form, as ``torch.nn.GELU()`` (used by ViT, DINOv2/v3 MLPs), not
    the tanh approximation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cache: tuple[Array, Array] | None = None  # (z, Phi(z))

    def forward(self, z: Array) -> Array:
        cdf = 0.5 * (1.0 + erf(z * (1.0 / math.sqrt(2.0))))
        self._cache = (z, cdf)

        return z * cdf

    def backward(self, dy: Array) -> Array:
        """gelu'(z) = Phi(z) + z * phi(z), phi the standard normal density."""
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        z, cdf = self._cache
        pdf = xp.exp(-0.5 * z * z) * (1.0 / math.sqrt(2.0 * math.pi))
        return dy * (cdf + z * pdf)
