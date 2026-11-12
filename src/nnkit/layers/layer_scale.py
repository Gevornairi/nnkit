"""LayerScale: a learned per-channel scale on a residual branch."""

from ..base.backend import Array, xp
from ..base.module import Module
from ..base.param import Param


class LayerScale(Module):
    """``y = gamma * x`` with ``gamma`` (d_model,) starting small (Touvron et al., 2021, CaiT).

    Put on each residual branch (``x + LayerScale(attn(norm(x)))``), so a deep
    ViT starts close to the identity. DINOv2/v3 initialise it to 1e-5.
    """

    def __init__(self, d_model: int, init: float = 1e-5) -> None:
        super().__init__()
        self.gamma = Param(xp.full((d_model,), init))
        self.register_params(self.gamma)
        self._x: Array | None = None

    def forward(self, x: Array) -> Array:
        self._x = x

        return self.gamma.value * x

    def backward(self, dy: Array) -> Array:
        if self._x is None:
            raise RuntimeError("backward() called before forward()")

        dim = self._x.shape[-1]
        self.gamma.grad += (dy * self._x).reshape(-1, dim).sum(axis=0)

        return dy * self.gamma.value
