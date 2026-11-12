"""Layer normalisation."""

from ..base.backend import Array, xp
from ..base.module import Module
from ..base.param import Param


class LayerNorm(Module):
    """``y = gain * (x - mean(x)) / sqrt(var(x) + eps) + bias`` over the last axis.

    The original transformer / ViT norm; ``gain`` starts at 1 and ``bias`` at 0.
    RMSNorm drops the mean subtraction and the bias.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gain = Param(xp.ones(d_model))
        self.bias = Param(xp.zeros(d_model))
        self.register_params(self.gain, self.bias)
        self.eps = eps
        self._cache: tuple[Array, Array] | None = None  # (x_hat, 1/std)

    def forward(self, x: Array) -> Array:
        centered = x - x.mean(axis=-1, keepdims=True)
        inv_std = 1.0 / xp.sqrt(
            (centered * centered).mean(axis=-1, keepdims=True) + self.eps
        )
        x_hat = centered * inv_std
        self._cache = (x_hat, inv_std)

        return self.gain.value * x_hat + self.bias.value

    def backward(self, dy: Array) -> Array:
        """dx = (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat)) / std, dx_hat = dy * gain."""

        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        x_hat, inv_std = self._cache
        dim = x_hat.shape[-1]
        self.gain.grad += (dy * x_hat).reshape(-1, dim).sum(axis=0)
        self.bias.grad += dy.reshape(-1, dim).sum(axis=0)
        dx_hat = dy * self.gain.value
        dx = inv_std * (
            dx_hat
            - dx_hat.mean(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)
        )

        return dx
