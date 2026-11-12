"""GELU position-wise feed-forward network (the ViT MLP)."""

from ..activations.gelu import GELU
from ..base.backend import Array
from ..base.module import Module
from .dropout import Dropout
from .linear import Linear


class GELUFeedForward(Module):
    """``y = W_2( dropout( gelu(x W_1 + b_1) ) ) + b_2``, applied to each position.

    The two-matrix MLP of ViT and DINOv2/v3 (biases on by default, as there),
    with d_ff = 4 * d_model. SwiGLUFeedForward is the gated alternative.
    Shapes: (..., d_model) -> (..., d_model).
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        self.w_1 = Linear(d_model, d_ff, bias)
        self.w_2 = Linear(d_ff, d_model, bias)
        self.gelu = GELU()
        self.dropout = Dropout(dropout)
        self.register_modules(self.w_1, self.w_2, self.gelu, self.dropout)

    def forward(self, x: Array) -> Array:
        hidden = self.gelu.forward(self.w_1.forward(x))

        return self.w_2.forward(self.dropout.forward(hidden))

    def backward(self, dy: Array) -> Array:
        d_hidden = self.dropout.backward(self.w_2.backward(dy))

        return self.w_1.backward(self.gelu.backward(d_hidden))
