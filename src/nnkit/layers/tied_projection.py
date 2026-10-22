"""Output projection that reuses an Embedding's table (tied input/output embeddings)."""

from ..base.backend import Array
from ..base.module import Module
from .embedding import Embedding


class TiedProjection(Module):
    """Logits from an embedding table: ``logits = x @ E^T``, one score per vocabulary word.

    Shares ``embedding.weight`` (vocab, d_model) instead of owning a weight, so the
    lookup's and this projection's gradients add into the same ``grad``. The lookup's
    sqrt(d_model) scale is not applied here. Shapes: (..., d_model) -> (..., vocab).
    """

    def __init__(self, embedding: Embedding) -> None:
        super().__init__()
        self.weight = embedding.weight
        # registered here too, so this module alone lists the table; parameters()
        # de-duplicates shared Params, so a model holding both still sees it once
        self.register_params(self.weight)
        self._x: Array | None = None

    def forward(self, x: Array) -> Array:
        self._x = x

        return x @ self.weight.value.T

    def backward(self, d_logits: Array) -> Array:
        """dE += d_logits^T x (summed over every leading dim); returns dx = d_logits E."""
        x = self._x
        if x is None:
            raise RuntimeError("backward() called before forward()")
        d_model, vocab = x.shape[-1], d_logits.shape[-1]
        self.weight.grad += d_logits.reshape(-1, vocab).T @ x.reshape(-1, d_model)

        return d_logits @ self.weight.value
