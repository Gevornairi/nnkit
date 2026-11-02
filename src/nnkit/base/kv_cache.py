"""Per-layer key/value cache for incremental decoding."""

from .backend import Array, xp


class KVCache:
    """Keys and values of the tokens one self-attention layer has already seen, for decoding.

    Passed to ``MultiHeadAttention.forward``, it makes the input the tokens that
    follow the cached ones: RoPE and temperature scaling start at position
    ``length``, and the new keys/values are appended before attending. Keys are
    stored after RoPE / QK-Norm and before the grouped-query repeat. Use one
    cache per layer and per batch of sequences; it holds no gradients.
    """

    def __init__(self) -> None:
        self.k: Array | None = None
        self.v: Array | None = None

    @property
    def length(self) -> int:
        """Number of tokens cached so far."""
        return 0 if self.k is None else self.k.shape[-2]

    def append(self, k: Array, v: Array) -> tuple[Array, Array]:
        """Add the new tokens' keys/values (time on axis -2); returns all keys and values."""
        if self.k is not None:
            k = xp.concatenate([self.k, k], axis=-2)
            v = xp.concatenate([self.v, v], axis=-2)
        self.k, self.v = k, v

        return k, v
