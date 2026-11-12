"""2-D axial rotary position embedding for image patches (DINOv3)."""

import numpy as np

from ..base.backend import DTYPE, Array, asarray, xp
from ..base.module import Module


class RoPE2D(Module):
    """RoPE over a patch grid: a quarter of the frequencies follow the row, a quarter the column.

    As DINOv3: patch centres get coordinates in [-1, 1] along each axis
    (normalised separately, so any resolution maps to the same box), and with
    ``d_k / 4`` periods ``base^(2j / (d_k / 2))`` the angles are
    ``2 pi * [row, col] / period``, giving d_k / 2 angles. Dimension i is
    rotated with i + d_k / 2 (the "half" pairing of ``RoPE``), so a query-key
    score depends only on the difference of their (row, col) coordinates.

    The first ``n_prefix`` tokens (CLS and registers) are not rotated.

    Call ``set_grid`` before each forward of a batch: the tables are kept until
    the next call, which nnkit's forward-then-backward order allows. One
    instance can serve every layer. Box augmentation (training only) scales
    and shifts the coordinates; DINOv3 rescales both axes by
    ``exp(uniform(-log 2, log 2))``.
    """

    def __init__(self, d_k: int, base: float = 100.0) -> None:
        super().__init__()
        if d_k % 4:
            raise ValueError(f"d_k={d_k} must be divisible by 4 for RoPE2D")
        self.half = d_k // 2
        self.periods = base ** (2 * np.arange(d_k // 4, dtype=np.float64) / (d_k // 2))
        self.cos: Array | None = None
        self.sin: Array | None = None

    def set_grid(
        self,
        h: int,
        w: int,
        n_prefix: int = 0,
        scale: tuple[float, float] = (1.0, 1.0),
        shift: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Tables for ``n_prefix`` unrotated tokens followed by an h x w grid in row-major order.

        Coordinates are ``(2 * (index + 0.5) / size - 1 + shift) * scale`` per axis.
        """
        rows = 2 * (np.arange(h) + 0.5) / h - 1
        cols = 2 * (np.arange(w) + 0.5) / w - 1
        coords = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1).reshape(
            -1, 2
        )
        coords = (coords + np.asarray(shift)) * np.asarray(scale)
        angles = 2 * np.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.reshape(h * w, self.half)  # row angles, then column angles
        angles = np.concatenate([angles, angles], axis=-1)  # (h * w, d_k)
        angles = np.concatenate([np.zeros((n_prefix, 2 * self.half)), angles])
        self.cos = asarray(np.cos(angles), DTYPE)
        self.sin = asarray(np.sin(angles), DTYPE)

    def _rotate_half(self, x: Array) -> Array:
        """[x1, x2] -> [-x2, x1], where x1 and x2 are the two halves of the last axis."""
        x1, x2 = x[..., : self.half], x[..., self.half :]

        return xp.concatenate([-x2, x1], axis=-1)

    def _tables(self, x: Array) -> tuple[Array, Array]:
        if self.cos is None or self.sin is None:
            raise RuntimeError("call set_grid() before forward()")
        if x.shape[-2] != self.cos.shape[0]:
            raise ValueError(
                f"{x.shape[-2]} tokens, but set_grid() covered {self.cos.shape[0]}"
            )
        return self.cos, self.sin

    def forward(self, x: Array, offset: int = 0) -> Array:
        """x (batch, heads, T, d_k) -> the same shape; T = n_prefix + h * w.

        ``offset`` exists for ``MultiHeadAttention``'s interface; KV-cached
        decoding doesn't apply to images, so it must be 0.
        """
        if offset:
            raise ValueError("RoPE2D does not support KV-cache offsets")
        cos, sin = self._tables(x)

        return x * cos + self._rotate_half(x) * sin

    def backward(self, dy: Array) -> Array:
        """Rotate the gradient back by the same angle: dx = dy*cos - rotate_half(dy*sin)."""
        cos, sin = self._tables(dy)
        return dy * cos - self._rotate_half(dy * sin)
