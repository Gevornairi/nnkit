"""Patch embedding: an image to a sequence of patch tokens."""

from ..base.backend import Array
from ..base.module import Module
from .linear import Linear


class PatchEmbedding(Module):
    """Cut the image into ``patch_size`` x ``patch_size`` patches and project each to ``d_model``.

    The same as ViT's strided convolution (kernel = stride = patch size, with
    bias), written as a reshape and a Linear layer. Images are channels-last;
    tokens come out in row-major order of the (H / p, W / p) patch grid, which
    is the order ``RoPE2D.set_grid`` assumes.

    Shapes: (batch, H, W, channels) -> (batch, (H / p) * (W / p), d_model).
    """

    def __init__(self, patch_size: int, channels: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = Linear(patch_size * patch_size * channels, d_model, bias=True)
        self.register_modules(self.proj)
        self._shape: tuple[int, ...] | None = None  # input shape, for backward

    def forward(self, x: Array) -> Array:
        b, h, w, c = x.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError(f"image {h}x{w} is not divisible by patch_size={p}")
        self._shape = x.shape
        patches = (
            x.reshape(b, h // p, p, w // p, p, c)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(b, (h // p) * (w // p), p * p * c)
        )

        return self.proj.forward(patches)

    def backward(self, dy: Array) -> Array:
        """Undo the patch reshape on the projection's input gradient; returns d_image."""
        if self._shape is None:
            raise RuntimeError("backward() called before forward()")

        b, h, w, c = self._shape
        p = self.patch_size
        d_patches = self.proj.backward(dy)

        return (
            d_patches.reshape(b, h // p, w // p, p, p, c)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(b, h, w, c)
        )
