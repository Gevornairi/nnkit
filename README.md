# nnkit

A from-scratch neural network toolkit with hand-written `forward` and `backward` passes, running on either the Apple GPU (`mlx`) or CPU (`numpy`).

Built as the shared foundation for from-scratch paper implementations (`transformer-remastered`, `dino-remastered`, etc.).

## Subpackages

- **`nnkit.base`**: Dual array backend (`mlx` / `numpy`), `Module`, `Param`, and attention masks (`padding_mask`, `causal_mask`, `decoder_mask`).
- **`nnkit.activations`**: `SiLU`, `softmax`, `log_softmax`, `softmax_backward`.
- **`nnkit.layers`**: `Linear`, `Embedding`, `Dropout`, `RMSNorm`, `RoPE`, `SwiGLUFeedForward`, `MultiHeadAttention` (with GQA/MQA), `EncoderLayer`, `DecoderLayer`.
- **`nnkit.losses`**: `LabelSmoothingLoss`.
- **`nnkit.optimisers`**: `AdamW` (with global-norm gradient clipping), `CosineSchedule`, `NoamSchedule`.

## Gradient Check

```sh
# Central finite differences in float64 on CPU
NNKIT_BACKEND=numpy NNKIT_DTYPE=float64 uv run python -m tests.gradcheck

# MLX autodiff on Apple GPU
uv run python -m tests.gradcheck
```
