# nnkit

A from-scratch neural network toolkit with hand-written `forward` and `backward` passes, running on either the Apple GPU (`mlx`) or CPU (`numpy`).

Built as the shared foundation for from-scratch paper implementations (`transformer-remastered`, `dino-remastered`, etc.).

## Installation

```sh
uv add git+https://github.com/Gevornairi/nnkit --tag v0.4.0
```

For co-developing nnkit alongside a paper repo, point `[tool.uv.sources]` at a local checkout instead:

```toml
nnkit = { path = "../nnkit", editable = true }
```

## Backend

The backend is chosen from environment variables when `nnkit` is first imported:

| Variable | Values | Default |
| --- | --- | --- |
| `NNKIT_BACKEND` | `mlx`, `numpy` | `mlx` |
| `NNKIT_DTYPE` | `float32`, `float64` (NumPy only) | `float32` |

## Subpackages

- **`nnkit.base`**: Dual array backend (`mlx` / `numpy`) with helpers such as `scatter_add` and `top_k_indices`, `Module`, `Param`, `Buffer` (non-trainable state saved in checkpoints), and attention masks (`padding_mask`, `causal_mask`, `decoder_mask`).
- **`nnkit.activations`**: `SiLU`, `sigmoid`, `sigmoid_backward`, `softcap`, `softcap_backward` (logit soft-capping), `softmax`, `log_softmax`, `softmax_backward`.
- **`nnkit.layers`**: `Linear`, `Embedding`, `Dropout`, `L2Norm` (unit length), `RMSNorm` (optional gain), `RoPE`, `SwiGLUFeedForward`, `MoEFeedForward` (shared + top-k routed experts, sigmoid gating, bias-based load balancing, sparse dispatch), `MultiHeadAttention` (with GQA/MQA, QK-Norm with or without a learned gain, iRoPE temperature scaling, and differential attention), `EncoderLayer`, `DecoderLayer`.
- **`nnkit.losses`**: `LabelSmoothingLoss`.
- **`nnkit.optimisers`**: `AdamW` (with global-norm gradient clipping), `CosineSchedule`, `NoamSchedule`.
- **`nnkit.training`**: `Prefetcher` (builds batches on a background thread from any `make_epoch(rng)` function), `save_checkpoint` / `load_checkpoint` (model, optimizer state and arbitrary metadata in one `.npz`), `read_checkpoint_metadata`.

## Gradient Check

```sh
# Central finite differences in float64 on CPU
NNKIT_BACKEND=numpy NNKIT_DTYPE=float64 uv run python -m tests.gradcheck

# MLX autodiff on Apple GPU
uv run python -m tests.gradcheck

# Checkpoint round trip and prefetcher
uv run python -m tests.training_check
```
