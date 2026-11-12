# nnkit

A from-scratch neural network toolkit with hand-written `forward` and `backward` passes, running on the Apple GPU (mlx), NVIDIA GPUs (cupy, CUDA) or CPU (numpy).

Built as the shared foundation for from-scratch paper implementations (`transformer-remastered`, `dino-remastered`, etc.).

## Installation

```sh
uv add git+https://github.com/Gevornairi/nnkit --tag v0.9.0 --extra cuda
```

For co-developing nnkit alongside a paper repo, point `[tool.uv.sources]` at a local checkout instead:

```toml
nnkit = { path = "../nnkit", editable = true }
```

## Backend

The backend is chosen from environment variables when `nnkit` is first imported:

| Variable | Values | Default |
| --- | --- | --- |
| `NNKIT_BACKEND` | `mlx`, `cupy`, `numpy` | `mlx` |
| `NNKIT_DTYPE` | `float32`, `float64` (NumPy or CuPy) | `float32` |

## Subpackages

- **`nnkit.base`**: Array backend (`mlx` / `cupy` / `numpy`) with helpers such as `scatter_add` and `top_k_indices`, `Module`, `Param`, `Buffer` (non-trainable state saved in checkpoints), `KVCache` (per-layer keys/values for incremental decoding), and attention masks (`padding_mask`, `causal_mask`, `chunked_causal_mask`, `sliding_window_mask`, `document_causal_mask` for packed sequences, `decoder_mask`).
- **`nnkit.activations`**: `GELU` (exact, via `erf`), `SiLU`, `sigmoid`, `sigmoid_backward`, `softcap`, `softcap_backward` (logit soft-capping), `softmax`, `log_softmax`, `softmax_backward`.
- **`nnkit.layers`**: `Linear` (optional bias), `Embedding` (optional √d scaling), `Dropout`, `L2Norm` (unit length), `RMSNorm` (optional gain), `RoPE`, `SwiGLUFeedForward`, `TiedProjection` (output logits from an `Embedding`'s table), `MoEFeedForward` (shared + top-k routed experts, sigmoid gating on the expert output or, as in Llama 4, its input, bias-based load balancing, sparse dispatch), `MultiHeadAttention` (with GQA/MQA, KV-cached decoding, QK-Norm with or without a learned gain, iRoPE temperature scaling, differential attention, gated attention, and ViT-style q/v/o biases), `EncoderLayer`, `DecoderLayer`. For vision transformers: `PatchEmbedding` (channels-last image to row-major patch tokens), `RoPE2D` (DINOv3 axial 2-D RoPE, prefix tokens unrotated, box jitter via `set_grid`), `LayerNorm`, `LayerScale`, `DropPath` (stochastic depth), `GELUFeedForward`.
- **`nnkit.losses`**: `LabelSmoothingLoss`, `z_loss` (keeps the softmax normaliser near 1), `soft_cross_entropy` (DINO/iBOT self-distillation), `koleo_loss` (KoLeo feature spreading), `gram_loss` (DINOv3 Gram anchoring), `GRPOLoss` with `group_advantages` and `token_logp` (group-relative policy optimisation, with an optional KL penalty to a frozen reference model).
- **`nnkit.optimisers`**: `AdamW` (with global-norm gradient clipping), `Muon` (orthogonalised momentum for weight matrices, with `newton_schulz` and `split_muon_params`), `CosineSchedule`, `NoamSchedule`, `WSDSchedule` (warmup, stable, short decay).
- **`nnkit.training`**: `Prefetcher` (builds batches on a background thread from any `make_epoch(rng)` function), `packed_batches` (language-model windows cut from a packed `doc EOS doc EOS ...` token stream, with `document_ids` for `document_causal_mask`), `save_checkpoint` / `load_checkpoint` (model, optimizer state and arbitrary metadata in one `.npz`), `read_checkpoint_metadata`, `sample_batch` (batched temperature sampling from a KV-cached model), and self-distillation teachers: `ema_update` (EMA teacher), `sinkhorn_knopp` (balanced teacher targets).

## Gradient Check

```sh
# Central finite differences in float64 on CPU
NNKIT_BACKEND=numpy NNKIT_DTYPE=float64 uv run python -m tests.gradcheck
# Central finite differences in float64 on an NVIDIA GPU
NNKIT_BACKEND=cupy NNKIT_DTYPE=float64 uv run python -m tests.gradcheck

# MLX autodiff on Apple GPU
uv run python -m tests.gradcheck

# Checkpoint round trip and prefetcher
uv run python -m tests.training_check

# KV-cached decoding matches the uncached forward
uv run python -m tests.cache_check

# Attention mask patterns
uv run python -m tests.mask_check

# RoPE2D, patch order, Sinkhorn-Knopp, EMA, DropPath, GELU and KoLeo properties
uv run python -m tests.vision_check

# Packed-stream batching for language models
uv run python -m tests.packing_check

# Optimisers and learning-rate schedules
uv run python -m tests.optim_check
```
