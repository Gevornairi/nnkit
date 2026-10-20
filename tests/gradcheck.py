"""Gradient check for nnkit layers and losses (finite differences in float64 & MLX autodiff)."""

import numpy as np

from nnkit.base import (
    BACKEND,
    Module,
    Param,
    asarray,
    decoder_mask,
    normal,
    padding_mask,
    seed,
    to_numpy,
    xp,
)
from nnkit.layers import (
    DecoderLayer,
    Embedding,
    EncoderLayer,
    L2Norm,
    Linear,
    MultiHeadAttention,
    RMSNorm,
    RoPE,
)
from nnkit.losses import LabelSmoothingLoss

PAD, BOS, VOCAB_SIZE = 0, 1, 7
SRC = [[3, 4, 5, 6], [5, 3, 0, 0]]
TGT_IN = [[BOS, 6, 5, 4, 3], [BOS, 3, 5, 0, 0]]
TGT_OUT = [[6, 5, 4, 3, 2], [3, 5, 2, 0, 0]]


def reference_grads(params: list[Param], loss_of_model) -> list[np.ndarray]:
    """Gradients of ``loss_of_model()`` w.r.t. ``params``: MLX autodiff, or central differences on NumPy."""
    if BACKEND == "mlx":

        def loss_of_values(values):
            for p, v in zip(params, values):
                p.value = v
            return loss_of_model()

        return [to_numpy(g) for g in xp.grad(loss_of_values)([p.value for p in params])]

    reference = []
    for p in params:
        num = np.zeros(p.shape)
        for ix in np.ndindex(*p.shape):
            old = p.value[ix]
            p.value[ix] = old + 1e-6
            loss_plus = loss_of_model().item()
            p.value[ix] = old - 1e-6
            loss_minus = loss_of_model().item()
            p.value[ix] = old
            num[ix] = (loss_plus - loss_minus) / 2e-6
        reference.append(num)
    return reference


def relative_error(params: list[Param], reference: list[np.ndarray]) -> float:
    """Largest gap between the hand-written and reference gradients, relative to the largest gradient."""
    scale = max(np.abs(r).max() for r in reference)
    errors = [np.abs(to_numpy(p.grad) - r).max() for p, r in zip(params, reference)]
    return float(max(errors)) / scale


def check(
    n_heads: int = 2, n_kv_heads: int | None = None, qk_norm: bool = False
) -> float:
    """Verify hand-written backward passes for all core layers against reference gradients."""
    seed(0)
    d_model, d_ff, max_len = 8, 16, 16
    embed = Embedding(VOCAB_SIZE, d_model)
    rope = RoPE(d_model // n_heads, max_len)
    enc_layer = EncoderLayer(d_model, n_heads, d_ff, 0.0, rope, n_kv_heads, qk_norm)
    enc_norm = RMSNorm(d_model)
    dec_layer = DecoderLayer(d_model, n_heads, d_ff, 0.0, rope, n_kv_heads, qk_norm)
    dec_norm = RMSNorm(d_model)
    proj = Linear(d_model, VOCAB_SIZE)
    loss_fn = LabelSmoothingLoss(VOCAB_SIZE, PAD, 0.1)

    modules = [embed, enc_layer, enc_norm, dec_layer, dec_norm, proj]
    params = [p for m in modules for p in m.parameters()]
    src, tgt_in, tgt_out = asarray(SRC), asarray(TGT_IN), asarray(TGT_OUT)
    s_mask = padding_mask(src, PAD)
    t_mask = decoder_mask(tgt_in, PAD)

    def forward_logits():
        mem = enc_norm.forward(enc_layer.forward(embed.forward(src), s_mask))
        dec = dec_norm.forward(
            dec_layer.forward(embed.forward(tgt_in), mem, s_mask, t_mask)
        )
        return proj.forward(dec)

    def loss_of_model():
        return loss_fn(forward_logits(), tgt_out)[0]

    for m in modules:
        m.zero_grad()
    _, d_logits = loss_fn(forward_logits(), tgt_out)
    d_dec = dec_norm.backward(proj.backward(d_logits))
    d_tgt_emb, d_mem = dec_layer.backward(d_dec)
    embed.backward(d_tgt_emb, tgt_in)
    d_src_emb = enc_layer.backward(enc_norm.backward(d_mem))
    embed.backward(d_src_emb, src)

    return relative_error(params, reference_grads(params, loss_of_model))


def check_attention(
    n_heads: int, n_kv_heads: int | None = None, **attn_kwargs
) -> float:
    """Masked self-attention on its own, for options the encoder/decoder layers don't expose.

    The input is a ``Param`` too, so ``d_query_in + d_kv_in`` is checked as well.
    Uses RoPE unless temperature scaling (a NoPE-only option) is on. 1-D params
    (gains, biases, lambda) are jittered so a gain of exactly 1 can't hide a bug.
    """
    seed(0)
    d_model, max_len = 16, 16
    d_qk = d_model // n_heads // (2 if attn_kwargs.get("diff_attn") else 1)
    rope = None if attn_kwargs.get("attn_scale_alpha") else RoPE(d_qk, max_len)
    attn = MultiHeadAttention(d_model, n_heads, 0.0, rope, n_kv_heads, **attn_kwargs)
    for p in attn.parameters():
        if p.value.ndim == 1:
            p.value = p.value + normal(0.3, p.shape)
    tgt_in = asarray(TGT_IN)
    mask = decoder_mask(tgt_in, PAD)
    x = Param(normal(1.0, (*tgt_in.shape, d_model)))
    weights = normal(1.0, (*tgt_in.shape, d_model))  # loss = sum(out * weights)
    params = [x, *attn.parameters()]

    def loss_of_model():
        return (attn.forward(x.value, x.value, mask) * weights).sum()

    attn.zero_grad()
    x.grad = xp.zeros_like(x.value)
    attn.forward(x.value, x.value, mask)
    d_q, d_kv = attn.backward(weights)
    x.grad = d_q + d_kv

    return relative_error(params, reference_grads(params, loss_of_model))


def check_norm(norm: Module) -> float:
    """A normalisation layer on its own, input gradient included."""
    seed(0)
    x = Param(normal(1.0, (2, 3, 8)))
    weights = normal(1.0, (2, 3, 8))
    params = [x, *norm.parameters()]
    for p in norm.parameters():
        p.value = p.value + normal(0.3, p.shape)

    def loss_of_model():
        return (norm.forward(x.value) * weights).sum()

    norm.zero_grad()
    norm.forward(x.value)
    x.grad = norm.backward(weights)

    return relative_error(params, reference_grads(params, loss_of_model))


def check_relative_position(**attn_kwargs) -> float:
    """RoPE's promise through the whole attention: scores depend only on n - m.

    Not a gradient check: a norm gain applied after RoPE has correct gradients
    but leaks absolute position. The input repeats with period P, so queries and
    keys at (m, n) and (m + P, n + P) are the same vectors; their log-attention
    over keys of the same period must agree up to a per-row constant. Norm gains
    are randomised, since a gain of all ones hides a misplaced norm.
    """
    seed(0)
    d_model, n_heads, period = 16, 4, 4
    rope = RoPE(d_model // n_heads, 2 * period)
    attn = MultiHeadAttention(d_model, n_heads, 0.0, rope, **attn_kwargs)
    for p in attn.parameters():
        if p.value.ndim == 1:
            p.value = p.value + normal(0.3, p.shape)
    x = normal(1.0, (1, period, d_model))
    x = xp.concatenate([x, x], axis=1)
    attn.forward(x, x)

    log_p = np.log(to_numpy(attn.attn_weights))
    first = log_p[..., :period, :period]
    second = log_p[..., period:, period:]
    gap = (first - first.mean(-1, keepdims=True)) - (
        second - second.mean(-1, keepdims=True)
    )
    return float(np.abs(gap).max())


def main() -> int:
    tol = 1e-4 if BACKEND == "mlx" else 1e-7
    ok = True

    def report(label: str, err: float, tol: float = tol) -> None:
        nonlocal ok
        ok &= err < tol
        status = "ok" if err < tol else "FAIL"
        print(f"[nnkit/{BACKEND}] {label}: max err {err:.2e} (tol {tol:.0e})  {status}")

    for h, g, qk in [(2, None, False), (4, 2, False), (4, 1, False), (4, 2, True)]:
        report(f"model heads={h} kv_heads={g or h} qk_norm={qk}", check(h, g, qk))

    nope_scale = {"attn_scale_alpha": 0.5, "attn_scale_floor": 2}
    for h, g, kwargs in [
        (4, 2, {"qk_norm": True}),
        (4, 2, {"qk_norm": True, "qk_norm_gain": True}),
        (4, 2, {"diff_attn": True}),
        (4, 1, {"diff_attn": True, "qk_norm": True}),
        (4, 1, {"diff_attn": True, "qk_norm": True, "qk_norm_gain": True}),
        (4, 2, nope_scale),
        (2, None, {"diff_attn": True, "qk_norm": True, **nope_scale}),
    ]:
        opts = " ".join(f"{k}={v}" for k, v in kwargs.items())
        report(
            f"attention heads={h} kv_heads={g or h} {opts}",
            check_attention(h, g, **kwargs),
        )

    for label, norm in [
        ("L2Norm", L2Norm()),
        ("RMSNorm", RMSNorm(8)),
        ("RMSNorm gain=False", RMSNorm(8, gain=False)),
    ]:
        report(f"norm {label}", check_norm(norm))

    # log-probabilities magnify float32 rounding to ~1e-3; a misplaced norm gain gives ~0.2-0.7
    rel_tol = 1e-2 if BACKEND == "mlx" else 1e-7
    for kwargs in [{}, {"qk_norm": True}, {"qk_norm": True, "qk_norm_gain": True}]:
        opts = " ".join(f"{k}={v}" for k, v in kwargs.items()) or "plain"
        err = check_relative_position(**kwargs)
        report(f"rope relative position {opts}", err, rel_tol)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
