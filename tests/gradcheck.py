"""Gradient check for nnkit layers and losses (finite differences in float64 & MLX autodiff)."""

import numpy as np

from nnkit.activations import sigmoid, softcap, softcap_backward
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
    top_k_indices,
    xp,
)
from nnkit.layers import (
    DecoderLayer,
    Embedding,
    EncoderLayer,
    L2Norm,
    Linear,
    MoEFeedForward,
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


def check_moe(
    top_k: int,
    normalize_gates: bool,
    dispatch: str = "sparse",
    dead_expert: bool = False,
) -> float:
    """MoE feed-forward with a non-zero bias that changes which experts are chosen.

    Finite differences must not flip a top-k choice, so the test asserts a
    margin between the k-th and (k+1)-th biased scores. With ``dead_expert``
    the last expert is never chosen, which the sparse path must skip cleanly.
    """
    seed(0)
    d_model, n_experts = 8, 4
    moe = MoEFeedForward(
        d_model, n_experts, 8, top_k, normalize_gates, dropout=0.0, dispatch=dispatch
    )
    bias = normal(0.3, (n_experts,))
    if dead_expert:
        bias = bias + xp.array([0.0] * (n_experts - 1) + [-10.0])
    moe.expert_bias.value = bias
    x = Param(normal(1.0, (2, 5, d_model)))
    weights = normal(1.0, (2, 5, d_model))  # loss = sum(out * weights)
    params = [x, *moe.parameters()]

    s = sigmoid(moe.router.forward(x.value.reshape(-1, d_model)))
    biased = s + moe.expert_bias.value
    ranked = np.sort(to_numpy(biased), axis=-1)[:, ::-1]
    if top_k < n_experts:
        assert (ranked[:, top_k - 1] - ranked[:, top_k]).min() > 1e-4, "top-k tie"
    plain = np.sort(to_numpy(top_k_indices(s, top_k)), axis=-1)
    with_bias = np.sort(to_numpy(top_k_indices(biased, top_k)), axis=-1)
    assert (plain != with_bias).any(), "bias should change some choices"

    def loss_of_model():
        return (moe.forward(x.value) * weights).sum()

    moe.zero_grad()
    moe.forward(x.value)
    x.grad = moe.backward(weights)
    if dead_expert:
        dead = moe.experts[-1].parameters()
        assert all(not np.any(to_numpy(p.grad)) for p in dead), "dead expert got grad"

    return relative_error(params, reference_grads(params, loss_of_model))


def check_moe_dispatch(top_k: int, normalize_gates: bool) -> float:
    """Sparse dispatch must match the dense reference: outputs, dx and every param grad.

    Returns the largest difference relative to the largest value of each array.
    On the MLX GPU this is ~1e-3, not ~1e-7: its batched matmul carries ~1e-3
    relative error while a one-row matmul is exact to float32, and an expert
    that receives a single token takes the one-row path. A dispatch bug (wrong
    token or gate) shows up as O(1).
    """
    results = []
    for dispatch in ("dense", "sparse"):
        seed(0)
        moe = MoEFeedForward(
            8, 4, 8, top_k, normalize_gates, dropout=0.0, dispatch=dispatch
        )
        moe.expert_bias.value = normal(0.3, (4,))
        x, weights = normal(1.0, (2, 5, 8)), normal(1.0, (2, 5, 8))
        y = moe.forward(x)
        dx = moe.backward(weights)
        results.append([y, dx, *(p.grad for p in moe.parameters())])
    errors = []
    for a, b in zip(*results):
        a, b = to_numpy(a), to_numpy(b)
        scale = max(float(np.abs(a).max()), 1e-12)  # unused experts have all-zero grads
        errors.append(float(np.abs(a - b).max()) / scale)
    return max(errors)


def check_softcap(cap: float = 5.0) -> float:
    """Soft-capping with inputs up to ~3x the cap, so the saturated region is covered."""
    seed(0)
    x = Param(normal(cap, (4, 16)))
    weights = normal(1.0, (4, 16))  # loss = sum(out * weights)
    saturated = np.abs(to_numpy(x.value)) > 2 * cap
    assert saturated.any(), "test inputs should reach the saturated region"

    def loss_of_model():
        return (softcap(x.value, cap) * weights).sum()

    x.grad = softcap_backward(softcap(x.value, cap), weights, cap)

    return relative_error([x], reference_grads([x], loss_of_model))


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

    report("softcap cap=5", check_softcap(5.0))

    # log-probabilities magnify float32 rounding to ~1e-3; a misplaced norm gain gives ~0.2-0.7
    rel_tol = 1e-2 if BACKEND == "mlx" else 1e-7
    for kwargs in [{}, {"qk_norm": True}, {"qk_norm": True, "qk_norm_gain": True}]:
        opts = " ".join(f"{k}={v}" for k, v in kwargs.items()) or "plain"
        err = check_relative_position(**kwargs)
        report(f"rope relative position {opts}", err, rel_tol)

    # MLX GPU batched matmul is ~1e-3 relative; a dispatch bug gives O(1)
    dispatch_tol = 1e-2 if BACKEND == "mlx" else 1e-12
    for top_k, normalize in [(1, False), (2, True), (2, False)]:
        for dispatch in ("dense", "sparse"):
            report(
                f"moe {dispatch} top_k={top_k} normalize_gates={normalize}",
                check_moe(top_k, normalize, dispatch),
            )
        report(
            f"moe sparse == dense top_k={top_k} normalize_gates={normalize}",
            check_moe_dispatch(top_k, normalize),
            dispatch_tol,
        )
    report(
        "moe sparse with an unused expert",
        check_moe(2, True, "sparse", dead_expert=True),
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
