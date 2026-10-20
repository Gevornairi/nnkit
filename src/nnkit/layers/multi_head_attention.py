"""Multi-head scaled dot-product attention with optional RoPE, grouped-query
attention, QK-Norm, NoPE temperature scaling, and differential attention."""

import math

from ..activations.softmax import softmax, softmax_backward
from ..base.backend import Array, xp
from ..base.module import Module
from ..base.param import Param
from .dropout import Dropout
from .linear import Linear
from .rms_norm import RMSNorm
from .rope import RoPE

NEG_INF = -1e9  # score given to masked positions; exp() of it underflows to 0


def diff_lambda_init(layer: int) -> float:
    """Differential attention's depth-dependent lambda_init for 0-indexed ``layer``."""
    return 0.8 - 0.6 * math.exp(-0.3 * layer)


class MultiHeadAttention(Module):
    """``softmax(Q K^T / sqrt(d_k) + mask) V`` over ``n_heads`` heads, then an output projection.

    Used both as self-attention (``query_in is kv_in``) and as cross-attention
    (queries from the decoder, keys/values from the encoder memory).

    With ``n_kv_heads < n_heads`` it is grouped-query attention: only
    ``n_kv_heads`` key/value heads are projected, and each serves
    ``n_rep = n_heads // n_kv_heads`` query heads (``n_kv_heads = 1`` is
    multi-query attention; Ainslie et al., 2023).

    If ``rope`` is given, queries and keys are rotated by position after the
    head split; pass it for self-attention only.

    If ``qk_norm`` is True, queries and keys are RMS-normalised per head:
    without a gain and after RoPE by default (Llama 4), or with a learned gain
    shared across heads and before RoPE if ``qk_norm_gain`` is True (Qwen3,
    OLMo 2, Gemma 3). A per-dimension gain applied after RoPE would scale the
    two dimensions of a rotated pair differently and make scores depend on
    absolute position, so the placement follows the flag.

    If ``attn_scale_alpha`` is non-zero, the query at position ``t``
    is scaled by ``1 + attn_scale_alpha * log(1 + floor((t + 1) / attn_scale_floor))``,
    Llama 4's temperature scaling for its NoPE layers, so ``rope`` must be None.

    With ``diff_attn`` it is differential attention (Ye et al., 2025): each
    head's query and key are split into two sub-heads of size ``d_k / 2``
    (RoPE and QK-Norm act on each, so pass ``RoPE(d_k // 2)``), giving two
    softmax maps A1 and A2. The head attends with ``A1 - lambda * A2``; its
    output goes through a shared RMSNorm and is scaled by ``1 - lambda_init``.
    Unlike the paper, lambda is a free per-head parameter initialised to
    ``lambda_init`` rather than ``exp(lq1.lk1) - exp(lq2.lk2) + lambda_init``;
    use ``diff_lambda_init(layer)`` for the paper's depth-dependent value.

    Shapes: query_in (batch, T_q, d_model), kv_in (batch, T_k, d_model),
    mask broadcastable to (batch, n_heads, T_q, T_k) -> (batch, T_q, d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        rope: RoPE | None = None,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        qk_norm_gain: bool = False,
        attn_scale_alpha: float = 0.0,
        attn_scale_floor: int | None = None,
        diff_attn: bool = False,
        lambda_init: float = 0.8,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        n_kv_heads = n_kv_heads or n_heads
        if n_heads % n_kv_heads:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
            )
        if qk_norm_gain and not qk_norm:
            raise ValueError("qk_norm_gain requires qk_norm=True")
        if attn_scale_alpha and (attn_scale_floor is None or attn_scale_floor <= 0):
            raise ValueError(
                "attn_scale_floor must be > 0 when attn_scale_alpha is set"
            )
        if attn_scale_alpha and rope is not None:
            raise ValueError("attn_scale_alpha is for NoPE layers; pass rope=None")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # query heads per key/value head
        self.d_k = d_model // n_heads
        if diff_attn and self.d_k % 2:
            raise ValueError(f"d_k={self.d_k} must be even for diff_attn")
        # size of the query/key vectors whose dot products form the scores
        self.d_qk = self.d_k // 2 if diff_attn else self.d_k
        if rope is not None and 2 * rope.half != self.d_qk:
            raise ValueError(
                f"rope is built for d_k={2 * rope.half}, needs {self.d_qk}"
            )
        self.scale = 1.0 / math.sqrt(self.d_qk)
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, n_kv_heads * self.d_k)
        self.w_v = Linear(d_model, n_kv_heads * self.d_k)
        self.w_o = Linear(d_model, d_model)
        self.attn_dropout = Dropout(dropout)
        self.register_modules(self.w_q, self.w_k, self.w_v, self.w_o, self.attn_dropout)
        self.rope = rope
        self.qk_norm = qk_norm
        self.qk_norm_before_rope = qk_norm_gain
        if qk_norm:
            self.q_norm = RMSNorm(self.d_qk, gain=qk_norm_gain)
            self.k_norm = RMSNorm(self.d_qk, gain=qk_norm_gain)
            self.register_modules(self.q_norm, self.k_norm)
        self.attn_scale_alpha = attn_scale_alpha
        self.attn_scale_floor = attn_scale_floor
        self.diff_attn = diff_attn
        self.lambda_init = lambda_init
        if diff_attn:
            self.diff_lambda = Param(xp.full((n_heads,), lambda_init))
            self.sub_norm = RMSNorm(self.d_k)
            self.register_params(self.diff_lambda)
            self.register_modules(self.sub_norm)
        # attention probabilities of the last forward, (batch, n_heads, T_q, T_k), for plotting
        self.attn_weights: Array | None = None
        self._cache = None

    def _split_heads(self, x: Array, n_heads: int) -> Array:
        """(batch, T, n_heads * d_k) -> (batch, n_heads, T, d_k)."""
        b, t, _ = x.shape
        return x.reshape(b, t, n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Array) -> Array:
        """(batch, n_heads, T, d_k) -> (batch, T, n_heads * d_k)."""
        b, h, t, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * d_k)

    def _split_halves(self, x: Array) -> Array:
        """Differential attention sub-heads: (batch, h, T, d_k) -> (batch, h, 2, T, d_k / 2)."""
        b, h, t, _ = x.shape
        return x.reshape(b, h, t, 2, self.d_qk).transpose(0, 1, 3, 2, 4)

    def _merge_halves(self, x: Array) -> Array:
        """(batch, h, 2, T, d_k / 2) -> (batch, h, T, d_k)."""
        b, h, _, t, _ = x.shape
        return x.transpose(0, 1, 3, 2, 4).reshape(b, h, t, self.d_k)

    def _repeat_kv(self, x: Array) -> Array:
        """(batch, n_kv_heads, ...) -> (batch, n_heads, ...); query head j uses kv head j // n_rep."""
        return x if self.n_rep == 1 else xp.repeat(x, self.n_rep, axis=1)

    def _sum_kv_grads(self, dx: Array) -> Array:
        """Backward of ``_repeat_kv``: sum the gradients of each kv head's copies."""
        if self.n_rep == 1:
            return dx
        b, _, *rest = dx.shape
        return dx.reshape(b, self.n_kv_heads, self.n_rep, *rest).sum(axis=2)

    def _temp_scale(self, q: Array) -> Array:
        """Position-dependent query scale (T_q, 1): 1 + alpha * log(1 + floor((t + 1) / floor_len))."""
        t = xp.arange(q.shape[-2], dtype=q.dtype)[:, None]
        steps = xp.floor((t + 1.0) / self.attn_scale_floor)
        return 1.0 + self.attn_scale_alpha * xp.log(1.0 + steps)

    def forward(
        self, query_in: Array, kv_in: Array, mask: Array | None = None
    ) -> Array:
        q = self._split_heads(self.w_q.forward(query_in), self.n_heads)
        k = self._split_heads(self.w_k.forward(kv_in), self.n_kv_heads)
        v = self._split_heads(self.w_v.forward(kv_in), self.n_kv_heads)

        if self.diff_attn:
            q, k = self._split_halves(q), self._split_halves(k)
            if mask is not None:
                mask = xp.expand_dims(mask, -3)  # broadcast over the two sub-heads

        if self.qk_norm and self.qk_norm_before_rope:
            q = self.q_norm.forward(q)
            k = self.k_norm.forward(k)

        if self.rope is not None:
            q = self.rope.forward(q)
            k = self.rope.forward(k)

        if self.qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm.forward(q)
            k = self.k_norm.forward(k)

        if self.attn_scale_alpha:
            q = q * self._temp_scale(q)

        k, v = self._repeat_kv(k), self._repeat_kv(v)

        scores = (q @ xp.swapaxes(k, -1, -2)) * self.scale
        if mask is not None:
            scores = xp.where(mask, scores, NEG_INF)
        probs = softmax(scores, axis=-1)  # (batch, n_heads, [2,] T_q, T_k)

        if self.diff_attn:
            attn = (
                probs[:, :, 0] - self.diff_lambda.value[:, None, None] * probs[:, :, 1]
            )
        else:
            attn = probs

        attn_dropped = self.attn_dropout.forward(attn)
        context = attn_dropped @ v
        if self.diff_attn:
            context = self.sub_norm.forward(context) * (1.0 - self.lambda_init)

        out = self.w_o.forward(self._merge_heads(context))
        self.attn_weights = attn
        # q and k are cached after RoPE / QK-Norm / temp scaling: d_scores is w.r.t. the final q and k
        self._cache = (q, k, v, probs, attn_dropped, mask)

        return out

    def backward(self, d_out: Array) -> tuple[Array, Array]:
        """Returns (d_query_in, d_kv_in).

        For self-attention the caller adds the two, since both inputs are the
        same tensor; for cross-attention d_kv_in is the gradient w.r.t. the
        encoder memory.
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        q, k, v, probs, attn_dropped, mask = self._cache

        d_context = self._split_heads(self.w_o.backward(d_out), self.n_heads)
        if self.diff_attn:
            d_context = self.sub_norm.backward(d_context * (1.0 - self.lambda_init))
        d_attn_dropped = d_context @ xp.swapaxes(v, -1, -2)
        d_v = xp.swapaxes(attn_dropped, -1, -2) @ d_context

        d_attn = self.attn_dropout.backward(d_attn_dropped)
        if self.diff_attn:
            # attn = A1 - lambda * A2
            self.diff_lambda.grad += -(d_attn * probs[:, :, 1]).sum(axis=(0, 2, 3))
            lam = self.diff_lambda.value[:, None, None]
            d_probs = xp.stack([d_attn, -lam * d_attn], axis=2)
        else:
            d_probs = d_attn

        d_scores = softmax_backward(probs, d_probs, axis=-1)
        if mask is not None:
            d_scores = xp.where(mask, d_scores, 0.0)
        d_scores = d_scores * self.scale

        d_q = d_scores @ k
        d_k = xp.swapaxes(d_scores, -1, -2) @ q
        d_k, d_v = self._sum_kv_grads(d_k), self._sum_kv_grads(d_v)

        if self.attn_scale_alpha:
            d_q = d_q * self._temp_scale(d_q)

        if self.qk_norm and not self.qk_norm_before_rope:
            d_q = self.q_norm.backward(d_q)
            d_k = self.k_norm.backward(d_k)

        if self.rope is not None:
            d_q = self.rope.backward(d_q)
            d_k = self.rope.backward(d_k)

        if self.qk_norm and self.qk_norm_before_rope:
            d_q = self.q_norm.backward(d_q)
            d_k = self.k_norm.backward(d_k)

        if self.diff_attn:
            d_q, d_k = self._merge_halves(d_q), self._merge_halves(d_k)

        d_query_in = self.w_q.backward(self._merge_heads(d_q))
        d_kv_from_k = self.w_k.backward(self._merge_heads(d_k))
        d_kv_from_v = self.w_v.backward(self._merge_heads(d_v))

        return d_query_in, d_kv_from_k + d_kv_from_v
