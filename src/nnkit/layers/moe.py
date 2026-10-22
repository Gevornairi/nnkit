"""Mixture-of-experts feed-forward: a shared SwiGLU expert plus top-k routed ones."""

import numpy as np

from ..activations.sigmoid import sigmoid, sigmoid_backward
from ..base.backend import (
    DTYPE,
    Array,
    asarray,
    scatter_add,
    to_numpy,
    top_k_indices,
    xp,
)
from ..base.buffer import Buffer
from ..base.module import Module
from .linear import Linear
from .swiglu_feed_forward import SwiGLUFeedForward


class MoEFeedForward(Module):
    """``y = shared(x) + sum over the top-k experts e of g_e(x) * expert_e(x)``.

    Router scores are ``s = sigmoid(x W_r)``. Experts are chosen by top-k of
    ``s + b``, where ``b`` (``expert_bias``) is a non-trainable load-balancing
    bias (DeepSeek-V3); the gates ``g`` use the unbiased ``s``, renormalised over
    the chosen experts if ``normalize_gates``. Llama 4: ``top_k=1``,
    ``normalize_gates=False``. Shapes: (..., d_model) -> (..., d_model).

    ``dispatch="sparse"`` (default) runs each expert only on its own tokens.
    ``dispatch="dense"`` runs every expert on every token and multiplies the
    unchosen outputs by a zero gate: n_experts times the compute, kept as the
    reference the sparse path is tested against.

    With ``gate_input`` the gate scales the token *entering* each expert instead of
    its output, ``expert_e(g_e * x)``, as in Llama 4's reference implementation.
    Experts are nonlinear, so weak routes are suppressed more strongly than with
    output gating.

    No gradient flows through the top-k choice or ``b``. In training mode,
    ``forward`` counts the tokens each expert receives; call ``update_bias``
    after each optimizer step to nudge ``b`` toward balanced load.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        d_ff_expert: int,
        top_k: int = 1,
        normalize_gates: bool = False,
        bias_update_rate: float = 1e-3,
        dropout: float = 0.1,
        dispatch: str = "sparse",
        gate_input: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k={top_k} must be in [1, n_experts={n_experts}]")
        if dispatch not in ("sparse", "dense"):
            raise ValueError(f"dispatch={dispatch!r} must be 'sparse' or 'dense'")
        self.n_experts = n_experts
        self.top_k = top_k
        self.normalize_gates = normalize_gates
        self.bias_update_rate = bias_update_rate
        self.dispatch = dispatch
        self.gate_input = gate_input
        self.router = Linear(d_model, n_experts)
        self.experts = [
            SwiGLUFeedForward(d_model, d_ff_expert, dropout) for _ in range(n_experts)
        ]
        self.shared = SwiGLUFeedForward(d_model, d_ff_expert, dropout)
        self.register_modules(self.router, *self.experts, self.shared)
        self.expert_bias = Buffer(xp.zeros(n_experts, dtype=DTYPE))
        self.register_buffers(self.expert_bias)
        # tokens routed to each expert since the last update_bias(), (n_experts,)
        self.load = xp.zeros(n_experts, dtype=DTYPE)
        self._cache = None

    def forward(self, x: Array) -> Array:
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        s = sigmoid(self.router.forward(x))
        idx = top_k_indices(s + self.expert_bias.value, self.top_k)
        mask = (idx[:, :, None] == xp.arange(self.n_experts)).astype(DTYPE).sum(axis=1)

        s_sel = s * mask
        denom = s_sel.sum(axis=-1, keepdims=True) if self.normalize_gates else None
        g = s_sel / denom if self.normalize_gates else s_sel

        if self.dispatch == "dense":
            routed, expert_cache = self._dense_forward(x, g)
        else:
            routed, expert_cache = self._sparse_forward(x, g, idx)

        y = self.shared.forward(x) + routed

        if self.training:
            self.load = self.load + mask.sum(axis=0)
        self._cache = (s, mask, denom, g, expert_cache, shape)

        return y.reshape(shape)

    def backward(self, dy: Array) -> Array:
        """Through the shared expert, each expert's gated output, and the router.

        The gate of expert e is g_e = s_e (or s_e / sum of chosen s); its
        gradient reaches the router through the sigmoid. Unchosen experts get
        zero gradient from the mask.
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        s, mask, denom, g, expert_cache, shape = self._cache
        dy = dy.reshape(-1, shape[-1])

        if self.dispatch == "dense":
            dx, d_g = self._dense_backward(dy, g, expert_cache)
        else:
            dx, d_g = self._sparse_backward(dy, expert_cache)
        dx = dx + self.shared.backward(dy)
        if self.normalize_gates:
            # quotient rule for g = s_sel / sum(s_sel)
            d_s_sel = (d_g - (g * d_g).sum(axis=-1, keepdims=True)) / denom
        else:
            d_s_sel = d_g

        d_logits = sigmoid_backward(s, d_s_sel * mask)
        dx = dx + self.router.backward(d_logits)

        return dx.reshape(shape)

    def update_bias(self) -> None:
        """Raise the bias of under-loaded experts and lower over-loaded ones, then reset the count.

        Call once per optimizer step (after all gradient-accumulation
        micro-batches). Each bias moves by exactly ``bias_update_rate``, so a
        rate that is large relative to the gaps between router scores makes the
        load oscillate instead of settling.
        """
        imbalance = xp.sign(self.load.mean() - self.load)
        self.expert_bias.value = (
            self.expert_bias.value + self.bias_update_rate * imbalance
        )
        self.load = xp.zeros_like(self.load)

    def _dense_forward(self, x: Array, g: Array) -> tuple[Array, Array]:
        """Every expert on every token; returns (routed sum, cache for backward).

        The cache is the outputs (N, E, d) with output gating, or the input x with
        ``gate_input`` (unchosen experts then get a zero input, which SwiGLU without
        biases maps to zero).
        """
        if self.gate_input:
            routed = sum(
                e.forward(g[:, i : i + 1] * x) for i, e in enumerate(self.experts)
            )

            return routed, x

        outs = xp.stack([e.forward(x) for e in self.experts], axis=1)
        return (g[:, :, None] * outs).sum(axis=1), outs

    def _sparse_forward(self, x: Array, g: Array, idx: Array) -> tuple[Array, list]:
        """Each expert on its own tokens; returns (sum of gated outputs, per-expert cache).

        Dispatch indices are integer constants with no gradient, so they are
        built on the CPU (one sync of the (N, k) choices per layer); on MLX this
        also keeps autodiff from trying to differentiate through them.
        """
        flat_expert = to_numpy(idx).reshape(-1)  # entry i is (token i // k, slot i % k)
        order = np.argsort(flat_expert, kind="stable")
        tokens_by_expert = order // self.top_k  # token ids, grouped by expert
        ends = np.cumsum(np.bincount(flat_expert, minlength=self.n_experts))

        routed = xp.zeros_like(x)
        active = []  # (expert, rows, gate, output, input) for experts that got tokens
        for e in range(self.n_experts):
            start = ends[e - 1] if e else 0
            if start == ends[e]:
                continue  # no tokens: skipped in backward too, its cache is stale
            rows = asarray(tokens_by_expert[start : ends[e]])
            gate = g[rows, e][:, None]
            x_e = x[rows]
            if self.gate_input:
                out = self.experts[e].forward(gate * x_e)
                contribution = out
            else:
                out = self.experts[e].forward(x_e)
                contribution = gate * out
            # a token goes to k experts, so results are accumulated, not assigned
            routed = scatter_add(routed, rows, contribution)
            active.append((e, rows, gate, out, x_e))
        return routed, active

    def _dense_backward(self, dy: Array, g: Array, cache: Array) -> tuple[Array, Array]:
        """Returns (dx from the experts, d_g (N, E))."""
        dx = xp.zeros_like(dy)
        if self.gate_input:
            x, d_g_cols = cache, []
            for i, expert in enumerate(self.experts):
                du = expert.backward(dy)
                d_g_cols.append((du * x).sum(axis=-1))
                dx = dx + g[:, i : i + 1] * du
            return dx, xp.stack(d_g_cols, axis=1)

        outs = cache
        for e, expert in enumerate(self.experts):
            dx = dx + expert.backward(g[:, e : e + 1] * dy)
        return dx, (dy[:, None, :] * outs).sum(axis=-1)

    def _sparse_backward(self, dy: Array, active: list) -> tuple[Array, Array]:
        """Returns (dx from the experts, d_g (N, E)); unchosen gates get zero gradient."""
        n = dy.shape[0]
        dx = xp.zeros_like(dy)
        d_g_cols = [xp.zeros(n, dtype=dy.dtype) for _ in range(self.n_experts)]
        for e, rows, gate, out, x_e in active:
            dy_e = dy[rows]
            if self.gate_input:
                du = self.experts[e].backward(dy_e)
                d_gate, dx_e = (du * x_e).sum(axis=-1), gate * du
            else:
                d_gate = (dy_e * out).sum(axis=-1)
                dx_e = self.experts[e].backward(gate * dy_e)

            d_g_cols[e] = scatter_add(d_g_cols[e], rows, d_gate)
            dx = scatter_add(dx, rows, dx_e)
        return dx, xp.stack(d_g_cols, axis=1)
