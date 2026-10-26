"""z-loss: an auxiliary penalty that keeps the softmax normaliser near 1 (PaLM)."""

from ..base.backend import DTYPE, Array, xp


def z_loss(
    logits: Array, targets: Array, pad_id: int, coef: float = 1e-4
) -> tuple[Array, Array]:
    """``coef * mean((log Z)^2)`` over non-PAD positions, and its gradient w.r.t. the logits.

    ``log Z = logsumexp(logits)`` at each position. Softmax ignores a shift of
    all logits by a constant, so cross-entropy alone never stops them drifting
    together; this pins ``log Z`` near 0. Since d(log Z)/d(logits) = softmax,
    the gradient is ``2 * coef * log Z * softmax(logits) / n_tokens``.

    logits (batch, T, vocab), targets (batch, T) -> (0-d loss, d_logits). Add both
    to the main loss and its gradient; positions are averaged like the main loss.
    """

    b, t, v = logits.shape
    flat = logits.reshape(-1, v)
    shift = flat.max(axis=-1, keepdims=True)
    e = xp.exp(flat - shift)
    total = e.sum(axis=-1, keepdims=True)
    log_z = shift + xp.log(total)  # (b * t, 1)
    probs = e / total

    valid = (targets.reshape(-1) != pad_id).astype(DTYPE)[:, None]
    n_tokens = xp.maximum(valid.sum(), 1.0)
    loss = coef * (valid * log_z * log_z).sum() / n_tokens
    d_logits = 2.0 * coef * valid * log_z * probs / n_tokens

    return loss, d_logits.reshape(b, t, v).astype(DTYPE)
