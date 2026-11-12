"""Cross-entropy against soft targets: the DINO and iBOT self-distillation losses."""

from ..activations.softmax import log_softmax
from ..base.backend import DTYPE, Array, xp


def soft_cross_entropy(
    logits: Array,
    target_probs: Array,
    temperature: float = 1.0,
    weights: Array | None = None,
) -> tuple[Array, Array]:
    """``sum_i w_i * H(t_i, softmax(z_i / temperature))`` and its gradient w.r.t. the logits.

    ``target_probs`` are the teacher's distributions (no gradient), e.g. from
    ``sinkhorn_knopp``. DINO compares the student's CLS logits for one crop
    with the teacher's for another; iBOT compares masked patch logits with
    the teacher's at the same patches. ``weights`` (one per row, e.g.
    1 / masked patches of the row's image for iBOT) default to 1 / rows, a
    plain mean. The gradient is ``w_i * (sum(t_i) * softmax(z_i / T) - t_i) / T``.

    logits and target_probs (..., K), weights (...) -> (0-d loss, d_logits).
    """
    k = logits.shape[-1]
    z = logits.reshape(-1, k) / temperature
    t = target_probs.reshape(-1, k)
    if weights is None:
        w = xp.full((z.shape[0], 1), 1.0 / z.shape[0], dtype=DTYPE)
    else:
        w = weights.reshape(-1, 1)

    log_p = log_softmax(z, axis=-1)
    loss = -(w * t * log_p).sum()
    d_z = w * (t.sum(axis=-1, keepdims=True) * xp.exp(log_p) - t)

    return loss, (d_z / temperature).reshape(logits.shape).astype(DTYPE)
