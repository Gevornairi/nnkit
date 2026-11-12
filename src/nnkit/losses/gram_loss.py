"""Gram anchoring: keep patch-to-patch similarities close to a reference model's (DINOv3)."""

from ..base.backend import DTYPE, Array, xp


def _normalize(x: Array, eps: float) -> tuple[Array, Array]:
    inv_norm = 1.0 / xp.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)
    return x * inv_norm, inv_norm


def gram_loss(
    student: Array, reference: Array, remove_neg: bool = False, eps: float = 1e-12
) -> tuple[Array, Array]:
    """``mean((U U^T - V V^T)^2)`` over every entry, and its gradient w.r.t. ``student``.

    ``U`` and ``V`` are the student's and the reference's patch features,
    L2-normalised, so ``U U^T`` holds cosine similarities between patches.
    The reference ("Gram teacher", an earlier checkpoint) gets no gradient.
    Only similarities are matched, not the features, so the student's features
    may still move. With ``remove_neg`` both Gram matrices are clipped at 0.

    student, reference (..., n_patches, d): one Gram matrix per leading index,
    e.g. (batch, n, d) per image, or (1, batch * n, d) across the batch
    -> (0-d loss, d_student).
    """
    u, inv_norm = _normalize(student, eps)
    v, _ = _normalize(reference, eps)
    s = u @ xp.swapaxes(u, -1, -2)
    t = v @ xp.swapaxes(v, -1, -2)
    if remove_neg:
        s_kept = xp.maximum(s, 0.0)
        t = xp.maximum(t, 0.0)
    else:
        s_kept = s
    gap = s_kept - t
    loss = (gap * gap).mean()

    d_s = 2.0 * gap / gap.size
    if remove_neg:
        d_s = d_s * (s > 0)
    d_u = (d_s + xp.swapaxes(d_s, -1, -2)) @ u
    dx = inv_norm * (d_u - u * (d_u * u).sum(axis=-1, keepdims=True))

    return loss, dx.astype(DTYPE)
