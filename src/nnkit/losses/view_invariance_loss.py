"""Pull every view's embedding towards the global views' mean (LeJEPA's prediction term)."""

from ..base.backend import DTYPE, Array, xp


def view_invariance_loss(z: Array, n_global: int | None = None) -> tuple[Array, Array]:
    """``mean((z_v - center)^2)`` over views, samples and features, and its gradient.

    ``center`` is the mean of the first ``n_global`` views (all views by
    default) of each sample. It is not detached, so the gradient also reaches
    the global views through it; with ``n_global`` = all views that part sums
    to zero.

    z (views, N, d), global views first -> (0-d loss, dz).
    """
    n_global = n_global or z.shape[0]
    center = z[:n_global].mean(axis=0)
    gap = z - center
    loss = (gap * gap).mean()

    d_gap = 2.0 * gap / gap.size
    d_center = -d_gap.sum(axis=0) / n_global
    is_global = (xp.arange(z.shape[0]) < n_global).astype(z.dtype)[:, None, None]
    dz = d_gap + is_global * d_center

    return loss, dz.astype(DTYPE)
