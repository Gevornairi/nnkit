"""KoLeo regulariser: spread features out within a batch (Sablayrolles et al., 2019; DINOv2)."""

import numpy as np

from ..base.backend import DTYPE, INT, Array, asarray, scatter_add, to_numpy, xp


def koleo_loss(x: Array, eps: float = 1e-8) -> tuple[Array, Array]:
    """``-mean_i log(||u_i - u_nn(i)|| + eps)`` and its gradient w.r.t. ``x``.

    ``u`` is ``x`` L2-normalised and ``nn(i)`` the nearest other row of the
    same group. Pushing every point away from its nearest neighbour drives
    the features towards a uniform spread on the sphere. The neighbour choice
    is a constant for the gradient, which reaches both ends of each pair.

    x (..., n, d): the last-but-one axis is one group, e.g. (batch / 16, 16, d)
    for DINOv3's groups of 16 -> (0-d loss, dx).
    """

    *lead, n, d = x.shape
    inv_norm = 1.0 / xp.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)
    u = (x * inv_norm).reshape(-1, n, d)
    g = u.shape[0]

    # nearest neighbours on the CPU, so no gradient can flow through the indices
    sims = to_numpy(u @ xp.swapaxes(u, -1, -2)) - 3.0 * np.eye(n)  # skip self
    nn = np.argmax(sims, axis=-1) + np.arange(g)[:, None] * n
    nn_flat = asarray(nn.reshape(-1), INT)
    u_flat = u.reshape(-1, d)
    diff = u_flat - u_flat[nn_flat]
    dist = xp.sqrt((diff * diff).sum(axis=-1))
    count = g * n
    loss = -xp.log(dist + eps).sum() / count

    # d loss / d diff, then diff = u_i - u_nn(i) sends it to both rows
    d_diff = (-1.0 / (count * (dist + eps) * xp.maximum(dist, eps)))[:, None] * diff
    d_u = scatter_add(d_diff, nn_flat, -d_diff).reshape(*lead, n, d)
    u = u.reshape(*lead, n, d)
    dx = inv_norm * (d_u - u * (d_u * u).sum(axis=-1, keepdims=True))

    return loss, dx.astype(DTYPE)
