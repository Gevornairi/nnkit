"""SIGReg: push a batch of embeddings towards an isotropic Gaussian (LeJEPA, Balestriero & LeCun, 2025)."""

import math

from ..base.backend import DTYPE, Array, asarray, normal, xp


def random_directions(d: int, n_slices: int) -> Array:
    """``n_slices`` random unit vectors in R^d as the columns of a (d, n_slices) matrix."""
    a = normal(1.0, (d, n_slices))
    return a / xp.sqrt((a * a).sum(axis=0, keepdims=True))


def sigreg_loss(
    z: Array,
    n_slices: int = 256,
    n_knots: int = 17,
    t_max: float = 3.0,
    directions: Array | None = None,
) -> tuple[Array, Array]:
    """Sketched isotropic Gaussian regulariser and its gradient w.r.t. ``z``.

    Each row of ``z`` is projected onto random unit directions, and each 1-D
    projection is compared with N(0, 1) by the Epps-Pulley statistic

        N * integral over t of |phi_N(t) - exp(-t^2 / 2)|^2 * exp(-t^2 / 2) dt,

    where ``phi_N(t) = mean_n exp(i t x_n)`` is the empirical characteristic
    function of the N projections and ``exp(-t^2 / 2)`` that of N(0, 1). The
    integrand is even, so the integral over [-t_max, t_max] is twice a
    trapezoid rule on ``n_knots`` points of [0, t_max]. A collapsed or
    stretched batch is far from N(0, I) along some direction, so this alone
    prevents collapse: no teacher, stop-gradient or centering.

    ``directions`` (d, n_slices) are drawn fresh by default (``random_directions``),
    as LeJEPA resamples them every step; pass them to fix the slices.

    z (..., N, d): the leading axes are separate batches (e.g. one per view),
    each tested on its own -> (0-d loss averaged over batches and slices, dz).
    """
    *lead, n, d = z.shape
    a = random_directions(d, n_slices) if directions is None else directions
    m = a.shape[1]

    dt = t_max / (n_knots - 1)
    t = asarray([k * dt for k in range(n_knots)], DTYPE)
    gauss = xp.exp(-0.5 * t * t)  # N(0, 1)'s characteristic function, also the window
    ends = (0, n_knots - 1)
    trapezoid = asarray([dt if k in ends else 2 * dt for k in range(n_knots)], DTYPE)
    w = trapezoid * gauss

    x = (z.reshape(-1, n, d) @ a)[..., None] * t  # (batches, N, slices, knots)
    cos_x, sin_x = xp.cos(x), xp.sin(x)
    c = cos_x.mean(axis=1, keepdims=True) - gauss  # real part of phi_N - phi
    s = sin_x.mean(axis=1, keepdims=True)  # imaginary part
    count = math.prod(lead) * m
    loss = n * ((c * c + s * s) * w).sum() / count

    # d loss / d x_n = 2 / count * sum_t w t (s cos(t x_n) - c sin(t x_n))
    d_x = ((2.0 / count) * w * t * (s * cos_x - c * sin_x)).sum(axis=-1)
    dz = (d_x @ a.T).reshape(z.shape)

    return loss, dz.astype(DTYPE)
