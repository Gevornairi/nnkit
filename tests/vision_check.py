"""Properties of the vision / self-supervised pieces that gradient checks can't see.

NNKIT_BACKEND=numpy NNKIT_DTYPE=float64 python -m tests.vision_check   # exact
python -m tests.vision_check                                           # MLX

Exits non-zero if any check fails.
"""

import numpy as np

from nnkit.activations import GELU
from nnkit.base import BACKEND, DTYPE, asarray, normal, seed, to_numpy, xp
from nnkit.layers import DropPath, Linear, PatchEmbedding, RoPE2D
from nnkit.losses import (
    koleo_loss,
    random_directions,
    sigreg_loss,
    view_invariance_loss,
)
from nnkit.training import ema_update, sinkhorn_knopp

# MLX GPU matmuls are ~1e-3 relative and float32 sums ~1e-7; a wrong rotation,
# ordering or formula gives O(1)
if BACKEND == "mlx":
    TOLERANCE = 1e-2
else:
    TOLERANCE = 1e-10 if DTYPE == xp.float64 else 1e-5


def check_rope_2d() -> None:
    """Scores depend only on the (row, col) offset; CLS/register tokens are not rotated."""
    seed(0)
    d_k, h, w, n_prefix = 16, 3, 4, 2
    rope = RoPE2D(d_k)
    rope.set_grid(h, w, n_prefix)
    q = to_numpy(normal(1.0, (d_k,)))
    k = to_numpy(normal(1.0, (d_k,)))
    t = n_prefix + h * w
    qs = to_numpy(rope.forward(asarray(np.tile(q, (1, 1, t, 1)))))[0, 0]
    ks = to_numpy(rope.forward(asarray(np.tile(k, (1, 1, t, 1)))))[0, 0]
    assert np.abs(qs[:n_prefix] - q).max() < TOLERANCE, "prefix tokens were rotated"

    scores = qs[n_prefix:] @ ks[n_prefix:].T
    by_offset: dict[tuple[int, int], list[float]] = {}
    for i in range(h * w):
        for j in range(h * w):
            offset = (j // w - i // w, j % w - i % w)
            by_offset.setdefault(offset, []).append(scores[i, j])
    spread = max(max(v) - min(v) for v in by_offset.values())
    assert spread < TOLERANCE, f"scores depend on absolute position ({spread:.1e})"
    distinct = {round(v[0], 6) for v in by_offset.values()}
    assert len(distinct) == len(by_offset), "different offsets give equal scores"


def check_rope_2d_axes() -> None:
    """Rows and columns are normalised separately: a column step means the same on any grid height."""
    seed(0)
    d_k, w = 16, 4
    q = to_numpy(normal(1.0, (d_k,)))
    k = to_numpy(normal(1.0, (d_k,)))
    scores = []
    for h in (1, 4):
        rope = RoPE2D(d_k)
        rope.set_grid(h, w)
        qs = to_numpy(rope.forward(asarray(np.tile(q, (1, 1, h * w, 1)))))[0, 0]
        ks = to_numpy(rope.forward(asarray(np.tile(k, (1, 1, h * w, 1)))))[0, 0]
        scores.append(qs[0] @ ks[1])  # same row, neighbouring columns
    gap = abs(scores[0] - scores[1]) / abs(scores[0])
    assert gap < TOLERANCE, f"column positions depend on the grid height ({gap:.1e})"


def check_patch_order() -> None:
    """Token r * (W / p) + c is the patch at grid row r, column c, flattened (row, col, channel)."""
    seed(0)
    p, h, w, c = 2, 4, 6, 3
    patch = PatchEmbedding(p, c, 5)
    image = to_numpy(normal(1.0, (2, h, w, c)))
    tokens = to_numpy(patch.forward(asarray(image)))
    weight = to_numpy(patch.proj.weight.value)
    for r in range(h // p):
        for col in range(w // p):
            block = image[:, r * p : (r + 1) * p, col * p : (col + 1) * p, :]
            expected = block.reshape(2, -1) @ weight
            gap = np.abs(tokens[:, r * (w // p) + col] - expected).max()
            gap /= np.abs(expected).max()
            assert gap < TOLERANCE, f"patch ({r}, {col}) is out of order"


def check_sinkhorn() -> None:
    """Rows are distributions and prototypes are used about equally, even for skewed logits."""
    seed(0)
    # favours the last prototypes
    logits = normal(1.0, (64, 8)) + asarray(np.linspace(0, 3, 8))
    q = to_numpy(sinkhorn_knopp(logits, temperature=0.5, n_iters=10))
    assert np.abs(q.sum(axis=1) - 1).max() < 1e-4, "rows don't sum to 1"
    usage = q.sum(axis=0) / 64
    assert np.abs(usage - 1 / 8).max() < 1e-3, f"prototype usage {usage}"


def check_ema() -> None:
    """momentum 1 keeps the teacher, 0 copies the student, 0.9 mixes them."""
    seed(0)
    teacher, student = Linear(4, 3, bias=True), Linear(4, 3, bias=True)
    assert student.bias is not None
    student.bias.value = normal(1.0, (3,))
    t0 = [to_numpy(p.value) for p in teacher.parameters()]
    s0 = [to_numpy(p.value) for p in student.parameters()]
    ema_update(teacher, student, 1.0)
    for p, v in zip(teacher.parameters(), t0):
        assert np.abs(to_numpy(p.value) - v).max() < 1e-12
    ema_update(teacher, student, 0.9)
    for p, t, s in zip(teacher.parameters(), t0, s0):
        assert np.abs(to_numpy(p.value) - (0.9 * t + 0.1 * s)).max() < 1e-6
    ema_update(teacher, student, 0.0)
    for p, s in zip(teacher.parameters(), s0):
        assert np.abs(to_numpy(p.value) - s).max() < 1e-12
    assert len(Linear(4, 3).parameters()) == 1, "Linear must stay bias-free by default"


def check_drop_path() -> None:
    """Each sample's branch is all dropped or all scaled by 1/(1-p); backward reuses the mask."""
    seed(0)
    drop = DropPath(0.5)
    x = asarray(np.ones((64, 3, 4)))
    y = to_numpy(drop.forward(x))
    per_sample = y.reshape(64, -1)
    assert np.all(per_sample.min(axis=1) == per_sample.max(axis=1)), (
        "mask is not per sample"
    )
    assert set(np.unique(per_sample)) <= {0.0, 2.0}
    assert 0 < (per_sample[:, 0] == 0).sum() < 64
    assert np.array_equal(to_numpy(drop.backward(x)), y), (
        "backward used a different mask"
    )
    drop.eval()
    assert np.array_equal(to_numpy(drop.forward(x)), to_numpy(x)), (
        "not identity in eval"
    )


def check_gelu_and_koleo() -> None:
    """GELU matches known values; KoLeo is lower for spread-out features and works per group."""
    gelu = to_numpy(GELU().forward(asarray([-1.0, 0.0, 1.0, 3.0])))
    expected = [-0.15865525, 0.0, 0.84134475, 2.99595031]
    assert np.abs(gelu - expected).max() < 1e-6, gelu

    seed(0)
    spread = asarray(np.eye(4)[None])  # 4 orthogonal unit vectors
    clustered = asarray(np.eye(4)[None] * 0.1 + 1.0)
    assert float(koleo_loss(spread)[0]) < float(koleo_loss(clustered)[0])
    x = normal(1.0, (2, 5, 4))
    both = float(koleo_loss(x)[0])
    each = (float(koleo_loss(x[:1])[0]) + float(koleo_loss(x[1:])[0])) / 2
    assert abs(both - each) < 1e-5, "groups are not independent"


def check_sigreg() -> None:
    """Matches the Epps-Pulley integral computed directly; low for N(0, I), high for collapse."""
    seed(0)
    z = normal(1.0, (2, 64, 8))
    a = random_directions(8, 16)
    assert np.abs(np.linalg.norm(to_numpy(a), axis=0) - 1).max() < 1e-5

    # the statistic straight from its definition: complex characteristic function,
    # trapezoid over the whole symmetric interval [-3, 3]
    t = np.linspace(-3.0, 3.0, 33)
    x = (to_numpy(z) @ to_numpy(a))[..., None] * t  # (views, N, slices, knots)
    phi = np.exp(1j * x).mean(axis=1)
    integrand = np.abs(phi - np.exp(-(t**2) / 2)) ** 2 * np.exp(-(t**2) / 2)
    expected = (64 * np.trapezoid(integrand, t, axis=-1)).mean()
    loss = float(sigreg_loss(z, directions=a)[0])
    assert abs(loss - expected) / expected < TOLERANCE, (loss, expected)

    collapsed = float(sigreg_loss(z[:, :1] + 0 * z, directions=a)[0])
    stretched = float(sigreg_loss(3.0 * z, directions=a)[0])
    assert loss < 0.1 * min(collapsed, stretched), (loss, collapsed, stretched)


def check_view_invariance() -> None:
    """Zero when every view agrees; the center is the mean of the global views only."""
    seed(0)
    one = normal(1.0, (1, 3, 5))
    same = one + 0 * normal(1.0, (4, 3, 5))
    assert float(view_invariance_loss(same)[0]) < 1e-12

    z = to_numpy(normal(1.0, (4, 3, 5)))
    expected = ((z - z[:2].mean(axis=0)) ** 2).mean()
    loss = float(view_invariance_loss(asarray(z), n_global=2)[0])
    assert abs(loss - expected) / expected < TOLERANCE, (loss, expected)


def main():
    ok = True
    for check in (
        check_rope_2d,
        check_rope_2d_axes,
        check_patch_order,
        check_sinkhorn,
        check_ema,
        check_drop_path,
        check_gelu_and_koleo,
        check_sigreg,
        check_view_invariance,
    ):
        try:
            check()
            print(f"[nnkit/{BACKEND}] {check.__name__}: ok")
        except AssertionError as e:
            ok = False
            print(f"[nnkit/{BACKEND}] {check.__name__}: FAIL {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
