"""Check nnkit's optimisers and schedules: Muon, Newton-Schulz, the WSD schedule.

python -m tests.optim_check

Exits non-zero if any check fails.
"""

import itertools

import numpy as np

from nnkit.base import Module, Param, asarray, normal, seed, to_numpy, xp
from nnkit.layers import Embedding, Linear, RMSNorm
from nnkit.optimisers import AdamW, Muon, WSDSchedule, newton_schulz, split_muon_params


def check_newton_schulz() -> None:
    """Singular values pushed to ~1, singular vectors kept (close to U V^T of the SVD).

    Five iterations lift small singular values a lot but not all the way: a
    direction the input barely has (< 0.5% of the largest singular value) stays
    small, which is the intended behaviour. Only the others must land near 1.
    """
    seed(0)
    for shape in ((16, 32), (48, 12), (24, 24)):
        # uneven spectrum
        g = to_numpy(normal(1.0, shape) @ normal(1.0, (shape[1], shape[1])))
        u, s, vt = np.linalg.svd(g, full_matrices=False)
        o = to_numpy(newton_schulz(asarray(g)))
        # o's singular value along each of g's singular directions: u_i^T o v_i
        along = np.einsum("ji,jk,ik->i", u, o, vt)
        kept = s / s.max() > 5e-3
        assert np.all((0.5 < along[kept]) & (along[kept] < 1.3)), (shape, along[kept])
        assert np.all(along[~kept] < 1.3), (shape, along[~kept])
        cos = (o * (u @ vt)).sum() / (np.linalg.norm(o) * np.linalg.norm(u @ vt))
        assert cos > 0.95, (shape, cos)


def check_muon_update_rms() -> None:
    """One step's update has RMS ~0.2 * lr, the AdamW-matched scale, for any matrix shape."""
    seed(0)
    for shape in ((64, 32), (32, 128)):
        p = Param(normal(1.0, shape))
        p.grad = normal(1.0, shape) * 1e-3  # gradient size must not matter
        before = to_numpy(p.value)
        Muon([p], lambda step: 1.0, nesterov=False).step()
        rms = np.sqrt(((to_numpy(p.value) - before) ** 2).mean())
        assert 0.1 < rms < 0.3, (shape, rms)


def check_muon_momentum() -> None:
    """With a zero gradient the momentum buffer only decays; it must never grow."""
    p = Param(xp.ones((4, 4)))
    opt = Muon([p], lambda step: 0.0)
    opt.m[0] = xp.ones((4, 4))
    for _ in range(3):
        p.grad = xp.zeros((4, 4))
        opt.step()
    np.testing.assert_allclose(to_numpy(opt.m[0]), 0.95**3, rtol=1e-5)


def check_muon_rejects_vectors() -> None:
    try:
        Muon([Param(xp.ones(8))], lambda step: 1e-3)
    except ValueError:
        return
    raise AssertionError("Muon accepted a 1-D parameter")


def check_muon_converges() -> None:
    """Least squares X W ~ Y: Muon drives the loss down steadily."""
    seed(0)
    x, w_true = normal(1.0, (256, 16)), normal(1.0, (16, 8))
    y = x @ w_true
    w = Param(xp.zeros((16, 8)))
    opt = Muon([w], lambda step: 0.05)
    losses = []
    for _ in range(200):
        err = x @ w.value - y
        losses.append(float((err * err).mean()))
        w.grad = 2 * x.T @ err / err.size
        opt.step()
    assert losses[-1] < 0.01 * losses[0], (losses[0], losses[-1])


def check_muon_state_round_trip() -> None:
    seed(0)
    p = Param(normal(1.0, (8, 4)))
    opt = Muon([p], lambda step: 1e-2)
    p.grad = normal(1.0, (8, 4))
    opt.step()
    restored = Muon([Param(xp.zeros((8, 4)))], lambda step: 1e-2)
    restored.load_state(opt.state())
    assert restored.step_count == 1
    np.testing.assert_array_equal(to_numpy(restored.m[0]), to_numpy(opt.m[0]))


def check_split_muon_params() -> None:
    """Hidden matrices to Muon; the embedding table and all 1-D params to AdamW."""

    class Tiny(Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = Embedding(10, 8)
            self.hidden = Linear(8, 8)
            self.norm = RMSNorm(8)
            self.register_modules(self.embed, self.hidden, self.norm)

    model = Tiny()
    muon, adamw = split_muon_params(model, exclude=[model.embed.weight])
    assert muon == [model.hidden.weight], muon
    assert {id(p) for p in adamw} == {id(model.embed.weight), id(model.norm.gain)}
    # the two optimisers together cover every parameter exactly once
    Muon(muon, lambda s: 1e-3), AdamW(adamw, lambda s: 1e-3)
    assert len(muon) + len(adamw) == len(model.parameters())


def check_wsd_schedule() -> None:
    for shape in ("sqrt", "linear"):
        s = WSDSchedule(
            1.0, warmup=10, total_steps=100, decay_steps=20, min_lr=0.1, shape=shape
        )
        assert s(0) == 0.0 and s(5) == 0.5 and s(10) == 1.0  # linear warmup
        assert s(50) == 1.0 and s(79) == 1.0  # stable
        assert abs(s(80) - 1.0) < 1e-12 and abs(s(100) - 0.1) < 1e-12
        assert s(500) == s(100)  # stays at min_lr after the run
        decay = [s(t) for t in range(80, 101)]
        assert all(a >= b for a, b in itertools.pairwise(decay)), (
            "lr went up during decay"
        )
    # 1 - sqrt decays faster at first than linear
    sq = WSDSchedule(1.0, 0, 100, 20, shape="sqrt")
    lin = WSDSchedule(1.0, 0, 100, 20, shape="linear")
    assert sq(85) < lin(85)
    # decaying over the whole post-warmup range is allowed
    WSDSchedule(1.0, warmup=10, total_steps=100, decay_steps=90)
    for bad in ({"decay_steps": 0}, {"decay_steps": 95}, {"shape": "cos"}):
        args = {
            "max_lr": 1.0,
            "warmup": 10,
            "total_steps": 100,
            "decay_steps": 20,
        } | bad
        try:
            WSDSchedule(**args)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def main():
    ok = True
    for check in (
        check_newton_schulz,
        check_muon_update_rms,
        check_muon_momentum,
        check_muon_rejects_vectors,
        check_muon_converges,
        check_muon_state_round_trip,
        check_split_muon_params,
        check_wsd_schedule,
    ):
        try:
            check()
            print(f"{check.__name__}: ok")
        except Exception as e:  # noqa: BLE001 - report every failing check
            ok = False
            print(f"{check.__name__}: FAIL {type(e).__name__}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
