"""Check nnkit.training: checkpoint round trip and the background prefetcher.

python -m tests.training_check

Exits non-zero if any check fails.
"""

import tempfile
from pathlib import Path

import numpy as np

from nnkit.base import seed, to_numpy
from nnkit.layers import Linear, RMSNorm
from nnkit.optimisers import AdamW
from nnkit.training import (
    Prefetcher,
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)


class _Tiny(Linear):
    """A Linear plus an RMSNorm, so the checkpoint has 2D and 1D parameters."""

    def __init__(self) -> None:
        super().__init__(4, 3)
        self.norm = RMSNorm(3)
        self.register_modules(self.norm)


def _values(module) -> list[np.ndarray]:
    return [to_numpy(p.value) for p in module.parameters()]


def check_checkpoint_round_trip() -> None:
    seed(0)
    model = _Tiny()
    opt = AdamW(model.parameters(), lambda step: 1e-2, weight_decay=0.1)
    for p in model.parameters():  # one update so the optimizer state is non-trivial
        p.grad = p.value * 0 + 1
    opt.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.npz"
        save_checkpoint(path, model, opt, step=7, best_score=1.5, args='{"lr": 0.01}')

        seed(1)  # different init, so a successful load is visible
        restored = _Tiny()
        restored_opt = AdamW(restored.parameters(), lambda step: 1e-2)
        meta = load_checkpoint(path, restored, restored_opt)

        assert meta == {"step": 7, "best_score": 1.5, "args": '{"lr": 0.01}'}, meta
        assert read_checkpoint_metadata(path) == meta
        for a, b in zip(_values(model), _values(restored)):
            np.testing.assert_array_equal(a, b)
        assert restored_opt.step_count == opt.step_count == 1
        for a, b in zip(opt.m + opt.v, restored_opt.m + restored_opt.v):
            np.testing.assert_array_equal(to_numpy(a), to_numpy(b))
        assert not (Path(tmp) / "ckpt.tmp.npz").exists()

        # model-only checkpoint: loading into an optimizer is an error
        save_checkpoint(path, model, step=1)
        try:
            load_checkpoint(path, _Tiny(), AdamW(_Tiny().parameters(), lambda s: 0.0))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for missing optimizer state")

        # metadata may not shadow parameter or optimizer keys
        for bad in ({"p0": 1}, {"opt_m0": 1}):
            try:
                save_checkpoint(path, model, **bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for metadata {bad}")


def check_prefetcher() -> None:
    def make_epoch(rng):
        return [int(x) for x in rng.permutation(5)]

    batches = Prefetcher(make_epoch, seed=3, depth=4)
    got = [batches.next() for _ in range(10)]

    rng = np.random.default_rng(3)  # same seed, same sequence, epoch after epoch
    expected = [(0, b) for b in make_epoch(rng)] + [(1, b) for b in make_epoch(rng)]
    assert got == expected, (got, expected)

    def failing_epoch(rng):
        yield 1
        raise KeyError("boom")

    failing = Prefetcher(failing_epoch, seed=0)
    assert failing.next() == (0, 1)
    try:
        failing.next()
    except RuntimeError as e:
        assert isinstance(e.__cause__, KeyError)
    else:
        raise AssertionError("expected the producer's exception to reach next()")


def main():
    ok = True
    for check in (check_checkpoint_round_trip, check_prefetcher):
        try:
            check()
            print(f"{check.__name__}: ok")
        except Exception as e:  # noqa: BLE001 - report every failing check
            ok = False
            print(f"{check.__name__}: FAIL {type(e).__name__}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
