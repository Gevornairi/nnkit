"""Check nnkit.training's packed-stream batching: windows, document ids and the mask.

python -m tests.packing_check

Exits non-zero if any check fails.
"""

import numpy as np

from nnkit.base import asarray, document_causal_mask, to_numpy
from nnkit.training import document_ids, packed_batches, window_starts

EOS = 1


def packed_stream(n_docs: int, seed: int = 0) -> np.ndarray:
    """Random documents of 1-40 tokens (ids 2-49), each followed by EOS."""
    rng = np.random.default_rng(seed)
    docs = [
        np.append(rng.integers(2, 50, rng.integers(1, 41)), EOS) for _ in range(n_docs)
    ]
    return np.concatenate(docs).astype(np.uint16)


def check_document_ids() -> None:
    ids = document_ids(
        np.array([[5, 9, EOS, 7, 8, EOS, 4], [EOS, EOS, 3, 3, 3, 3, EOS]]), EOS
    )
    np.testing.assert_array_equal(ids, [[0, 0, 0, 1, 1, 1, 2], [0, 1, 2, 2, 2, 2, 2]])


def check_window_coverage() -> None:
    """A shuffled epoch predicts a contiguous run of tokens, each exactly once."""
    n, seq_len = 1000, 16
    for seed in range(5):
        starts = window_starts(n, seq_len, np.random.default_rng(seed))
        predicted = np.concatenate([np.arange(s + 1, s + seq_len + 1) for s in starts])
        assert len(np.unique(predicted)) == len(predicted), "a token predicted twice"
        assert predicted.max() - predicted.min() + 1 == len(predicted), "a gap"
        # less than one window left uncovered at either end, nothing past the end
        assert predicted.min() <= seq_len and n - 1 - predicted.max() < seq_len
        assert predicted.max() < n
    # validation: from 0, in order; the window at 32 ends exactly at token 48 < 50
    assert window_starts(50, 16).tolist() == [0, 16, 32]


def check_packed_batches() -> None:
    """Shapes, the one-token shift, and the same batches for the same seed."""
    stream, seq_len, batch_size = packed_stream(200), 32, 4
    first = list(
        packed_batches(stream, seq_len, batch_size, EOS, np.random.default_rng(0))
    )
    again = list(
        packed_batches(stream, seq_len, batch_size, EOS, np.random.default_rng(0))
    )
    other = list(
        packed_batches(stream, seq_len, batch_size, EOS, np.random.default_rng(1))
    )
    # the shift drawn from the same seed can leave one window fewer than shift 0
    n_windows = len(window_starts(len(stream), seq_len, np.random.default_rng(0)))
    assert len(first) == n_windows // batch_size
    for (x, y, d), (x2, _, _) in zip(first, again, strict=True):
        assert x.shape == y.shape == d.shape == (batch_size, seq_len)
        assert x.dtype == y.dtype == d.dtype == np.int32
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
        np.testing.assert_array_equal(x, x2)
    assert not np.array_equal(first[0][0], other[0][0])


def check_document_mask() -> None:
    """Query i sees key j exactly when j <= i and no EOS sits in inputs[j:i]."""
    x, _, d = next(packed_batches(packed_stream(100), 48, 3, EOS))
    mask = to_numpy(document_causal_mask(asarray(d)))[:, 0]
    for b in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                expected = j <= i and not (x[b, j:i] == EOS).any()
                assert mask[b, i, j] == expected, (b, i, j)


def main():
    ok = True
    for check in (
        check_document_ids,
        check_window_coverage,
        check_packed_batches,
        check_document_mask,
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
