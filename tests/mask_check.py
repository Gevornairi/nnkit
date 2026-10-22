"""Check nnkit's attention masks: their patterns and their effect inside attention.

python -m tests.mask_check

Exits non-zero if any check fails.
"""

import numpy as np

from nnkit.base import (
    asarray,
    causal_mask,
    chunked_causal_mask,
    document_causal_mask,
    normal,
    seed,
    sliding_window_mask,
    to_numpy,
)
from nnkit.layers import MultiHeadAttention


def check_chunked_pattern() -> None:
    # size 8, chunk 3: blocks 0-2, 3-5, 6-7 (the last one shorter), causal inside each
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ],
        dtype=bool,
    )
    mask = to_numpy(chunked_causal_mask(8, 3))
    assert mask.shape == (1, 1, 8, 8) and mask.dtype == bool, (mask.shape, mask.dtype)
    np.testing.assert_array_equal(mask[0, 0], expected)


def check_chunk_edge_cases() -> None:
    # one block covering everything is plain causal attention
    for chunk in (8, 100):
        np.testing.assert_array_equal(
            to_numpy(chunked_causal_mask(8, chunk)), to_numpy(causal_mask(8))
        )
    # chunk 1: every token sees only itself
    np.testing.assert_array_equal(
        to_numpy(chunked_causal_mask(5, 1))[0, 0], np.eye(5, dtype=bool)
    )
    for bad in (0, -2):
        try:
            chunked_causal_mask(8, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for chunk={bad}")


def check_chunked_attention() -> None:
    """Inside attention, keys outside the query's chunk get exactly zero weight."""
    seed(0)
    size, chunk = 10, 4
    attn = MultiHeadAttention(16, 4, dropout=0.0)
    x = normal(1.0, (2, size, 16))
    attn.forward(x, x, chunked_causal_mask(size, chunk))
    w = to_numpy(attn.attn_weights)  # (batch, heads, size, size)
    hidden = ~to_numpy(chunked_causal_mask(size, chunk))
    assert np.all(w[np.broadcast_to(hidden, w.shape)] == 0.0), (
        "masked keys got attention weight"
    )
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-5)


def check_sliding_pattern() -> None:
    # size 6, window 3: each query sees itself and the 2 keys before it
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    mask = to_numpy(sliding_window_mask(6, 3))
    assert mask.shape == (1, 1, 6, 6) and mask.dtype == bool, (mask.shape, mask.dtype)
    np.testing.assert_array_equal(mask[0, 0], expected)


def check_sliding_edge_cases() -> None:
    # a window covering everything is plain causal attention
    for window in (8, 100):
        np.testing.assert_array_equal(
            to_numpy(sliding_window_mask(8, window)), to_numpy(causal_mask(8))
        )
    # window 1: every token sees only itself
    np.testing.assert_array_equal(
        to_numpy(sliding_window_mask(5, 1))[0, 0], np.eye(5, dtype=bool)
    )
    for bad in (0, -2):
        try:
            sliding_window_mask(8, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for window={bad}")


def check_sliding_vs_chunked() -> None:
    """At a chunk boundary the chunked mask sees only itself; the window still sees window tokens."""
    size, n = 12, 4
    sliding, chunked = (
        to_numpy(sliding_window_mask(size, n)),
        to_numpy(chunked_causal_mask(size, n)),
    )
    for i in range(n, size):  # every query past the first window
        assert sliding[0, 0, i].sum() == n, (i, sliding[0, 0, i].sum())
    for start in range(n, size, n):  # first token of each later chunk
        assert chunked[0, 0, start].sum() == 1, (start, chunked[0, 0, start].sum())


def check_sliding_attention() -> None:
    """Inside attention, keys outside the window get exactly zero weight."""
    seed(0)
    size, window = 10, 3
    attn = MultiHeadAttention(16, 4, dropout=0.0)
    x = normal(1.0, (2, size, 16))
    attn.forward(x, x, sliding_window_mask(size, window))
    w = to_numpy(attn.attn_weights)  # (batch, heads, size, size)
    hidden = ~to_numpy(sliding_window_mask(size, window))
    assert np.all(w[np.broadcast_to(hidden, w.shape)] == 0.0), (
        "masked keys got attention weight"
    )
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-5)


# document number of each token in two packed rows, boundaries in different places
DOC_IDS = [[0, 0, 0, 1, 1, 1, 2], [0, 0, 1, 1, 1, 1, 1]]


def check_document_pattern() -> None:
    mask = to_numpy(document_causal_mask(asarray(DOC_IDS)))
    assert mask.shape == (2, 1, 7, 7) and mask.dtype == bool, (mask.shape, mask.dtype)
    row0 = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask[0, 0], row0)
    # row 1 has its boundary elsewhere, so it gets its own pattern
    assert not np.array_equal(mask[0, 0], mask[1, 0]), "rows should differ"
    np.testing.assert_array_equal(mask[1, 0, 2], [0, 0, 1, 0, 0, 0, 0])


def check_document_edge_cases() -> None:
    # a single document is plain causal attention
    one_doc = asarray([[0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(
        to_numpy(document_causal_mask(one_doc)), to_numpy(causal_mask(5))
    )
    # combined with a sliding window, a key is visible only if both rules allow it
    doc_ids = asarray(DOC_IDS)
    both = to_numpy(document_causal_mask(doc_ids) & sliding_window_mask(7, 2))
    doc, win = (
        to_numpy(document_causal_mask(doc_ids)),
        to_numpy(sliding_window_mask(7, 2)),
    )
    np.testing.assert_array_equal(both, doc & win)
    assert both.shape == (2, 1, 7, 7)


def check_document_attention() -> None:
    """Inside attention, keys in another document get exactly zero weight."""
    seed(0)
    mask = document_causal_mask(asarray(DOC_IDS))
    attn = MultiHeadAttention(16, 4, dropout=0.0)
    x = normal(1.0, (2, 7, 16))
    attn.forward(x, x, mask)
    w = to_numpy(attn.attn_weights)  # (batch, heads, 7, 7)
    hidden = ~to_numpy(mask)
    assert np.all(w[np.broadcast_to(hidden, w.shape)] == 0.0), (
        "keys in another document got attention weight"
    )
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-5)


def main():
    ok = True
    for check in (
        check_chunked_pattern,
        check_chunk_edge_cases,
        check_chunked_attention,
        check_sliding_pattern,
        check_sliding_edge_cases,
        check_sliding_vs_chunked,
        check_sliding_attention,
        check_document_pattern,
        check_document_edge_cases,
        check_document_attention,
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
