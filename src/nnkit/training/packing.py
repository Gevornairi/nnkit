"""Training windows cut from a packed token stream, with document ids for the mask.

A packed stream is one flat array of token ids, ``doc EOS doc EOS ...``, as
written by a data-preparation script (usually a ``np.memmap`` of a ``.bin``
file). Windows of ``seq_len + 1`` tokens give inputs and targets shifted by
one; a window may span several documents, which ``document_ids`` labels so
``nnkit.base.document_causal_mask`` keeps them apart. All NumPy on the CPU,
so batches can be built on a ``Prefetcher`` thread.
"""

from collections.abc import Iterator

import numpy as np


def document_ids(tokens: np.ndarray, eos: int) -> np.ndarray:
    """Per position, which document of the row it belongs to: 0, 1, 2, ...

    EOS stays with the document it ends: cumsum counts the EOS up to and
    including i, minus one if i itself is the EOS.
    """
    is_eos = (tokens == eos).astype(np.int32)
    return np.cumsum(is_eos, axis=-1, dtype=np.int32) - is_eos


def window_starts(
    n_tokens: int, seq_len: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Starts of windows of seq_len + 1 tokens, seq_len apart, covering the stream.

    Neighbouring windows share one token, so every token is predicted exactly
    once. With rng: a random shift in [0, seq_len) and shuffled order (a new
    cut every epoch). Without: from 0, in order (validation).
    """
    shift = int(rng.integers(seq_len)) if rng is not None else 0
    # a window needs start + seq_len + 1 <= n_tokens
    starts = np.arange(shift, n_tokens - seq_len, seq_len)
    if rng is not None:
        rng.shuffle(starts)

    return starts


def packed_batch(
    stream: np.ndarray, starts: np.ndarray, seq_len: int, eos: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inputs, targets, doc_ids) for the windows at ``starts``, each int32 (n, seq_len)."""
    windows = np.stack([stream[s : s + seq_len + 1] for s in starts]).astype(np.int32)
    inputs, targets = windows[:, :-1], windows[:, 1:]

    return inputs, targets, document_ids(inputs, eos)


def packed_batches(
    stream: np.ndarray,
    seq_len: int,
    batch_size: int,
    eos: int,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One epoch of ``packed_batch``es; the last partial batch is dropped.

    Pass the ``Prefetcher``'s rng for training and none for validation.
    """

    starts = window_starts(len(stream), seq_len, rng)
    for i in range(0, len(starts) - batch_size + 1, batch_size):
        yield packed_batch(stream, starts[i : i + batch_size], seq_len, eos)
