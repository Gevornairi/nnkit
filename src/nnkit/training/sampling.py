"""Batched sampling from a language model that decodes with a KV cache."""

from collections.abc import Callable

import numpy as np

from ..base.backend import Array, asarray, to_numpy


def sample_batch(
    step: Callable[[Array], Array],
    prompts: np.ndarray,
    max_new: int,
    eos_id: int,
    pad_id: int,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Continue every row of ``prompts`` (batch, P) for up to ``max_new`` tokens.

    ``step(ids)`` takes the next ids (batch, t) and returns their logits
    (batch, t, vocab), remembering earlier calls (a KV cache); the first call
    gets the whole prompt, later ones one token per row. All prompts have the
    same length, so every row is at the same position and needs no padding.

    Returns new ids (batch, max_new): each row's tokens up to and including its
    first EOS, then PAD. A row without EOS ran out of ``max_new``. Rows that
    are done keep being fed PAD, whose outputs are ignored, until all are done.
    ``temperature=0`` is greedy. Policy-gradient rollouts need ``temperature=1``,
    so the samples come from the distribution the loss differentiates.
    """
    rng = rng if rng is not None else np.random.default_rng()
    batch = prompts.shape[0]
    out = np.full((batch, max_new), pad_id, dtype=np.int32)
    done = np.zeros(batch, dtype=bool)
    logits = step(asarray(prompts.astype(np.int32)))
    for i in range(max_new):
        last = to_numpy(logits[:, -1]).astype(np.float64)
        if temperature == 0:
            token = last.argmax(axis=-1)
        else:
            z = last / temperature
            p = np.exp(z - z.max(axis=-1, keepdims=True))
            cdf = np.cumsum(p / p.sum(axis=-1, keepdims=True), axis=-1)
            # inverse-CDF sampling, one uniform draw per row
            token = np.minimum(
                (cdf < rng.random((batch, 1))).sum(axis=-1), p.shape[-1] - 1
            )
        token = np.where(done, pad_id, token)
        out[:, i] = token
        done |= token == eos_id
        if done.all() or i == max_new - 1:
            break
        logits = step(asarray(token[:, None].astype(np.int32)))

    return out
