"""Background thread that keeps a queue of ready batches."""

import queue
import threading
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


class Prefetcher:
    """Runs ``make_epoch`` on a background thread, forever, so batches are ready before they're needed.

    ``make_epoch(rng)`` returns one epoch of batches, in whatever form the
    training loop wants (e.g. padded NumPy arrays). It is called again for
    every epoch with the same ``np.random.Generator``, so a fixed ``seed``
    gives the same batch sequence on every run. Batches are built on the CPU
    while the GPU trains on the previous ones; ``depth`` bounds how many
    are queued.

    If ``make_epoch`` raises, the exception is re-raised by the next call to
    ``next`` instead of the thread dying silently and ``next`` blocking forever.
    """

    def __init__(
        self,
        make_epoch: Callable[[np.random.Generator], Iterable[Any]],
        seed: int,
        depth: int = 16,
    ) -> None:
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.epoch = 0
        args = (make_epoch, np.random.default_rng(seed))
        threading.Thread(target=self._run, args=args, daemon=True).start()

    def _run(self, make_epoch, rng) -> None:
        try:
            while True:
                for batch in make_epoch(rng):
                    self.queue.put((self.epoch, batch, None))
                self.epoch += 1
        except Exception as e:  # noqa: BLE001 - any failure is re-raised by next()
            self.queue.put((self.epoch, None, e))

    def next(self) -> tuple[int, Any]:
        """Blocks until a batch is ready; returns (epoch, batch)."""
        epoch, batch, error = self.queue.get()
        if error is not None:
            raise RuntimeError("batch producer failed") from error
        return epoch, batch
