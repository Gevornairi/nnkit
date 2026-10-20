"""Save and restore model + optimizer state and metadata as a single .npz file.

File layout: model parameters under ``p0, p1, ...`` and buffers under
``b0, b1, ...`` (``Module.state`` order), optimizer state under ``opt_*``,
and every metadata value under its own key.
"""

import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..base import Module

_MODEL_KEY = re.compile(r"[pb]\d+")  # parameters and buffers
_OPT_PREFIX = "opt_"


class Stateful(Protocol):
    """Anything with ``state`` / ``load_state``, such as ``nnkit.optimisers.AdamW``."""

    def state(self) -> dict[str, Any]: ...

    def load_state(self, state) -> None: ...


def _is_reserved(key: str) -> bool:
    return bool(_MODEL_KEY.fullmatch(key)) or key.startswith(_OPT_PREFIX)


def save_checkpoint(
    path: Path, model: Module, optimizer: Stateful | None = None, **metadata
) -> None:
    """Write model parameters, optimizer state (if given) and metadata to path.

    Metadata values are anything ``np.savez`` stores: numbers, strings (e.g.
    JSON) or arrays. The file is written to a temporary name and then renamed,
    so an interrupted save never leaves a half-written checkpoint behind.
    """
    reserved = [k for k in metadata if _is_reserved(k)]
    if reserved:
        raise ValueError(
            f"metadata keys {reserved} clash with parameter/buffer/optimizer keys"
        )
    arrays = dict(model.state())
    if optimizer is not None:
        arrays.update({_OPT_PREFIX + k: v for k, v in optimizer.state().items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays, **metadata)
    tmp.replace(path)


def _metadata(z) -> dict[str, Any]:
    # 0-d arrays (numbers, strings) come back as plain Python values
    return {
        k: (z[k].item() if z[k].ndim == 0 else z[k])
        for k in z.files
        if not _is_reserved(k)
    }


def load_checkpoint(
    path: Path, model: Module, optimizer: Stateful | None = None
) -> dict[str, Any]:
    """Restore model (and optimizer, if given) from path; returns the saved metadata.

    The model's architecture must match the one that was saved; mismatches
    raise ``ValueError`` (see ``Module.load_state``).
    """
    with np.load(path) as z:
        model.load_state({k: z[k] for k in z.files if _MODEL_KEY.fullmatch(k)})
        if optimizer is not None:
            opt_state = {
                k[len(_OPT_PREFIX) :]: z[k]
                for k in z.files
                if k.startswith(_OPT_PREFIX)
            }
            if not opt_state:
                raise ValueError(f"{path} has no optimizer state")
            optimizer.load_state(opt_state)
        return _metadata(z)


def read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """The metadata saved in path, without loading any parameters.

    Useful when the metadata says how to build the model before loading it.
    """
    with np.load(path) as z:
        return _metadata(z)
