"""Training utilities: packed-stream batching, background prefetching, checkpointing,
sampling and self-distillation teachers."""

from .checkpoint import load_checkpoint, read_checkpoint_metadata, save_checkpoint
from .packing import document_ids, packed_batch, packed_batches, window_starts
from .prefetcher import Prefetcher
from .sampling import sample_batch
from .self_distillation import ema_update, sinkhorn_knopp

__all__ = [
    "Prefetcher",
    "document_ids",
    "ema_update",
    "load_checkpoint",
    "packed_batch",
    "packed_batches",
    "read_checkpoint_metadata",
    "sample_batch",
    "save_checkpoint",
    "sinkhorn_knopp",
    "window_starts",
]
