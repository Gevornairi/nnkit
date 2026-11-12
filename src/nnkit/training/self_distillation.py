"""Teacher-side helpers for self-distillation (DINO, iBOT): no gradients involved."""

from ..base.backend import Array, xp
from ..base.module import Module


def ema_update(teacher: Module, student: Module, momentum: float) -> None:
    """``teacher = momentum * teacher + (1 - momentum) * student``, parameter by parameter.

    The teacher is a second instance of the student's architecture (e.g.
    built from the same config and loaded with ``student.state()``); DINO
    raises ``momentum`` from ~0.99 towards 1 over training.
    """
    t_params, s_params = teacher.parameters(), student.parameters()
    if len(t_params) != len(s_params):
        raise ValueError(
            f"teacher has {len(t_params)} parameters, student {len(s_params)}"
        )
    for t, s in zip(t_params, s_params):
        if t.shape != s.shape:
            raise ValueError(f"teacher {t.shape} vs student {s.shape}")
        t.value = momentum * t.value + (1.0 - momentum) * s.value


def sinkhorn_knopp(logits: Array, temperature: float, n_iters: int = 3) -> Array:
    """Teacher targets with prototypes used equally across the batch (SwAV; DINOv2/v3).

    Starts from ``exp(logits / temperature)`` and alternately rescales so each
    prototype (column) gets total mass rows / K and each row sums to 1. Unlike
    softmax-with-centering, this cannot put the whole batch on a few
    prototypes, which is how self-distillation collapses.

    logits (rows, K): teacher CLS logits, or the teacher's logits at the masked
    patches for iBOT -> (rows, K) target distributions.
    """
    q = xp.exp((logits - logits.max()) / temperature)  # the global shift cancels below
    rows, k = q.shape
    q = q / q.sum()
    for _ in range(n_iters):
        q = q / (q.sum(axis=0, keepdims=True) * k)  # every prototype: total 1 / K
        q = q / (q.sum(axis=1, keepdims=True) * rows)  # every row: total 1 / rows

    return q * rows
