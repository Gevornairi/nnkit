"""Warmup, a constant (stable) phase, then a short decay (MiniCPM; Hägele et al., 2024)."""

import math


class WSDSchedule:
    """Learning rate as a function of the step number.

    Rises linearly from 0 to ``max_lr`` over ``warmup`` steps, stays at
    ``max_lr``, then decays to ``min_lr`` over the last ``decay_steps`` of
    ``total_steps``, and stays there. ``shape="sqrt"`` decays as
    ``1 - sqrt(progress)`` (usually slightly better), ``"linear"`` as a straight line.

    Unlike cosine, the schedule doesn't depend on the run length until the
    decay starts: a run can be extended by staying in the stable phase, and a
    short decay can be branched off any stable checkpoint to evaluate it.
    """

    def __init__(
        self,
        max_lr: float,
        warmup: int,
        total_steps: int,
        decay_steps: int,
        min_lr: float = 0.0,
        shape: str = "sqrt",
    ) -> None:
        if shape not in ("sqrt", "linear"):
            raise ValueError(f"shape={shape!r} must be 'sqrt' or 'linear'")
        if not 0 < decay_steps <= total_steps - warmup:
            raise ValueError(
                f"decay_steps={decay_steps} must be in (0, total_steps - warmup]"
            )
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup = warmup
        self.total_steps = total_steps
        self.decay_steps = decay_steps
        self.decay_start = total_steps - decay_steps
        self.shape = shape

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.max_lr * step / self.warmup
        if step < self.decay_start:
            return self.max_lr
        progress = min(1.0, (step - self.decay_start) / self.decay_steps)
        remaining = (
            1.0 - math.sqrt(progress) if self.shape == "sqrt" else 1.0 - progress
        )
        return self.min_lr + (self.max_lr - self.min_lr) * remaining
