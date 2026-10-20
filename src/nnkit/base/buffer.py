"""Non-trainable module state: a value array saved in checkpoints but never optimised."""

from .backend import Array, xp


class Buffer:
    """State a layer updates itself (e.g. the MoE load-balancing bias).

    Unlike ``Param`` it has no gradient and is not in ``Module.parameters``, so
    the optimizer never sees it; ``Module.state`` still saves it. Layers replace
    ``value`` rather than mutating it, since MLX arrays are immutable. The dtype
    is kept as given, so integer counters stay integers.
    """

    __slots__ = ("value",)

    def __init__(self, value: Array) -> None:
        self.value = xp.array(value)
