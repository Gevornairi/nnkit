"""Base class for every layer: parameter/child bookkeeping, train/eval mode, checkpoints."""

from __future__ import annotations

from .backend import Array, asarray, to_numpy, xp
from .buffer import Buffer
from .param import Param


class Module:
    """A node in the model tree with its own parameters and child modules.

    There is no autograd: each subclass implements ``forward``, which caches
    whatever its ``backward`` needs, and ``backward``, which takes the gradient
    of the loss w.r.t. the output, adds parameter gradients into ``Param.grad``
    and returns the gradient w.r.t. the input(s). ``forward`` and ``backward``
    signatures vary per layer, so they are not declared here.

    Subclasses must call ``register_params`` / ``register_buffers`` /
    ``register_modules`` for every parameter, buffer and child; ``parameters``,
    ``buffers``, ``train``/``eval`` and checkpointing only see what was registered.
    """

    def __init__(self) -> None:
        self._params: list[Param] = []
        self._children: list[Module] = []
        self._buffers: list[Buffer] = []
        self.training = True

    def register_params(self, *params: Param) -> None:
        """Make ``params`` part of this module's trainable parameters."""
        self._params.extend(params)

    def register_modules(self, *modules: Module) -> None:
        """Make ``modules`` children of this module."""
        self._children.extend(modules)

    def register_buffers(self, *buffers: Buffer) -> None:
        """Make ``buffers`` part of this module's saved, non-trainable state."""
        self._buffers.extend(buffers)

    def parameters(self) -> list[Param]:
        """All parameters of this module and its descendants, depth-first.

        Each ``Param`` appears once even if it is shared (e.g. the embedding
        table used by encoder, decoder and output projection). The order is
        deterministic for a given architecture, which checkpoints rely on.
        """
        seen: set[int] = set()
        out: list[Param] = []

        for p in self._params:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)

        for child in self._children:
            for p in child.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)

        return out

    def buffers(self) -> list[Buffer]:
        """All buffers of this module and its descendants, depth-first, each once.

        Same order guarantee as ``parameters``, which checkpoints rely on.
        """
        seen: set[int] = set()
        out: list[Buffer] = []

        for b in self._buffers:
            if id(b) not in seen:
                seen.add(id(b))
                out.append(b)

        for child in self._children:
            for b in child.buffers():
                if id(b) not in seen:
                    seen.add(id(b))
                    out.append(b)

        return out

    def zero_grad(self) -> None:
        """Reset every parameter's gradient to zeros."""
        for p in self.parameters():
            p.grad = xp.zeros_like(p.value)

    def train(self, mode: bool = True) -> Module:
        """Set training mode (dropout active) on this module and all descendants."""
        self.training = mode
        for child in self._children:
            child.train(mode)

        return self

    def eval(self) -> Module:
        """Set evaluation mode (dropout off) on this module and all descendants."""
        return self.train(False)

    def state(self) -> dict[str, Array]:
        """Values as NumPy arrays: parameters keyed ``p0, p1, ...`` in ``parameters()``
        order, buffers keyed ``b0, b1, ...`` in ``buffers()`` order."""
        state = {}

        for i, p in enumerate(self.parameters()):
            state[f"p{i}"] = to_numpy(p.value)

        for i, b in enumerate(self.buffers()):
            state[f"b{i}"] = to_numpy(b.value)

        return state

    def load_state(self, state) -> None:
        """Load values saved by ``state``; the architecture must match exactly.

        The number and shapes of parameters and of buffers must both match, so a
        model without buffers still loads checkpoints saved before buffers existed.
        """
        for prefix, slots in (("p", self.parameters()), ("b", self.buffers())):
            n_saved = sum(1 for k in state if k[0] == prefix and k[1:].isdigit())
            if n_saved != len(slots):
                raise ValueError(
                    f"checkpoint has {n_saved} '{prefix}' arrays, model has {len(slots)}"
                )
            for i, s in enumerate(slots):
                v = state[f"{prefix}{i}"]
                if tuple(v.shape) != tuple(s.value.shape):
                    raise ValueError(
                        f"{prefix}{i}: checkpoint {v.shape} vs model {s.value.shape}"
                    )
                s.value = asarray(v, s.value.dtype)

    def modules(self) -> list[Module]:
        """This module and all its descendants, depth-first, each once.

        Used to find layers of one kind in a model, e.g. every MoE layer whose
        load-balancing bias the training loop updates after each step.
        """

        seen: set[int] = set()
        out: list[Module] = []

        def visit(m: Module) -> None:
            if id(m) in seen:
                return
            seen.add(id(m))
            out.append(m)
            for child in m._children:
                visit(child)

        visit(self)

        return out
