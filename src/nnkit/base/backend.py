"""Array backend shared by every layer.

All math goes through ``xp``, which is ``mlx.core``, ``cupy`` or ``numpy``. All
three expose the same NumPy-style API for everything this project uses; the
few operations whose APIs differ (random numbers, scatter-add, erf, copying
to the host, waiting for the device) are wrapped by the helpers below. Code
outside this file moves data between host and device only through
``asarray`` and ``to_numpy``: CuPy refuses implicit conversions.

The backend is picked from environment variables at import time, so set them
before ``nnkit`` is imported:

    NNKIT_BACKEND=mlx      (default) Apple GPU via MLX, float32 only
    NNKIT_BACKEND=cupy     NVIDIA GPU via CuPy (CUDA)
    NNKIT_BACKEND=numpy    CPU via NumPy
    NNKIT_DTYPE=float64    NumPy or CuPy; used for finite-difference gradient checks
"""

import math
import os
from typing import Any

import numpy as np

BACKEND = os.environ.get("NNKIT_BACKEND", "mlx")

xp: Any  # the array module: mlx.core, cupy or numpy
if BACKEND == "mlx":
    import mlx.core as xp
elif BACKEND == "cupy":
    import cupy as xp  # pyright: ignore[reportMissingImports]

elif BACKEND == "numpy":
    xp = np
else:
    raise ValueError(f"unknown NNKIT_BACKEND={BACKEND!r} (use mlx, cupy or numpy)")

Array = Any  # an mx.array, cupy.ndarray or np.ndarray

_dtype_name = os.environ.get("NNKIT_DTYPE", "float32")
if BACKEND == "mlx" and _dtype_name != "float32":
    raise ValueError("MLX on GPU only supports NNKIT_DTYPE=float32")
DTYPE = getattr(xp, _dtype_name)  # float dtype of parameters and activations
INT = xp.int32  # dtype of token ids

_np_rng = np.random.default_rng()


def seed(s: int) -> None:
    """Seed the random generator of the active backend."""
    global _np_rng
    _np_rng = np.random.default_rng(s)
    if BACKEND in ("mlx", "cupy"):
        xp.random.seed(s)


def asarray(x, dtype=None) -> Array:
    """Convert ``x`` (list, NumPy array, ...) to a backend array, optionally casting."""
    a = xp.array(x)
    return a if dtype is None else a.astype(dtype)


def to_numpy(x: Array) -> np.ndarray:
    """Copy a backend array to a NumPy array (forces evaluation on MLX, a device copy on CuPy)."""
    if BACKEND == "cupy":
        return xp.asnumpy(x)
    return np.array(x)


def normal(std: float, shape: tuple[int, ...]) -> Array:
    """Samples from N(0, std^2) with the given shape, in ``DTYPE``."""
    if BACKEND == "mlx":
        return (xp.random.normal(shape) * std).astype(DTYPE)
    if BACKEND == "cupy":
        return xp.random.normal(0.0, std, shape, dtype=DTYPE)
    return _np_rng.normal(0.0, std, shape).astype(DTYPE)


def uniform(low: float, high: float, shape: tuple[int, ...]) -> Array:
    """Samples from U(low, high) with the given shape, in ``DTYPE``."""
    if BACKEND == "mlx":
        return xp.random.uniform(low, high, shape).astype(DTYPE)
    if BACKEND == "cupy":
        return xp.random.uniform(low, high, shape, dtype=DTYPE)
    return _np_rng.uniform(low, high, shape).astype(DTYPE)


def keep_mask(p_keep: float, shape: tuple[int, ...]) -> Array:
    """A 0/1 mask in ``DTYPE`` where each entry is 1 with probability ``p_keep``."""
    if BACKEND == "mlx":
        return xp.random.bernoulli(p_keep, shape).astype(DTYPE)
    if BACKEND == "cupy":
        return (xp.random.random(shape) < p_keep).astype(DTYPE)
    return (_np_rng.random(shape) < p_keep).astype(DTYPE)


def scatter_add(target: Array, ids: Array, values: Array) -> Array:
    """``target[ids] += values``, accumulating repeated ids.

    Plain ``target[ids] += values`` keeps only one write per repeated id, which
    would drop gradient for tokens that appear more than once in a batch.
    Returns the updated array (a new one on MLX; ``target`` itself on NumPy and CuPy).
    """
    if BACKEND == "mlx":
        return target.at[ids].add(values)
    if BACKEND == "cupy":
        import cupyx  # pyright: ignore[reportMissingImports]

        cupyx.scatter_add(target, ids, values)
        return target

    np.add.at(target, ids, values)
    return target


def materialize(*trees) -> None:
    """Force MLX's lazy graph to run so it doesn't grow across steps; on CuPy, wait for the
    queued CUDA kernels, so per-step timings are real (no-op on NumPy)."""
    if BACKEND == "mlx":
        xp.eval(*trees)
    elif BACKEND == "cupy":
        xp.cuda.Device().synchronize()


def erf(x: Array) -> Array:
    """Gauss error function, elementwise (NumPy has none, so ``math.erf`` per element there)."""
    if BACKEND == "mlx":
        return xp.erf(x)
    if BACKEND == "cupy":
        import cupyx.scipy.special  # pyright: ignore[reportMissingImports]

        return cupyx.scipy.special.erf(x)

    return np.vectorize(math.erf, otypes=[x.dtype])(x)


def top_k_indices(x: Array, k: int) -> Array:
    """Indices of the ``k`` largest entries along the last axis, in no particular order.

    Shape (..., n) -> (..., k). Ties are broken arbitrarily.
    """
    return xp.argpartition(-x, k - 1, axis=-1)[..., :k]
