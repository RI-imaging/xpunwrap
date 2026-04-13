from __future__ import annotations

import numpy as _np

try:  # Optional cupy detection for type checking
    import cupy as _cp
except Exception:  # pragma: no cover - cupy optional
    _cp = None

from .._dtype_utils import real_pi
from .._ndarray_backend import xp


def _use_cupy_backend(arr) -> bool:
    return _cp is not None and isinstance(arr, _cp.ndarray)


def restore_mean_plane(
    phase_unwrapped: xp.ndarray,
    phase_wrapped: xp.ndarray,
) -> xp.ndarray:
    """
    Reintroduce the mean wrapped gradient plane removed by
    Poisson-like solvers.

    Supports 2D (H, W) or stacked (N, H, W) inputs. The output matches the
    shape of ``phase_unwrapped``.
    """
    input_2d = False
    backend = xp if _use_cupy_backend(phase_unwrapped) else _np

    if phase_unwrapped.ndim == 2:
        input_2d = True
        phase_unwrapped = backend.expand_dims(phase_unwrapped, axis=0)
        phase_wrapped = backend.expand_dims(phase_wrapped, axis=0)
    elif phase_unwrapped.ndim != 3:
        raise ValueError(
            "phase_unwrapped must have shape (H, W) or (N, H, W)."
        )

    dtype = phase_unwrapped.dtype
    pi = real_pi(backend, dtype)
    two_pi = dtype.type(2) * pi

    gx = backend.diff(phase_wrapped, axis=2, append=phase_wrapped[:, :, -1:])
    gy = backend.diff(phase_wrapped, axis=1, append=phase_wrapped[:, -1:, :])
    gx = (gx + pi) % two_pi - pi
    gy = (gy + pi) % two_pi - pi

    gx_mean = gx.mean(axis=(1, 2), keepdims=True)
    gy_mean = gy.mean(axis=(1, 2), keepdims=True)

    N, H, W = phase_unwrapped.shape
    x_idx = backend.arange(W, dtype=dtype).reshape(1, 1, W)
    y_idx = backend.arange(H, dtype=dtype).reshape(1, H, 1)
    plane = gx_mean * x_idx + gy_mean * y_idx
    anchor = (
        phase_wrapped[:, 0, 0]
        - (phase_unwrapped[:, 0, 0] + plane[:, 0, 0])
    )
    out = phase_unwrapped + plane + anchor[:, None, None]

    if input_2d:
        out = out[0]
    return out
