from __future__ import annotations

"""
Thin wrapper around ``skimage.restoration.unwrap_phase`` to match the project's
algorithm interface.
"""

from typing import Any

from .._ndarray_backend import xp


def algo_skimage_unwrap(
    phase_wrapped: xp.ndarray,
    restore_plane: bool = False,
    **kwargs: Any,
) -> xp.ndarray:
    """
    2D phase unwrapping using scikit-image's CPU implementation.

    Parameters
    ----------
    phase_wrapped : xp.ndarray
        Wrapped phase, shape (H, W) or (N, H, W), values in [-pi, pi).
    restore_plane : bool, optional
        Ignored for this algorithm; scikit-image already preserves the global
        ramp (Poisson null-space is resolved internally). Kept for interface
        compatibility. Default False.
    kwargs :
        Passed through to ``skimage.restoration.unwrap_phase`` (e.g. spacing,
        wrap_around, seed).

    Returns
    -------
    xp.ndarray
        Unwrapped phase, same shape and backend as the input.

    Notes
    -----
    - Requires ``scikit-image`` to be installed in the active environment.
    - Computation is performed on CPU; Cupy inputs are transferred to NumPy and
      converted back to the active backend on return.
    """
    if phase_wrapped.ndim == 2:
        return _unwrap_single(phase_wrapped, restore_plane, **kwargs)
    if phase_wrapped.ndim != 3:
        raise ValueError("phase_wrapped must have shape (H, W) or (N, H, W).")

    out = []
    for i in range(phase_wrapped.shape[0]):
        out.append(_unwrap_single(phase_wrapped[i], restore_plane, **kwargs))
    return xp.stack(out, axis=0)


def _unwrap_single(arr: xp.ndarray, restore_plane: bool, **kwargs: Any) -> xp.ndarray:
    if xp.is_cupy():
        np_arr = xp.asnumpy(arr)
    else:
        import numpy as np

        np_arr = np.asarray(arr)

    try:
        from skimage.restoration import unwrap_phase as sk_unwrap
    except Exception as err:  # ImportError or runtime issues
        raise ImportError(
            "scikit-image is required for algo_skimage_unwrap. "
            "Install with `pip install scikit-image` or the project extra "
            "`pip install unwrap_phase_gpu[examples]`."
        ) from err

    np_unwrapped = sk_unwrap(np_arr, **kwargs)

    if restore_plane:
        # scikit-image unwrap_phase already returns a ramp-preserved solution;
        # restoring again would double-count the plane.
        pass

    return xp.asarray(np_unwrapped) if xp.is_cupy() else np_unwrapped
