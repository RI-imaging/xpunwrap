from .._ndarray_backend import xp
from ._ls_common import wrap_phase


def restore_mean_plane(
    phase_unwrapped: xp.ndarray,
    phase_wrapped: xp.ndarray,
) -> xp.ndarray:
    """Reintroduce the mean wrapped gradient plane removed by Poisson solvers.

    Least-squares Poisson solvers discard the null-space component (a linear
    ramp). This function estimates that ramp from the mean wrapped gradients of
    the input and adds it back, anchored so that the value at pixel ``(0, 0)``
    matches the original wrapped phase.

    Parameters
    ----------
    phase_unwrapped : xp.ndarray
        Solver output, shape (H, W) or (N, H, W).
    phase_wrapped : xp.ndarray
        Original wrapped phase, same shape as ``phase_unwrapped``.

    Returns
    -------
    xp.ndarray
        Phase with the mean gradient plane restored, same shape as input.

    Raises
    ------
    ValueError
        If ``phase_unwrapped`` is not 2-D or 3-D.
    """
    input_2d = False
    if phase_unwrapped.ndim == 2:
        input_2d = True
        phase_unwrapped = xp.expand_dims(phase_unwrapped, axis=0)
        phase_wrapped = xp.expand_dims(phase_wrapped, axis=0)
    elif phase_unwrapped.ndim != 3:
        raise ValueError(
            "phase_unwrapped must have shape (H, W) or (N, H, W)."
        )

    dtype = phase_unwrapped.dtype

    gx = wrap_phase(xp.diff(phase_wrapped, axis=2,
                            append=phase_wrapped[:, :, -1:]))
    gy = wrap_phase(xp.diff(phase_wrapped, axis=1,
                            append=phase_wrapped[:, -1:, :]))

    gx_mean = gx.mean(axis=(1, 2), keepdims=True)
    gy_mean = gy.mean(axis=(1, 2), keepdims=True)

    _, H, W = phase_unwrapped.shape
    x_idx = xp.arange(W, dtype=dtype).reshape(1, 1, W)
    y_idx = xp.arange(H, dtype=dtype).reshape(1, H, 1)
    plane = gx_mean * x_idx + gy_mean * y_idx
    anchor = (
        phase_wrapped[:, 0, 0]
        - (phase_unwrapped[:, 0, 0] + plane[:, 0, 0])
    )
    out = phase_unwrapped + plane + anchor[:, None, None]

    if input_2d:
        out = out[0]
    return out
