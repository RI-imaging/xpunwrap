from .._ndarray_backend import xp
from ._ls_common import divergence_stack, poisson_solve_fft_stack, wrap_phase
from ._plane_utils import restore_mean_plane


def algo_ls_poisson(
    phase_wrapped: xp.ndarray,
    restore_plane: bool = False,
) -> xp.ndarray:
    """
    Batched 2D phase unwrapping using a least-squares Poisson solver.

    Parameters
    ----------
    phase_wrapped : xp.ndarray
        Wrapped phase, shape (H, W) or (N, H, W), values in [-pi, pi).
    restore_plane : bool, optional
        If True, add back the mean wrapped gradient plane. Default False.

    Returns
    -------
    phase_unwrapped : xp.ndarray
        Unwrapped phase, same shape as input.

    Notes
    -----
    The FFT Poisson solver assumes periodic boundary conditions: the input
    domain is treated as if its left/right and top/bottom edges connect. No
    zero-padding is applied before the FFT. If the wrapped phase is not
    periodic at the boundaries, the solver may produce artifacts (ringing or
    slope errors) near the domain edges. For non-periodic data consider
    cropping the region of interest away from the edges or using
    :func:`algo_ls_weighted`, which can suppress discontinuous boundary regions
    via its border-weight mask.

    References
    ----------
    .. [1] D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
       Theory, Algorithms, and Software," Wiley, 1998.

    """
    input_2d = False
    if phase_wrapped.ndim == 2:
        input_2d = True
        phase_wrapped = xp.expand_dims(phase_wrapped, axis=0)
    elif phase_wrapped.ndim != 3:
        raise ValueError("phase_wrapped must have shape (H, W) or (N, H, W).")

    dtype = phase_wrapped.dtype
    # Wrapped gradients with periodic forward differences.
    # This is the discrete operator used by the FFT Poisson solver.
    gx = wrap_phase(xp.roll(phase_wrapped, -1, axis=2) - phase_wrapped)
    gy = wrap_phase(xp.roll(phase_wrapped, -1, axis=1) - phase_wrapped)

    rhs = divergence_stack(gx, gy)

    # Solve the discrete Poisson equation in the Fourier domain.
    # The FFT helper returns the positive-denominator form, so apply -1 to
    # match the Laplacian sign convention here.
    phase_unwrapped = poisson_solve_fft_stack(rhs)
    phase_unwrapped *= dtype.type(-1)

    if restore_plane:
        phase_unwrapped = restore_mean_plane(phase_unwrapped, phase_wrapped)

    if input_2d:
        phase_unwrapped = phase_unwrapped[0]
    return phase_unwrapped
