from .._ndarray_backend import xp
from ._ls_common import divergence_stack, poisson_solve_fft_stack, wrap_phase
from ._plane_utils import restore_mean_plane


def algo_ls_poisson_pg(
    phase_wrapped: xp.ndarray,
    restore_plane: bool = False,
) -> xp.ndarray:
    """
    Least-squares unwrapping with periodic gradient enforcement.

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
    This implementation enforces periodic boundary conditions on the wrapped
    gradients prior to solving the Poisson equation.

    References
    ----------
    - D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
      Theory, Algorithms, and Software," Wiley, 1998.
    """
    input_2d = False
    if phase_wrapped.ndim == 2:
        input_2d = True
        phase_wrapped = xp.expand_dims(phase_wrapped, axis=0)
    elif phase_wrapped.ndim != 3:
        raise ValueError("phase_wrapped must have shape (H, W) or (N, H, W).")

    # Periodic gradients.
    gx, gy = wrapped_gradients_stack(phase_wrapped)
    gx, gy = enforce_periodic_gradients_stack(gx, gy)
    rhs = divergence_stack(gx, gy)

    # The FFT helper returns the positive-denominator form, so apply -1 to
    # match the Laplacian sign convention here.
    phi = poisson_solve_fft_stack(rhs)
    phi *= phi.dtype.type(-1)

    if restore_plane:
        phi = restore_mean_plane(phi, phase_wrapped)
    if input_2d:
        phi = phi[0]
    return phi


def wrapped_gradients_stack(
        phi: xp.ndarray,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Wrapped forward gradients for a stack of phases.

    Parameters
    ----------
    phi : xp.ndarray
        Wrapped phase, shape (N, H, W).

    Returns
    -------
    gx, gy : xp.ndarray
        Wrapped gradients with shape (N, H, W).
    """
    gx = wrap_phase(phi[:, :, 1:] - phi[:, :, :-1])
    gy = wrap_phase(phi[:, 1:, :] - phi[:, :-1, :])

    gx = xp.pad(gx, ((0, 0), (0, 0), (0, 1)))
    gy = xp.pad(gy, ((0, 0), (0, 1), (0, 0)))

    return gx, gy


def enforce_periodic_gradients_stack(
        gx: xp.ndarray,
        gy: xp.ndarray,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Enforce periodic boundary conditions on wrapped gradients.

    Parameters
    ----------
    gx, gy : xp.ndarray
        Gradient stacks, shape (N, H, W).

    Returns
    -------
    gx, gy : xp.ndarray
        Periodic gradients, shape (N, H, W).
    """
    # x-direction periodicity
    gx[:, :, -1] = gx[:, :, 0]

    # y-direction periodicity
    gy[:, -1, :] = gy[:, 0, :]

    return gx, gy
