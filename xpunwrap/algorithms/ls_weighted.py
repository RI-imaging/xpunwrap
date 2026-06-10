from .._ndarray_backend import xp
from ._ls_common import wrap_phase
from ._plane_utils import restore_mean_plane


def algo_ls_weighted(
        phase_wrapped: xp.ndarray,
        border_thresh=1.5,
        n_iter=200,
        restore_plane: bool = False,
):
    """
    Weighted least-squares unwrapping with border masking.

    This is a simplified binary-weight Jacobi variant of the weighted LS
    method from Ghiglia and Romero.

    Parameters
    ----------
    phase_wrapped : xp.ndarray
        Wrapped phase, shape (H, W) or (N, H, W), values in [-pi, pi).
    border_thresh : float
        Gradient magnitude threshold for border detection.
    n_iter : int
        Jacobi iterations for the weighted Poisson solve.
    restore_plane : bool, optional
        If True, add back the mean wrapped-gradient plane. Default False.

    Returns
    -------
    phase_unwrapped : xp.ndarray
        Unwrapped phase, same shape as input.

    References
    ----------
    - D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and
      unweighted phase unwrapping that uses fast transforms and iterative
      methods," J. Opt. Soc. Am. A, vol. 11, no. 1, pp. 107-117, 1994.
    - D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
      Theory, Algorithms, and Software," Wiley, 1998.
    """
    input_2d = False
    if phase_wrapped.ndim == 2:
        input_2d = True
        phase_wrapped = xp.expand_dims(phase_wrapped, axis=0)
    elif phase_wrapped.ndim != 3:
        raise ValueError("phase_wrapped must have shape (H, W) or (N, H, W).")

    gx = wrap_phase(xp.diff(phase_wrapped, axis=2,
                            append=phase_wrapped[:, :, -1:]))
    gy = wrap_phase(xp.diff(phase_wrapped, axis=1,
                            append=phase_wrapped[:, -1:, :]))

    w = _phase_border_weights(gx, gy, thresh=border_thresh)

    f = _weighted_divergence(gx, gy, w)

    phi = _weighted_poisson_solver(f, w, n_iter=n_iter)

    if restore_plane:
        phi = restore_mean_plane(phi, phase_wrapped)

    if input_2d:
        phi = phi[0]
    return phi


def _weighted_poisson_solver(
        f: xp.ndarray,
        w: xp.ndarray,
        n_iter: int = 200,
) -> xp.ndarray:
    """
    Solve div(w * grad(phi)) = f with Jacobi iterations.

    Parameters
    ----------
    f : xp.ndarray
        Right-hand side, shape (N, H, W).
    w : xp.ndarray
        Weights, shape (N, H, W).

    Returns
    -------
    xp.ndarray
        Unwrapped phase estimate, shape (N, H, W).
    """
    phi = xp.zeros_like(f)

    for _ in range(n_iter):
        phi_xp = xp.roll(phi, -1, axis=2)
        phi_xm = xp.roll(phi, 1, axis=2)
        phi_yp = xp.roll(phi, -1, axis=1)
        phi_ym = xp.roll(phi, 1, axis=1)

        w_xp = xp.roll(w, -1, axis=2)
        w_xm = xp.roll(w, 1, axis=2)
        w_yp = xp.roll(w, -1, axis=1)
        w_ym = xp.roll(w, 1, axis=1)

        denom = w_xp + w_xm + w_yp + w_ym
        denom = xp.where(denom == 0, denom.dtype.type(1), denom)

        phi = (
                      w_xp * phi_xp +
                      w_xm * phi_xm +
                      w_yp * phi_yp +
                      w_ym * phi_ym -
                      f
              ) / denom

    return phi


def _weighted_divergence(
        gx: xp.ndarray,
        gy: xp.ndarray,
        w: xp.ndarray,
) -> xp.ndarray:
    """
    Compute the divergence of the weighted wrapped gradients.

    Parameters
    ----------
    gx, gy : xp.ndarray
        Wrapped gradients, shape (N, H, W).
    w : xp.ndarray
        Weights, shape (N, H, W).

    Returns
    -------
    xp.ndarray
        Weighted divergence, shape (N, H, W).
    """
    wx = w * gx
    wy = w * gy

    div_x = wx - xp.roll(wx, 1, axis=2)
    div_y = wy - xp.roll(wy, 1, axis=1)

    return div_x + div_y


def _phase_border_weights(
        gx: xp.ndarray,
        gy: xp.ndarray,
        thresh: float = 1.5,
) -> xp.ndarray:
    """
    Detect phase borders using gradient magnitude.

    Parameters
    ----------
    gx, gy : xp.ndarray
        Wrapped gradients, shape (N, H, W).
    thresh : float
        Gradient magnitude threshold.

    Returns
    -------
    xp.ndarray
        Binary weights, shape (N, H, W).
    """
    grad_mag = xp.sqrt(gx ** 2 + gy ** 2)

    w = xp.ones_like(grad_mag)
    w[grad_mag > thresh] = 0.0

    return w
