from .._dtype_utils import real_pi
from .._ndarray_backend import xp


def algo_ls_weighted(
        phase_wrapped: xp.ndarray,
        border_thresh=1.5,
        n_iter=200,
):
    """
    Weighted least-squares unwrapping with border masking.

    Parameters
    ----------
    phase_wrapped : xp.ndarray
        Wrapped phase, shape (N, H, W) or (H, W).
    border_thresh : float
        Gradient magnitude threshold for border detection.
    n_iter : int
        Jacobi iterations for the weighted Poisson solve.

    Returns
    -------
    xp.ndarray
        Unwrapped phase, same shape as input.

    References
    ----------
    .. [1] D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and
       unweighted phase unwrapping that uses fast transforms and iterative
       methods," J. Opt. Soc. Am. A, vol. 11, no. 1, pp. 107-117, 1994.
    .. [2] D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
       Theory, Algorithms, and Software," Wiley, 1998.
    """
    input_2d = False
    if phase_wrapped.ndim == 2:
        input_2d = True
        phase_wrapped = xp.expand_dims(phase_wrapped, axis=0)
    elif phase_wrapped.ndim != 3:
        raise ValueError("phase_wrapped must have shape (H, W) or (N, H, W).")

    dtype = phase_wrapped.dtype
    pi = real_pi(xp, dtype)
    two_pi = dtype.type(2) * pi

    gx = xp.diff(phase_wrapped, axis=2, append=phase_wrapped[:, :, -1:])
    gy = xp.diff(phase_wrapped, axis=1, append=phase_wrapped[:, -1:, :])

    gx = (gx + pi) % two_pi - pi
    gy = (gy + pi) % two_pi - pi

    w = _phase_border_weights(phase_wrapped, thresh=border_thresh)

    f = _weighted_divergence(gx, gy, w)

    phi = _weighted_poisson_solver(f, w, n_iter=n_iter)

    if input_2d:
        phi = phi[0]
    return phi


def _weighted_poisson_solver(
        f,
        w,
        n_iter=200,
):
    """
    Solve div(w * grad(phi)) = f using Jacobi iterations.

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


def _weighted_divergence(gx, gy, w):
    """
    Compute divergence of weighted wrapped gradients.

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

    div_x = xp.diff(wx, axis=2, prepend=wx[:, :, :1])
    div_y = xp.diff(wy, axis=1, prepend=wy[:, :1, :])

    return div_x + div_y


def _phase_border_weights(wrapped, thresh=1.5):
    """
    Detect phase borders using gradient magnitude.

    Parameters
    ----------
    wrapped : xp.ndarray
        Wrapped phase, shape (N, H, W).
    thresh : float
        Gradient magnitude threshold.

    Returns
    -------
    xp.ndarray
        Binary weights, shape (N, H, W).
    """
    dtype = wrapped.dtype
    pi = real_pi(xp, dtype)
    two_pi = dtype.type(2) * pi

    gx = xp.diff(wrapped, axis=2, append=wrapped[:, :, -1:])
    gy = xp.diff(wrapped, axis=1, append=wrapped[:, -1:, :])

    gx = (gx + pi) % two_pi - pi
    gy = (gy + pi) % two_pi - pi

    grad_mag = xp.sqrt(gx ** 2 + gy ** 2)

    w = xp.ones_like(grad_mag)
    w[grad_mag > thresh] = 0.0

    return w
