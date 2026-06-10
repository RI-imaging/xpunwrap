from __future__ import annotations


def real_pi(xp, dtype):
    """Return pi cast to the requested real dtype.

    Parameters
    ----------
    xp : module
        Active ndarray backend (``numpy`` or ``cupy``).
    dtype : dtype-like
        Target real floating-point dtype.

    Returns
    -------
    scalar
        ``pi`` as a scalar of ``dtype``.
    """
    dt = xp.dtype(dtype)
    return dt.type(xp.pi)


def complex_dtype_for_real(xp, dtype):
    """Map a real dtype to the corresponding complex dtype.

    ``float16`` and ``float32`` map to ``complex64``; ``float64`` maps to
    ``complex128``. Complex inputs are returned unchanged. Uncommon real
    dtypes fall back to ``complex128``.

    Parameters
    ----------
    xp : module
        Active ndarray backend (``numpy`` or ``cupy``).
    dtype : dtype-like
        Input dtype, real or complex.

    Returns
    -------
    dtype
        Corresponding complex dtype.
    """
    dt = xp.dtype(dtype)
    if dt.kind == "c":
        return dt
    if dt == xp.float16 or dt == xp.float32:
        return xp.complex64
    if dt == xp.float64:
        return xp.complex128
    return xp.complex128
