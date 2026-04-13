from __future__ import annotations


def real_pi(xp, dtype):
    """
    Return pi in the requested real dtype.
    """
    dt = xp.dtype(dtype)
    return dt.type(xp.pi)


def complex_dtype_for_real(xp, dtype):
    """
    Map real dtypes to an appropriate complex dtype.
    """
    dt = xp.dtype(dtype)
    if dt.kind == "c":
        return dt
    if dt == xp.float16 or dt == xp.float32:
        return xp.complex64
    if dt == xp.float64:
        return xp.complex128
    # Fallback: default to complex128 for uncommon real dtypes
    return xp.complex128
