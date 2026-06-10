from collections.abc import Callable
from typing import Any

from ._ndarray_backend import get_ndarray_backend, set_ndarray_backend

from .algorithms import (
    algo_ls_poisson,
    algo_ls_poisson_pg,
    algo_ls_weighted,
    algo_skimage_unwrap,
)


def algos_available() -> dict[str, Callable[..., Any]]:
    """Return all available phase unwrapping algorithms.

    Returns
    -------
        Mapping from algorithm name to callable, in the order:
        ``algo_ls_poisson``, ``algo_ls_poisson_pg``, ``algo_ls_weighted``,
        ``algo_skimage_unwrap``.
    """
    algos = {}
    for algo in [algo_ls_poisson,
                 algo_ls_poisson_pg,
                 algo_ls_weighted,
                 algo_skimage_unwrap]:
        algos[algo.__name__] = algo
    return algos
