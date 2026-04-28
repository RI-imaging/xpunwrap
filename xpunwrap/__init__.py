from ._ndarray_backend import get_ndarray_backend, set_ndarray_backend

from .algorithms import (
    algo_ls_poisson,
    algo_ls_poisson_pg,
    algo_ls_weighted,
    algo_skimage_unwrap,
)


def algos_available():
    algos_available = {}
    for algo in [algo_ls_poisson,
                 algo_ls_poisson_pg,
                 algo_ls_weighted,
                 algo_skimage_unwrap]:
        algos_available[algo.__name__] = algo
    return algos_available
