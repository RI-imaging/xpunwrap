from ._ndarray_backend import get_ndarray_backend, set_ndarray_backend

from .algorithms import (
    algo_ls_poisson,
    algo_ls_poisson_periodic_grad,
    algo_ls_weighted,
    algo_tvl1,
)


def algos_available():
    algos_available = {}
    for algo in [algo_ls_poisson,
                 algo_ls_poisson_periodic_grad,
                 algo_ls_weighted,
                 algo_tvl1]:
        algos_available[algo.__name__] = algo
    return algos_available
