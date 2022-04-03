# Importing packages
from functools import partial
from time import time
import numpy as np
from numba import jit, njit
import ruptures as rpt
from ruptures.base import BaseCost
from pychange.costs import nonparametric_cost, NonParametricCost
from pychange.numba_costs import nonparametric_cost as aot_nonparametric_cost
#from pychange_cython.cython_costs import cython_normal_mean_var_cost
#from pychange.numba_costs import scalar_normal_mean_var_cost as aot_scalar_normal_mean_var_cost
#from pychange.costs import scalar_normal_mean_var_cost#, ParametricCost
from pychange.preprocess import create_partial_sums
from pychange.nonparametric_segment import pelt


if __name__ == '__main__':

    # Creating test series
    x = np.concatenate([np.random.normal(-1, 1, (10000,)),
                        np.random.normal(1, 2, (10000,)),
                        np.random.normal(-1, 1, (10000,)),
                        np.random.normal(-3, 1, (10000,)),
                        np.random.normal(1, 2, (10000,)),
                        np.random.normal(-2, 1, (10000,)),
                        np.random.normal(-1, 1, (10000,))])

    # Creating summary stats
    k = min([x.shape[0], int(np.ceil(4 * np.log(x.shape[0])))])
    n = x.shape[0]
    jit_create_partial_sums = njit(fastmath=True)(create_partial_sums)
    sum_stats = jit_create_partial_sums(x, k)
    print('jitted sum_stats')

    # Defining pure python
    #def normal_mean_var_cost(x):
    #    return x[:, 3] * (np.log(2 * np.pi) + np.log(np.fmax((x[:, 1] - ((x[:, 0] * x[:, 0]) / x[:, 3]))/ x[:, 3], 1e-8) + 1))

    # Jitting cost function
    jit_nonparametric_cost = njit(fastmath=True)(nonparametric_cost)
    _ = jit_nonparametric_cost(sum_stats[100, :], 0, 100, k, x.shape[0])
    print('jitted cost_fn')

    # Jitting binary segmentation
    jit_pelt = njit(fastmath=True)(pelt)
    _ = jit_pelt(x[:100], 30, 100, k, jit_create_partial_sums, jit_nonparametric_cost)
    print('jitted pelt')

    # Testing each method with pure python binseg
    # test_dict = {'Numba (AOT)': aot_nonparametric_cost,
    #              'Numba (JIT)': jit_nonparametric_cost,
    #              #'Numba Scalar (AOT)': aot_scalar_normal_mean_var_cost,
    #              #'Cython': cython_normal_mean_var_cost,
    #              'Pure Python': nonparametric_cost,
    #              #'Pure Python (Scalar)': scalar_normal_mean_var_cost,
    #              }

    # # Segmentation timing
    # print('\nScoring timing')
    # for m, v in test_dict.items():
    #     start_time = time()
    #     for _ in range(1000):
    #         _ = [v(sum_stats, 0, i, k, x.shape[0]) for i in range(10, sum_stats.shape[0])]
    #     print(f'{m}: {(time() - start_time)}')

    test_dict = {'Numba (AOT)': aot_nonparametric_cost,
                 'Numba (JIT)': jit_nonparametric_cost,
                 #'Numba Scalar (AOT)': aot_scalar_normal_mean_var_cost,
                 #'Cython': cython_normal_mean_var_cost,
                 'Pure Python': nonparametric_cost,
                 #'Pure Python (Scalar)': scalar_normal_mean_var_cost,
                 }
    print('Starting pelt timing')

    # Binseg timing
    print('\nPelt Timing')
    for m, v in test_dict.items():
        start_time = time()
        #for _ in range(1000):
        _ = pelt(x, 30, 100, k, create_partial_sums, v)
        print(f'{m}: {(time() - start_time)}')

    # Testing jit binseg
    start_time = time()
    _ = jit_pelt(x, 30, 100, k, jit_create_partial_sums, jit_nonparametric_cost)
    print(f'Jitted Pelt: {(time() - start_time)}')

    # Testing ruptures
    print('\nRuptures Pelt Timing')
    for m, v in test_dict.items():
        start_time = time()
        algo = rpt.Pelt(custom_cost=NonParametricCost(cost_fn=v, k=k)).fit(x)
        _ = algo.predict(pen=100)
        print(f'{m}: {(time() - start_time)}')
