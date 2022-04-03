# Importing packages
from functools import partial
from time import time
import numpy as np
from numba import jit, njit
import ruptures as rpt
from ruptures.base import BaseCost
from pychange.costs import normal_mean_var_cost, scalar_normal_mean_var_cost, ParametricCost
from pychange.numba_costs import normal_mean_var_cost as aot_normal_mean_var_cost
#from pychange_cython.cython_costs import cython_normal_mean_var_cost
from pychange.numba_costs import scalar_normal_mean_var_cost as aot_scalar_normal_mean_var_cost
from pychange.costs import scalar_normal_mean_var_cost#, ParametricCost
from pychange.segment import create_summary_stats, pelt


if __name__ == '__main__':

    # Creating test series
    x = np.concatenate([np.random.normal(-1, 1, (10000,)),
                        np.random.normal(1, 2, (10000,)),
                        np.random.normal(-1, 1, (10000,)),
                        np.random.normal(1, 2, (10000,)),
                        np.random.normal(-1, 1, (10000,)),
                        np.random.normal(1, 2, (10000,))])

    # Creating summary stats
    sum_stats = create_summary_stats(x)

    # Jitting sum stats
    jit_sum_stats = njit(fastmath=True)(create_summary_stats)
    _ = jit_sum_stats(x)
    
    # Defining pure python
    #def normal_mean_var_cost(x):
    #    return x[:, 3] * (np.log(2 * np.pi) + np.log(np.fmax((x[:, 1] - ((x[:, 0] * x[:, 0]) / x[:, 3]))/ x[:, 3], 1e-8) + 1))

    # Jitting cost function
    jit_normal_mean_var_cost = njit(fastmath=True)(normal_mean_var_cost)
    _ = jit_normal_mean_var_cost(sum_stats)
    print('jitted cost_fn')

    # Jitting binary segmentation
    jit_pelt = njit(fastmath=True)(pelt)
    _ = jit_pelt(x[:1000], 30, 100, jit_sum_stats, jit_normal_mean_var_cost)
    print('jitted pelt')

    # Testing each method with pure python binseg
    test_dict = {'Numba (AOT)': aot_normal_mean_var_cost,
                 #'Numba (JIT)': jit_normal_mean_var_cost,
                 'Numba Scalar (AOT)': aot_scalar_normal_mean_var_cost,
                 #'Cython': cython_normal_mean_var_cost,
                 #'Pure Python': normal_mean_var_cost,
                 #'Pure Python (Scalar)': scalar_normal_mean_var_cost,
                 }

    # Segmentation timing
    # print('\nScoring timing')
    # for k, v in test_dict.items():
    #     if 'Scalar' not in k:
    #         start_time = time()
    #         for _ in range(1000):
    #             _ = v(sum_stats)
    #         print(f'{k}: {(time() - start_time)}')
    #     else:
    #         start_time = time()
    #         for _ in range(1000):
    #             _ = [v(sum_stats[i, :]) for i in range(sum_stats.shape[0])]
    #         print(f'{k}: {(time() - start_time)}')

    test_dict = {'Numba (AOT)': aot_normal_mean_var_cost,
            'Numba (JIT)': jit_normal_mean_var_cost,
            #'Cython': cython_normal_mean_var_cost,
            'Pure Python': normal_mean_var_cost,
            }
    print('Starting pelt timing')

    # Binseg timing
    print('\nPelt Timing')
    for k, v in test_dict.items():
        start_time = time()
        #for _ in range(1000):
        _ = pelt(x, 30, 100, jit_sum_stats, v)
        print(f'{k}: {(time() - start_time)}')

    # Testing jit binseg
    start_time = time()
    _ = jit_pelt(x, 30, 100, jit_sum_stats, jit_normal_mean_var_cost)
    print(f'Jitted Pelt: {(time() - start_time)}')

    # Jitting scaler
    jit_scalar_normal_mean_var_cost = njit(fastmath=True)(scalar_normal_mean_var_cost)
    _ = jit_scalar_normal_mean_var_cost(sum_stats[100, :])

    test_dict = {'Numba (AOT)': aot_scalar_normal_mean_var_cost,
            'Numba (JIT)': jit_scalar_normal_mean_var_cost,
            #'Cython': cython_normal_mean_var_cost,
            'Pure Python': scalar_normal_mean_var_cost,
            }

    # Testing ruptures
    print('\nRuptures Pelt Timing')
    for k, v in test_dict.items():
        start_time = time()
        algo = rpt.Pelt(custom_cost=ParametricCost(cost_fn=v)).fit(x)
        _ = algo.predict(pen=100)
        print(f'{k}: {(time() - start_time)}')
