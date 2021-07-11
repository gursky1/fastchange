# Importing packages
from functools import partial
from time import time
import numpy as np
from numba import jit, njit
import ruptures as rpt
#from pychange.costs import normal_mean_var_cost
from pychange.numba_costs import normal_mean_var_cost as aot_normal_mean_var_cost
from pychange_cython.cython_costs import cython_normal_mean_var_cost
from pychange.numba_costs import scalar_normal_mean_var_cost as aot_scalar_normal_mean_var_cost
from pychange.costs import scalar_normal_mean_var_cost
from pychange.segment import create_summary_stats, binary_segmentation


if __name__ == '__main__':
    # Creating test series
    x = np.concatenate([np.random.normal(-1, 1, (2000,)),
                        np.random.normal(1, 2, (2000,)),
                        np.random.normal(-1, 1, (2000,)),
                        np.random.normal(1, 2, (2000,)),
                        np.random.normal(-1, 1, (2000,)),
                        np.random.normal(1, 2, (2000,))])

    # Creating summary stats
    sum_stats = create_summary_stats(x)
    sum_stats[:, 3] = sum_stats[:, 3] + 1
    
    # Defining pure python
    def normal_mean_var_cost(x):
        return x[:, 3] * (np.log(2 * np.pi) + np.log(np.fmax((x[:, 1] - ((x[:, 0] * x[:, 0]) / x[:, 3]))/ x[:, 3], 1e-8) + 1))

    # Jitting cost function
    jit_normal_mean_var_cost = njit(fastmath=True)(normal_mean_var_cost)
    _ = jit_normal_mean_var_cost(sum_stats)

    # Jitting binary segmentation
    jit_binary_segmentation = njit(fastmath=True, parallel=True)(binary_segmentation)
    _ = jit_binary_segmentation(x, 30, 100, 100, jit_normal_mean_var_cost)

    # Testing each method with pure python binseg
    test_dict = {'Numba (AOT)': aot_normal_mean_var_cost,
                 'Numba (JIT)': jit_normal_mean_var_cost,
                 'Numba Scalar (AOT)': aot_scalar_normal_mean_var_cost,
                 'Cython': cython_normal_mean_var_cost,
                 'Pure Python': normal_mean_var_cost,
                 'Pure Python (Scalar)': scalar_normal_mean_var_cost,
                 }

    # Segmentation timing
    print('\nScoring timing')
    for k, v in test_dict.items():
        if 'Scalar' not in k:
            start_time = time()
            for _ in range(1000):
                _ = v(sum_stats)
            print(f'{k}: {(time() - start_time)}')
        else:
            start_time = time()
            for _ in range(1000):
                _ = [v(sum_stats[i, :]) for i in range(sum_stats.shape[0])]
            print(f'{k}: {(time() - start_time)}')

    test_dict = {'Numba (AOT)': aot_normal_mean_var_cost,
            'Numba (JIT)': jit_normal_mean_var_cost,
            'Cython': cython_normal_mean_var_cost,
            'Pure Python': normal_mean_var_cost,
            }

    # Binseg timing
    print('\nBinary Segmentation Timing')
    for k, v in test_dict.items():
        start_time = time()
        #for _ in range(1000):
        _ = binary_segmentation(x, 30, 100, 100, v)
        print(f'{k}: {(time() - start_time)}')

    # Testing jit binseg
    start_time = time()
    _ = jit_binary_segmentation(x, 30, 100, 100, jit_normal_mean_var_cost)
    print(f'Jitted BinSeg: {(time() - start_time)}')

    # Testing ruptures
    start_time = time()
    algo = rpt.Binseg(model="l2").fit(x)
    _ = algo.predict(pen=100)
    print(f'Ruptures: {(time() - start_time)}')
