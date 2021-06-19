# Importing packages
from functools import partial
from time import time
import numpy as np
from numba import njit
from pychange.costs import iter_sumstats
from pychange.numba_costs import iter_sumstats as aot_iter_sumstats
from pychange_cython.cython_costs import iter_sumstats as cy_iter_sumstats
from pychange.segment import create_summary_stats


if __name__ == '__main__':
    # Creating test series
    x = np.concatenate([np.random.normal(-1, 1, (2000,)), np.random.normal(1, 2, (2000,))])

    # Creating summary stats
    sum_stats = create_summary_stats(x)#[1000, :]

    # Jitting iter function
    jit_iter_sumstats = njit(fastmath=True)(iter_sumstats)
    _ = jit_iter_sumstats(sum_stats, 10, 3900)

    # Testing each method
    test_dict = {'Numba (AOT)': aot_iter_sumstats,
                 'Numba (JIT)': jit_iter_sumstats,
                 'Cython': cy_iter_sumstats,
                 'Pure Python': iter_sumstats,
                 }
    for k, v in test_dict.items():
        start_time = time()
        for _ in range(1000):
            v(sum_stats, 10, 3900)
        print(f'{k}: {(time() - start_time)}')
