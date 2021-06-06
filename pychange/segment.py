# Importing packages
import numpy as np
from numba import njit

@njit(fastmath=True)
def create_summary_stats(x):
    x = np.stack([np.append(0, x.cumsum()),
                  np.append(0, (x ** 2).cumsum()),
                  np.append(0, ((x - x.mean()) ** 2).cumsum())],
                  axis=-1)
    return x

def amoc_segment(x, min_len, cost_fn):

    n = x.shape[0]
    sum_stats = create_summary_stats(x)
    null_cost = cost_fn(sum_stats[n, :], n)
    costs = np.empty((n - 1 - 2 * min_len, ))
    best_ind = 0
    for i in range(min_len, n - 1 - min_len):
        costs[i - min_len] = cost_fn(sum_stats[i, :] - sum_stats[0, :], i) + cost_fn(sum_stats[n, :] - sum_stats[i, :], n - i)
    
    if costs.min() < null_cost:
        best_ind = np.argmin(costs) + min_len
    return best_ind
