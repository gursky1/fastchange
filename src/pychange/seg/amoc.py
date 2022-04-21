# Importing packages
import numpy as np
import numba as nb

from .base import BaseSeg, seg_sig


class AmocSeg(BaseSeg):
    
    @staticmethod
    @nb.njit(seg_sig(), fastmath=True, nogil=True, parallel=True)
    def seg_fn(cost, sumstats, cost_args, penalty, min_len, max_cps, n):
        
        # Getting cost with no changepoints
        null_cost = cost(0, n, sumstats, cost_args)
        
        # Finding possible changepoint locations
        costs = np.empty(n - 2 * min_len, dtype=np.float64)
        for i in nb.prange(min_len, n - min_len):
            costs[i - min_len] = cost(0, i, sumstats, cost_args) + cost(i, n, sumstats, cost_args) + penalty
        
        # Determining if best changepoint better than null
        best_ind = np.argmin(costs)
        if costs[best_ind] < null_cost:
            cp = np.int64([best_ind + min_len])
        else:
            cp = np.int64([-1])
            
        return cp
