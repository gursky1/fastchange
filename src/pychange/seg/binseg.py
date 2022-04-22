# Importing packages
import numpy as np
import numba as nb

from .base import BaseSeg, seg_sig


class BinSeg(BaseSeg):
    
    @staticmethod
    @nb.njit(seg_sig(), fastmath=True)
    def seg_fn(cost, sumstats, cost_args, penalty, min_len, max_cps, n):
        
        # Creating partial cost function
        def _cost_fn(start, end,):
            return cost(start, end, sumstats, cost_args)

        # Initializing our array of changepoint candidates and costs
        tau = np.empty((max_cps + 2,), dtype=np.int64)
        tau[0] = 0
        tau[1] = n
        
        # Initializing array of found changepoints and their associated costs
        cps = np.empty((max_cps,), dtype=np.int64)
        cps_costs = np.zeros((max_cps,), dtype=np.float64)
        
        # Iterating to a max depth of our max changepoint limit
        for q in range(max_cps):
            
            # Setting best cost to compare against
            best_ind = 0
            best_cost = 0.0
            
            # Iterating over each of the current segments
            for ind in nb.prange(q + 1):
                
                # Setting start and end
                start = tau[ind] + 1
                end = tau[ind + 1]
                
                # Getting null cost
                null_cost = _cost_fn(start, end)

                # Adjusting for min len
                start_ind = start + min_len
                end_ind = end - min_len + 1
                
                # Iterating over candidate points
                for j in nb.prange(start_ind, end_ind):
                    _cost = _cost_fn(start, j) + _cost_fn(j, end) - null_cost
                    if _cost < best_cost:
                        best_ind = j
                        best_cost = _cost

            # Finding the best changepoint candidate from this run
            cps[q] = best_ind
            
            # If better than the previous best cost, add to change point list
            if best_cost < cps_costs[q]:
                cps_costs[q] = best_cost
                
            # Adding changepoint to our list of endpoints
            tau[q + 2] = best_ind
            tau[: q + 3].sort()

        # Pruning changepoints by penalty
        valid_cps = cps_costs <= -1 * penalty
        n_cps = np.sum(valid_cps)
        if n_cps == 0:
            cps = np.int64([-1])
        else:
            cps = cps[:n_cps]
            cps.sort()
        return cps
