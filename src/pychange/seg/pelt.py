# Importing packages
import numpy as np
import numba as nb

from .base import BaseSeg, seg_sig


class PeltSeg(BaseSeg):
        
    @staticmethod
    @nb.njit(seg_sig(), fastmath=True)
    def seg_fn(cost, sumstats, cost_args, penalty, min_len, max_cps, n):
        
        # Creating partial cost function
        def _cost_fn(start, end):
            return cost(start, end, sumstats, cost_args)
        
        # Hold cost values at each iteration
        f = np.empty(n + 1, dtype=np.float64)
        f[0] = -1 * penalty
        f[1:min_len] = 0.0
        for i in range(min_len, 2 * min_len):
            f[i] = _cost_fn(0, i)
        
        # Setting number of changepoints found
        n_cps = np.empty(n + 1, dtype=np.int64)
        n_cps[: min_len] = 0
        n_cps[min_len: 2 * min_len] = 1
        
        # Last changepoints that we found
        cps = np.empty(n + 1, dtype=np.int64)
        cps[: 2 * min_len] = 0
        
        # Array to hold costs temporarily at each iteration
        _costs = np.empty(n, dtype=np.float64)
        
        # Array for tracking valid indices
        r = np.empty(n, dtype=np.int64)
        r[0] = 0
        r[1] = min_len
        r_len = 2

        # Starting loop for search over valid indices
        for tau_star in range(2 * min_len, n + 1):

            # Finding cost of valid indices
            for i in range(r_len):
                _costs[i] = _cost_fn(r[i], tau_star) + f[r[i]] + penalty
                    
            # Getting best cost and index
            best_ind = np.argmin(_costs[:r_len])
            best_cost = _costs[best_ind]

            # Writing best score
            f[tau_star] = best_cost
            cps[tau_star] = r[best_ind]
            n_cps[tau_star] = n_cps[cps[tau_star]] + 1
            
            # Updating our values of R
            new_r_len = 0
            for i in range(r_len):
                if _costs[i] <= f[tau_star] + penalty:
                    r[new_r_len] = r[i]
                    new_r_len += 1
            r_len = new_r_len
            
            # Adding the next element to our candidate list
            r[r_len] = tau_star - min_len - 1
            r_len += 1
            
        # Getting final set of changepoints
        n_valid_cps = n_cps[n]
        if n_valid_cps == 1:
            cps_out = np.int64([-1])
        else:
            cps_out = np.empty(n_valid_cps + 1, dtype=np.int64)
            valid_cp = cps[n]
            ind = 0
            while valid_cp > min_len:
                cps_out[ind] = valid_cp
                valid_cp = cps[valid_cp]
                ind += 1
            cps_out = cps_out[:ind]
            cps_out.sort()
        
        return cps_out
