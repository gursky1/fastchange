# Importing packages
import math
import numpy as np
import numba as nb
from numba.types import FunctionType

from .costs import preprocess_sig, cost_sig


def seg_sig(*args):
    return np.int64[:](nb.float64[:], *args, FunctionType(preprocess_sig), FunctionType(cost_sig), np.float64[:])


class BaseSeg():
    
    def __init__(self, cost, min_len=1, max_cps=-1):
        self.cost = cost
        self.min_len = min_len
        self.max_cps = max_cps
        
    def fit(self, X):
        self.cost.fit(X)
        return self
    
    def predict(self):
        return self.seg_fn(self.cost.cost, self.min_len, self.cost.n)


class AmocSeg(BaseSeg):
    
    @staticmethod
    @nb.njit(seg_sig(nb.int64, nb.int64), fastmath=True, nogil=True, parallel=True)
    def seg_fn(cost, min_len, n):
        
        costs = np.empty(n - 2 * min_len, dtype=np.float64)
        for i in nb.prange(min_len, n - min_len):
            costs[i - min_len] = cost(0, i) + cost(i, n)
        return np.argmin(costs) + min_len



@nb.njit(fastmath=True, nogil=True)
def _binseg(start_ind, end_ind, null_cost, cost_array, cps, cost, penalty, min_len, max_cps):

    # Calculating valid costs over relevant window
    for i in range(start_ind + min_len, end_ind - min_len):
        cost_array[i, 0] = cost.cost(start_ind, i)
        cost_array[i, 1] = cost.cost(i, end_ind)
        cost_array[i, 2] = cost_array[i, 0] + cost_array[i, 1]
        
    # Getting best split value for this iteration
    best_split = np.argmin(cost_array[start_ind + min_len: end_ind - min_len, 2])
    best_ind = best_split + start_ind + min_len
    best_cost = cost_array[best_ind, 2]
    
    if best_cost + penalty >= null_cost:
        return
    
    else:

        # Adding changepoint to list
        if max_cps > 0:
            if cps.sum() >= max_cps:
                return
        cps[best_ind] = 1

        # Calculating next splits
        min_cand_len = 2 * min_len + 1

        # Pre segment
        if best_ind - start_ind >= min_cand_len:
            _binseg(start_ind, best_ind, cost_array[best_ind, 0], cost_array, cps, cost, penalty, min_len, max_cps)

        # Post segment
        if end_ind - best_ind >= min_cand_len:
            _binseg(best_ind, end_ind, cost_array[best_ind, 1], cost_array, cps, cost, penalty, min_len, max_cps)
    return cps


@nb.njit(fastmath=True, nogil=True)
def binary_segment(cost, penalty, min_len, max_cps):
    
    # Setting up values for segmentations
    n = cost.n
    cost_array = np.empty((n, 3), dtype=np.float64)
    null_cost = cost.cost(0, n)
    cps = np.zeros((n,))
    cps[n] = True
    
    # Running binary segmentation
    _binseg(0, n, null_cost, cost_array, cps, cost, penalty, min_len, max_cps)
    
    return np.nonzero(cps)[0]


class BinSeg(BaseSeg):
        
    def fit(self, X):
        self.cost.fit(X)
        if not isinstance(self.penalty, float):
            self._pen = self.penalty(self.cost.n, self.cost.n_params)
        else:
            self._pen = self.penalty
        self.cps = binary_segment(self.cost, self._pen, self.min_len, self.max_cps)
        return self
    
    def predict(self):
        return self.cps


@nb.njit(fastmath=True, nogil=True)
def pelt_segment(cost, max_cps, penalty, mbic):
    """Pruned exact linear time changepoint segmentation"""
    
    # Initializing parameters for segmentation
    n = cost.n
    
    # Initializing storage for costs and changepoints
    f = np.empty(n, dtype=np.float64)
    f[0] = -1 * penalty
    _costs = np.empty(n, dtype=np.float64)
    n_cps = np.zeros(n, dtype=np.int64)
    cps = np.empty((n, max_cps), dtype=np.int64)
    r = np.empty(n, dtype=np.int64)
    r[0] = 0
    r_len = 1

    # Starting loop for search
    for tau_star in range(1, n):

        # Calculating each candidate cost
        for j in nb.prange(r_len):
            tau = r[j]
            _costs[j] = cost.cost(tau, tau_star) + f[tau] + penalty
            if mbic:
                _costs[j] += math.log(tau_star - tau)

        # Finding best candidate
        best_tau = np.argmin(_costs[: r_len])
        f[tau_star] = _costs[best_tau]
        best_r = r[best_tau]
        
        # Updating our changepoint array
        swap_cps = n_cps[best_r]
        cps[tau_star, : swap_cps] = cps[best_r, : swap_cps]
        old_swap_cps = swap_cps
        cps[tau_star, swap_cps] = best_r
        n_cps[tau_star] = swap_cps + 1
        
        # Updating costs and prepping for next loop
        new_r_len = 0
        prune_cost = f[tau_star] + penalty
        for j in range(r_len):
            if _costs[j] <= prune_cost:
                r[new_r_len] = r[j]
                new_r_len += 1

        r[new_r_len] = tau_star
        r_len = new_r_len + 1
        
    cps[tau_star, swap_cps + 1] = n
    return cps[tau_star, 1: swap_cps + 2]

class PeltSeg(BaseSeg):
        
    def fit(self, X):
        self.cost.fit(X)
        if not isinstance(self.penalty, float):
            self._pen = self.penalty(X.shape[0], self.cost.n_params)
        else:
            self._pen = self.penalty
        self.cps = pelt_segment(self.cost, self.max_cps, self._pen, self.mbic)
        return self
    
    def predict(self):
        return self.cps