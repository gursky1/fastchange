# Importing packages
import math
import numpy as np
import numba as nb


@nb.njit(fastmath=True, nogil=True, parallel=True)
def amoc_segment(cost, min_len, penalty, mbic):

    n = cost.n
    
    costs = np.empty(n - 2 * min_len, dtype=np.float64)
    for i in nb.prange(min_len, n - min_len):
        costs[i - min_len] = cost.cost(0, i) + cost.cost(i, n) + penalty
        if mbic:
            costs[i - min_len] += math.log(i) + math.log(n - i)
    return np.argmin(costs) + min_len

class AmocSeg:
    
    def __init__(self, cost, penalty, min_len, mbic):
        self.cost = cost
        self.penalty = penalty
        self.min_len = min_len
        self.mbic = mbic
        
    def fit(self, X):
        self.cost.fit(X)
        if not isinstance(self.penalty, float):
            self._pen = self.penalty(self.cost.n, self.cost.n_params)
        else:
            self._pen = self.penalty
        self.cps = amoc_segment(self.cost, self.min_len, self._pen, self.mbic)
        return self
    
    def predict(self):
        return np.array([self.cps, self.cost.n])


@nb.njit(fastmath=True, nogil=True, parallel=True)
def binary_segment(cost, min_len, penalty):
    """Runs binary segmentation on time series"""

    # Setting up summary statistics and objects
    n = cost.n
    cps = np.empty(n, dtype=np.int64)
    cps[0] = n - 1
    n_cps = 1
    best_total_cost = cost.cost(0, n - 1)
    part_costs = np.empty(n, dtype=np.float64)
    part_costs[0] = best_total_cost

    # Initializing for out first run
    pre_part_cands = np.arange(min_len, n - min_len - 1)
    post_part_cands = np.empty(0, dtype=np.int64)
    cand_len = n - 2 * min_len
    pre_cps_start = 0
    pre_cps_end = n - 1
    post_cps_start = 0
    post_cps_end = n - 1
    pre_costs = 0.0
    post_costs = 0.0
    _pre_costs = np.empty(n, dtype=np.float64)
    _post_costs = np.empty(n, dtype=np.float64)
    _total_costs = np.full(n, fill_value=np.inf, dtype=np.float64)

    # Iterating through changepoints until convergence
    while True:

        # Checking if there are candidates within the last created partitions
        for i in nb.prange(pre_part_cands.size):

            # Calculating new partition costs
            _pre_cand = pre_part_cands[i]
            _pre_costs[_pre_cand] = cost.cost(pre_cps_start, _pre_cand)
            _post_costs[_pre_cand] = cost.cost(_pre_cand, pre_cps_end)
            _total_costs[_pre_cand] = _pre_costs[_pre_cand] + _post_costs[_pre_cand] + pre_costs

        # Checking if there are candidates within the last created partitions
        for i in nb.prange(post_part_cands.size):

            # Calculating new partition costs
            _post_cand = post_part_cands[i]
            _pre_costs[_post_cand] = cost.cost(post_cps_start, _post_cand)
            _post_costs[_post_cand] = cost.cost(_post_cand, post_cps_end)
            _total_costs[_post_cand] = _pre_costs[_post_cand] + _post_costs[_post_cand] + post_costs
        
        # Checking all candidates to compare against
        best_ind = np.argmin(_total_costs)
        best_cost = _total_costs[best_ind]

        # Checking if the new changepoint is worth it
        if best_cost >= best_total_cost - penalty:
            break

        # New changepoint detected
        cps[n_cps] = best_ind
        old_cost_sum = part_costs[: n_cps].sum()
        n_cps += 1
        sorted_inds = np.argsort(cps[: n_cps])
        new_cp_ind = np.argsort(sorted_inds)[n_cps - 1]
        cps[: n_cps] = cps[sorted_inds]
        part_costs[: n_cps] = part_costs[sorted_inds]
        pre_cp_ind = new_cp_ind - 1
        post_cp_ind = new_cp_ind + 1
        part_costs[new_cp_ind] = _pre_costs[best_ind]
        part_costs[post_cp_ind] = _post_costs[best_ind]

        # Setting values for next run
        if pre_cp_ind == -1:
            pre_cps_start = 0
        else:
            pre_cps_start = cps[pre_cp_ind]
        pre_cps_end = best_ind
        post_cps_start = best_ind
        post_cps_end = cps[post_cp_ind]
        cost_sum = part_costs[: n_cps].sum()
        cost_change = cost_sum - old_cost_sum
        pre_costs = cost_sum - part_costs[new_cp_ind]
        post_costs = cost_sum - part_costs[post_cp_ind]

        # Setting up new candidate arrays
        pre_part_cands = np.arange(pre_cps_start + min_len, pre_cps_end - min_len)
        post_part_cands = np.arange(post_cps_start + min_len, post_cps_end - min_len)

        # Updating total cost change with outer partitions
        _total_costs[: pre_cps_start] += cost_change
        _total_costs[post_cps_end: ] += cost_change
        _total_costs[best_ind - min_len: best_ind + min_len] = np.inf
        best_total_cost = best_cost

    return cps[: n_cps]

class BinSeg:
    
    def __init__(self, cost, penalty, min_len):
        self.cost = cost
        self.penalty = penalty
        self.min_len = min_len
        
    def fit(self, X):
        self.cost.fit(X)
        if not isinstance(self.penalty, float):
            self._pen = self.penalty(self.cost.n, self.cost.n_params)
        else:
            self._pen = self.penalty
        self.cps = binary_segment(self.cost, self.min_len, self._pen)
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

class PeltSeg:
    
    def __init__(self, cost, penalty, max_cps, mbic):
        self.cost = cost
        self.penalty = penalty
        self.max_cps = max_cps
        self.mbic = mbic
        
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