# Importing packages
import math
import numpy as np
import numba as nb

@nb.njit(fastmath=True, nogil=True)
def amoc_segment(cost, min_len, penalty):
    n = cost.n
    costs = np.empty(n - 2 * min_len, dtype=np.float64)
    for i in range(min_len, n - min_len):
        pre_cost = cost.cost(0, i)
        post_cost = cost.cost(i, n)
        costs[i - min_len] = pre_cost + post_cost + penalty
    return np.argmin(costs) + min_len


@nb.njit(fastmath=True, nogil=True, parallel=True)
def amoc_segment_parallel(cost, min_len, penalty):
    n = cost.n
    costs = np.empty(n - 2 * min_len, dtype=np.float64)
    for i in nb.prange(min_len, n - min_len):
        pre_cost = cost.cost(0, i)
        post_cost = cost.cost(i, n)
        costs[i - min_len] = pre_cost + post_cost + penalty
    return np.argmin(costs) + min_len


@nb.njit(fastmath=True, nogil=True)
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
    pre_part_cands = np.arange(min_len - 1, n - min_len - 1)
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
        for i in range(pre_part_cands.size):

            # Calculating new partition costs
            _pre_cand = pre_part_cands[i]
            _pre_costs[_pre_cand] = cost.cost(pre_cps_start, _pre_cand)
            _post_costs[_pre_cand] = cost.cost(_pre_cand, pre_cps_end)
            _total_costs[_pre_cand] = _pre_costs[_pre_cand] + _post_costs[_pre_cand] + pre_costs

        # Checking if there are candidates within the last created partitions
        for i in range(post_part_cands.size):

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


@nb.njit(fastmath=True, nogil=True, parallel=True)
def binary_segment_parallel(cost, min_len, penalty):
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
    pre_part_cands = np.arange(min_len - 1, n - min_len - 1)
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


@nb.njit(fastmath=True, nogil=True)
def pelt_segment(cost, min_len, max_cps, penalty, jump):
    """Pruned exact linear time changepoint segmentation"""
    
    # Initializing parameters for segmentation
    n = cost.n
    cands = np.arange(0, n + jump, jump)
    cands[-1] = n
    n_cands = len(cands)
    min_cand_len = math.ceil(min_len / jump)
    
    # Initializing cost matrix
    costs = np.full(n_cands, fill_value=np.inf, dtype=np.float64)
    costs[0] = 0.0
    
    # Initializing partitions for cp locations
    n_cps = n_cands * (max_cps - min_cand_len)
    cps = np.empty(n_cps, dtype=np.int64)
    cps_starts = np.zeros(n_cands + 1, dtype=np.int64)
    r = np.array([0], dtype=np.int64)
    
    # Starting loop for search
    tau_star_range = np.arange(min_cand_len, n_cands)
    for tau_ind in range(tau_star_range.size):
        
        # Initializing for cost calc
        tau_star = tau_star_range[tau_ind]
        r_len = len(r)
        f = np.empty(r_len, dtype=np.float64)
        
        # Calculating each candidate cost
        for j in range(r_len):
            tau = r[j]
            f[j] = cost.cost(cands[tau], cands[tau_star]) + costs[tau] + penalty

        # Finding best candidate
        best_tau = np.argmin(f)
        best_cost = f[best_tau]
        best_r = r[best_tau]
        
        # Checking for zero condition
        if best_r == 0:
            r_part = np.empty(0, dtype=np.int64)
        else:
            r_start = cps_starts[best_r]
            r_end = cps_starts[best_r + 1]
            r_part = cps[r_start: r_end]
        
        # Checking if we are done segmenting
        if tau_star == n_cands - 1:
            return cands[r_part]
        
        # Updating changepoint partitions
        _part_len = len(r_part) + 1
        _part = np.empty(_part_len, dtype=np.int64)
        _part[: -1] = r_part
        _part[-1] = tau_star
        _part_start = cps_starts[tau_star]
        _part_end = _part_start + _part_len
        
        # Checking if we need to expand the max changepoint limit
        # TODO try replacing this with a typed list in numba
        while _part_end > n_cps:
            _cps = cps
            cps = np.empty(n_cps * 2, dtype=np.int64)
            cps[: n_cps] = _cps
            n_cps *= 2
            
        cps[_part_start: _part_end] = _part
        cps_starts[tau_star + 1] = _part_end
        
        costs[tau_star] = best_cost
        
        _r = r[f <= best_cost + penalty]
        r = np.empty(len(_r) + 1, dtype=np.int64)
        r[: -1] = _r
        r[-1] = tau_ind + 1

@nb.njit(fastmath=True, nogil=True)
def pelt_segment_parallel(cost, min_len, max_cps, penalty, jump):
    """Pruned exact linear time changepoint segmentation"""
    
    # Initializing parameters for segmentation
    n = cost.n
    cands = np.arange(0, n + jump, jump)
    cands[-1] = n
    n_cands = len(cands)
    min_cand_len = math.ceil(min_len / jump)
    
    # Initializing cost matrix
    costs = np.full(n_cands, fill_value=np.inf, dtype=np.float64)
    costs[0] = 0.0
    
    # Initializing partitions for cp locations
    n_cps = n_cands * (max_cps - min_cand_len)
    cps = np.empty(n_cps, dtype=np.int64)
    cps_starts = np.zeros(n_cands + 1, dtype=np.int64)
    r = np.array([0], dtype=np.int64)
    
    # Starting loop for search
    tau_star_range = np.arange(min_cand_len, n_cands)
    for tau_ind in range(tau_star_range.size):
        
        # Initializing for cost calc
        tau_star = tau_star_range[tau_ind]
        r_len = len(r)
        f = np.empty(r_len, dtype=np.float64)
        
        # Calculating each candidate cost
        for j in nb.prange(r_len):
            tau = r[j]
            f[j] = cost.cost(cands[tau], cands[tau_star]) + costs[tau] + penalty

        # Finding best candidate
        best_tau = np.argmin(f)
        best_cost = f[best_tau]
        best_r = r[best_tau]
        
        # Checking for zero condition
        if best_r == 0:
            r_part = np.empty(0, dtype=np.int64)
        else:
            r_start = cps_starts[best_r]
            r_end = cps_starts[best_r + 1]
            r_part = cps[r_start: r_end]
        
        # Checking if we are done segmenting
        if tau_star == n_cands - 1:
            return cands[r_part]
        
        # Updating changepoint partitions
        _part_len = len(r_part) + 1
        _part = np.empty(_part_len, dtype=np.int64)
        _part[: -1] = r_part
        _part[-1] = tau_star
        _part_start = cps_starts[tau_star]
        _part_end = _part_start + _part_len
        
        # Checking if we need to expand the max changepoint limit
        # TODO try replacing this with a typed list in numba
        while _part_end > n_cps:
            _cps = cps
            cps = np.empty(n_cps * 2, dtype=np.int64)
            cps[: n_cps] = _cps
            n_cps *= 2
            
        cps[_part_start: _part_end] = _part
        cps_starts[tau_star + 1] = _part_end
        
        costs[tau_star] = best_cost
        
        _r = r[f <= best_cost + penalty]
        new_r = np.empty(len(_r) + 1, dtype=np.int64)
        new_r[: -1] = _r
        new_r[-1] = tau_ind + 1
        r = new_r