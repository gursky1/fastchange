# Importing packages
import math
import numpy as np
import numba as nb

@nb.njit(fastmath=True, nogil=True)
def amoc_segment(cost, min_len, penalty):
    n = cost.n
    costs = np.empty(n - 2 * min_len, dtype=np.float64)
    for ind, i in enumerate(range(min_len, n - min_len)):
        pre_cost = cost.cost(0, i)
        post_cost = cost.cost(i, n)
        costs[ind] = pre_cost + post_cost + penalty
    return np.argmin(costs) + min_len


@nb.njit(fastmath=True, nogil=True)
def binary_segment(cost, min_len, max_cps, penalty):
    """Runs binary segmentation on time series"""

    # Setting up summary statistics and objects
    n = cost.n
    is_candidate = np.arange(min_len, n - min_len)
    cps = np.zeros(shape=(n,))
    costs = np.full(shape=n, fill_value=0.0)
    cps[-1] = 1
    cps[0] = 1
    costs[-1] = cost.cost(0, n)

    # Iterating through changepoints until convergence
    while True:

        # Single Loop Iteration
        _cps = np.flatnonzero(cps)
        best_cand, best_cost, best_next_cost, best_next = 0, 0, 0, 0
        best_total_cost = costs.sum()

        # Looping over candidates
        for c1, c2 in np.stack((_cps[:-1], _cps[1:]), axis=-1):
            _cands = is_candidate[(is_candidate > c1) & (is_candidate < c2)]
            if _cands.shape[0] == 0:
                continue
            _costs = np.empty(shape=(_cands.shape[0], 3), dtype=np.float64)
            _other_costs = costs[: (c1 + 1)].sum() + costs[(c2 + 1):].sum()
            _costs[:, 0] = [cost.cost(c1, i) for i in _cands]
            _costs[:, 1] = [cost.cost(i, c2) for i in _cands]
            _costs[:, 2] = _costs[:, 0] + _costs[:, 1] + _other_costs + penalty
            _best_cand = np.argmin(_costs[:, 2])
            if _costs[_best_cand, 2] < best_total_cost:
                best_cand = _cands[_best_cand]
                best_cost = _costs[_best_cand, 0]
                best_next_cost = _costs[_best_cand, 1]
                best_total_cost = _costs[_best_cand, 2]
                best_next = c2

        if best_cand == 0:
            break
        else:
            cps[best_cand] = True
            costs[best_cand] = best_cost
            costs[best_next] = best_next_cost
            is_candidate[(best_cand - min_len): (best_cand + min_len)] = False
            if np.flatnonzero(cps).shape[0] > max_cps + 2:
                break
        
    return np.flatnonzero(cps)[1:-1]

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
    tau_star_range = range(min_cand_len, n_cands)
    for tau_ind, tau_star in enumerate(tau_star_range):
        
        # Initializing for cost calc
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
    