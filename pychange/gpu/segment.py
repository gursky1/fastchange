# Importing packages
import cupy as cp

from .preprocess import create_summary_stats

def binary_segmentation(x, min_len, max_cp, penalty, preprocess_fn, cost_fn):
    """Runs binary segmentation on time series"""

    # Setting up summary statistics and objects
    n = x.shape[0]
    sum_stats = preprocess_fn(x)
    is_candidate = cp.arange(min_len, n - min_len)
    cps = cp.zeros(shape=(n,))
    costs = cp.full(shape=n, fill_value=0.0)
    cps[-1] = 1
    cps[0] = 1
    costs[-1] = cost_fn(sum_stats[-1:, :])[0]

    # Iterating through changepoints until convergence
    while True:

        # Single Loop Iteration
        _cps = cp.flatnonzero(cps)
        best_cand, best_cost, best_next_cost, best_next = 0, 0, 0, 0
        best_total_cost = costs.sum()

        # Looping over candidates
        for c1, c2 in cp.stack((_cps[:-1], _cps[1:]), axis=-1):
            _cands = is_candidate[(is_candidate > c1) & (is_candidate < c2)]
            _costs = cp.empty(shape=(_cands.shape[0], 3), dtype=cp.float64)
            _other_costs = costs[: (c1 + 1)].sum() + costs[(c2 + 1):].sum()
            _costs[:, 0] = cost_fn(sum_stats[_cands, :] - sum_stats[c1, :])
            _costs[:, 1] = cost_fn(sum_stats[c2, :] - sum_stats[_cands, :])
            _costs[:, 2] = _costs[:, 0] + _costs[:, 1] + _other_costs + penalty
            _best_cand = cp.argmin(_costs[:, 2])
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
            if cp.flatnonzero(cps).shape[0] > max_cp + 2:
                break
        
    return cp.flatnonzero(cps)[1:-1]


def pelt(x, min_len, penalty, preprocess_fn, cost_fn):
    """Pruned exact linear time changepoint segmentation"""
    
    # Setting up summary statistics and objects
    n = x.shape[0]
    sum_stats = preprocess_fn(x)
    
    # Initializing pelt parameters
    f = cp.empty(shape=(n,), dtype=cp.float64)
    f[0] = -penalty
    costs = cp.empty(shape=(n,), dtype=cp.float64)
    #cp = np.array([np.array([], dtype=np.int32)], dtype=object)
    cps = cp.zeros(shape=(n, 1), dtype=cp.bool8)
    r = cp.array([0])
    
    
    # Entering main loop
    for tau_star in cp.arange(1, n):
        
        # Calculating minimum segment cost
        costs[r] = cost_fn(sum_stats[tau_star] - sum_stats[r])
        _costs = costs[r] + f[r] + penalty
        
        f[tau_star] = _costs.min()
        tau_l = r[cp.argmin(_costs)]
        
        # Setting new changepoints
        _cps = cps[:, cp.array([tau_l])].copy()
        _cps[tau_l, :] = True
        cps = cp.concatenate((cps, _cps), axis=1)
        #cp = np.append(cp, np.append(cp[tau_l], np.array([tau_l], dtype=np.int32)))
        
        # Setting new candidate points
        r = r[(f[r] + costs[r] + penalty) <= f[tau_star]]
        cps = cps[:, r]
        r = cp.append(r, tau_star)
        
    return cps[:, -1]
    