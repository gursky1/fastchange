# Importing packages
import numpy as np

def create_summary_stats(x):
    n = x.shape[0]
    sum_stats = np.stack((np.append(0, x.cumsum()),
                          np.append(0, (x ** 2).cumsum()),
                          np.append(0, ((x - x.mean()) ** 2).cumsum()),
                          np.arange(0, n + 1)),
                         axis=-1)
    return sum_stats

def binary_segmentation(x, min_len, max_cp, penalty, cost_fn):
    """Runs binary segmentation on time series"""

    # Setting up summary statistics and objects
    n = x.shape[0]
    sum_stats = np.stack((np.append(0, x.cumsum()),
                          np.append(0, (x ** 2).cumsum()),
                          np.append(0, ((x - x.mean()) ** 2).cumsum()),
                          np.arange(0, n + 1)),
                         axis=-1)
    is_candidate = np.arange(min_len, n - min_len)
    cps = np.zeros(shape=(n,))
    costs = np.full(shape=n, fill_value=0.0)
    cps[-1] = 1
    cps[0] = 1
    costs[-1] = cost_fn(sum_stats[-1:, :])[0]

    # Iterating through changepoints until convergence
    while True:

        # Single Loop Iteration
        _cps = np.flatnonzero(cps)
        best_cand, best_cost, best_next_cost, best_next = 0, 0, 0, 0
        best_total_cost = costs.sum()

        # Looping over candidates
        for c1, c2 in np.stack((_cps[:-1], _cps[1:]), axis=-1):
            _cands = is_candidate[(is_candidate > c1) & (is_candidate < c2)]
            _costs = np.empty(shape=(_cands.shape[0], 3), dtype=np.float64)
            _other_costs = costs[: (c1 + 1)].sum() + costs[(c2 + 1):].sum()
            _costs[:, 0] = cost_fn(sum_stats[_cands, :] - sum_stats[c1, :])
            _costs[:, 1] = cost_fn(sum_stats[c2, :] - sum_stats[_cands, :])
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
            if np.flatnonzero(cps).shape[0] > max_cp + 2:
                break
        
    return np.flatnonzero(cps)[1:-1]


def pelt(x, min_len, penalty, cost_fn):
    """Pruned exact linear time changepoint segmentation"""
    
    # Setting up summary statistics and objects
    n = x.shape[0]
    sum_stats = np.stack((np.append(0, x.cumsum()),
                          np.append(0, (x ** 2).cumsum()),
                          np.append(0, ((x - x.mean()) ** 2).cumsum()),
                          np.arange(0, n + 1)),
                         axis=-1)
    
    # Initializing pelt parameters
    f = np.empty(shape=(n,), dtype=np.float64)
    f[0] = -penalty
    cp = []
    r = np.array([0])
    
    
    # Entering main loop
    for tau_star in np.arange(n):
        
        # Calculating minimum segment cost
        _costs = np.array([f[tau_star] + penalty + cost_fn(sum_stats[(tau + 1): tau_star]) for tau in r[tau_star]])
        f[tau_star] = _costs.min()
        tau_l = np.argmin(_costs)
        
        # Setting new changepoints
        cp[tau_star] = cp[tau_l] + tau_l
        
        # Setting new candidate points
        r = np.array([i for i, j in zip(r[tau_star], _costs) if f[i] + j <= (f[tau_star] - penalty)])
        r = np.append(r, tau_star)
        
    return cp[-1]
    