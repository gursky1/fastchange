# Importing packages
import numpy as np

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

def binary_segmentation(x, min_len, max_cp, penalty, cost_fn):
    """Runs binary segmentation on time series"""

    # Setting up summary statistics and objects
    n = x.shape[0]
    sum_stats = create_summary_stats(x)
    is_candidate = np.ones((n, ), dtype=np.bool)
    is_candidate = np.arange(min_len, n - min_len)
    cps = np.zeros((n, ), dtype=bool)
    costs = np.zeros((n, ), dtype=np.float64)
    cps[-1] = True
    cps[0] = True
    costs[-1] = cost_fn(sum_stats[n, :, n])

    # Iterating through changepoints until convergence
    while True:

        _cps = np.flatnonzero(cps)

        best_cand, best_cost, best_next_cost, best_next = 0, 0, 0
        best_total_cost = costs.sum()

        for c1, c2 in zip(cps[:-1], cps[1:]):
            _cands = is_candidate[np.in1d(is_candidate, [c1, c2])]
            _costs = np.empty((_cands.shape[0], 3), dtype=np.float64)
            _other_costs = costs[:c1].sum() + costs[(c2 + 1):].sum()
            _costs[:, 0] = np.array([cost_fn(sum_stats[i, :] - sum_stats[c1, :], i - c1) for i in _cands])
            _costs[:, 1] = np.array([cost_fn(sum_stats[c2, :] - sum_stats[i, :], c2 - i) for i in _cands])
            _costs[:, 2] = _costs[:, 0] + _costs[:, 0] + _other_costs
            _best_cand = np.argmin(_costs[:, 2])
            if costs[_best_cand, 2] < best_total_cost:
                best_cand = _cands[_best_cand]
                best_cost = _costs[_best_cand, 0]
                best_next_cost = _costs[_best_cand, 1]
                best_next = c2

        if best_cand == 0:
            break
        else:
            cps[best_cand] = True
            costs[best_cand] = best_cost
            costs[best_next] = best_next_cost
            is_candidate[(best_cand - min_len): (best_cand + min_len)] = False
        
        return np.flatnonzero(cps)



