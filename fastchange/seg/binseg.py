#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
from typing import Callable

import numpy as np
import numba as nb

from .base import BaseSeg, seg_sig


@nb.njit(seg_sig(), fastmath=True)
def binary_seg(cost: Callable[[int, int, np.ndarray, np.ndarray], float], sumstats: np.ndarray, cost_args: np.ndarray, penalty: Callable[[int, int], float], min_len: int, max_cps: int, n: int) -> np.ndarray:
    """Binary segmentation algorithm
    
    Recursively identifies change points in a signal and on each resulting subsignal, until the no more change points are discovered or we find the max number of change points. Note this method is not exact, as change point indices may not be globally optimal, but is faster than exact methods like PELT.
    
    A. J. Scott and M. Knott. “A Cluster Analysis Method for Grouping Means in the Analysis of Variance”. In: Biometrics 30.3 (1974), pp. 507-512. issn: 0006341X, 15410420. url: http://www.jstor.org/stable/2529204 (visited on 04/23/2022)

    Args:
        cost (Callable[[int, int, np.ndarray, np.ndarray], float]): Cost function for segments
        sumstats (np.ndarray): Summary statistics of signal according to cost function
        cost_args (np.ndarray): Arguments to pass to cost function
        penalty (Callable[[int, int], float]): Complexity penalty
        min_len (int): Minimum segment length
        max_cps (int): Maximum number of changepoints to return
        n (int): Number of points in signal

    Returns:
        np.ndarray: Indices of change points
    """
    
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


class BinSeg(BaseSeg):
    """Binary segmentation method"""
    
    def __init__(self, *args, max_cps: int=10, **kwargs):
        super().__init__(*args, max_cps=max_cps, **kwargs)
    
    seg_fn = staticmethod(binary_seg)
