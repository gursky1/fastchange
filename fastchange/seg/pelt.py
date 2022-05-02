#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
from typing import Callable

import numpy as np
import numba as nb

from .base import BaseSeg, seg_sig


@nb.njit(seg_sig(), cache=True, fastmath=True)
def pelt_seg(cost: Callable[[int, int, np.ndarray, np.ndarray], float], sumstats: np.ndarray, cost_args: np.ndarray, penalty: Callable[[int, int], float], min_len: int, max_cps: int, n: int) -> np.ndarray:
    """Pruned exact linear time segmentation algorithm
    
    Exact method for detecting multiple change points in a signal in linear time.
    
    R. Killick, P. Fearnhead, and I. A. Eckley. “Optimal Detection of Changepoints With a Linear Computational Cost”. In: Journal of the American Statistical Association 107.500 (2012), pp. 1590-1598. doi: 10.1080/01621459.2012.737745. eprint: https://doi.org/ 10.1080/01621459.2012.737745. url: https://doi.org/10.1080/01621459.2012.737745

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


class PeltSeg(BaseSeg):
    """Pruned exact linear time segmentation algorithm"""

    seg_fn = staticmethod(pelt_seg)
