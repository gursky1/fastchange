#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
from typing import Callable

import numpy as np
import numba as nb

from .base import BaseSeg, seg_sig


@nb.njit(seg_sig(), cache=True, fastmath=True, nogil=True, parallel=True)
def amoc_seg(cost: Callable[[int, int, np.ndarray, np.ndarray], float], sumstats: np.ndarray, cost_args: np.ndarray, penalty: Callable[[int, int], float], min_len: int, max_cps: int, n: int) -> np.ndarray:
    """At-most One Changepoint segmentation algorithm
    
    Detects at most one changepoint in a given signal, performing an exhaustive search over all points in a signal, and returning the index (if one exists) that minimizes the cost and penalty functions.

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
    def _cost_fn(start, end):
        return cost(start, end, sumstats, cost_args)
    
    # Getting cost with no changepoints
    null_cost = cost(0, n, sumstats, cost_args)
    
    # Finding possible changepoint locations
    costs = np.empty(n - 2 * min_len, dtype=np.float64)
    for i in nb.prange(min_len, n - min_len):
        costs[i - min_len] = _cost_fn(0, i, sumstats, cost_args) + _cost_fn(i, n, sumstats, cost_args) + penalty
    
    # Determining if best changepoint better than null
    best_ind = np.argmin(costs)
    if costs[best_ind] < null_cost:
        cp = np.int64([best_ind + min_len])
    else:
        cp = np.int64([-1])
        
    return cp


class AmocSeg(BaseSeg):
    """At-most one change point segmentation method"""

    seg_fn = staticmethod(amoc_seg)
