# Importing packages
import math

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig



@nb.njit(['f8[:, :](f8[:], i8, i8)'], fastmath=True, nogil=True)
def make_partial_sums(x, n, k):
    
    # Initializing array to hold partial sum values
    partial_sums = np.zeros(shape=(n + 1, k), dtype=np.float64)
    x_sorted = np.sort(x)
    
    # Calculating constants
    yK = -1 + (2 * ((np.arange(k)) / k) - (1 / k))
    c = -1.0 * np.log(2 * n - 1)
    pK = 1 / (1 + np.exp(c * yK))
    
    # Iterating over quantiles
    for i in range(k):
        j = int((n - 1) * pK[i] + 1)
        partial_sums[1:, i] = np.cumsum(x < x_sorted[j]) + 0.5 * np.cumsum(x == x_sorted[j])

    return partial_sums


class EmpiricalCost(BaseCost):
    
    n_params = 1
    
    def __init__(self, *args, k: int=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = np.float64([k])
    
    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y: np.ndarray, args):
        sumstats = np.empty((y.shape[0] + 1, 1), np.float64)
        sumstats[:, 0] = np.append(0.0, y.cumsum())

        n = y.shape[0]
        c = 2.0 * (-np.log(2 * n - 1)) / args[0]
        sumstats = make_partial_sums(y, n, args[0])
        return sumstats, np.float64([args[0], c])


    @staticmethod
    @nb.njit(cost_sig, fastmath=True, nogil=True)
    def cost_fn(start, end, sumstats, cost_args):
        cost = 0.0
        for j in nb.prange(cost_args[0]):
            a_sum = sumstats[end, j] - sumstats[start, j]
            if a_sum != 0.0:
                diff = end - start
                a_half = 0.5 * a_sum
                if a_half != diff:
                    f = a_half / diff
                    fi = 1.0 - a_half / diff
                    l = f * math.log(f) + fi * math.log(fi)
                    cost += diff * l
        cost *= cost_args[1]
        return cost
