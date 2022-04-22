# Importing packages
import math

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig


@nb.njit(cost_sig, fastmath=True)
def empirical_cost(s, e, y, cost_args):
    cost = 0.0
    k = int(cost_args[0])
    for j in range(k):
        a_sum = y[e, j] - y[s, j]
        if a_sum != 0.0:
            n = e - s
            a_half = 0.5 * a_sum
            if a_half != n:
                f = a_half / n
                fi = 1.0 - f
                l = f * math.log(f) + fi * math.log(fi)
                cost += n * l
    cost *= cost_args[1]
    return cost


@nb.njit(cost_sig, fastmath=True)
def empirical_cost_mbic(s, e, y, cost_args):

    cost = empirical_cost(s, e, y, cost_args)
    return cost + math.log(e - s)


class EmpiricalCost(BaseCost):
    
    n_params = 1
    
    def __init__(self, *args, k: int=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = np.float64([k])
    
    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y, args):

        # Getting some needed scalars
        n = y.shape[0]
        k = int(args[0])
        yK = -1 + (2 * (np.arange(k) -1) / k)
        c = -1.0 * np.log(2 * n - 1)
        pK = 1 / (1 + np.exp(c * yK))

        # Initializing array to hold partial sum values
        sumstats = np.zeros(shape=(n + 1, k), dtype=np.float64)
        y_sort = np.sort(y)

        # Iterating over quantiles
        for i in range(k):
            j = int((n - 1) * pK[i] + 1)
            sumstats[1:, i] = np.cumsum(y < y_sort[j]) + 0.5 * np.cumsum(y == y_sort[j])
    
        return sumstats, np.float64([k, 2 * c / k])

    cost_fn = staticmethod(empirical_cost)
    cost_fn_mbic = staticmethod(empirical_cost_mbic)
