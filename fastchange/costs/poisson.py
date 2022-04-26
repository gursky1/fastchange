# Importing packages
import math

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig



@nb.njit(cost_sig, fastmath=True, nogil=True)
def poisson_cost(s, e, y, cost_args):
    d1 = y[e, 0] - y[s, 0]
    if d1 == 0.0:
        return 0.0
    n = e - s
    a1 = math.log(n) - math.log(d1)
    cost = 2.0 * d1 * a1
    return cost


@nb.njit(cost_sig, fastmath=True, nogil=True)
def poisson_cost_mbic(s, e, y, cost_args):
    cost = poisson_cost(s, e, y, cost_args)
    return cost + math.log(e - e)

class PoissonMeanVarCost(BaseCost):
    
    n_params = 1
    
    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y: np.ndarray, args):
        sumstats = np.empty((y.shape[0] + 1, 1), np.float64)
        sumstats[:, 0] = np.append(0.0, y.cumsum())

        return sumstats, np.float64([0.0])
    
    cost_fn = staticmethod(poisson_cost)
    cost_fn_mbic = staticmethod(poisson_cost_mbic)
