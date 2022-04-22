# Importing packages
import math

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig


@nb.njit(cost_sig, fastmath=True)
def gamma_cost(s, e, y, cost_args):
    d1 = y[e, 0] - y[s, 0]
    n = e - s
    cost = 2.0 * n * cost_args[0] * (math.log(d1) - math.log(n * cost_args[0]))
    return cost


@nb.njit(cost_sig, fastmath=True)
def gamma_cost_mbic(s, e, y, cost_args):
    cost = gamma_cost(s, e, y, cost_args)
    return cost + math.log(e - s)


class GammaMeanVarCost(BaseCost):
    
    n_params = 1
    
    def __init__(self, *args, shape=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_args = np.float64([shape])
    
    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y: np.ndarray, args):
        sumstats = np.empty((y.shape[0] + 1, 1), np.float64)
        sumstats[:, 0] = np.append(0.0, y.cumsum())

        return sumstats, args
    
    cost_fn = staticmethod(gamma_cost)
    cost_fn_mbic = staticmethod(gamma_cost_mbic)
