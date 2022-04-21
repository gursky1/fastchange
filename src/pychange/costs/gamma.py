# Importing packages
import math

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig

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
    
    @staticmethod
    @nb.njit(cost_sig, fastmath=True, nogil=True)
    def cost_fn(start, end, sumstats, cost_args):
        d1 = sumstats[end, 0] - sumstats[start, 0]
        n = end - start
        cost = 2.0 * n * cost_args[0] * (math.log(d1) - math.log(n * cost_args[0]))
        return cost
