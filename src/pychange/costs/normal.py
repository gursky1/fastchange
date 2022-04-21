# Importing packages
import math

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig

class NormalMeanCost(BaseCost):
    
    n_params = 1
    
    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y: np.ndarray, args):
        sumstats = np.empty((y.shape[0] + 1, 2), np.float64)
        sumstats[0, :] = 0.0
        sumstats[1:, 0] = y.cumsum()
        sumstats[1:, 1] = (y ** 2).cumsum()

        return sumstats, np.float64([0.0])

    @staticmethod
    @nb.njit(cost_sig, fastmath=True)
    def cost_fn(s, e, y, cost_args):
        n = e - s
        d1 = y[e, 0] - y[s, 0]
        d2 = y[e, 1] - y[s, 1]
        return d2 - math.pow(d1, 2) / n

class NormalVarCost(BaseCost):
    
    n_params = 1

    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y: np.ndarray, args):
        sumstats = np.empty((y.shape[0] + 1, 1), np.float64)
        sumstats[:, 0] = np.append(0.0, ((y - y.mean()) ** 2).cumsum())
        return sumstats, np.float64([0.0])
    
    @staticmethod
    @nb.njit(cost_sig, fastmath=True, nogil=True)
    def cost_fn(s, e, y, cost_args):        
        n = e - s
        d = y[e, 0] - y[s, 0]
        a1 = 2.837877066 + math.log(d / n)
        cost = n * a1
        return cost


class NormalMeanVarCost(BaseCost):
    
    n_params = 2
    
    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y: np.ndarray, args):
        sumstats = np.empty((y.shape[0] + 1, 2), np.float64)
        sumstats[0, :] = 0.0
        sumstats[1:, 0] = y.cumsum()
        sumstats[1:, 1] = (y ** 2).cumsum()

        return sumstats, np.float64([0.0])
    
    @staticmethod
    @nb.njit(cost_sig, fastmath=True, nogil=True)
    def cost_fn(s, e, y, cost_args):
        
        n = e - s
        d1 = y[e, 0] - y[s, 0]
        d2 = y[e, 1] - y[s, 1]
        a1 = (d1 ** 2.0) / n
        a2 = d2 - a1
        if a2 <= 0.0:
            a2 = 1e-8
        a3 = 2.837877066 + math.log(a2 / n)
        cost = n * a3
        return cost
