# Importing packages
import math

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig


@nb.njit(cost_sig, fastmath=True)
def normal_mean_cost(s, e, y, cost_args):
    n = e - s
    d1 = y[e, 0] - y[s, 0]
    d2 = y[e, 1] - y[s, 1]
    return d2 - math.pow(d1, 2) / n


@nb.njit(cost_sig, fastmath=True)
def normal_mean_cost_mbic(s, e, y, cost_args):
    cost = normal_mean_cost(s, e, y, cost_args)
    return cost + math.log(e - s)


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

    cost_fn = staticmethod(normal_mean_cost)
    cost_fn_mbic = staticmethod(normal_mean_cost_mbic)



@nb.njit(cost_sig, fastmath=True, nogil=True)
def normal_var_cost(s, e, y, cost_args):        
    n = e - s
    d = y[e, 0] - y[s, 0]
    a1 = 2.837877066 + math.log(d / n)
    cost = n * a1
    return cost


@nb.njit(cost_sig, fastmath=True, nogil=True)
def normal_var_cost_mbic(s, e, y, cost_args):

    cost = normal_var_cost(s, e, y, cost_args) 
    return cost + math.log(e - s)


class NormalVarCost(BaseCost):
    
    n_params = 1

    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y: np.ndarray, args):
        sumstats = np.empty((y.shape[0] + 1, 1), np.float64)
        sumstats[:, 0] = np.append(0.0, ((y - y.mean()) ** 2).cumsum())
        return sumstats, np.float64([0.0])

    cost_fn = staticmethod(normal_var_cost)
    cost_fn_mbic = staticmethod(normal_var_cost_mbic)




@nb.njit(cost_sig, fastmath=True, nogil=True)
def normal_meanvar_cost(s, e, y, cost_args):
    
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


@nb.njit(cost_sig, fastmath=True, nogil=True)
def normal_meanvar_cost_mbic(s, e, y, cost_args):
    
    cost = normal_meanvar_cost(s, e, y, cost_args)
    return cost + math.log(e - s)


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
    
    cost_fn = staticmethod(normal_meanvar_cost)
    cost_fn_mbic = staticmethod(normal_meanvar_cost_mbic)
