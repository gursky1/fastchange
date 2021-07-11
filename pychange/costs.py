# Importing packages
import numpy as np
from numba.pycc import CC
from ruptures.base import BaseCost

from .preprocess import create_partial_sums, create_summary_stats

cc = CC('numba_costs')

@cc.export('normal_mean_cost', 'f8[:](f8[:, :])')
def normal_mean_cost(x):
    return x[:, 1] - (x[:, 0] * x[:, 0]) / x[:, 3]

@cc.export('normal_var_cost', 'f8[:](f8[:, :])')
def normal_var_cost(x):
    return x[:, 3] * (np.log(2 * np.pi) + np.log(np.fmax(x[:, 2], 1e-8) / x[:, 3]) + 1)

@cc.export('normal_mean_var_cost', 'f8[:](f8[:, :])')
def normal_mean_var_cost(x):
    return x[:, 3] * (np.log(2 * np.pi) + np.log(np.fmax((x[:, 1] - ((x[:, 0] * x[:, 0]) / x[:, 3]))/ x[:, 3], 1e-8) + 1))

@cc.export('poisson_mean_var_cost', 'f8[:](f8[:, :])')
def poisson_mean_var_cost(x):
    return 2 * x[:, 0] * (np.log(x[:, 3]) - np.log(x[:, 0]))

@cc.export('scalar_normal_mean_var_cost', 'f8(f8[:])')
def scalar_normal_mean_var_cost(x):
    return x[3] * (np.log(2 * np.pi) + np.log(np.fmax((x[1] - ((x[0] * x[0]) / x[3]))/ x[3], 1e-8) + 1))

@cc.export('nonparametric_cost', 'f8(i4[:, :], i4, i4, i4, i4)')
def nonparametric_cost(x, start, end, k, n):

    _cost = 0
    for i in np.arange(k):

        actual_sum = x[i, end] - x[i, start]
        if actual_sum not in [0, 2 * (end - start)]:
            f = (actual_sum * 0.5) / (end - start)
            _cost += (end - start) * (f * np.log(f) + (1 - f) * np.log(1 - f))
    c = -np.log(2 * n - 1)
    return 2.0 * (c / k) * _cost


class ParametricCost(BaseCost):

    model = ""
    min_size = 2

    def __init__(self, cost_fn):
        self.cost_fn = cost_fn

    def fit(self, signal):
        self.signal = signal
        self.stats = create_summary_stats(signal)

    def error(self, start, end):
        return self.cost_fn(self.stats[end, :] - self.stats[start, :])

class NonParametricCost(BaseCost):

    model = ""
    min_size = 2

    def __init__(self, cost_fn, k):
        self.cost_fn = cost_fn
        self.k = k

    def fit(self, signal):
        self.signal = signal
        self.stats = create_partial_sums(signal, self.k)

    def error(self, start, end):
        return self.cost_fn(self.stats, start, end, self.k, self.signal.shape[0])
