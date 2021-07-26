# Importing packages
import cupy as cp
from ruptures.base import BaseCost

from .preprocess import create_partial_sums, create_summary_stats

def normal_mean_cost(x):
    return x[:, 1] - (x[:, 0] * x[:, 0]) / x[:, 3]

def normal_var_cost(x):
    return x[:, 3] * (cp.log(2 * cp.pi) + cp.log(cp.fmax(x[:, 2], 1e-8) / x[:, 3]) + 1)

def normal_mean_var_cost(x):
    return x[:, 3] * (cp.log(2 * cp.pi) + cp.log(cp.fmax((x[:, 1] - ((x[:, 0] * x[:, 0]) / x[:, 3]))/ x[:, 3], 1e-8) + 1))

def poisson_mean_var_cost(x):
    return 2 * x[:, 0] * (cp.log(x[:, 3]) - cp.log(x[:, 0]))

def scalar_normal_mean_var_cost(x):
    return x[3] * (cp.log(2 * cp.pi) + cp.log(cp.fmax((x[1] - ((x[0] * x[0]) / x[3]))/ x[3], 1e-8) + 1))

def nonparametric_cost(x, start, end, k, n):

    _cost = 0
    for i in cp.arange(k):

        actual_sum = x[i, end] - x[i, start]
        if actual_sum not in [0, 2 * (end - start)]:
            f = (actual_sum * 0.5) / (end - start)
            _cost += (end - start) * (f * cp.log(f) + (1 - f) * cp.log(1 - f))
    c = -cp.log(2 * n - 1)
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
