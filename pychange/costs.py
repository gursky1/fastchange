# Importing packages
import numpy as np
import numba as nb
from numba import njit
from ruptures.base import BaseCost

from .preprocess import create_partial_sums, create_summary_stats

parametric_sig = ['f8[::1](f8[:, ::1], i4[::1], i4)', 'f4[::1](f4[:, ::1], i4[::1], i4)']

#@cc.export('normal_mean_cost', parametric_sig)
@njit(parametric_sig, fastmath=True)
def normal_mean_cost(x, start, end):
    _x = x[end, :] - x[start, :]
    return _x[:, 1] - ((_x[:, 0] * _x[:, 0]) / _x[:, 3])

#@cc.export('normal_var_cost', parametric_sig)
@njit(parametric_sig, fastmath=True)
def normal_var_cost(x, start, end):
    _x = x[end, :] - x[start, :]
    _pi_const = x.dtype.type(np.log(2 * np.pi))
    return _x[:, 3] * (_pi_const + np.log(np.fmax(_x[:, 2] / _x[:, 3], x.dtype.type(1e-8)) + 1))

#@cc.export('normal_mean_var_cost', parametric_sig)
@njit(parametric_sig, fastmath=True)
def normal_mean_var_cost(x, start, end):
    _x = x[end, :] - x[start, :]
    _pi_const = x.dtype.type(np.log(2 * np.pi))
    return _x[:, 3] * (_pi_const + np.log(np.fmax((_x[:, 1] - ((_x[:, 0] * _x[:, 0]) / _x[:, 3]))/ _x[:, 3], x.dtype.type(1e-8)) + 1))

#@cc.export('poisson_mean_var_cost', parametric_sig)
@njit(parametric_sig, fastmath=True)
def poisson_mean_var_cost(x, start, end):
    _x = x[end, :] - x[start, :]
    return x.dtype.type(2) * _x[:, 0] * (np.log(_x[:, 3]) - np.log(_x[:, 0]))

#@cc.export('scalar_normal_mean_var_cost', 'f8(f8[:, :], i4, i4)')
# @njit(parametric_sig, fastmath=True)
# def scalar_normal_mean_var_cost(x, start, end):
#     _x = x[end, :] - x[start, :]
#     _pi_const = x.dtype.type(np.log(2 * np.pi))
#     return _x[3] * (np.log(2 * np.pi) + np.log(np.fmax((_x[1] - ((_x[0] * _x[0]) / _x[3]))/ _x[3], x.dtype.type(1e-8)) + 1))

#@cc.export('nonparametric_cost', 'f8[:](f8[:, :], i4[:], i4, f8)')
nonparametric_sig = ['f8[::1](f8[:, ::1], i4[::1], i4, f8)', 'f4[::1](f4[:, ::1], i4[::1], i4, f4)']
@njit(nonparametric_sig, fastmath=True)
def nonparametric_cost(x, start, end, c):
    _d = (end - start).astype(x.dtype)
    f = ((x[end, :] - x[start, :]).T * x.dtype.type(0.5)) / _d
    _t = _d * (f * np.log(f) + (1 - f) * np.log(1 - f)).sum(axis=0)
    return c * _t


@nb.experimental.jitclass({'k': nb.int32, 'c': nb.float32, 'stats': nb.float32[:, :]})
class JitNonParametricCost:

    def __init__(self, k):
        self.k = k

    def fit(self, signal):
        self.stats = create_partial_sums(signal, self.k)
        self.c = 2.0 * (-np.log(2 * signal.shape[0] - 1) / self.k)

    def error(self, start, end):
        return nonparametric_cost(self.stats, start, end, self.c)


class ParametricCost(BaseCost):

    model = ""
    min_size = 2

    def __init__(self, cost_fn):
        self.cost_fn = cost_fn

    def fit(self, signal):
        self.signal = signal
        self.stats = create_summary_stats(signal)

    def error(self, start, end):
        return self.cost_fn(self.stats, start, end)

class NonParametricCost(BaseCost):

    model = ""
    min_size = 2

    model = ""
    min_size = 2

    def __init__(self, cost_fn, k):
        self.cost_fn = cost_fn
        self.k = k

    def fit(self, signal):
        self.signal = signal
        self.stats = create_partial_sums(signal, self.k)
        self.c = 2.0 * (-np.log(2 * signal.shape[0] - 1) / self.k)

    def error(self, start, end):
        return self.cost_fn(self.stats, start, end, self.c)