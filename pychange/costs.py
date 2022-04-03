# Importing packages
import math
import numpy as np
import numba as nb

_core_cost_vars = [('n', nb.int64), ('n_params', nb.int64)]

@nb.experimental.jitclass(_core_cost_vars + [
    ('y', nb.float64[:]),
])
class L1Cost:
    """L1 cost class"""

    def __init__(self):
        self.n_params = 1

    def fit(self, x: np.ndarray):
        """Fit method for L1 cost

        Args:
            x (np.ndarray): Input time series

        Returns:
            L1Cost: self
        """
        self.y = x
        self.n = x.shape[0]
        return self

    def cost(self, start, end):
        return _l1_cost(self.y[start: end])


@nb.njit(['f8(f8[:])'], fastmath=True, nogil=True)
def _l1_cost(y):
    """"""
    return np.abs(y - np.median(y)).sum()


@nb.experimental.jitclass(_core_cost_vars + [
    ('y', nb.float64[:]),
])
class L2Cost:

    def __init__(self):
        self.n_params = 1

    def fit(self, x):
        self.y = x
        self.n = x.shape[0]
        return self
    
    def cost(self, start, end):
        return _l2_cost(self.y[start: end], start, end)
    
@nb.njit(['f8(f8[:], i8, i8)'], fastmath=True, nogil=True)
def _l2_cost(y, start, end):
    return (end - start) * y.var()


@nb.experimental.jitclass(_core_cost_vars + [
    ('y1', nb.float64[:]),
    ('y2', nb.float64[:]),
])
class NormalMeanCost:
    
    def __init__(self):
        self.n_params = 1
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y1 = np.append(0.0, x.cumsum())
        self.y2 = np.append(0.0, (x ** 2).cumsum())
        return self
    
    def cost(self, start, end):
        return _normal_mean_cost(self.y1[start],
                                self.y1[end],
                                self.y2[start],
                                self.y2[end],
                                start,
                                end)


@nb.njit(['f8(f8, f8, f8, f8, i8, i8)'], fastmath=True, nogil=True)
def _normal_mean_cost(y1s, y1e, y2s, y2e, start, end):
    n = end - start
    d1 = y1e - y1s
    d2 = y2e - y2s
    a1 = (d1 ** 2) / n
    cost = d2 - a1
    return cost


@nb.experimental.jitclass(_core_cost_vars + [
    ('y', nb.float64[:])
])
class NormalVarCost:
    
    def __init__(self):
        self.n_params = 1
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y = np.append(0.0, ((x - x.mean()) ** 2).cumsum())
        return self
    
    def cost(self, start, end):
        return _normal_var_cost(self.y[start],
                                self.y[end],
                                start,
                                end)


@nb.njit(['f8(f8, f8, i8, i8)'], fastmath=True, nogil=True)
def _normal_var_cost(ys, ye, start, end):
    n = end - start
    d = ye - ys
    a1 = math.log(d / n)
    a2 = 2.8378771 + math.log(a1)
    cost = n * a2
    return cost


@nb.experimental.jitclass(_core_cost_vars + [
    ('y1', nb.float64[:]),
    ('y2', nb.float64[:]),
])
class NormalMeanVarCost:
    
    def __init__(self):
        self.n_params = 2
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y1 = np.append(0.0, x.cumsum())
        self.y2 = np.append(0.0, (x ** 2).cumsum())
        return self
    
    def cost(self, start, end):
        return _normal_mean_var_cost(self.y1[start],
                                     self.y1[end],
                                     self.y2[start],
                                     self.y2[end],
                                     start,
                                     end)

    
@nb.njit(['f8(f8, f8, f8, f8, i8, i8)'], fastmath=True, nogil=True)
def _normal_mean_var_cost(y1s, y1e, y2s, y2e, start, end):
    n = end - start
    d1 = y1e - y1s
    d2 = y2e - y2s
    a1 = (d1 ** 2) / n
    a2 = d2 - a1
    a3 = a2 / n
    if a3 <= 0.0:
        a3 = 1e-8
    a4 = 2.8378771 + math.log(a3)
    cost = n * a4
    return cost


@nb.experimental.jitclass(_core_cost_vars + [
    ('y', nb.float64[:]),
])
class PoissonMeanVarCost:
    
    def __init__(self):
        self.n_params = 1
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y = np.append(0.0, x.cumsum())
        return self
    
    def cost(self, start, end):
        return _poisson_mean_var_cost(self.y[start],
                                      self.y[end],
                                      start,
                                      end)
        
@nb.njit(['f8(f8, f8, i8, i8)'], fastmath=True, nogil=True)
def _poisson_mean_var_cost(ys, ye, start, end):

    d1 = ye - ys
    if d1 == 0.0:
        return 0.0
    n = end - start
    a1 = math.log(n) - math.log(d1)
    cost = 2.0 * d1 * a1
    return cost


@nb.experimental.jitclass(_core_cost_vars + [
    ('y', nb.float64[:])
])
class ExponentialMeanVarCost:

    def __init__(self):
        self.n_params = 1
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y = np.append(0.0, x.cumsum())
        return self
    
    def cost(self, start, end):
        return _exponential_mean_var_cost(self.y[start], self.y[end], start, end)
    

@nb.njit(['f8(f8, f8, i8, i8)'], fastmath=True, nogil=True)
def _exponential_mean_var_cost(ys, ye, start, end):
    d1 = ye - ys
    n = end - start
    cost = 2.0 * n * (math.log(d1) - math.log(n))
    return cost


@nb.experimental.jitclass(_core_cost_vars + [
    ('shape', nb.float64),
    ('y', nb.float64[:])
])
class GammaMeanVarCost:

    def __init__(self, shape=1.0):
        self.shape = shape
        self.n_params = 1

    def fit(self, x):
        self.n = x.shape[0]
        self.y = np.append(0.0, x.cumsum())
        return self

    def cost(self, start, end):
        return _gamma_mean_var_cost(self.y[start], self.y[end], start, end, self.shape)


@nb.njit(['f8(f8, f8, i8, i8, f8)'], fastmath=True, nogil=True)
def _gamma_mean_var_cost(ys, ye, start, end, shape):
    d1 = ye - ys
    n = end - start
    cost = 2.0 * n * shape * (math.log(d1) - math.log(n * shape))
    return cost


@nb.experimental.jitclass(_core_cost_vars + [
    ('k', nb.int64),
    ('y', nb.float64[:, :]),
    ('c', nb.float64)
])
class EmpiricalCost:
    
    def __init__(self, k=10):
        self.k = k
        self.n_params = 1
    
    def fit(self, x):
        self.n = x.shape[0]
        self.c = 2.0 * (-np.log(2 * self.n - 1)) / self.k
        self.y = _make_partial_sums(x, self.n, self.k)
        return self
    
    def cost(self, start, end):
        return _empirical_cost(self.y[start, :], self.y[end, :], start, end, self.k, self.c)


@nb.njit(['f8[:, :](f8[:], i8, i8)'], fastmath=True, nogil=True)
def _make_partial_sums(x, n, k):
    
    # Initializing array to hold partial sum values
    partial_sums = np.zeros(shape=(n + 1, k), dtype=np.float64)
    x_sorted = np.sort(x)
    
    # Iterating over quantiles
    for i in nb.prange(k):
        
        # Calculating constants
        z = -1 + (2 * i + 1.0) / k
        p = 1.0 / (1 + (2 * n - 1) ** (-z))
        t = x_sorted[int((n - 1) * p)]
        
        # Iterating over sorted observations
        for j in range(1, n + 1):

            partial_sums[j, i] = partial_sums[j - 1, i]
            if x[j - 1] < t:
                partial_sums[j, i] += 2
            if x[j - 1] == t:
                partial_sums[j, i] += 1

    return partial_sums


@nb.njit(['f8(f8[:], f8[:], i8, i8, i8, f8)'], fastmath=True, nogil=True)
def _empirical_cost(ys, ye, start, end, k, c):
    cost = 0.0
    for j in nb.prange(k):
        a_sum = ye[j] - ys[j]
        if a_sum != 0.0:
            diff = end - start
            a_half = 0.5 * a_sum
            if a_half != diff:
                f = a_half / diff
                fi = 1.0 - f
                flog = math.log(f)
                filog = math.log(fi)
                t = f * flog
                ti = fi * filog
                l = t + ti
                ld = diff * l
                cost += ld
    cost *= c
    return cost
