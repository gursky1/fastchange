# Importing packages
import math
import numpy as np
import numba as nb

@nb.experimental.jitclass([
    ('n', nb.int64),
    ('y', nb.float64[:])
])
class L1Cost:

    def __init__(self):
        pass

    def fit(self, x):
        self.y = x
        self.n = x.shape[0]
        return self

    def cost(self, start, end):
        _x = self.y[start: end]
        _med = np.median(_x)
        _diff = np.abs(_x - _med)
        return _diff.sum()


@nb.experimental.jitclass([
    ('n', nb.int64),
    ('y', nb.float64[:])
])
class L2Cost:

    def __init__(self):
        pass

    def fit(self, x):
        self.y = x
        self.n = x.shape[0]
        return self

    def cost(self, start, end):
        _x = self.y[start: end]
        _n = end - start
        return _n * _x.var()


@nb.experimental.jitclass([
    ('n', nb.int64),
    ('y1', nb.float64[:]),
    ('y2', nb.float64[:]),
])
class NormalMeanCost:
    
    def __init__(self):
        pass
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y1 = np.append(0.0, x.cumsum())
        self.y2 = np.append(0.0, (x ** 2).cumsum())
        return self
    
    def cost(self, start, end):
        
        n = end - start
        d1 = self.y1[end] - self.y1[start]
        d2 = self.y2[end] - self.y2[start]
        a1 = (d1 ** 2) / n
        return d2 - a1

@nb.experimental.jitclass([
    ('n', nb.int64),
    ('y1', nb.float64[:]),
    ('y2', nb.float64[:]),
    ('y3', nb.float64[:])
])
class NormalVarCost:
    
    def __init__(self):
        pass
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y1 = np.append(0.0, x.cumsum())
        self.y2 = np.append(0.0, (x ** 2).cumsum())
        self.y3 = np.append(0.0, ((x - x.mean()) ** 2).cumsum())
        return self
    
    def cost(self, start, end):
        
        n = end - start
        d1 = self.y1[end] - self.y1[start]
        d2 = self.y2[end] - self.y2[start]
        d3 = self.y3[end] - self.y3[start]
        a1 = math.log(d3 / n)
        a2 = 2.8378771 + math.log(a1)
        return n * a2

@nb.experimental.jitclass([
    ('n', nb.int64),
    ('y1', nb.float64[:]),
    ('y2', nb.float64[:]),
])
class NormalMeanVarCost:
    
    def __init__(self):
        pass
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y1 = np.append(0.0, x.cumsum())
        self.y2 = np.append(0.0, (x ** 2).cumsum())
        return self
    
    def cost(self, start, end):
        
        n = end - start
        d1 = self.y1[end] - self.y1[start]
        d2 = self.y2[end] - self.y2[start]
        a1 = (d1 ** 2) / n
        a2 = d2 - a1
        a3 = a2 / n
        if a3 <= 0.0:
            a3 = 1e-8
        a4 = 2.8378771 + math.log(a3)
        return n * a4

@nb.experimental.jitclass([
    ('n', nb.int64),
    ('y1', nb.float64[:]),
])
class PoissonMeanVarCost:
    
    def __init__(self):
        pass
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y1 = np.append(0.0, x.cumsum())
        return self
    
    def cost(self, start, end):

        d1 = self.y1[end] - self.y1[start]
        if d1 == 0.0:
            return 0.0
        n = end - start
        a1 = math.log(n) - math.log(d1)
        return 2.0 * d1 * a1


@nb.experimental.jitclass([
    ('n', nb.int64),
    ('y1', nb.float64[:])
])
class ExponentialMeanVarCost:

    def __init__(self):
        pass
    
    def fit(self, x):
        self.n = x.shape[0]
        self.y1 = np.append(0.0, x.cumsum())
        return self
    
    def cost(self, start, end):

        d1 = self.y1[end] - self.y1[start]
        n = end - start
        return 2.0 * n * (math.log(d1) - math.log(n))


@nb.experimental.jitclass([
    ('shape', nb.float64),
    ('n', nb.int64),
    ('y1', nb.float64[:])
])
class GammaMeanVarCost:

    def __init__(self, shape=1.0):
        self.shape = shape

    def fit(self, x):
        self.n = x.shape[0]
        self.y1 = np.append(0.0, x.cumsum())
        return self

    def cost(self, start, end):

        d1 = self.y1[end] - self.y1[start]
        n = end - start
        return 2.0 * n * self.shape * (math.log(d1) - math.log(n * self.shape))


@nb.experimental.jitclass([
    ('k', nb.int64),
    ('n', nb.int64),
    ('y', nb.float64[:, :]),
    ('c', nb.float64)
])
class NonParametricCost:
    
    def __init__(self, k):
        self.k = k
    
    def fit(self, x):
        self.n = x.shape[0]
        partial_sums = np.zeros(shape=(self.n + 1, self.k), dtype=x.dtype)
        sorted_data = np.sort(x)

        for i in np.arange(self.k):

            z = -1 + (2 * i + 1.0) / self.k
            p = 1.0 / (1 + (2 * self.n - 1) ** (-z))
            t = sorted_data[int((self.n - 1) * p)]

            for j in np.arange(1, self.n + 1):

                partial_sums[j, i] = partial_sums[j - 1, i]
                if x[j - 1] < t:
                    partial_sums[j, i] += 2
                if x[j - 1] == t:
                    partial_sums[j, i] += 1
        self.c = 2.0 * (-np.log(2 * self.n - 1)) / self.k
        self.y = partial_sums
        return self
    
    def cost(self, start, end):
        cost = 0.0
        ys = self.y[start, :]
        ye = self.y[end, :]
        for j in range(self.k):
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
        return self.c * cost