# Importing packages
import numpy as np
from numba.pycc import CC

cc = CC('numba_costs')

@cc.export('normal_mean_cost', 'f8(f8[:], i4)')
def normal_mean_cost(x, n):
    return x[1] - (x[0] * x[0]) / n

@cc.export('normal_var_cost', 'f8(f8[:], i4)')
def normal_var_cost(x, n):
    return n * (np.log(2 * np.pi) + np.log(np.fmax(x[2], 1e-8) / n) + 1)

@cc.export('normal_mean_var_cost', 'f8(f8[:], i4)')
def normal_mean_var_cost(x, n):
    return n * (np.log(2 * np.pi) + np.log(np.fmax((x[1] - ((x[0] * x[0]) / n))/ n, 1e-8) + 1))

@cc.export('poisson_mean_var_cost', 'f8(f8[:], i4)')
def poisson_mean_var_cost(x, n):
    return 2 * x[0] * (np.log(n) - np.log(x[0]))

@cc.export('iter_sumstats', 'f8(f8[:, :], i4, i4)')
def iter_sumstats(sumstats, start, end):

    for i in range(start, end):
        cost = i * (np.log(2 * np.pi) + np.log(np.fmax((sumstats[i, 1] - ((sumstats[i, 0] * sumstats[i, 0]) / i))/ i, 1e-8) + 1))
    return cost

def mbic_cost(cost_fn, x, n):
    return cost_fn(x, n) + np.log(n)
