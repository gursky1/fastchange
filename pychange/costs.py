# Importing packages
import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC('numba_costs')
cc.verbose = True
cc.target_cpu = 'host'

def normal_mean_cost(x, n):
    return x[1] - (x[0] * x[0]) / n

def normal_var_cost(x, n):
    return n * (np.log(2 * np.pi) + np.log(np.fmax(x[2], 1e-8) / n) + 1)

def normal_mean_var_cost(x, n):
    return n * (np.log(2 * np.pi) + np.log(np.fmax((x[1] - ((x[0] * x[0]) / n))/ n, 1e-8) + 1))

def poisson_mean_var_cost(x, n):
    return 2 * x[0] * (np.log(n) - np.log(x[0]))

def mbic_cost(cost_fn, x, n):
    return cost_fn(x, n) + np.log(n)

# Jitting
jit_normal_mean_cost = njit(fastmath=True)(normal_mean_cost)
jit_normal_var_cost = njit(fastmath=True)(normal_var_cost)
jit_normal_mean_var_cost = njit(fastmath=True)(normal_mean_var_cost)
jit_poisson_mean_var_cost = njit(fastmath=True)(poisson_mean_var_cost)

# Exporting
aot_normal_mean_cost = cc.export('aot_normal_mean_cost', 'f8(f8[:], i4)')(normal_mean_cost)
aot_normal_var_cost = cc.export('aot_normal_var_cost', 'f8(f8[:], i4)')(normal_var_cost)
aot_normal_mean_var_cost = cc.export('aot_normal_mean_var_cost', 'f8(f8[:], i4)')(normal_mean_var_cost)
aot_poisson_mean_var_cost = cc.export('aot_poisson_mean_var_cost', 'f8(f8[:], i4)')(poisson_mean_var_cost)
