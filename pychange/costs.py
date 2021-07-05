# Importing packages
import numpy as np
from numba.pycc import CC

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
