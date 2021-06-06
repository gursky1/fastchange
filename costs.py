# Importing packages
import numpy as np
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
