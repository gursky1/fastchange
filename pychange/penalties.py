# Importing packages
import math
import numpy as np
import numba as nb


@nb.njit(['f8(i8, i8)'], fastmath=True)
def bic0_penalty(n, n_params):
    return n_params * math.log(n)

@nb.njit(['f8(i8, i8)'], fastmath=True)
def bic_penalty(n, n_params):
    return (n_params + 1) * math.log(n)

@nb.njit(['f8(i8, i8)'], fastmath=True)
def mbic_penalty(n, n_params):
    return (n_params + 2) * math.log(n)

@nb.njit(['f8(i8, i8)'], fastmath=True)
def aic0_penalty(n, n_params):
    return 2 * n_params

@nb.njit(['f8(i8, i8)'], fastmath=True)
def aic_penalty(n, n_params):
    return 2 * (n_params + 1)

@nb.njit(['f8(i8, i8)'], fastmath=True)
def hq0_penalty(n, n_params):
    return 2 * n_params * math.log(math.log(n))

@nb.njit(['f8(i8, i8)'], fastmath=True)
def hq_penalty(n, n_params):
    return 2 * (n_params + 1) * math.log(math.log(n))
