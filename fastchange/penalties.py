# Importing packages
import math


def bic_penalty(n, n_params):
    return (n_params + 1) * math.log(n)


def mbic_penalty(n, n_params):
    return (n_params + 2) * math.log(n)


def aic_penalty(n, n_params):
    return 2 * (n_params + 1)


def hq_penalty(n, n_params):
    return 2 * (n_params + 1) * math.log(math.log(n))
