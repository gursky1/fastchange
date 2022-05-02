#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
import math
from typing import Tuple

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig



@nb.njit(cost_sig, fastmath=True, nogil=True)
def poisson_cost(s: int, e: int, y: np.ndarray, cost_args: np.ndarray) -> float:
    """Poisson distribution cost of a proposed segment

    Jie Chen and Arjun K. Gupta. “Univariate Normal Model”. In: Parametric Statistical Change Point Analysis: With Applications to Genetics, Medicine, and Finance. Boston: Birkh ̈auser Boston, 2012, pp. 7-88. isbn: 978-0-8176-4801-5. doi: 10 . 1007 / 978 - 0 - 8176 - 4801-5_2. url: https://doi.org/10.1007/978-0-8176-4801-5_2

    Args:
        s (int): Start index of segment
        e (int): End index of segment
        y (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Arguments of cost function. Unused.

    Returns:
        float: Segment cost
    """
    d1 = y[e, 0] - y[s, 0]
    if d1 == 0.0:
        return 0.0
    n = e - s
    a1 = math.log(n) - math.log(d1)
    cost = 2.0 * d1 * a1
    return cost


@nb.njit(cost_sig, fastmath=True, nogil=True)
def poisson_cost_mbic(s: int, e: int, y: np.ndarray, cost_args: np.ndarray) -> float:
    """Poisson cost of a proposed segment with an MBIC penalty

    Jie Chen and Arjun K. Gupta. “Univariate Normal Model”. In: Parametric Statistical Change Point Analysis: With Applications to Genetics, Medicine, and Finance. Boston: Birkh ̈auser Boston, 2012, pp. 7-88. isbn: 978-0-8176-4801-5. doi: 10 . 1007 / 978 - 0 - 8176 - 4801-5_2. url: https://doi.org/10.1007/978-0-8176-4801-5_2

    Args:
        s (int): Start index of segment
        e (int): End index of segment
        y (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Arguments of cost function. Unused.

    Returns:
        float: Segment cost
    """
    cost = poisson_cost(s, e, y, cost_args)
    return cost + math.log(e - e)

class PoissonMeanVarCost(BaseCost):
    """Change in the mean and variance of a Poisson distribution"""
    
    n_params = 1
    
    @staticmethod
    @nb.njit(preprocess_sig, fastmath=True)
    def preprocess(y: np.ndarray, args: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sumstats = np.empty((y.shape[0] + 1, 1), np.float64)
        sumstats[:, 0] = np.append(0.0, y.cumsum())

        return sumstats, np.float64([0.0])
    
    cost_fn = staticmethod(poisson_cost)
    cost_fn_mbic = staticmethod(poisson_cost_mbic)
