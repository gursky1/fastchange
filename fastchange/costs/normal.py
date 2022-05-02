#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
import math
from typing import Tuple

import numpy as np
import numba as nb

from .base import BaseCost, preprocess_sig, cost_sig


@nb.njit(cost_sig, cache=True, fastmath=True)
def normal_mean_cost(s: int, e: int, y: np.ndarray, cost_args: np.ndarray) -> float:
    """Normal mean change cost of a proposed segment

    Jie Chen and Arjun K. Gupta. “Univariate Normal Model”. In: Parametric Statistical Change Point Analysis: With Applications to Genetics, Medicine, and Finance. Boston: Birkh ̈auser Boston, 2012, pp. 7-88. isbn: 978-0-8176-4801-5. doi: 10 . 1007 / 978 - 0 - 8176 - 4801-5_2. url: https://doi.org/10.1007/978-0-8176-4801-5_2

    Args:
        s (int): Start index of segment
        e (int): End index of segment
        y (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Arguments of cost function. Unused.

    Returns:
        float: Segment cost
    """
    n = e - s
    d1 = y[e, 0] - y[s, 0]
    d2 = y[e, 1] - y[s, 1]
    return d2 - math.pow(d1, 2) / n


@nb.njit(cost_sig, cache=True, fastmath=True)
def normal_mean_cost_mbic(s: int, e: int, y: np.ndarray, cost_args: np.ndarray) -> float:
    """Normal mean change cost of a proposed segment with an MBIC penalty

    Jie Chen and Arjun K. Gupta. “Univariate Normal Model”. In: Parametric Statistical Change Point Analysis: With Applications to Genetics, Medicine, and Finance. Boston: Birkh ̈auser Boston, 2012, pp. 7-88. isbn: 978-0-8176-4801-5. doi: 10 . 1007 / 978 - 0 - 8176 - 4801-5_2. url: https://doi.org/10.1007/978-0-8176-4801-5_2


    Args:
        s (int): Start index of segment
        e (int): End index of segment
        y (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Arguments of cost function. Unused.

    Returns:
        float: Segment cost
    """
    cost = normal_mean_cost(s, e, y, cost_args)
    return cost + math.log(e - s)


class NormalMeanCost(BaseCost):
    """Change in the mean of a normal distribution"""
    
    n_params = 1
    
    @staticmethod
    @nb.njit(preprocess_sig, cache=True, fastmath=True)
    def preprocess(y: np.ndarray, args: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sumstats = np.empty((y.shape[0] + 1, 2), np.float64)
        sumstats[0, :] = 0.0
        sumstats[1:, 0] = y.cumsum()
        sumstats[1:, 1] = (y ** 2).cumsum()

        return sumstats, np.float64([0.0])

    cost_fn = staticmethod(normal_mean_cost)
    cost_fn_mbic = staticmethod(normal_mean_cost_mbic)



@nb.njit(cost_sig, cache=True, fastmath=True, nogil=True)
def normal_var_cost(s: int, e: int, y: np.ndarray, cost_args: np.ndarray) -> float:   
    """Normal variance change cost of a proposed segment

    Jie Chen and Arjun K. Gupta. “Univariate Normal Model”. In: Parametric Statistical Change Point Analysis: With Applications to Genetics, Medicine, and Finance. Boston: Birkh ̈auser Boston, 2012, pp. 7-88. isbn: 978-0-8176-4801-5. doi: 10 . 1007 / 978 - 0 - 8176 - 4801-5_2. url: https://doi.org/10.1007/978-0-8176-4801-5_2

    Args:
        s (int): Start index of segment
        e (int): End index of segment
        y (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Arguments of cost function. Unused.

    Returns:
        float: Segment cost
    """     
    n = e - s
    d = y[e, 0] - y[s, 0]
    a1 = 2.837877066 + math.log(d / n)
    cost = n * a1
    return cost


@nb.njit(cost_sig, cache=True, fastmath=True, nogil=True)
def normal_var_cost_mbic(s: int, e: int, y: np.ndarray, cost_args: np.ndarray) -> float:
    """Normal variance change cost of a proposed segment with an MBIC penalty

    Jie Chen and Arjun K. Gupta. “Univariate Normal Model”. In: Parametric Statistical Change Point Analysis: With Applications to Genetics, Medicine, and Finance. Boston: Birkh ̈auser Boston, 2012, pp. 7-88. isbn: 978-0-8176-4801-5. doi: 10 . 1007 / 978 - 0 - 8176 - 4801-5_2. url: https://doi.org/10.1007/978-0-8176-4801-5_2


    Args:
        s (int): Start index of segment
        e (int): End index of segment
        y (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Arguments of cost function. Unused.

    Returns:
        float: Segment cost
    """

    cost = normal_var_cost(s, e, y, cost_args) 
    return cost + math.log(e - s)


class NormalVarCost(BaseCost):
    """Change in variance of a normal distribution"""
    
    n_params = 1

    @staticmethod
    @nb.njit(preprocess_sig, cache=True, fastmath=True)
    def preprocess(y: np.ndarray, args: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sumstats = np.empty((y.shape[0] + 1, 1), np.float64)
        sumstats[:, 0] = np.append(0.0, ((y - y.mean()) ** 2).cumsum())
        return sumstats, np.float64([0.0])

    cost_fn = staticmethod(normal_var_cost)
    cost_fn_mbic = staticmethod(normal_var_cost_mbic)




@nb.njit(cost_sig, cache=True, fastmath=True, nogil=True)
def normal_meanvar_cost(s: int, e: int, y: np.ndarray, cost_args: np.ndarray) -> float:
    """Normal mean and variance change cost of a proposed segment

    Jie Chen and Arjun K. Gupta. “Univariate Normal Model”. In: Parametric Statistical Change Point Analysis: With Applications to Genetics, Medicine, and Finance. Boston: Birkh ̈auser Boston, 2012, pp. 7-88. isbn: 978-0-8176-4801-5. doi: 10 . 1007 / 978 - 0 - 8176 - 4801-5_2. url: https://doi.org/10.1007/978-0-8176-4801-5_2

    Args:
        s (int): Start index of segment
        e (int): End index of segment
        y (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Arguments of cost function. Unused.

    Returns:
        float: Segment cost
    """
    
    n = e - s
    d1 = y[e, 0] - y[s, 0]
    d2 = y[e, 1] - y[s, 1]
    a1 = (d1 ** 2.0) / n
    a2 = d2 - a1
    if a2 <= 0.0:
        a2 = 1e-8
    a3 = 2.837877066 + math.log(a2 / n)
    cost = n * a3
    return cost


@nb.njit(cost_sig, cache=True, fastmath=True, nogil=True)
def normal_meanvar_cost_mbic(s: int, e: int, y: np.ndarray, cost_args: np.ndarray) -> float:
    """Normal mean and variance change cost of a proposed segment with an MBIC penalty

    Jie Chen and Arjun K. Gupta. “Univariate Normal Model”. In: Parametric Statistical Change Point Analysis: With Applications to Genetics, Medicine, and Finance. Boston: Birkh ̈auser Boston, 2012, pp. 7-88. isbn: 978-0-8176-4801-5. doi: 10 . 1007 / 978 - 0 - 8176 - 4801-5_2. url: https://doi.org/10.1007/978-0-8176-4801-5_2


    Args:
        s (int): Start index of segment
        e (int): End index of segment
        y (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Arguments of cost function. Unused.

    Returns:
        float: Segment cost
    """
    
    cost = normal_meanvar_cost(s, e, y, cost_args)
    return cost + math.log(e - s)


class NormalMeanVarCost(BaseCost):
    """Change in mean and variance of a normal distribution"""
    
    n_params = 2
    
    @staticmethod
    @nb.njit(preprocess_sig, cache=True, fastmath=True)
    def preprocess(y: np.ndarray, args: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sumstats = np.empty((y.shape[0] + 1, 2), np.float64)
        sumstats[0, :] = 0.0
        sumstats[1:, 0] = y.cumsum()
        sumstats[1:, 1] = (y ** 2).cumsum()

        return sumstats, np.float64([0.0])
    
    cost_fn = staticmethod(normal_meanvar_cost)
    cost_fn_mbic = staticmethod(normal_meanvar_cost_mbic)
