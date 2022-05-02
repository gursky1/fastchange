#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
import numpy as np
import numba as nb
from numba.types import Tuple


# Types we will use for all preprocess and cost functions
preprocess_sig = Tuple((nb.float64[:, :], nb.float64[:]))(nb.float64[:], nb.float64[:])
cost_sig = nb.float64(nb.int64, nb.int64, nb.float64[:, :], nb.float64[:])

class BaseCost:
    """Base class for all cost functions
    
    Attributes:
        n_params (int): Number of parameters in cost function to estimate
        args (np.ndarray): Arguments to pass to preprocessing method
        sumstats (np.ndarray): Summary statistics of signal
        cost_args (np.ndarray): Calculated arguments to pass to cost method
        
    """
    
    n_params = 1
    
    def __init__(self):
        """Initialization method"""
        self.args = np.float64([0.0])
        
    def fit(self, y: np.ndarray) -> None:
        """Create summary statistics and arguments for cost function

        Args:
            y (np.ndarray): Signal
        """
        self.sumstats, self.cost_args = self.preprocess(y, self.args)
        return self
    
    def cost(self, start: int, end: int) -> float:
        """Find the cost of segment [start, end] for the signal

        Args:
            start (int): Starting index
            end (int): Ending index

        Returns:
            float: Cost of segment
        """
        return self.cost_fn(start, end, self.sumstats, self.cost_args)
