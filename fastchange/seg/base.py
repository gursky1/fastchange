#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
import numpy as np
import numba as nb
from numba.types import Type, FunctionType

from ..costs.base import BaseCost, cost_sig
from ..penalties import mbic_penalty


def seg_sig(*args: nb.types.Type) -> "Signature":
    """Numba signature definition for segmentation functions
    
    Args:
        *args (nb.types.Type): Additional argument types

    Returns:
        Signature: Numba function signature
    """
    return nb.int64[:](FunctionType(cost_sig), nb.float64[:, :], nb.float64[:], nb.float64, nb.int64, nb.int64, nb.int64, *args)


class BaseSeg:
    """Base class for all segmentation methods"""
    
    def __init__(self, cost: BaseCost, penalty: float=0.0, min_len: int=1, max_cps: int=-1):
        """Initialization method

        Args:
            cost (BaseCost): Cost function for segmentation
            penalty (float, optional): Complexity penalty. Defaults to 0.0.
            min_len (int, optional): Minimum segment length. Defaults to 1.
            max_cps (int, optional): Maximum number of change points. Defaults to -1.
        """
        self.cost = cost
        self.penalty = penalty
        self.min_len = min_len
        self.max_cps = max_cps
        self.mbic = penalty == mbic_penalty
        
    def fit(self, y: np.ndarray) -> None:
        """Find all change points in a signal

        Args:
            y (np.ndarray): Signal in which to detect change points
        """
        # Fitting penalty
        self.n = y.shape[0]
        if callable(self.penalty):
            self.pen = self.penalty(y.shape[0], self.cost.n_params)
        else:
            self.pen = self.penalty
            
        # Fitting cost function
        self.cost.fit(y)
        
        # Running segmentation
        cost_fn = self.cost.cost_fn_mbic if self.mbic else self.cost.cost_fn
        self.cps = self.seg_fn(cost_fn, self.cost.sumstats, self.cost.cost_args, self.pen, self.min_len, self.max_cps, self.n)
        if self.cps.shape[0] == 1 and self.cps[0] == -1:
            self.cps = np.int64([])
        return self
    
    def predict(self) -> np.ndarray:
        """Get array of change point indices

        Returns:
            np.ndarray: Change point indices in given signal
        """
        return self.cps
