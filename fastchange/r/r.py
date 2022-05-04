#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
import numpy as np

# import rpy2's package module
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

# import R's utility package
try:
    utils = rpackages.importr('utils')
    rcp = rpackages.importr('changepoint')
    rcpnp = rpackages.importr('changepoint.np')
    rocp = rpackages.importr('ocp')

    _r_cpt_methods = {
        "mean": rcp.cpt_mean,
        "var": rcp.cpt_var,
        "meanvar": rcp.cpt_meanvar,
        "np": rcpnp.cpt_np
    }
except:
    print('Failed to get R installations, please install necessary packages')
    _r_cpt_methods = {}

class ROfflineChangepoint:
    """Rpy2-based interface for calling R change point libraries"""


    def __init__(self, cost_method='meanvar', **kwargs):
        """Initialization method
        
        Keyword arguments will be passed to the underlying R function.

        Args:
            cost_method (str, optional): What cost to use, must be 'mean', 'var', 'meanvar', or 'np' ('np' corresponds to changepoint.np package). Defaults to 'meanvar'.
        """
        self.cost_method = cost_method
        self.cpt_fn = _r_cpt_methods[cost_method]
        self.kwargs = kwargs

    def fit(self, y: np.ndarray) -> None:
        """Fit model and find changepoints

        Args:
            y (np.ndarray): Signal of which to find change points
        """
        self.cps = self.cpt_fn(robjects.FloatVector(y), **self.kwargs)
        return self

    def predict(self) -> np.ndarray:
        """Get indices of change points in signal

        Returns:
            np.ndarray: Indices of change points
        """
        return np.int64(rcp.cpts(self.cps))

class ROCP:
    """Online change point detection calling the r bocp library"""

    def __init__(self, **kwargs):
        """Initialization method
        
        Keyword arguments will be passed to the underlying R function.
        """
        self.kwargs = kwargs
    
    def fit(self, y: np.ndarray) -> None:
        """Fit model and find changepoints

        Args:
            y (np.ndarray): Signal of which to find change points
        """
        _x = robjects.FloatVector(y)
        self.cp = rocp.onlineCPD(_x)
        return self
    
    def predict(self) -> np.ndarray:
        """Get indices of change points in signal

        Returns:
            np.ndarray: Indices of change points
        """
        return np.array(self.cp.rx2('changepoint_lists').rx2('colmaxes'))[0].astype(np.int64)[1:]
