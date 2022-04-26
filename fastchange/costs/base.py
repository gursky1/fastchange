# Importing packages
import numpy as np
import numba as nb
from numba.types import Tuple


# Types we will use for all preprocess and cost functions
preprocess_sig = Tuple((nb.float64[:, :], nb.float64[:]))(nb.float64[:], nb.float64[:])
cost_sig = nb.float64(nb.int64, nb.int64, nb.float64[:, :], nb.float64[:])

class BaseCost:
    
    n_params = 1
    
    def __init__(self):
        self.args = np.float64([0.0])
        
    def fit(self, y: np.ndarray):
        self.sumstats, self.cost_args = self.preprocess(y, self.args)
        return self
    
    def cost(self, start, end):
        return self.cost_fn(start, end, self.sumstats, self.cost_args)
