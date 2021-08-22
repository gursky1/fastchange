# Importing packages
import numpy as np

# import rpy2's package module
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

# import R's utility package
utils = rpackages.importr('utils')
rcp = rpackages.importr('changepoint')
rcpnp = rpackages.importr('changepoint.np')

_r_cpt_methods = {
    "mean": rcp.cpt_mean,
    "var": rcp.cpt_var,
    "meanvar": rcp.cpt_meanvar,
    "np": rcpnp.cpt_np
}

class RChangepoint:


    def __init__(self, method='meanvar', **kwargs):
        self.method = method
        self.cpt_fn = _r_cpt_methods[method]
        self.kwargs = kwargs

    def fit(self, signal):
        self.cp = self.cpt_fn(robjects.FloatVector(signal), **self.kwargs)

    def predict(self):
        return np.array(rcp.cpts(self.cp))
