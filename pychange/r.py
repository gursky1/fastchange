# Importing packages
import os
import numpy as np

# Setting Rpy2 environment
#os.environ['R_HOME'] = r"C:\Program Files\R\R-4.1.1"
#os.environ['R_LIBS_USER'] = r"C:\Users\15072\Documents\R\win-library\4.1"
#os.environ['R_USER'] = r"C:\Users\15072\anaconda3\envs\8801\Lib\site-packages\rpy2"

# import rpy2's package module
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

# import R's utility package
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

class ROfflineChangepoint:


    def __init__(self, cost_method='meanvar', **kwargs):
        self.cost_method = cost_method
        self.cpt_fn = _r_cpt_methods[cost_method]
        self.kwargs = kwargs

    def fit(self, signal):
        self.cp = self.cpt_fn(robjects.FloatVector(signal), **self.kwargs)
        return self

    def predict(self):
        return np.array(rcp.cpts(self.cp))

class ROCP:

    def __init(self, **kwargs):
        self.kwargs = kwargs
    
    def fit(self, signal):
        _x = robjects.FloatVector(signal)
        self.cp = rocp.onlineCPD(_x)
        return self
    
    def predict(self):
        return np.array(self.cp.rx2('changepoint_lists').rx2('colmaxes'))[0].astype(np.int64)[1:]
