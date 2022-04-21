# Importing packages
import numpy as np

def create_synthetic_series(dist, n_segments, **kwargs):

    return np.vstack([dist(**kwargs) for i in n_segments])
