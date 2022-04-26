# Importing packages
from time import time
import math

import numpy as np
import pandas as pd
import ruptures as rpt

#from ..pychange.costs.normal import NormalMeanCost, NormalVarCost, NormalMeanVarCost
from ..pychange.costs.normal import NormalMeanVarCost
# from ..pychange.costs.gamma import GammaMeanVarCost
# from ..pychange.costs.poisson import PoissonMeanVarCost
# from ..pychange.costs.exp import ExponentialMeanVarCost
from ..pychange.costs.emp import EmpiricalCost
from ..pychange.seg.amoc import AmocSeg
from ..pychange.seg.binseg import BinSeg
from ..pychange.seg.pelt import PeltSeg
from ..pychange.penalties import mbic_penalty

from ..pychange.r import ROfflineChangepoint

if __name__ == '__main__':

    repeats = 5

    def normal_data(x):
        return np.hstack([
            np.random.normal(0, 1, (x,)),
            np.random.normal(10, 4, (x,)),
            np.random.normal(1, 2, (x,)),
            np.random.normal(5, 1, (x,)),
            np.random.normal(-2, 2, (x,)),
        ])

    def poisson_data(x):
        return np.hstack([
            np.random.poisson(0, (x,)),
            np.random.poisson(10, (x,)),
            np.random.poisson(2, (x,)),
        ])

    def exp_data(x):
        return np.hstack([
            np.random.exponential(0, (x,)),
            np.random.exponential(10, (x,)),
            np.random.exponential(2, (x,)),
        ])

    def gamma_data(x):
        return np.hstack([
            np.random.gamma(5, 1, (x,)),
            np.random.gamma(10, 1, (x,)),
            np.random.gamma(2, 1, (x,)),
        ])

    # Setting segment lengths
    seg_n = {
        (AmocSeg, 'Normal'): map(int, [5e3, 1e4, 5e4, 1e5, 5e5, 1e6]),
        (BinSeg, 'Normal'): map(int, [5e2, 7.5e2, 1e3, 2.5e3, 5e3, 7.5e3]),
        (PeltSeg, 'Normal'): map(int, [5e2, 1e3, 5e3, 1e4, 5e4, 1e5]),
        (PeltSeg, 'Empirical'): map(int, [100, 350, 700, 1000, 1350, 1750, 2000])
    }

    with open('synthetic_profile.csv', 'w') as f:
        f.write('n,seg,dist,stat,penalty,py_time,r_time\n')

    for seg_py, seg_r in [(AmocSeg, 'AMOC'), (BinSeg, 'BinSeg'), (PeltSeg, 'PELT')]:
        for dist, dist_fn in [('Normal', normal_data), ('Empirical', normal_data)]:

            for stat in ['MeanVar']:
                if dist != 'Normal' and stat != 'MeanVar':
                    continue
                if dist == 'Empirical' and seg_r != 'PELT':
                    continue
                print('\n')
                n_list = seg_n[(seg_py, dist)]
                for n in list(n_list):
                    try:

                        py_times = []
                        r_times = []
                        for _ in range(repeats):
                            data = dist_fn(n).astype(np.float64)
                            if dist == 'Empirical':
                                start_time = time()
                                for _ in range(5):
                                    seg_py(EmpiricalCost(k=10), mbic_penalty, min_len=10, max_cps=10).fit(data).predict()
                                end_time = time() - start_time
                                py_times.append(end_time)

                                start_time = time()
                                for _ in range(5):
                                    ROfflineChangepoint('np', penalty='MBIC', method=seg_r, nquantiles=10, minseglen=10).fit(data).predict()
                                end_time = time() - start_time
                                r_times.append(end_time)
                            
                            else:
                                start_time = time()
                                for _ in range(10):
                                    seg_py(NormalMeanVarCost(), mbic_penalty, min_len=10, max_cps=10).fit(data).predict()
                                end_time = time() - start_time
                                py_times.append(end_time)

                                start_time = time()
                                for _ in range(10):
                                    ROfflineChangepoint('meanvar', penalty='MBIC', test_stat='Normal', Q=10, method=seg_r, minseglen=10).fit(data).predict()
                                end_time = time() - start_time
                                r_times.append(end_time)

                        py_mean = np.mean(py_times)
                        r_mean = np.mean(r_times)

                        with open('synthetic_profile.csv', 'a') as f:
                            f.write(f'{n},{seg_r},{dist},{stat},MBIC,{py_mean},{r_mean}\n')
                        print(n, seg_r, dist, stat, r_mean, py_mean, (r_mean / py_mean).round(4))

                    except KeyboardInterrupt:
                        quit()
                    except Exception as e:
                        print(e)
