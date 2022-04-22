# Importing packages
import timeit
import math

import numpy as np
import pandas as pd
import ruptures as rpt

from ..pychange.costs.normal import NormalMeanCost, NormalVarCost, NormalMeanVarCost
from ..pychange.costs.gamma import GammaMeanVarCost
from ..pychange.costs.poisson import PoissonMeanVarCost
from ..pychange.costs.exp import ExponentialMeanVarCost
from ..pychange.costs.emp import EmpiricalCost
from ..pychange.seg.amoc import AmocSeg
from ..pychange.seg.binseg import BinSeg
from ..pychange.seg.pelt import PeltSeg
from ..pychange.penalties import mbic_penalty, bic_penalty, aic_penalty

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

    with open('synthetic_profile.csv', 'w') as f:
        f.write('n,seg,dist,stat,penalty,py_time,r_time,rpt_time\n')

    #for seg_py, seg_r in [('AmocSeg', 'AMOC'), ('BinSeg', 'BinSeg'), ('PeltSeg', 'PELT')]:
    for seg_py, seg_r, seg_rpt in [('AmocSeg', 'AMOC', None),('PeltSeg', 'PELT', 'Pelt')]:
        #for dist, dist_fn in [('Normal', normal_data), ('Poisson', poisson_data), ('Exponential', exp_data), ('Gamma', gamma_data), ('Empirical', normal_data)]:
        for dist, dist_fn, n_list in [('Normal', normal_data, map(int, [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5])), ('Empirical', normal_data, map(int, [20, 350, 700, 1000, 1350, 1750, 2000]))]:
            #for stat in ['Mean', 'Var', 'MeanVar']:
            for stat in ['MeanVar']:
                if dist != 'Normal' and stat != 'MeanVar':
                    continue
                if dist == 'Empirical' and seg_py != 'PeltSeg':
                    continue
                for pen in ['BIC']:
                    print('\n')
                    for n in n_list:
                        try:
                            data = dist_fn(n).astype(np.float64)

                            if dist == 'Empirical':
                                py_times = timeit.repeat(f'{seg_py}({dist}Cost(k=10), {pen.lower()}_penalty, min_len=10, max_cps=10).fit(data).predict()', repeat=repeats, number=1, globals=globals()) 
                                r_times = timeit.repeat(f"ROfflineChangepoint('np', penalty='{pen}', method='{seg_r}', nquantiles=10, minseglen=10).fit(data).predict()", repeat=repeats, number=1, globals=globals())
                                rpt_times = None
                            else:
                                py_times = timeit.repeat(f'{seg_py}({dist}{stat}Cost(), {pen.lower()}_penalty, min_len=10, max_cps=10).fit(data).predict()', repeat=repeats, number=1, globals=globals()) 
                                r_times = timeit.repeat(f"ROfflineChangepoint('{stat.lower()}', penalty='{pen}', method='{seg_r}', test_stat='{dist}', minseglen=10, Q=10).fit(data).predict()", repeat=repeats, number=1, globals=globals())
                                if seg_rpt is not None and dist != 'Empirical' and n < 1000:
                                    rpt_times = timeit.repeat(f"rpt.Pelt(model='normal', min_size=10, jump=1).fit(data).predict(pen=3 * math.log(data.shape[0]))", repeat=repeats, number=1, globals=globals())
                                else:
                                    rpt_times = None
                            py_mean = np.mean(py_times)
                            r_mean = np.mean(r_times)
                            if rpt_times is not None:
                                rpt_mean = np.mean(rpt_times)
                            else:
                                rpt_mean = None

                            with open('synthetic_profile.csv', 'a') as f:
                                f.write(f'{n},{seg_py},{dist},{stat},{pen},{py_mean},{r_mean},{rpt_mean}\n')
                            print(n, seg_py, dist, stat, (r_mean / py_mean).round(4), '' if rpt_mean is None else (rpt_mean / py_mean).round(4))

                        except KeyboardInterrupt:
                            quit()
                        except Exception as e:
                            print(e)
