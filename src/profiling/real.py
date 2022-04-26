# Importing packages
from time import time

import numpy as np
import pandas as pd

from ..pychange.costs.normal import NormalMeanVarCost
from ..pychange.costs.emp import EmpiricalCost
from ..pychange.seg.amoc import AmocSeg
from ..pychange.seg.binseg import BinSeg
from ..pychange.seg.pelt import PeltSeg
from ..pychange.penalties import mbic_penalty

from ..pychange.r import ROfflineChangepoint

if __name__ == '__main__':

    repeats = 5

    with open('real_profile.csv', 'w') as f:
        f.write('dataset,seg,dist,stat,penalty,py_time,r_time\n')

    for data_name in ['wave_c44137', 'ftse100', 'HC1']:
        print('\n')
        data = pd.read_csv(f'./data/{data_name}.csv').iloc[:, -1].values.astype(np.float64)
        for seg_py, seg_r in [(AmocSeg, 'AMOC'), (BinSeg, 'BinSeg'), (PeltSeg, 'PELT')]:
            for dist in ['Normal', 'Empirical']:

                for stat in ['MeanVar']:
                    if dist != 'Normal' and stat != 'MeanVar':
                        continue
                    if dist == 'Empirical' and seg_r != 'PELT':
                        continue
                    
                    try:

                        py_times = []
                        r_times = []
                        for _ in range(repeats):
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

                        with open('real_profile.csv', 'a') as f:
                            f.write(f'{data_name},{seg_r},{dist},{stat},MBIC,{py_mean},{r_mean}\n')
                        print(data_name, seg_r, dist, stat, r_mean, py_mean, (r_mean / py_mean).round(4))

                    except KeyboardInterrupt:
                        quit()
                    except Exception as e:
                        print(e)
