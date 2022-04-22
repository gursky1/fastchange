# Importing packages
import timeit

import numpy as np
import pandas as pd

from ..pychange.costs.normal import NormalMeanCost, NormalVarCost, NormalMeanVarCost
from ..pychange.seg.amoc import AmocSeg
from ..pychange.seg.binseg import BinSeg
from ..pychange.seg.pelt import PeltSeg
from ..pychange.penalties import bic_penalty

from ..pychange.r import ROfflineChangepoint

if __name__ == '__main__':

    repeats = 20

    with open('real_profile.csv', 'w') as f:
        f.write('dataset,seg,dist,stat,penalty,py_time,r_time\n')

    for data_name in ['wave_c44137', 'ftse100', 'HC1', 'Lai2005fig3', 'Lai2005fig4']:
        data = pd.read_csv(f'./data/{data_name}.csv').iloc[:, -1].values.astype(np.float64)

        try:
            print('\n', data_name)

            for seg_py, seg_r in [('AmocSeg', 'AMOC'), ('BinSeg', 'BinSeg'), ('PeltSeg', 'PELT')]:
                for dist in ['Normal']:
                    for stat in ['Mean', 'Var', 'MeanVar']:
                        for pen in ['BIC']:

                            py_times = timeit.repeat(f'{seg_py}({dist}{stat}Cost(), {pen.lower()}_penalty, min_len=10, max_cps=50).fit(data).predict()', repeat=repeats, number=1, globals=globals())
                            r_times = timeit.repeat(f"ROfflineChangepoint('{stat.lower()}', penalty='{pen}', method='{seg_r}', test_stat='{dist}', minseglen=10, Q=50).fit(data).predict().astype(np.int64)", repeat=repeats, number=1, globals=globals())

                            py_mean = np.mean(py_times)
                            r_mean = np.mean(r_times)

                            with open('real_profile.csv', 'a') as f:
                                f.write(f'{data_name},{seg_py},{dist},{stat},{pen},{py_mean},{r_mean}\n')
                            print(seg_py, dist, stat, (r_mean / py_mean).round(4))

        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(e)
