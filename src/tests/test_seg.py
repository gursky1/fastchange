# Importing packages
import numpy as np
import pytest

import src.pychange.costs as pyc_costs
import src.pychange.segment as pyc_seg
from src.pychange.r import ROfflineChangepoint
# Importing r stuff
import rpy2.robjects.packages as rpackages

rcp = rpackages.importr('changepoint')


# Creating normal test data
norm_data = np.hstack((np.random.normal(0, 1, (100,)), np.random.normal(10, 2, (100,))))

@pytest.mark.parametrize(
        'seg', [('AMOC', pyc_seg.AmocSeg),
                ('BinSeg', pyc_seg.BinSeg),
                ('PELT', pyc_seg.PeltSeg),
                ]
    )
@pytest.mark.parametrize(
        'cost', [('mean', 1, pyc_costs.NormalMeanCost()),
                  ('var', 2, pyc_costs.NormalVarCost()),
                  ('meanvar', 2, pyc_costs.NormalMeanVarCost())
                  ],
    )
def test_seg_norm(seg, cost):
    r_cps = ROfflineChangepoint(cost_method=cost[0], method=seg[0], minseglen=cost[1], Q=100, pen_value=100.0, penalty='Manual').fit(norm_data).predict()
    pyc_cps = seg[1](cost[2], penalty=100.0, mbic=False, min_len=cost[1], max_cps=100).fit(norm_data).predict()
    assert r_cps.shape[0] == pyc_cps[:-1].shape[0]
    assert np.allclose(r_cps, pyc_cps[:-1], atol=1)
