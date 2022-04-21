# Importing packages
import pytest
import numpy as np

from src.pychange.costs import EmpiricalCost

from src.pychange.segment import PeltSeg
from src.pychange.r import ROfflineChangepoint
# Importing r stuff
import rpy2.robjects.packages as rpackages

rcp = rpackages.importr('changepoint')

# Note this is the sumstat creation function from the np.changepoint package in R
# This is only available as an inner function and thus not importable, so we
# directly translate to python to test
def nonparametric_ed_sumstat(data,K):
    n = data.shape[0]
    if K > n:
        K = n
    Q = np.zeros((K, n + 1))
    x = np.sort(data)
    yK = -1 + (2 * (np.arange(K)) / K - 1 / K)
    c = -1.0 * np.log(2 * n - 1)
    pK = 1 / (1 + np.exp(c * yK))
    for i in range(K):
        j = int((n - 1) * pK[i] + 1)
        Q[i, 1:] = np.cumsum(data < x[j]) + 0.5 * np.cumsum(data == x[j])
    return Q

@pytest.mark.parametrize(
        'k', [10, 12, 15, 20]
    )
def test_np_sumstat(k):
    data = np.hstack((np.random.normal(0, 1, (100,)), np.random.normal(1, 2, (100,))))
    py_sumstats = EmpiricalCost(k).fit(data).y
    r_sumstats = nonparametric_ed_sumstat(data, k)
    assert np.allclose(py_sumstats.T, r_sumstats)

# TODO write empirical test
# def test_np():
#     data = np.hstack((np.random.normal(0.0, 1.0, (100,)), np.random.normal(1.0, 2.0, (100,))))
#     pychange_cost = EmpiricalCost(k=10).fit(data).cost(0, len(data))
#     r_sumstats = nonparametric_ed_sumstat(data, 10)
#     r_cost = mll_nonparametric_ed(r_sumstats[:, data.shape[0]], 10, data.shape[0])
#     assert pytest.approx(pychange_cost) == pytest.approx(r_cost)

# Creating normal test data
data = np.hstack((np.random.normal(0, 1, (100,)), np.random.normal(10, 2, (100,))))

def test_seg_norm():
    r_cps = ROfflineChangepoint(cost_method='np', method='PELT', minseglen=1, pen_value=100, penalty='Manual', nquantiles=10).fit(data).predict()
    pyc_cps = PeltSeg(EmpiricalCost(k=10), penalty=50.0, mbic=False, min_len=1, max_cps=100).fit(data).predict()
    assert np.allclose(r_cps, pyc_cps[:-1], atol=1)
