# Importing packages
import numpy as np
import pandas as pd
import pytest

import src.pychange.costs as pyc

# Importing r stuff
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

rcp = rpackages.importr('changepoint')

data_name = 'wave_c44137'
data = pd.read_csv(f'../data/{data_name}.csv').iloc[:, -1].values[:61036]


def test_norm_mean():
    #data = np.hstack((np.random.normal(0.0, 1.0, (100,)), np.random.normal(1.0, 1.0, (100,))))
    pychange_cost = pyc.NormalMeanCost().fit(data).cost(0, len(data))
    r_cost = rcp.single_mean_norm_calc(robjects.FloatVector(data), minseglen=1)[1]
    assert pytest.approx(pychange_cost) == pytest.approx(r_cost)


def test_norm_var():
    #data = np.hstack((np.random.normal(0.0, 1.0, (100,)), np.random.normal(0.0, 2.0, (100,))))
    pychange_cost = pyc.NormalVarCost().fit(data).cost(0, 61036)
    r_cost = rcp.single_var_norm_calc(robjects.FloatVector(data), minseglen=1, mu=float(data.mean()))[1]
    assert pytest.approx(pychange_cost) == pytest.approx((len(data) * (np.log(2 * np.pi) + 1)) + r_cost)

 
def test_norm_mean_var():
    #data = np.hstack((np.random.normal(0.0, 1.0, (100,)), np.random.normal(1.0, 2.0, (100,))))
    pychange_cost = pyc.NormalMeanVarCost().fit(data).cost(0, 61036)
    r_cost = rcp.single_meanvar_norm_calc(robjects.FloatVector(data), minseglen=1)[1]
    assert pytest.approx(pychange_cost) == pytest.approx((len(data) * (np.log(2 * np.pi) + 1)) + r_cost)


def test_poisson_mean_var():
    data = np.hstack((np.random.poisson(1.0, (100,)), np.random.poisson(2.0, (100,))))
    pychange_cost = pyc.PoissonMeanVarCost().fit(data).cost(0, len(data))
    r_cost = rcp.single_meanvar_poisson_calc(robjects.FloatVector(data), minseglen=1)[1]
    assert pytest.approx(pychange_cost) == pytest.approx(r_cost)


def test_exp_mean_var():
    data = np.hstack((np.random.exponential(1.0, (100,)), np.random.exponential(2.0, (100,))))
    pychange_cost = pyc.ExponentialMeanVarCost().fit(data).cost(0, len(data))
    r_cost = rcp.single_meanvar_exp_calc(robjects.FloatVector(data), minseglen=1)[1]
    assert pytest.approx(pychange_cost) == pytest.approx(r_cost)


def test_gamma_mean_var():
    data = np.hstack((np.random.gamma(1.0, 1.0, (100,)), np.random.gamma(2.0, 2.0, (100,))))
    pychange_cost = pyc.GammaMeanVarCost().fit(data).cost(0, len(data))
    r_cost = rcp.single_meanvar_gamma_calc(robjects.FloatVector(data), minseglen=1)[1]
    assert pytest.approx(pychange_cost) == pytest.approx(r_cost)
