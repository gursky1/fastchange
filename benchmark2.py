# Importing packages
from math import cos
from time import time
from functools import partial
import numpy as np
from pychange.costs import L1Cost, L2Cost, NormalMeanCost, NormalVarCost, NormalMeanVarCost, PoissonMeanVarCost, ExponentialMeanVarCost, GammaMeanVarCost, EmpiricalCost
from pychange.segment import AmocSeg, BinSeg, PeltSeg, pelt_segment
from pychange.online import ConstantHazard, StudentTProb, OnlineCP
from pychange.r import ROfflineChangepoint, ROCP
from pychange.penalties import mbic_penalty
import ruptures as rpt
from ruptures.metrics import randindex, precision_recall

# Setting hyperparams
min_len = 1
k = 20
pen = 250.0
mbic = False
max_cps = 50
n_runs = 1


def normal_random(seg_len, n_cps, loc_limit=50, scale_var=5):
    loc = np.random.uniform(-1 * loc_limit, loc_limit, size=(n_cps + 1,))
    scale = np.random.uniform(0, scale_var, size=(n_cps + 1,))
    return np.random.normal(loc, scale, size=(seg_len, n_cps + 1)).flatten('F')


def poisson_random(seg_len, n_cps, lam_var=10):
    lam = np.random.uniform(0, lam_var, size=(n_cps + 1,))
    return np.random.poisson(lam, size=(seg_len, n_cps + 1)).flatten('F')


def exponential_random(seg_len, n_cps, scale_var=10):
    scale = np.random.uniform(0, scale_var, size=(n_cps + 1,))
    return np.random.exponential(scale, size=(seg_len, n_cps + 1)).flatten('F')


def gamma_random(seg_len, n_cps, shape_var=15):
    shape = np.random.uniform(0, shape_var, size=(n_cps + 1,))
    return np.random.gamma(shape, size=(seg_len, n_cps + 1)).flatten('F')


def format_time(x):
    
    if x > 1.0:
        t = round(x, 3)
        return f'{t} seconds'
    elif x > 1e-3:
        t = round(x * 1e3, 3)
        return f'{t} milliseconds'
    elif x > 10e-6:
        t = round(x * 10e6, 3)
        return f'{t} microseconds'
    else:
        t = round(x * 10e9, 3)
        return f'{t} nanoseconds'

cost_methods = {'l1': (L1Cost,
                       None,
                       [],
                       {'model': 'l1'},
                       normal_random),
                'l2': (L2Cost,
                       None,
                       [],
                       {'model': 'l2'},
                       normal_random),
                'normal_mean': (
                    NormalMeanCost,
                    {"cost_method": "mean", "test_stat": "Normal"},
                    ['amoc', 'binseg', 'pelt'],
                    None,
                    normal_random),
                'normal_var': (
                    NormalVarCost,
                    {"cost_method": "var", "test_stat": "Normal"},
                    ['amoc', 'binseg', 'pelt'],
                    None,
                    normal_random),
                'normal_mean_var': (
                    NormalMeanVarCost,
                    {"cost_method": "meanvar", "test_stat": "Normal"},
                    ['amoc', 'binseg', 'pelt'],
                    {'model': 'normal'},
                    normal_random),
                'poisson_mean_var': (
                    PoissonMeanVarCost,
                    {"cost_method": "meanvar", "test_stat": "Poisson"},
                    ['amoc', 'binseg', 'pelt'],
                    None,
                    poisson_random),
                'exponential_mean_var': (
                    ExponentialMeanVarCost,
                    {"cost_method": "meanvar", "test_stat": "Exponential"},
                    ['amoc', 'binseg', 'pelt'],
                    None,
                    exponential_random),
                'gamma_mean_var': (
                    GammaMeanVarCost,
                    {"cost_method": "meanvar", "test_stat": "Gamma"},
                    ['amoc', 'binseg', 'pelt'],
                    None,
                    gamma_random),
                'empirical': (
                    lambda: EmpiricalCost(k=k),
                    {"cost_method": "np", "nquantiles": k},
                    ['pelt_np'],
                    None,
                    normal_random
                )}


seg_methods = {
    # 'amoc': {
    #     'pychange': lambda cost, data: AmocSeg(cost(), mbic_penalty, min_len, True).fit(data).predict(),
    #     'r': lambda params, data: ROfflineChangepoint(penalty='MBIC', method='AMOC', minseglen=min_len, **params).fit(data).predict(),
    #     'ruptures': None,
    #     'data_params': (30000, 1) 
    # },
    # 'binseg': {
    #     'pychange': lambda cost, data: BinSeg(cost(), 100.0, min_len).fit(data).predict(),
    #     'r': lambda params, data: ROfflineChangepoint(penalty='Manual', pen_value=100.0, method='BinSeg', minseglen=min_len, Q=100, **params).fit(data).predict(),
    #     'ruptures': None,#lambda params, data: rpt.Binseg(min_size=min_len, jump=1, **params).fit(data).predict(pen=pen),
    #     'data_params': (10000, 3)
    #     },
    'pelt': {
        'pychange': lambda cost, data: pelt_segment(cost().fit(data), max_cps, 250.0, False),#PeltSeg(cost(), 200.0, max_cps, False).fit(data).predict(),
        'r': lambda params, data: ROfflineChangepoint(penalty='Manual', pen_value=250.0, method='PELT', minseglen=min_len, **params).fit(data).predict(),
        'ruptures': None,#lambda params, data: rpt.Pelt(min_size=min_len, jump=1, **params).fit(data).predict(pen=pen),
        'data_params': (10000, 3)
    },
    'pelt_np': {
        'pychange': lambda cost, data: PeltSeg(cost(), mbic_penalty, max_cps, True).fit(data).predict(),
        'r': lambda params, data: ROfflineChangepoint(penalty='MBIC', method='PELT', minseglen=min_len, **params).fit(data).predict(),
        'ruptures': None,#lambda params, data: rpt.Pelt(min_size=min_len, jump=1, **params).fit(data).predict(pen=pen),
        'data_params': (500, 3)
    }
}
    

if __name__ == '__main__':
    
    all_results = {}
    
    for seg_type, seg_dict in seg_methods.items():
        all_results[seg_type] = {}
        
        for cost_type, cost_dict in cost_methods.items():
            
            if seg_type not in cost_dict[2] and seg_dict['ruptures'] is None:
                continue
            print(f'\nTiming {seg_type} - {cost_type}')
            # Creating data
            _data = cost_dict[-1](*seg_dict['data_params'])
            true_seg = np.arange(seg_dict['data_params'][0], seg_dict['data_params'][0] * (seg_dict['data_params'][1] + 1) + 1, step=seg_dict['data_params'][0])
            true_seg[-1] = _data.shape[0]
            print(true_seg)
            
            # Compiling
            _ = seg_dict['pychange'](cost_dict[0], _data)
            
            # Initializing results dictionary
            results = {'pychange': [], 'r': [], 'ruptures': []}
            
            # Running pychange
            for i in range(n_runs):
                start_time = time()
                pychange_seg = seg_dict['pychange'](cost_dict[0], _data)
                end_time = time() - start_time
                results['pychange'].append(end_time)
            pychange_mean = np.median(results["pychange"])
            print(pychange_seg)
            print(f'Pychange: {format_time(pychange_mean)} | {len(pychange_seg) - 1} / {seg_dict["data_params"][1]} changepoints detected | Precision-Recall: {precision_recall(pychange_seg, true_seg, margin=50)}')
            
            # Running R
            if seg_type in cost_dict[2]:
                #try:
                for i in range(n_runs):
                    start_time = time()
                    r_seg = seg_dict['r'](cost_dict[1], _data)
                    end_time = time() - start_time
                    results['r'].append(end_time)
                r_mean = np.median(results["r"])
                r_seg = np.append(r_seg.astype(int), len(_data))
                print(f'R: {format_time(r_mean)} | {len(r_seg) - 1} / {seg_dict["data_params"][1]} changepoints detected | Precision-Recall: {precision_recall(r_seg, true_seg, margin=50)}')
                print(r_seg)
                r_diff = round(r_mean / pychange_mean if pychange_mean < r_mean else pychange_mean / r_mean, 2)
                print(f'Pychange is {r_diff}x {"faster" if pychange_mean < r_mean else "slower"} than R')
                #except:
                #    print('R failed')
            
            # Running ruptures
            if cost_dict[3] is not None and seg_dict['ruptures'] is not None:
                for i in range(n_runs):
                    start_time = time()
                    _ = seg_dict['ruptures'](cost_dict[3], _data)
                    end_time = time() - start_time
                    results['ruptures'].append(end_time)
                ruptures_mean = np.median(results["ruptures"])
                print(f'Ruptures: {format_time(ruptures_mean)}')
                ruptures_diff = ruptures_mean / pychange_mean if pychange_mean < ruptures_mean else pychange_mean / ruptures_mean
                print(f'Pychange is {round(ruptures_diff, 2)}x {"faster" if pychange_mean < ruptures_mean else "slower"} than ruptures')
            all_results[seg_type][cost_type] = results