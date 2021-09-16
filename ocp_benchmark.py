# Importing packages
from math import cos
from time import time
import numpy as np
from pychange.online import ConstantHazard, StudentTProb, OnlineCP
from pychange.r import ROfflineChangepoint, ROCP
from pychange.penalties import mbic_penalty
import ruptures as rpt
from ruptures.metrics import randindex, precision_recall
import bocd

# Setting hyperparams
seg_lens = [50, 100, 500, 2000]
n_cps = [5, 5, 5, 5]
n_runs = 5

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

def normal_random(seg_len, n_cps, loc_limit=50, scale_var=5):
    loc = np.random.uniform(-1 * loc_limit, loc_limit, size=(n_cps + 1,))
    scale = np.random.uniform(0, scale_var, size=(n_cps + 1,))
    return np.random.normal(loc, scale, size=(seg_len, n_cps + 1)).flatten('F')

if __name__ == '__main__':

    for s, n in zip(seg_lens, n_cps):

        data = normal_random(s, n)

        # Pychange benchmark
        results = []
        _ = OnlineCP(ConstantHazard(100.0), StudentTProb(), 10, 0.5).update(data).get_cps()
        for i in range(n_runs):
            start_time = time()
            _ = OnlineCP(ConstantHazard(), StudentTProb(), 10, 0.5).update(data).get_cps()
            results.append(time() - start_time)

        print(f"Seglen {s} n_cps {n} Pychange: {format_time(np.mean(results))}")

        # BOCD benchmark
        results = []
        for i in range(n_runs):
            start_time = time()
            bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(100.0), bocd.StudentT(mu=0.1, kappa=0.01, alpha=0.01, beta=1e-4))
            rt_mle = np.empty(data.shape)
            for i, d in enumerate(data):
                bc.update(d)
                rt_mle[i] = bc.rt
            results.append(time() - start_time)

        print(f"Seglen {s} n_cps {n} BOCD: {format_time(np.mean(results))}")

        # R benchmark
        results = []
        for i in range(n_runs):
            try:
                start_time = time()
                _ = ROCP().fit(data).predict()
                results.append(time() - start_time)
            except:
                continue

        print(f"Seglen {s} n_cps {n} R: {format_time(np.mean(results))}")
