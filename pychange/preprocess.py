# Importing packages
import numpy as np

def create_summary_stats(x):
    n = x.shape[0]
    sum_stats = np.stack((np.append(0, x.cumsum()),
                          np.append(0, (x ** 2).cumsum()),
                          np.append(0, ((x - x.mean()) ** 2).cumsum()),
                          np.arange(0, n + 1)),
                         axis=-1)
    return sum_stats

def create_partial_sums(x, k):

    n = x.shape[0]
    partial_sums = np.zeros(shape=(n + 1, k), dtype=np.float64)
    sorted_data = np.sort(x)

    for i in np.arange(k):

        z = -1 + (2 * i + 1.0) / k
        p = 1.0 / (1 + (2 * n - 1) ** (-z))
        t = sorted_data[int((n - 1) * p)]

        for j in np.arange(1, n + 1):

            partial_sums[j, i] = partial_sums[j - 1, i]
            if x[j - 1] < t:
                partial_sums[j, i] += 2
            if x[j - 1] == t:
                partial_sums[j, i] += 1
    return partial_sums
