# cython: language_level=3

# Importing packages
import numpy as np 
cimport numpy as np 
cimport cython

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

from libc.math cimport log, M_PI, fmax

@cython.wraparound(False)
@cython.binding(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.overflowcheck(False)
cdef float _cython_normal_mean_var_cost(np.ndarray[DTYPE_t, ndim=1] x, int n):
    return(n * (log(2 * M_PI) + log(fmax(x[2], 1e-8) / n) + 1) + log(n))

def cython_normal_mean_var_cost(np.ndarray[DTYPE_t, ndim=1] x, int n):
    return _cython_normal_mean_var_cost(x, n)

@cython.wraparound(False)
@cython.binding(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.overflowcheck(False)
cdef float _iter_sumstats(np.ndarray[DTYPE_t, ndim=2] sumstats, int start, int end):
    for i in range(start, end):
        cost = _cython_normal_mean_var_cost(sumstats[i, :], i)
    return cost

def iter_sumstats(np.ndarray[DTYPE_t, ndim=2] sumstats, int start, int end):
    return _iter_sumstats(sumstats, start, end)
