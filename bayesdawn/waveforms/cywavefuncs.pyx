# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2018
# This code is written to compute LISA's waveform in with Cython

import numpy as np
cimport numpy as np
cimport cython
# from cython.parallel import prange
# cimport scipy.special.cython_special as csc
from libc.math cimport pi, exp, sin
cdef double complex I = 1j

DTYPE = np.complex128

# ctypedef fused my_type:
#     int
#     double
#     long long

cdef extern from "complex.h":
    double complex cexp(double complex)

def sinc(double x):
    cdef double out
    out = sin(pi * x) / x
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
# def v_func_gb_mono(np.ndarray [double, ndim=2] array_f, double f_0,
#                    double tobs, double ts):
def v_func_gb_mono(double[:, ::1] array_f, double f_0, double tobs, double ts):

    cdef Py_ssize_t x_max = array_f.shape[0]
    cdef Py_ssize_t y_max = array_f.shape[1]
    # array_f.shape is now a C array
    cdef np.ndarray [np.complex128_t, ndim=2] result = np.empty_like(array_f, dtype=DTYPE)
    # result = np.zeros((x_max, y_max), dtype=DTYPE)
    # cdef double complex[:, :] result_view = result

    cdef double complex tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
    # for x in prange(x_max, nogil=True):
        for y in range(y_max):
            tmp = cexp(I * pi * (f_0 - array_f[x, y]) * tobs) * sinc((f_0 - array_f[x, y]) * tobs) * tobs / ts
            result[x, y] = tmp
            # result_view[x, y] = tmp

    return result
