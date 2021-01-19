# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# This code implements simple simluations of spectral leakage effects
import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fftpack import fft, ifft
# Enable the cache to save FFTW plan to perform faster fft for the subsequent calls of pyfftw
pyfftw.interfaces.cache.enable()


def periodogram_mean(func, fe, n_data, f_zero=None):
    """
    Function calculating the theoretical mean of the periodogram (defined as the
    squared modulus of the fft devided by fe*n_data) given the theoretical PSD (func)
    , the sampling frequency fe and the number of points n_data.


    @param func: function of one parameter giving the PSD as a function of frequency
    @type func : function
    @param fe: sampling frequency
    @type fe : scalar (float)
    @param n_data: number of points of the periodogram
    @type n_data : scalar (integer)

    @return:
        P_mean : Periodogram expectation (n_data-vector)
    """

    # 1. Calculation of the autocovariance function Rn
    power = np.int(np.log(n_data) / np.log(2.)) + 4
    # Number of points for the integration
    N_points = 2 ** power
    # N_points = 3*n_data

    k_points = np.arange(0, N_points)
    frequencies = fe * (k_points / np.float(N_points) - 0.5)

    if f_zero is None:
        f_zero = fe / (N_points * 10.)
    i = np.where(frequencies == 0)
    frequencies[i] = f_zero
    Z = func(frequencies)
    n = np.arange(0, n_data)
    Z_ifft = ifft(Z)
    R = fe / np.float(N_points) * (
            Z[0] * 0.5 * (np.exp(1j * np.pi * n) - np.exp(-1j * np.pi * n)) + N_points * Z_ifft[0:n_data] * np.exp(
            -1j * np.pi * n))
    # 2. Calculation of the of the periodogram mean vector
    X = R[0:n_data] * (1. - np.abs(n) / np.float(n_data))

    return 1. / fe * (fft(X) + n_data * ifft(X) - R[0]), R[0:n_data]


def periodogram_mean_masked(func, fe, n_data, n_freq, mask, 
                            n_points=None, n_conv=None, normal=True):
    """
    Function calculating the theoretical mean of the periodogram of a masked
    signal (defined as the squared modulus of the fft devided by fe*n_data) 
    given the theoretical PSD (func), the sampling frequency fe and the number
    of points n_data.

    @param func: function of one parameter giving the PSD as a function of
    frequency
    @type func : function
    @param fe: sampling frequency
    @type fe : scalar (float)
    @param n_data: number of points of the periodogram
    @type n_data : scalar (integer)
    @param mask: mask vetor  M[i] = 1 if data is available, 0 otherwise
    @type mask : (n_data x 1) array
    @param n_freq: number of frequency point where to compute the periodogram
    @type n_freq : scalar (integer)

    @return:
        P_mean : Periodogram expectation (n_data-vector)
    """

    if n_points == None:
        # 1. Calculation of the autocovariance function Rn
        power = np.int(np.log(2 * n_data) / np.log(2.))  # + 1
        # Number of points for the integration
        n_points = 2 ** power

    k_points = np.arange(0, n_points)
    frequencies = fe * (k_points / np.float(n_points) - 0.5)
    i = np.where(frequencies == 0)
    frequencies[i] = fe / (n_points)
    Z = func(frequencies)
    n = np.arange(0, n_data)
    Z_ifft = ifft(Z)
    R = fe / np.float(n_points) * (Z[0] * 0.5 * (np.exp(1j * np.pi * n) \
                                                 - np.exp(-1j * np.pi * n)) + n_points * Z_ifft[0:n_data] * np.exp(
        -1j * np.pi * n))

    if n_conv == None:
        n_conv = 2 * n_data - 1
    # 2. Calculation of the sample autocovariance of the mask
    fx = fft(mask, n_conv)
    # print("FFT of M is done with N_points")
    # fx = fft(M, N_points)

    if normal:
        K2 = np.sum(mask ** 2)
    else:
        K2 = n_data
    lambda_N = np.real(ifft(fx * np.conj(fx))) / K2

    # 3. Calculation of the of the periodogram mean vector
    X = R[0:n_data] * lambda_N[0:n_data]

    Pm = 1. / fe * (fft(X, n_freq) + n_freq * ifft(X, n_freq) - R[0] * lambda_N[0])

    return Pm