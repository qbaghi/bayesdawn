#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:40:47 2019

@author: qbaghi
"""

import numpy as np
from scipy import special


def v_func_gb(f, f_0, f_dot, T, ts):
    """
    function of frequency giving the Fourier transform of exp(-j*Phi(t)),
    where Phi(t) is the phase of the gravitatonal wave

    Parameters
    ----------
    f : array_like
        vector frequencies (size n_data) where to evalutate the function w(f)
    f_0 : scalar float
        wave frequency
    f_dot : scalar float
        wave frequency derivative
    T : scalar float
        integration time
    ts : scalar float
        sampling time (cadence)



    Returns
    -------
    v : numpy array
        vector of values of v (size n_data) calculated at given frequencies

    """

    if f_dot != 0:
        f_shift = f + f_0
        a = np.pi * f_dot
        coeff = np.exp(3j * np.pi / 4) / np.sqrt(a)
        common_factor = - np.exp(1j*np.pi/4)/(2*np.sqrt(a))*np.sqrt(np.pi) \
                        * np.exp(1j*np.pi**2 * f_shift**2 / a)

        erfi_0 = special.erfi(coeff * (np.pi*f_shift))
        erfi_T = special.erfi(coeff * (a*T+np.pi*f_shift))

        return common_factor*(erfi_T - erfi_0)/ts

    else:
        return np.exp(-1j*np.pi*(f_0+f)*T)*np.sinc((f_0+f)*T)*T/ts


def v_func_gb_star(f, f_0, f_dot, T, ts):
    """
    function of frequency giving the Fourier transform of exp(+j*Phi(t)),
    where Phi(t) is the phase of the gravitatonal wave

    Parameters
    ----------
    f : array_like
        vector frequencies (size n_data) where to evalutate the function w(f)
    f_0 : scalar float
        wave frequency
    f_dot : scalar float
        wave frequency derivative
    T : scalar float
        integration time
    ts : scalar float
        sampling time (cadence)

    Returns
    -------
    v : numpy array
        vector of values of v (size n_data) calculated at given frequencies

    """

    return np.conjugate(v_func_gb(-f, f_0, f_dot, T, ts))


def v_func_gb_seg(f, f_0, f_dot, ts, T1, T2):
    """
    function of frequency giving the Fourier transform of exp(-j*Phi(t)),
    where Phi(t) is the phase of the gravitatonal wave, computed on a segment
    of data starting at time T1 and ending at time T2

    Parameters
    ----------
    f : array_like
        vector frequencies (size n_data) where to evalutate the function w(f)
    f_0 : scalar float
        wave frequency
    f_dot : scalar float
        wave frequency derivative
    ts : scalar float
        sampling time (cadence)
    T1 : numpy 1d array
        vector of times at which the segment starts
    T2 : numpy 1d array
        vector of times at which the segment ends


    Returns
    -------
    v : numpy array
        vector of values of v (size n_data) calculated at given frequencies

    """

    if f_dot != 0:
        f_shift = f + f_0
        a = np.pi*f_dot
        coeff = np.exp(3j*np.pi/4) / np.sqrt(a)
        common_factor = - np.exp(1j*np.pi/4)/(2*np.sqrt(a))*np.sqrt(np.pi)
        common_factor = common_factor * np.exp(1j*np.pi**2 * f_shift**2 / a)

        erfi_T1 = special.erfi(coeff * (a*T1+np.pi*f_shift))
        erfi_T2 = special.erfi(coeff * (a*T2+np.pi*f_shift))

        return common_factor*(erfi_T2 - erfi_T1)/ts

    else:
        f_shift = f + f_0

        return window_tf(f_shift, T1, T2)/ts


def window_tf(f, T_start, T_end):
    """

    Fourier transform of the rectangular window function between times
    T_start and T_end


    Parameters
    ----------
    f : array_like
        vector frequencies (size n_data) where to evalutate the function w(f)
    T_start : numpy 1d array
        vector of times at which each gap starts
    T_end : numpy 1d array
        vector of times at which each gap ends

    Returns
    -------
    v : numpy array
        vector of values of the window TF calculated at given frequencies


    """

    dT = T_end - T_start

    return dT * np.exp(-np.pi*f*1j*(T_end+T_start))*np.sinc(f*dT)


def v_func_gb_conj(f, f_0, f_dot, ts, t1, t2):
    """
    function of frequency giving the Fourier transform of exp(+j*Phi(t)),
    where Phi(t) is the phase of the gravitatonal wave, computed on a segment
    of data starting at time T1 and ending at time T2

    Parameters
    ----------
    f : array_like
        vector frequencies (size n_data) where to evalutate the function w(f)
    f_0 : scalar float
        wave frequency
    f_dot : scalar float
        wave frequency derivative
    T : scalar float
        integration time
    ts : scalar float
        sampling time (cadence)
    t1 : numpy 1d array
        vector of times at which the segment starts
    t2 : numpy 1d array
        vector of times at which the segment ends


    Returns
    -------
    v : numpy array
        vector of values of v (size n_data) calculated at given frequencies

    """

    if f_dot != 0:
        f_shift = f_0 - f
        coeff = np.exp(1j*np.pi/4) * np.sqrt(np.pi / f_dot)
        common_factor = np.exp(1j*np.pi/4 - 1j*np.pi * f_shift**2 / f_dot)
        common_factor = common_factor / (2 * np.sqrt(f_dot))

        erfi_T1 = special.erfi(coeff * (f_dot * t1 + f_shift))
        erfi_T2 = special.erfi(coeff * (f_dot * t2 + f_shift))

        return common_factor * (erfi_T2 - erfi_T1)/ts

    else:
        dT = t2 - t1
        f_shift = f_0 - f
        out = dT * np.exp(np.pi * f_shift * 1j
                          * (t2 + t1)) * np.sinc(f_shift * dT) / ts

        return out
    # return np.conj(v_func_gb(-f, f_0, f_dot, ts, t1, t2))


def v_func_GB_mono(f, f_0, f_dot, ts, T1, T2):

    dT = T2 - T1
    f_shift = f + f_0

    return dT * np.exp(-np.pi*f_shift*1j*(T2+T1))*np.sinc(f_shift*dT) / ts


def integral0(f, T1, T2, ts):
    """

    Itegral of
    0.5 *exp(-2jpif) dt from t = T1 to T2

    """

    dT = T2 - T1
    return - dT/2*np.exp(-1j*np.pi*f*(T1+T2))*np.sinc(f*dT) / ts


def integral1(f, T, tw, ts):
    """

    Itegral of
    0.5 * cos(pi*(t-T)*/tw) exp(-2jpif) dt from t = T to T+tw

    """

    return -1j*tw/4.*np.exp(1j*np.pi*(2*T-tw)*f)*(
        np.sinc(1/2.-f*tw) - np.sinc(1/2.+f*tw)) / ts


def v_func_GB_wind(f, f_0, f_dot, ts, T1, T2, tw):
    """
    function of frequency giving the Fourier transform of W(t)*exp(-j*Phi(t)),
    where Phi(t) is the phase of the gravitatonal wave, computed on a segment of
    data starting at time T1 and ending at time T2, and where W(t) is a
    smooth Tukey windowing function, going to zero at T1 and T2.

    W(t) = 0.5*( 1 - cos(2*pi*(t-T1)/(2*tw))) for T1 <= t < T1+tw
         = 1                                  for T1+tw <= t < T2-tw
         = 0.5( 1 - cos(2*pi*(t-T2+2*tw)/(2*tw)) ) for T2-tw <= t < T2
         = 0                                  otherwise

    Parameters
    ----------
    f : array_like
        vector frequencies (size n_data) where to evalutate the function w(f)
    f_0 : scalar float
        wave frequency
    T : scalar float
        integration time
    ts : scalar float
        sampling time (cadence)
    T1 : numpy 1d array
        vector of times at which the segment starts
    T2 : numpy 1d array
        vector of times at which the segment ends
    tw : scalar float
        window time constant


    Returns
    -------
    v : numpy array
        vector of values of v (size n_data) calculated at given frequencies

    """
    f_shift = f + f_0

    integr = integral0(f, T1, T1+tw, ts) - integral1(f_shift, T1, tw, ts)
    integr += integral0(f_shift, T1+tw, T2-tw, ts) + integral0(f_shift, T2-tw,
                                                               T2, ts)
    integr -= integral1(f_shift, T2, tw, ts)

    return integr
