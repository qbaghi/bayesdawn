#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:40:47 2019

@author: qbaghi
"""

import numpy as np
from scipy import special


def v_func_gb(f, f_0, f_dot, ts, T1, T2):
    """
    function of frequency giving the Fourier transform of exp(-j*Phi(t)),
    where Phi(t) is the phase of the gravitatonal wave, computed on a segment of
    data starting at time T1 and ending at time T2

    Parameters
    ----------
    f : array_like
        vector frequencies (size N) where to evalutate the function w(f)
    f_0 : scalar float
        wave frequency
    f_dot : scalar float
        wave frequency derivative
    T : scalar float
        integration time
    ts : scalar float
        sampling time (cadence)
    T1 : numpy 1d array
        vector of times at which the segment starts
    T2 : numpy 1d array
        vector of times at which the segment ends


    Returns
    -------
    v : numpy array
        vector of values of v (size N) calculated at given frequencies

    """

    if f_dot != 0:
        #return exp_square_fourier_trunc(f+f_0,np.pi*f_dot,T,ts)

        f_shift = f + f_0
        a = np.pi * f_dot
        coeff = np.exp(3j*np.pi/4) / np.sqrt(a)
        common_factor = - np.exp(1j*np.pi/4)/(2*np.sqrt(a))*np.sqrt(np.pi) \
        * np.exp(1j*np.pi**2 * f_shift**2 / a )

        erfi_T1 = special.erfi(coeff * (a*T1+np.pi*f_shift))
        erfi_T2 = special.erfi(coeff * (a*T2+np.pi*f_shift))

        return common_factor*(erfi_T2 - erfi_T1)/ts

    else:
        
        dT = T2 - T1
        f_shift = f + f_0

        return dT * np.exp(-np.pi*f_shift*1j*(T2+T1))*np.sinc(f_shift*dT) / ts


def v_func_GB_mono(f, f_0, f_dot, ts, T1, T2):

    dT = T2 - T1
    f_shift = f + f_0

    return dT * np.exp(-np.pi*f_shift*1j*(T2+T1))*np.sinc(f_shift*dT) / ts


def I0(f, T1, T2, ts):
    """
    
    Itegral of 
    0.5 *exp(-2jpif) dt from t = T1 to T2
    
    """
    
    dT = T2 - T1
    
    #return -tw/2*np.exp(-1j*np.pi*f*(2*T-tw))*np.sinc(f*tw) / ts
    return - dT/2*np.exp(-1j*np.pi*f*(T1+T2))*np.sinc(f*dT) / ts
    

def I1(f, T, tw, ts):
    """
    
    Itegral of 
    0.5 * cos(pi*(t-T)*/tw) exp(-2jpif) dt from t = T to T+tw
    
    """
    
    return -1j*tw/4.*np.exp(1j*np.pi*(2*T-tw)*f)*( np.sinc(1/2.-f*tw)  \
                  - np.sinc(1/2.+f*tw) ) / ts
    
    
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
        vector frequencies (size N) where to evalutate the function w(f)
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
        vector of values of v (size N) calculated at given frequencies

    """
    f_shift = f + f_0
    
    I =  I0(f,T1,T1+tw,ts) - I1(f_shift,T1,tw,ts) \
    + I0(f_shift,T1+tw,T2-tw,ts) \
    + I0(f_shift,T2-tw,T2,ts) - I1(f_shift,T2,tw,ts)
    
    return I