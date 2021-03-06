#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:11:56 2019

@author: qbaghi
"""
import numpy as np

from scipy import linalg as LA
from . import samplers

# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft


def gls(mat_fft, psd, y_fft):
    """
    Generalized least-squares estimate
    Parameters
    ----------
    mat_fft : bytearray
        design matrix in frequency domain
    psd : bytearray
        noise spectrum
    y_fft : bytearray
        discrete-fourier transformed data

    Returns
    -------
    amp : complex
        estimated parameter vector

    """

    mat_fft_weighed = mat_fft / np.array([psd]).T

    return LA.pinv(mat_fft_weighed.conj().T.dot(mat_fft)).dot(mat_fft_weighed.conj().T.dot(y_fft))


class GWModel(object):
    """

    This class provide all the functions necessary to calcualte the likelihood
    of the gravitational wave data model

    """

    def __init__(self,
                 smodel,
                 tobs,
                 del_t,
                 names=['theta', 'phi', 'f_0', 'f_dot'],
                 channels=['X1'],
                 fmin=1e-5,
                 fmax=0.5e-1,
                 nsources=1,
                 order=None,
                 reduced=False):
        """

        Parameters
        ----------
        smodel : instance of waveform.lisaresp.GWwaveform
            GW LISA response model class
        names : list of strings
            parameter names
        timevect : numpy array
            time vector
        channels : list of strings
            type of tdi channels to use
        fmin : scalar float
            mininum frequency where to compute the likelihood
        fmax : scalar float
            maximum frequency where to compute the likelihood
        nsources : scalar integer
            number of considered sources in the analysis
        order : list
            list of integers that specify an inequality constraint on the parameters, that will be reflected in the
            log-prior distribution.
        reduced : bool
            whether to use the full or the reduced likelihood model
        """


        # ======================================================================
        # Initialization of fixed data / parameters
        # ======================================================================
        # Number of GW sources
        self.nsources = nsources
        # Names of parameters
        self.names = names
        # Possible inequality constraint
        self.order = order
        # Total number of parameters
        self.ndim_tot = len(names)
        # Number of parameters per source
        self.ndim = np.int(self.ndim_tot/nsources)
        # Type of TDI channel
        self.channels = channels
        # Type of likelihood model
        self.reduced = reduced

        # ======================================================================
        # Parameters for waveform computation
        # ======================================================================
        self.ts = del_t
        self.fs = 1/self.ts
        self.tobs = tobs
        self.N = np.int(tobs/del_t)
        # Waveform model
        self.smodel = smodel

        # Considered bandwidth for the fit
        self.fmin = fmin
        self.fmax = fmax
        # Frequency vector
        self.f = np.fft.fftfreq(self.N)*self.fs
        # Find corresponding indices in the frequency vector
        self.inds_pos = np.where((self.f >= fmin) & (self.f <= fmax))[0]
        self.inds_neg = self.N - self.inds_pos
        self.inds = np.concatenate((self.inds_pos, self.inds_neg))
        print("The estimation domain has size " + str(len(self.inds_pos)))
        # Number of positive frequencies considered
        self.npos = len(self.inds_pos)
        # Vector of analysed frequencies
        self.f_pos = self.f[self.inds_pos]
        # Way of concatenating data and models depending number of analysed channels
        if len(channels) == 1:
            self.concatenate_model = lambda x: x
            self.concatenate_data_pos = lambda x: x[self.inds_pos]
        elif len(channels) > 1:
            self.concatenate_data_pos = lambda x_list: np.concatenate([x[self.inds_pos] for x in x_list])
            self.concatenate_model = lambda x_list: np.concatenate(x_list)
        else:
            raise ValueError("Please indicate at least one channel")

    def matrix_model(self, params):
        """
    
        function of frequency f, source model parameters params describing the features of the analyzed data
    
        Parameters
        ----------
        params : bytearray
            vector of waveform parameters
        Returns
        -------
        mat : bytearray
            design matrix in frequency domain

    
        """

        return self.smodel.design_matrix_freq(self.f_pos, params, self.ts, self.tobs, channel=self.channels, complex=True)


    def compute_frequency_signal(self, params):

        y_gw_fft = np.zeros(self.N, dtype=np.complex128)

        y_gw_fft[self.inds_pos] = self.smodel.compute_signal_freq(self.f_pos, params, self.ts, self.tobs,
                                                                  channel=self.channels)
        y_gw_fft[self.inds_neg] = np.conj(y_gw_fft[self.inds_pos])

        return y_gw_fft

    def compute_time_signal(self, params):

        y_gw_fft = self.compute_frequency_signal(params)

        return np.real(ifft(y_gw_fft))

    def log_likelihood(self, params, spectrum, y_fft):
        """
        Logarithm of the likelihood, optimized for FREQUENCY domain computations, depending on all parameters

        Parameters
        ----------
        params : array_like
            vector of parameters
        spectrum : array_like
            noise spectrum, equal to fs * PSD / 2
        y_fft : array_like
            discrete fourier transform of the data


        Returns
        -------
        logL : scalar float
            logarithm of the likelihood

        """

        # Update the frequency domain residuals in the case of intrinsic parameters
        # Stack the channel data DFTs
        y_fft_stack = self.concatenate_data_pos(y_fft)
        # For a full likelihood model
        if not self.reduced:
            # Stack the computed waveform DFTs
            s_fft_stack = self.concatenate_model(self.smodel.compute_signal_freq(self.f_pos, params, self.ts, self.tobs,
                                                                                 channel='TDIAET', ldc=False, tref=0))
        # For a reduced likelihood model
        else:
            # Compute design matrices
            mat_list = self.matrix_model(params)

            # Compute extrinsinc amplitudes for each channel
            amplitudes = [gls(mat_list[i], spectrum[i][self.inds_pos], y_fft[i][self.inds_pos])
                          for i in range(len(mat_list))]

            # Stack the modeled signals
            s_fft_stack = self.concatenate_model([mat_list[i].dot(amplitudes[i]) for i in range(len(mat_list))])
            # # Data with real and imaginary part separated
            # yr = np.concatenate((y_fft[self.inds_pos].real, y_fft[self.inds_pos].imag))
            # N_bar = mat_freq_w.conj().T.dot(yr)
            #
            # return np.real(N_bar.conj().T.dot(ZI.dot(N_bar)))

        # Compute periodogram for relevant frequencies
        per_inds = np.abs(y_fft_stack - s_fft_stack)**2/self.N

        # Update reduced likelihood
        # return np.real(-0.5*np.sum(np.log(S[self.inds_pos]) + I_inds/S[self.inds_pos]))
        return np.real(-0.5 * np.sum(per_inds / self.concatenate_data_pos(spectrum)))

    def draw_single_freq_signal_onBW(self, params, psd, y_fft):
        """
        Compute deterministic signal model on the restricted bandwidth,
        for one single source only
        """
        # Update design matrix and derived quantities
        a_freq, a_freq_w, zi = self.compute_matrices(params, psd)

        #ZI = alfastfreq.compute_inverse_normal(mat_freq,ApS)
        beta = self.draw_beta(zi, a_freq_w, y_fft)

        # Return the frequency domain GW signal
        #return np.dot(mat_freq,beta)   
        y_gw_fft_2 = np.dot(a_freq, beta)

        return y_gw_fft_2[0:self.npos] + 1j*y_gw_fft_2[self.npos:]

    def draw_frequency_signal_onBW(self, params, psd, y_fft):
        """
        Compute deterministic signal model on the restricted bandwidth only
        """

        if self.nsources > 0:
            y_gw_fft = self.draw_single_freq_signal_onBW(params, psd, y_fft)
        else:
            y_gw_fft = np.zeros(self.npos, dtype=np.complex128)
                
        # Return the frequency domain GW signal
        return y_gw_fft     

    def draw_frequency_signal(self, params, psd, y_fft):
        """
        Compute deterministic signal model on the full Fourier grid
        """

        y_gw_fft = np.zeros(self.N, dtype=np.complex128)

        y_gw_fft[self.inds_pos] = self.draw_frequency_signal_onBW(params, psd, y_fft)
        y_gw_fft[self.inds_neg] = np.conj(y_gw_fft[self.inds_pos])

        return y_gw_fft

    def compute_matrices(self, params, s):
        """

        Compute the design matrix, its weighted version, and the inverse normal
        matrix

        """
        if self.nsources == 1:
            # Update frequency model of the waveform
            #A_computed = self.matmodel(self.f[self.inds_pos],params,self)
            #mat_freq = np.vstack((A_computed,np.conj(A_computed)))
            mat_freq = self.matrix_model(params)

        elif self.nsources > 1:
            
            mat_freq = np.hstack([self.matrix_model(params[i*self.ndim:(i+1)*self.ndim]) for i in range(self.nsources)])
        
        # Weight matrix columns by the inverse of the PSD
        #mat_freq_w = np.array([mat_freq[:,j]/S[self.inds] for j in range(mat_freq.shape[1])]).T
        s2 = np.concatenate((s[self.inds_pos], s[self.inds_pos]))
        mat_freq_w = np.array([mat_freq[:, j]/s2 for j in range(mat_freq.shape[1])]).T
        
        # Inverse normal matrix
        ZI = LA.pinv(np.dot(np.transpose(mat_freq).conj(), mat_freq_w))

        return mat_freq, mat_freq_w, ZI

    def ls_estimate_beta(self, ZI, ApS, data_fft):
        """
        Generalized least-square estimate of the extrinsic parameters
        """
        
        data_real = np.concatenate((data_fft[self.inds_pos].real,
                                    data_fft[self.inds_pos].imag))
        #return np.real(np.dot( ZI, np.dot(np.transpose(ApS).conj(),data_fft[self.inds])))
        return np.dot(ZI, np.dot(np.transpose(ApS).conj(), data_real))
        #return np.dot( ZI, np.dot(np.transpose(ApS).conj(),data_fft[self.inds_pos]))

    def draw_beta(self, ZI, ApS, data_fft):
        """
        Draw the GW amplitude parameter vector beta from its conditional distribution

        Parameters
        ----------
        ZI : 2d numpy array
            inverse weighted normal matrix (A^* C^-1 A )^-1
        ApS : 2d numpy array
            weighted design matrix
        data_fft : array_like
            discrete fourier transform of data

        Returns
        -------
        beta : 1d numpy array
            random drawn of the conditional distribution of beta

        """
        return np.random.multivariate_normal(self.ls_estimate_beta(ZI, ApS, data_fft), ZI)
        #return np.real(np.dot( ZI, np.dot(np.transpose(ApS).conj(),data_fft[self.inds])))

    def compute_time_signal(self, y_gw_fft):
        """

        Compute the GW signal in the time domain given the
        Fourier-domain model in the relevant bandwidth.


        Parameters
        ----------
        y_gw_fft : array_like
            estimated GW signal in the frequency domain, in the Fourier grid

        Returns
        -------
        y_model : array_like
            estimated GW signal in the time domain


        """
        #y_gw_fft = np.zeros(self.N, dtype = np.complex128)
        #y_gw_fft[self.inds] = y_gw_fft_inds

        return np.real(ifft(y_gw_fft))



