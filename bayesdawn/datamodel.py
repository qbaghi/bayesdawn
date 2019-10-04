#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:24:27 2019

@author: qbaghi

This module provide classes to perform missing data imputation steps based on 
Gaussian conditional model
"""


from bayesdawn.gaps import gapgenerator
from numpy import ndarray
import numpy as np
from scipy import signal
import time
from scipy import linalg as LA
# from mecm import noise
import copy
# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft
# import librosa
# from pycbc.filter.qtransform import qtiling, qplane
# from scipy.interpolate import interp2d


# class TimeSeries(np.array):
#
#     def __init__(self, *args):
#
# class NDTimeSeries(ndarray):


class NdTimeSeries(ndarray):

    # def __init__(self, object, del_t=1.0, dtype=None, copy=True, order='K', subok=False, ndmin=0):

    def __init__(self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, del_t=1.0):

        ndarray.__init__(self, shape, dtype=dtype, buffer=buffer, offset=offset, strides=strides, order=order)
        # super().__init__(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

        self.del_t = del_t
        self.fs = 1 / del_t
        self.n = self.shape[0]
        self.tobs = self.n * del_t
        self.t = np.arange(0, self.n) * del_t
        self.f = np.fft.fftfreq(self.n) * self.fs
        # Time windowing
        self.w = 1

    def set_sampling_time(self, del_t):

        self.del_t = del_t
        self.fs = 1 / del_t
        self.n = self.shape[0]
        self.tobs = self.n * del_t
        self.t = np.arange(0, self.n) * del_t
        self.f = np.fft.fftfreq(self.n) * self.fs

    def compute_window(self, wind='tukey', n_wind=500):

        if wind == 'tukey':
            w = signal.tukey(self.n)
        elif wind == 'hanning':
            w = np.hanning(self.n)
        elif wind == 'blackman':
            w = np.blackman(self.n)
        elif wind == 'modified_hann':
            w = gapgenerator.modified_hann(self.n, n_wind=n_wind)
        elif (wind == 'rectangular') | (wind == 'rect'):
            w = np.ones(self.n)

        return w

    def dft(self, wind='tukey', n_wind=500, normalized=True):

        self.w = self.compute_window(wind=wind, n_wind=n_wind)

        if normalized:
            norm = np.sum(self.w) / (self.del_t * 2)
        else:
            norm = 1.0

        return fft(self * self.w) / norm

    def periodogram(self, wind='tukey'):
        """

        Parameters
        ----------
        wind : basestring
            type of time windowing

        Returns
        -------
        freq : numpy array
            frequency vector
        per : numpy array
            periodogram of the time series expressed in A / Hz where A is the unit of x

        """

        w = self.compute_window(wind=wind)
        k2 = np.sum(w ** 2)

        return np.abs(fft(self * w)) ** 2 / (k2 * self.fs)

    def qtransform(self, delta_t=None, delta_f=None, logfsteps=None, frange=None, qrange=(4,64), mismatch=0.2,
                   return_complex=False):
        """ Return the interpolated 2d qtransform of this data

        FROM PYCBC

        Parameters
        ----------
        delta_t : {self.delta_t, float}
            The time resolution to interpolate to
        delta_f : float, Optional
            The frequency resolution to interpolate to
        logfsteps : int
            Do a log interpolation (incompatible with delta_f option) and set
            the number of steps to take.
        frange : {(30, nyquist*0.8), tuple of ints}
            frequency range
        qrange : {(4, 64), tuple}
            q range
        mismatch : float
            Mismatch between frequency tiles
        return_complex: {False, bool}
            return the raw complex series instead of the normalized power.

        Returns
        -------
        times : numpy.ndarray
            The time that the qtransform is sampled.
        freqs : numpy.ndarray
            The frequencies that the qtransform is sampled.
        qplane : numpy.ndarray (2d)
            The two dimensional interpolated qtransform of this time series.
        """

        z = np.abs(librosa.cqt(self, sr=1/self.fs, fmin=1/self.tobs, hop_length=2**10, window='hann'))

        return z


def TimeSeries(object, del_t=1.0):

    out = object.view(NdTimeSeries) # dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
    out.set_sampling_time(del_t)

    # out = NdTimeSeries(object.shape, dtype=float, buffer=object, offset=0, strides=None, order=None, del_t=del_t)

    return out


class GaussianStationaryProcess(object):
    """

    Implement the (naive) nearest-neighboor method for missing data imputation.


    """
    def __init__(self, y, mask, Na=150, Nb=150):
        """

        Parameters
        ----------
        y : array_lilke
            vector of masked data y = x * M
        mask : array_like
            binary mask
        Na : scalar integer
            number of points to consider before each gap (for the conditional
            distribution of gap data)
        Na : scalar integer
            number of points to consider after each gap
        """

        # Masked data
        self.y = copy.deepcopy(y)
        # The binary mask
        self.mask = mask[:]
        # Total length of the data
        self.N = len(mask)
        # Check whether there are gaps
        if np.any(self.mask == 0):
            # Starting and ending points of gaps
            self.N_starts, self.N_ends = gapgenerator.findEnds(mask)
            gap_lengths = self.N_ends - self.N_starts
            self.N_max = np.int( Na + Nb + np.max(gap_lengths) )
            if self.N_max > 2000:
                raise ValueError("The maximum size of gap + conditional set is too high.")

            self.Na = Na
            self.Nb = Nb
            # Number of gaps
            N_gaps = len(self.N_starts)
            # Indices of missing data
            self.ind_mis = np.where(mask == 0)[0]
            # Indices of observed data
            self.ind_obs = np.where(mask == 1)[0]
        else:
            self.N_starts, self.N_ends = 0, self.N
            self.N_max = 0
            N_gaps = 0
            self.ind_mis = []
            self.ind_obs = np.arange(0, self.N).astype(np.int)

        # Indices of embedding segments around each gap
        # The edges of each segment is set such that there are Na + Nb observed
        # data around, unless another gap is present.
        if N_gaps == 0:
            self.indices = None
            print("Time series does not contain gaps.")

        elif N_gaps == 1:
            # 2 segments
            self.indices = [np.arange(np.int(np.max([self.N_starts[0] - Na, 0])),
                                      np.int(np.min([self.N_ends[0] + Nb, self.N])))]

        elif N_gaps > 1:
            # first segment
            self.indices = [np.arange(np.int(np.max([self.N_starts[0] - Na, 0])), np.int(np.min([self.N_ends[0]+Nb,
                                                                                                 self.N_starts[1]])))]
            # most of the segments
            self.indices = self.indices + [np.arange(np.int(np.max([self.N_starts[j] - Na, self.N_ends[j-1]])),
                                                     np.int(np.min([self.N_ends[j]+Nb, self.N_starts[j+1]])))
                                           for j in range(1, N_gaps-1)]
            # last segment
            self.indices = self.indices + [np.arange(np.int(np.max([self.N_starts[N_gaps-1] - Na,
                                                                    self.N_ends[N_gaps-2]])),
                                                     np.int(np.min([self.N_ends[N_gaps-1]+Nb, self.N])))]
        # self.indices = [np.arange(np.int(np.max([self.N_starts[j]- Na,0])),
        # np.int(np.min([self.N_ends[j]+Nb,N]))) for j in range(len(self.N_starts))]

        # The imputation method
        if type(y) == np.ndarray:
            # If there is only one single channel
            if N_gaps > 0:
                self.impute = self.draw_missing_data
            else:
                self.impute = lambda y_model, psd: self.y
            self.dft = lambda x, w: fft(x * w)

        elif type(y) == list:
            # If there are several
            if N_gaps > 0:
                self.impute = lambda y_model_list, psd_list: [self.draw_missing_data(y_model_list[i], psd_list[i])
                                                              for i in range(len(y_model_list))]
            else:
                self.impute = lambda y_model_list, psd_list: self.y

            self.dft = lambda x_list, w: [fft(x * w) for x in x_list]


    def conditionalDraw(self, z_o, S_2N, CooI, C_mo, ind_obs, ind_mis, mask, C):
        """
        Function performing random draws of the complete data noise vector
        conditionnaly on the observed data.

        Parameters
        ----------
        z_o : numpy array
            vector of observed residuals (size No)
        S_2N : numpy array (size P >= 2N)
            PSD vector
        CooI : 2d numpy array
            Inverse of covariance matrix of observed data
        C_mo : 2d numpy array
            Matrix of covariance between missing data with observed data
        ind_obs : array_like (size No)
            vector of chronological indices of the observed data points in the
            complete data vector
        ind_mis : array_like (size No)
            vector of chronological indices of the missing data points in the
            complete data vector
        mask : numpy array (size N)
            mask vector (with entries equal to 0 or 1)

        Returns
        -------
        eps : numpy array (size Nm)
            realization of the vector of missing noise given the observed data


        References
        ----------
        J. Stroud et al, Bayesian and Maximum Likelihood Estimation for Gaussian
        Processes on an Incomplete Lattice, 2014


        """

        # the size of the vector that is randomly drawn is
        # equal to the size of the mask.
        # e = np.real(noise.generateNoiseFromDSP(np.sqrt(S_2N*2.), 1.)[0:M.shape[0]])
        e = np.random.multivariate_normal(np.zeros(mask.shape[0]), C[0:mask.shape[0], 0:mask.shape[0]])

        # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
        eps = e[ind_mis] + C_mo.dot(CooI.dot(z_o - e[ind_obs]))

        return eps

    def draw_missing_data(self, y_model, psd):
        """

        Draw the missing data from their conditional distributions on the
        observed data

        Parameters
        ----------
        y_model : array_like
            vector of modelled signal (size N)
        psd : PSD_spline instance
            class to compute the noise PSD


        Returns
        -------
        y_rec : array_like
            realization of the full data vector conditionnally to the observed
            data

        """
        # Compute the autocovariance from the full PSD and restrict it to N_max
        # points
        t1 = time.time()
        autocorr = psd.calculate_autocorr(self.N)[0:self.N_max]
        t2 = time.time()
        print("Computation of autocovariance took " + str(t2-t1))
        # Compute the spectrum on 2*N_max points
        s2 = psd.calculate(2*self.N_max)
        t1 = time.time()
        # Impute the missing data: estimation of missing residuals
        y_mis_res = self.imputation(self.y - y_model, autocorr, s2)
        # Construct the full imputed data vector
        # at observed value this is the same
        y_rec = np.zeros(self.N, dtype=np.float64)
        y_rec[self.ind_obs] = self.y[self.ind_obs]
        y_rec[self.ind_mis] = y_mis_res + y_model[self.ind_mis]
        t2 = time.time()
        print("Missing data imputation took " + str(t2-t1))

        return y_rec

    def imputation(self, y, R, S2):
        """

        Nearest neighboor imputation

        Parameters
        ----------
        y : array_like
            observed residuals
        R : array_like
            autocovariance function until lag N_max
        S2 : array_like
            values of the noise spectrum calculated on a Fourier grid of size
            2 N_max

        Returns
        -------
        y_mis : 1d numpy array
            imputed missing value

        """

        C = LA.toeplitz(R)
        # ======================================================================
        # For precomputations
        # ======================================================================
        #indj_obs = np.where(M[indices[0]]==1)[0]
        #indj_mis = np.where(M[indices[0]]==0)[0]
        #y_mis = np.array([]) #np.zeros(len(ind_mis),dtype = np.float64)
        #C_oo = C[np.ix_(indj_obs,indj_obs)]
        #CooI = LA.inv(C_oo)

        # ======================================================================
        # Gap per gap imputation
        # ======================================================================
        results = [self.single_imputation(y[indj], self.mask[indj], C, S2) for indj in self.indices]
        y_mis = np.concatenate(results)
        # y_rec = np.zeros(N, dtype = np.float64)
        # y_rec[self.ind_obs] = y[self.ind_obs]
        # y_rec[self.ind_mis] = y_mis

        #return y_rec
        return y_mis

    def single_imputation(self, yj, maskj, C, S_2N):
        """

        Sample the missing data distribution conditionally on the observed data

        """
        # Local indices of missing and observed data
        ind_obsj = np.where(maskj == 1)[0]
        ind_misj = np.where(maskj == 0)[0]

        C_mo = C[np.ix_(ind_misj, ind_obsj)]
        #C_mm = C[np.ix_(ind_misj,ind_misj)]
        CooI = LA.inv(C[np.ix_(ind_obsj, ind_obsj)])

        out = self.conditionalDraw(yj[ind_obsj], S_2N, CooI, C_mo, ind_obsj, ind_misj, maskj, C)
        #out = conditionalDraw2(yj[ind_obsj],C_mm,C_mo,CooI)

        return out



