# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:24:27 2019

@author: qbaghi

This module provide classes to perform missing data imputation steps based on
Gaussian conditional model
"""

from .algebra import matrixalgebra, fastoeplitz
from .gaps import gapgenerator, operators
from numpy import ndarray
import numpy as np
from scipy import signal
import time
from scipy import linalg
import copy
import warnings
# FTT modules
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
pyfftw.interfaces.cache.enable()

# import librosa
# from pycbc.filter.qtransform import qtiling, qplane
# from scipy.interpolate import interp2d


# class time_series(np.array):
#
#     def __init__(self, *args):
#
# class NDTimeSeries(ndarray):
def generate_freq_noise_from_psd(psd, fe, myseed=None):
    """
    Generate noise in the frequency domain from the values of the DSP.
    """


    """
    Function generating a colored noise from a vector containing the DSP.
    The DSP contains Np points such that Np > 2N and the output noise should
    only contain N points in order to avoid boundary effects. However, the
    output is a 2N vector containing all the generated data. The troncature
    should be done afterwards.

    References : Timmer & König, "On generating power law noise", 1995

    Parameters
    ----------
    DSP : array_like
        vector of size N_DSP continaing the noise DSP calculated at frequencies
        between -fe/N_DSP and fe/N_DSP where fe is the sampling frequency and N
        is the size of the time series (it will be the size of the returned
        temporal noise vector b)
    N : scalar integer
        Size of the output time series
    fe : scalar float
        sampling frequency
    myseed : scalar integer or None
        seed of the random number generator

    Returns
    -------
        bf : numpy array
        frequency sample of the colored noise (size N)
    """

    # Size of the DSP
    n_psd = len(psd)
    # Initialize seed for generating random numbers
    np.random.seed(myseed)

    n_fft = np.int((n_psd-1)/2)
    # Real part of the Noise fft : it is a gaussian random variable
    noise_ft_real = np.sqrt(0.5)*psd[0:n_fft+1]*np.random.normal(loc=0.0, 
                                                                 scale=1.0, 
                                                                 size=n_fft+1) 
    # Imaginary part of the Noise fft :
    noise_ft_imag = np.sqrt(0.5)*psd[0:n_fft+1]*np.random.normal(loc=0.0, 
                                                               scale=1.0, 
                                                               size=n_fft+1)
    # The Fourier transform must be real in f = 0
    noise_ft_imag[0] = 0.
    noise_ft_real[0] = noise_ft_real[0]*np.sqrt(2.)

    # Create the NoiseTF complex numbers for positive frequencies
    Noise_TF = noise_ft_real + 1j*noise_ft_imag

    # To get a real valued signal we must have NoiseTF(-f) = NoiseTF*
    if n_psd % 2 == 0 :
        # The TF at Nyquist frequency must be real in the case of an even number of data
        Noise_sym0 = np.array([ psd[n_fft+1]*np.random.normal(0,1) ])
        # Add the symmetric part corresponding to negative frequencies
        Noise_TF = np.hstack( (Noise_TF, Noise_sym0, np.conj(Noise_TF[1:n_fft+1])[::-1]) )

    else :

        # Noise_TF = np.hstack( (Noise_TF, Noise_sym[::-1]) )
        Noise_TF = np.hstack( (Noise_TF, np.conj(Noise_TF[1:n_fft+1])[::-1]) )

    return np.sqrt(n_psd*fe/2.) * Noise_TF


def generate_noise_from_psd(DSP, fe, myseed=None) :
    """
    Function generating a colored noise from a vector containing the DSP.
    The DSP contains Np points such that Np > 2N and the output noise should
    only contain N points in order to avoid boundary effects. However, the
    output is a 2N vector containing all the generated data. The troncature
    should be done afterwards.

    References : Timmer & König, "On generating power law noise", 1995

    Parameters
    ----------
    DSP : array_like
        vector of size N_DSP continaing the noise DSP calculated at frequencies
        between -fe/N_DSP and fe/N_DSP where fe is the sampling frequency and N
        is the size of the time series (it will be the size of the returned
        temporal noise vector b)
    N : scalar integer
        Size of the output time series
    fe : scalar float
        sampling frequency
    myseed : scalar integer or None
        seed of the random number generator

    Returns
    -------
        b : numpy array
        time sample of the colored noise (size N)
    """

    return ifft(generate_freq_noise_from_psd(DSP,fe,myseed = myseed))


class NdTimeSeries(ndarray):

    # def __init__(self, object, del_t=1.0, dtype=None, copy=True, order='K', subok=False, ndmin=0):

    def __init__(self, shape, dtype=float, buffer=None, offset=0, strides=None,
                 order=None, del_t=1.0):

        ndarray.__init__(self, shape, dtype=dtype, buffer=buffer,
                         offset=offset, strides=strides, order=order)
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
            periodogram of the time series expressed in a_mat / Hz where a_mat is the
            unit of x

        """

        w = self.compute_window(wind=wind)
        k2 = np.sum(w ** 2)

        return np.abs(fft(self * w)) ** 2 / (k2 * self.fs)

    def qtransform(self, delta_t=None, delta_f=None, logfsteps=None,
                   frange=None, qrange=(4,64), mismatch=0.2,
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

        z = np.abs(librosa.cqt(self, sr=1/self.fs, fmin=1/self.tobs,
                               hop_length=2**10, window='hann'))

        return z


def time_series(object, del_t=1.0):
    """
    Transform a ndarray into a TimeSeries object


    Parameters
    ----------
    object
    del_t

    Returns
    -------

    """

    out = object.view(NdTimeSeries)
    # dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
    out.set_sampling_time(del_t)
    # out = NdTimeSeries(object.shape, dtype=float, buffer=object, offset=0,
    # strides=None, order=None, del_t=del_t)

    return out


def cm_direct(s2, c, ind_mis, ind_obs, mask):

    c_mo = c[np.ix_(ind_mis, ind_obs)]

    return lambda v: c_mo.dot(v)


def cm_fft(s2, c, ind_mis, ind_obs, mask):

    return lambda v: matrixalgebra.mat_vect_prod(v, ind_obs, ind_mis, mask, s2)


def toeplitz(r, inds):

    ix, iy = np.meshgrid(inds, inds)

    indx = np.abs(ix - iy)

    return np.vstack([r[indx[i, :]] for i in range(indx.shape[0])])
    # c = np.asarray(c).ravel()
    # if r is None:
    #     r = c.conjugate()
    # else:
    #     r = np.asarray(r).ravel()
    # # Form a 1D array of values to be used in the matrix, containing a reversed
    # # copy of r[1:], followed by c.
    # vals = np.concatenate((r[-1:0:-1], c))
    # a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
    # indx = a + b
    # # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
    # # that `vals[indx]` is the Toeplitz matrix.
    # return vals[indx]


class GaussianStationaryProcess(object):
    """

    Implement the (naive) nearest-neighboor method for missing data imputation.


    """

    def __init__(self, y_mean, mask, psd_cls,
                 method='nearest', na=150, nb=150, p=60,
                 tol=1e-6, n_it_max=150):
        """

        Parameters
        ----------
        y_mean : array_like
            mean vector of the Gaussian process, size n
        mask : array_like
            binary mask
        psd_cls : psdmodel.PSD instance
            power spectral density class
        method : str
            method to use to perform imputation. 'nearest': nearest neighboors,
            approximate method.
        na : scalar integer
            number of points to consider before each gap (for the conditional
            distribution of gap data)
        na : scalar integer
            number of points to consider after each gap
        p : int
            number of points to keep before truncation for the preconditionner
            (only if 'PCG' method is chosen)
        """

        # Masked data
        self.y_mean = copy.deepcopy(y_mean)
        # The binary mask
        self.mask = copy.deepcopy(mask)
        # The PSD
        self.psd_cls = copy.deepcopy(psd_cls)
        # Total length of the data
        self.n = len(mask)
        # Imputation method
        self.method = method
        # Tappering number for sparse approximation of the covariance
        self.p = p
        # Error tolerance to reach to end PCG algorithm iterations
        self.tol = tol
        # Maximum number of iterations for the PCG algorithm
        self.n_it_max = n_it_max
        # Check whether there are gaps
        if np.any(self.mask == 0):
            # Starting and ending points of gaps
            self.n_starts, self.n_ends = gapgenerator.find_ends(mask)
            gap_lengths = self.n_ends - self.n_starts
            self.n_max = np.int(na + nb + np.max(gap_lengths))
            self.na = na
            self.nb = nb
            # Number of gaps
            self.n_gaps = len(self.n_starts)
            # Indices of missing data
            self.ind_mis = np.where(mask == 0)[0]
            # Indices of observed data
            self.ind_obs = np.where(mask == 1)[0]
        else:
            self.n_starts, self.n_ends = 0, self.n
            self.n_max = 0
            self.n_gaps = 0
            self.ind_mis = []
            self.ind_obs = np.arange(0, self.n).astype(np.int)
        # If the method is exact (not nearest neighboors) you need the full autocovariance
        if self.method != 'nearest':
            self.n_max = len(mask)
        else:
            if self.n_max > 2000:
                warnings.warn("The maximum size of gap + conditional is high.", UserWarning)

        # Indices of embedding segments around each gap
        # The edges of each segment is set such that there are Na + Nb observed
        # data around, unless another gap is present.
        if self.n_gaps == 0:
            self.indices = None
            print("Time series does not contain gaps.")

        elif self.n_gaps == 1:
            # 2 segments
            self.indices = [np.arange(np.int(np.max([self.n_starts[0] - na, 0])),
                                      np.int(np.min([self.n_ends[0] + nb, self.n])))]

        elif self.n_gaps > 1:
            # first segment
            self.indices = [np.arange(np.int(np.max([self.n_starts[0] - na, 0])),
                                       np.int(np.min([self.n_ends[0] + nb,
                                                      self.n_starts[1]])))]
            # most of the segments
            self.indices = self.indices + [
                np.arange(
                    np.int(np.max([self.n_starts[j] - na, self.n_ends[j - 1]])),
                    np.int(np.min([self.n_ends[j] + nb, self.n_starts[j + 1]])))
                for j in range(1, self.n_gaps-1)]
            # last segment
            self.indices = self.indices + [
                np.arange(
                    np.int(np.max([self.n_starts[self.n_gaps - 1] - na,
                                   self.n_ends[self.n_gaps - 2]])),
                    np.int(np.min([self.n_ends[self.n_gaps - 1] + nb, self.n])))]

        # ==
        # Store quantities that can be computed offline
        # ==
        
        # Autocovariance
        self.autocorr = None
        # Power spectral density computed on a frequency grid of size 2n
        self.s2 = None
        # Preconditionner for PCG or tapered methods
        self.solve = None
        # Inverted matrix for woodbury method
        self.sig_inv_mm_inv = None
        self.w_m_cls = None
        self.a = None
        self.lamdba_n = None

        
    def update_psd(self, psd_cls):
        """
        Update the PSD class of the Gaussian stationary process

        Parameters
        ----------
        psd_cls : psdmodel.PSD instance
            New PSD class
        """
        
        self.psd_cls = copy.deepcopy(psd_cls)
        
    def update_mean(self, y_mean):
        """
        Update the mean vector of the Gaussian stationary process

        Parameters
        ----------
        y_mean : ndarray or list
            Mean vector (deterministic part).
        """
        
        self.y_mean = y_mean[:]
        
    def compute_offline(self):
        """
        Performs all necessary offline computations that depend on PSD and 
        mean vector.
        """
        
        # Compute the autocovariance from the full PSD and restrict it to N_max
        # points
        if type(self.psd_cls) != list:
            t1 = time.time()
            self.autocorr = self.psd_cls.calculate_autocorr(self.n)[0:self.n_max]
            t2 = time.time()
            # Compute the spectrum on 2*N_max points
            self.s2 = self.psd_cls.calculate(2 * self.n_max)
        else:
            t1 = time.time()
            self.autocorr = [psd.calculate_autocorr(self.n)[0:self.n_max]
                             for psd in self.psd_cls]
            t2 = time.time()
            self.s2 = [psd.calculate(2 * self.n_max) for psd in self.psd_cls]
        print("Computation of autocovariance + PSD took " + str(t2-t1))
        
        if self.method == 'woodbury':
            if len(self.ind_mis) < 1e3:
                print("Start Toeplitz system precomputations...")
                self.w_m_cls = operators.MappingOperator(self.ind_mis, self.n)
                w_m = self.w_m_cls.build_matrix(sp=False)
                s_n = self.psd_cls.calculate(self.n_max)
                sigma_inv_wmt = ifft(fft(w_m.T, axis=0) / np.array([s_n]).T, axis=0)
                # Precompute quantities for calculating the inverse of Sigma
                self.lambda_n, self.a = fastoeplitz.teopltiz_precompute(
                    self.autocorr,  p=self.p, nit=self.n_it_max, tol=self.tol)
                # sigma_inv_wmt = fastoeplitz.multiple_toepltiz_inverse(
                #     w_m.T, self.lambda_n, self.a)
                self.sig_inv_mm_inv = linalg.pinv(w_m.dot(sigma_inv_wmt))
            else:
                msg = "Number of missing data is too large for woodbury method."
                raise ValueError(msg)
        
    def compute_preconditioner(self):
        """
        Precompute the pre-conditioner operator that looks like Coo

        """

        # Precompute solver if necessary
        if (self.method == 'PCG') | (self.method == 'tapered'):
            print("Build preconditionner...")
            if type(self.autocorr) != list:
                self.solve = matrixalgebra.compute_precond(self.autocorr, 
                                                           self.mask, 
                                                           p=self.p,
                                                           taper='Wendland2')
            else:
                self.solve = [matrixalgebra.compute_precond(autocorr, 
                                                            self.mask, 
                                                            p=self.p,
                                                            taper='Wendland2')
                              for autocorr in self.autocorr]              
            print("Preconditionner built.")

    def impute(self, y):
        """

        Parameters
        ----------
        y : ndarray or list
            masked data vector, size n

        Returns
        -------
        y_rec : array_like
            realization of the full data vector conditionnally to the observed data

        """

        # If there is only one single channel
        if self.n_gaps > 0:
            return self.draw_missing_data(y)
        else:
            return y

    def conditional_draw(self, z_o, psd_2n, c_oo_inv, c_mo, 
                         ind_obs, ind_mis, mask, c):
        """
        Function performing random draws of the complete data noise vector
        conditionnaly on the observed data.

        Parameters
        ----------
        z_o : numpy array
            vector of observed residuals (size No)
        psd_2n : numpy array (size P >= 2N)
            PSD vector
        c_oo_inv : 2d numpy array
            Inverse of covariance matrix of observed data
        c_mo : callable
            function computing the product of Matrix of covariance between 
            missing data with observed data with any vector: Cmo.x
        ind_obs : array_like (size No)
            vector of chronological indices of the observed data points in the
            complete data vector
        ind_mis : array_like (size No)
            vector of chronological indices of the missing data points in the
            complete data vector
        mask : numpy array (size n_data)
            mask vector (with entries equal to 0 or 1)

        Returns
        -------
        eps : numpy array (size Nm)
            realization of the vector of missing noise given the observed data


        References
        ----------
        n_knots. Stroud et al, Bayesian and Maximum Likelihood Estimation for Gaussian
        Processes on an Incomplete Lattice, 2014


        """

        # the size of the vector that is randomly drawn is
        # equal to the size of the mask.
        # e = np.real(noise.generateNoiseFromDSP(np.sqrt(psd_2n*2.), 1.)[0:mask.shape[0]])
        e = np.random.multivariate_normal(np.zeros(mask.shape[0]), 
                                          c[0:mask.shape[0], 0:mask.shape[0]])

        # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
        eps = e[ind_mis] + c_mo.dot(c_oo_inv.dot(z_o - e[ind_obs]))

        return eps

    def conditional_draw_fast(self, z_o, psd_2n, c_oo_inv, c_mo, 
                              ind_obs, ind_mis, mask):

        e = np.real(generate_noise_from_psd(np.sqrt(psd_2n*2.), 1.)[0:mask.shape[0]])

        # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
        eps = e[ind_mis] + c_mo(c_oo_inv.dot(z_o - e[ind_obs]))

        return eps

    def draw_missing_data(self, y):
        """

        Draw the missing data from their conditional distributions on the
        observed data

        Parameters
        ----------
        y : ndarray or list
            masked data y = mask * x

        Returns
        -------
        y_rec : array_like
            realization of the full data vector conditionnally to the observed
            data

        """

        if self.autocorr is None:
            self.compute_offline()
        if ((self.method == 'PCG') | (self.method == 'tapered')) & (self.solve is None):
            self.compute_preconditioner()
        # If there is only one array
        if type(y) == np.ndarray:
            t1 = time.time()
            # Impute the missing data: estimation of missing residuals
            y_mis_res = self.imputation(y - self.y_mean, 
                                        self.autocorr, 
                                        self.s2)
            # Construct the full imputed data vector
            # at observed value this is the same
            y_rec = copy.deepcopy(y)
            y_rec[self.ind_mis] = y_mis_res + self.y_mean[self.ind_mis]
            t2 = time.time()
            print("Missing data imputation took " + str(t2-t1))
            
        elif type(y) == list:
            
            y_mis_res = [self.imputation(y[i] - self.y_mean[i], 
                                         self.autocorr[i], self.s2[i]) 
                         for i in range(len(y))]
            y_rec = copy.deepcopy(y)
            
            for i in range(len(y)):
                y_rec[i][self.ind_mis] = y_mis_res[i] + self.y_mean[i][self.ind_mis]
            
        return y_rec

    def imputation(self, y, r, s2):
        """

        Nearest neighboor imputation

        Parameters
        ----------
        y : array_like
            masked residuals (size n_data)
        r : array_like
            autocovariance function until lag N_max
        s2 : array_like
            values of the noise spectrum calculated on a Fourier grid of size
            2 N_max

        Returns
        -------
        y_mis : 1d numpy array
            imputed missing value

        """

        if self.method == 'nearest':
            # =================================================================
            # Gap per gap imputation
            # =================================================================
            if self.n_max <= 2000:
                
                c = linalg.toeplitz(r)
                
                results = [self.single_imputation(y[indj], self.mask[indj], c,
                                                  s2) 
                           for indj in self.indices]
            else:
                # If the number of points inside the gaps is too large, use a
                # FFT-based method
                results = [self.single_imputation_fast(y[indj],
                                                       self.mask[indj],
                                                       r,
                                                       s2)
                           for indj in self.indices]
            y_mis = np.concatenate(results)
            # y_rec = np.zeros(n_data, dtype = np.float64)
            # y_rec[self.ind_obs] = y[self.ind_obs]
            # y_rec[self.ind_mis] = y_mis
        elif self.method == 'tapered':
            # Sparse approximation of the covariance
            print("Build preconditionner...")
            self.solve = matrixalgebra.compute_precond(r, 
                                                       self.mask, p=self.p,
                                                       taper='Wendland2')
            print("Preconditionner built.")
            # Approximately solve the linear system C_oo x = eps
            u = self.solve(y[self.ind_obs])
            # Compute the missing data estimate via z | o = Cmo Coo^-1 z_o
            y_mis = matrixalgebra.mat_vect_prod(u, self.ind_obs, self.ind_mis,
                                                self.mask, s2)
        elif self.method == 'PCG':
            # Precompute solver if necessary
            if self.solve is None:
                self.compute_preconditioner(r)
            # First guess
            x0 = np.zeros(len(self.ind_obs))
            # Solve the linear system C_oo x = eps
            u, _ = matrixalgebra.pcg_solve(self.ind_obs, self.mask, s2,
                                           y[self.ind_obs], x0,
                                           self.tol, self.n_it_max,
                                           self.solve,
                                          'scipy')
            # Compute the missing data estimate via z | o = Cmo Coo^-1 z_o
            y_mis = matrixalgebra.mat_vect_prod(u, self.ind_obs, self.ind_mis,
                                                self.mask, s2)
            
        elif self.method == 'woodbury':
            x = self.mask * y
            # Apply inverse sigma
            a_o = fastoeplitz.toepltiz_inverse_jain(x, self.lambda_n, self.a)
            b_o = self.sig_inv_mm_inv.dot(a_o[self.ind_mis])
            y_o = np.zeros(self.n)
            y_o[self.ind_mis] = b_o
            # e_o = a_o - sigma_inv.dot(y_o)
            e_o = a_o - fastoeplitz.toepltiz_inverse_jain(
                y_o, self.lambda_n, self.a)
            # Multiply my the mask and them by the covariance Sigma
            eps_given_o = ifft(fft(self.mask * e_o, 2*self.n_max) * self.s2)
            
            y_mis = eps_given_o[self.ind_mis]

        return y_mis

    def single_imputation(self, yj, maskj, c, psd_2n):
        """
        Sample the missing data distribution conditionally on the observed
        data, using direct brute-force computation.

        Parameters
        ----------
        yj : ndarray
            segment of masked data
        maskj : ndarray
            local mask
        c : ndarray
            covariance matrix of sized nj x nj
        psd_2n : ndarray
            psd computed on a Fourier grid of size 2nj

        Returns
        -------
        out : ndarray
            imputed missing data, of size len(np.where(maskj == 0)[0])

        """

        # Local indices of missing and observed data
        ind_obsj = np.where(maskj == 1)[0]
        ind_misj = np.where(maskj == 0)[0]

        c_mo = c[np.ix_(ind_misj, ind_obsj)]
        #C_mm = C[np.ix_(ind_misj,ind_misj)]
        c_oo_inv = linalg.inv(c[np.ix_(ind_obsj, ind_obsj)])

        out = self.conditional_draw(yj[ind_obsj], psd_2n, c_oo_inv, c_mo,
                                    ind_obsj, ind_misj, maskj, c)
        #out = conditionalDraw2(yj[ind_obsj],C_mm,c_mo,c_oo_inv)

        return out

    def single_imputation_fast(self, yj, maskj, r, psd_2n):
        """
        Sample the missing data distribution conditionally on the observed data, computed usin g

        Parameters
        ----------
        yj : ndarray
            segment of masked data
        maskj : ndarray
            local mask
        r : ndarray
            autocovariance computed until lag n_max
        psd_2n : ndarray
            psd computed on a Fourier grid of size 2nj

        Returns
        -------
        out : ndarray
            imputed missing data, of size len(np.where(maskj == 0)[0])

        """

        # Local indices of missing and observed data
        ind_obsj = np.where(maskj == 1)[0]
        ind_misj = np.where(maskj == 0)[0]
        # Covariance of observed data and its inverse
        c_oo = toeplitz(r, ind_obsj)
        c_oo_inv = linalg.inv(c_oo)
        # Covariance missing / observed data : matrix operator
        c_mo = lambda v: matrixalgebra.mat_vect_prod(v, ind_obsj, ind_misj,
                                                     maskj, psd_2n)

        return self.conditional_draw_fast(yj[ind_obsj], psd_2n, c_oo_inv, c_mo,
                                          ind_obsj, ind_misj, maskj)


class GSP(object):
    
    def __init__(self, mu, cov):
        """
        
        New Gaussian stationary process class

        Parameters
        ----------
        mu : callable
            Mean function (of time)
        cov : bayesdawn.operator.CovarianceOperator instance
            Covariance operator of the Gaussian process
        """
        
        self.mu = mu
        self.cov = cov
        
        