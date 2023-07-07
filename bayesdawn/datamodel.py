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


def generate_freq_noise_from_psd(psd, fs, myseed=None):
    """
    Generate noise in the frequency domain from the values of the DSP.
    """


    """
    Function generating a colored noise from a vector containing the DSP.
    The PSD contains Np points such that Np > 2N and the output noise should
    only contain N points in order to avoid boundary effects. However, the
    output is a 2N vector containing all the generated data. The troncature
    should be done afterwards.

    References : Timmer & König, "On generating power law noise", 1995

    Parameters
    ----------
    psd : array_like
        vector of size N_DSP continaing the noise one-sided PSD calculated at frequencies
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
    noise_ft_real = np.sqrt(0.5 * psd[0:n_fft+1])*np.random.normal(loc=0.0, 
                                                                 scale=1.0, 
                                                                 size=n_fft+1) 
    # Imaginary part of the Noise fft :
    noise_ft_imag = np.sqrt(0.5 * psd[0:n_fft+1])*np.random.normal(loc=0.0, 
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
        Noise_sym0 = np.array([ np.sqrt(psd[n_fft+1])*np.random.normal(0,1) ])
        # Add the symmetric part corresponding to negative frequencies
        Noise_TF = np.hstack( (Noise_TF, Noise_sym0, np.conj(Noise_TF[1:n_fft+1])[::-1]) )

    else :

        # Noise_TF = np.hstack( (Noise_TF, Noise_sym[::-1]) )
        Noise_TF = np.hstack( (Noise_TF, np.conj(Noise_TF[1:n_fft+1])[::-1]) )

    return np.sqrt(n_psd*fs/2.) * Noise_TF


def generate_noise_from_psd(psd, fs, myseed=None) :
    """
    Function generating a colored noise from a vector containing the DSP.
    The PSD contains Np points such that Np > 2N and the output noise should
    only contain N points in order to avoid boundary effects. However, the
    output is a 2N vector containing all the generated data. The troncature
    should be done afterwards.

    References : Timmer & König, "On generating power law noise", 1995

    Parameters
    ----------
    psd : array_like
        vector of size N_DSP continaing the one-sided 
        noise DSP calculated at frequencies between -fe/N_DSP and fe/N_DSP where 
        fe is the sampling frequency and N is the size of the time series 
        (it will be the size of the returned temporal noise vector b)
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

    return ifft(generate_freq_noise_from_psd(psd, fs, myseed=myseed))


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


class GaussianStationaryProcess(object):
    """

    Implement the (naive) nearest-neighboor method for missing data imputation.


    """

    def __init__(self, y_mean, mask, psd_cls,
                 method='nearest', precond='taper', na=150, nb=150, p=60,
                 tol=1e-6, n_it_max=1000, n_wood_max=5000):
        """

        Parameters
        ----------
        y_mean : array_like
            mean vector of the Gaussian process, size n
        mask : array_like
            binary mask
        psd_cls : psdmodel.PSD instance or callable
            power spectral density class. Should have a method called 
            calculate() that takes a frequency vector as input.
            Alternatively, it can be a function that takes a frequency vector 
            as input.
        method : str
            method to use to perform imputation. 
            'nearest': nearest neighboors, approximate method.
            'PCG': preconjugate gradient, iterative exact method.
            'woodbury': low-rank formulation, non-iterative, exact method.
        precond : str
            Preconditionning methods among {'taper', 'circulant'}.
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
        # Preconditionning method
        self.precond = precond
        # Tappering number for sparse approximation of the covariance
        self.p = p
        # Error tolerance to reach to end PCG algorithm iterations
        self.tol = tol
        # Maximum number of iterations for the PCG algorithm
        self.n_it_max = n_it_max
        # Maximum missing data length accepted by Woodbury method 
        self.n_wood_max = n_wood_max
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
                warnings.warn("The maximum size of gap + conditional is high.", 
                              UserWarning)

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
        #print("Computation of autocovariance + PSD took " + str(t2-t1))        
        
        if self.method == 'woodbury':
            if len(self.ind_mis) <= self.n_wood_max:
                print("Start Toeplitz system precomputations...")
                self.w_m_cls = operators.MappingOperator(self.ind_mis, self.n)
                w_m = self.w_m_cls.build_matrix(sp=False)
                # s_n = self.psd_cls.calculate(self.n_max)
                # sigma_inv_wmt = ifft(fft(w_m.T, axis=0) / np.array([s_n]).T, axis=0)
                if type(self.psd_cls) != list:
                    autocorr = self.autocorr[:]
                else:
                    # Assume same autocovariance for every channel
                    autocorr = self.autocorr[0]
                # Precompute quantities for calculating the inverse of Sigma
                self.lambda_n, self.a = fastoeplitz.teopltiz_precompute(
                    autocorr,  p=self.p, nit=self.n_it_max, tol=self.tol,
                    method='levinson',
                    precond=self.precond)
                sigma_inv_wmt = fastoeplitz.multiple_toepltiz_inverse(
                    w_m.T, self.lambda_n, self.a)
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
            # # For now, use the same preconditionner for all channels           
            # self.solve = matrixalgebra.compute_precond(self.autocorr, 
            #                                             self.mask, 
            #                                             p=self.p,
            #                                             taper='Wendland2') 
            print("Preconditionner built.")

    def impute(self, y, draw=True):
        """

        Draw the missing data from their conditional distributions on the
        observed data. The difference with the draw_missing_data method is that
        it checks whether there are gaps or not. If not, this function is 
        identity.
        
        Parameters
        ----------
        y : ndarray or list
            masked data vector, size n
        draw : bool
            if True (default), the data vector is drawn from the conditional 
            distribution given the observed data. If False, the expectation of 
            the conditional distribution is returned (in that case the output 
            is deterministic, as it does not involved any random number 
            generation.)

        Returns
        -------
        y_rec : array_like
            realization of the full data vector conditionnally to the observed 
            data, or its mean.

        """

        # If there is only one single channel
        if self.n_gaps > 0:
            return self.draw_missing_data(y, draw=draw)
        else:
            return y

    def draw_missing_data(self, y, draw=True):
        """

        Draw the missing data from their conditional distributions on the
        observed data

        Parameters
        ----------
        y : ndarray or list of ndarrays
            masked data y = mask * x. If a list is given, draw as many 
            vectors as there are arrays in the list.
        draw : bool
            if True (default), the data vector is drawn from the conditional 
            distribution given the observed data. If False, the expectation of 
            the conditional distribution is returned (in that case the output 
            is deterministic, as it does not involved any random number 
            generation.)

        Returns
        -------
        y_rec : array_like
            realization of the full data vector conditionnally to the observed
            data

        """

        if self.autocorr is None:
            #print('recomputing offline elements')
            self.compute_offline()
        if ((self.method == 'PCG') | (self.method == 'tapered')) & (self.solve is None):
            self.compute_preconditioner()
            #print('recomputing preconditioner')
        # If there is only one array
        if type(y) == np.ndarray:
            #t1 = time.time()
            # Impute the missing data: estimation of missing residuals
            y_mis_res = self.imputation(y - self.y_mean, 
                                        self.autocorr, 
                                        self.s2,
                                        solve=self.solve,
                                        draw=draw)
            # Construct the full imputed data vector
            # at observed value this is the same
            y_rec = copy.deepcopy(y)
            y_rec[self.ind_mis] = y_mis_res + self.y_mean[self.ind_mis]
            #t2 = time.time()
            #print("Missing data imputation took " + str(t2-t1))
            
        elif type(y) == list:
            
            y_mis_res = [self.imputation(y[i] - self.y_mean[i], 
                                         self.autocorr[i], self.s2[i],
                                         solve=self.solve[i], draw=draw) 
                         for i in range(len(y))]
            y_rec = copy.deepcopy(y)
            
            for i in range(len(y)):
                y_rec[i][self.ind_mis] = y_mis_res[i] + self.y_mean[i][self.ind_mis]
                
        else:
            raise ValueError("Unknown input type for y")
            
        return y_rec
    
    def apply_coo_inv(self, z_o, s2, solve=None):
        """

        Operator performing the product Coo^{-1} z on any vector z

        Parameters
        ----------
        z_o : array_like
            vector of size n_obs
        s2 : array_like
            One-sided PSD values calculated on a Fourier grid of size 2 N_max
            WARNING: used to be S(f) * fs / 2. Now the normalization is done
            inside the function.
        solve : linear operator
            preconditionner

        Returns
        -------
        x : 1d numpy array
            vector of size n_obs, such that x = Coo^{-1} z

        """
        
        # Compute the DFT covariances from the one-sided PSD
        # The actual covariance is npoints x S(f) * fs / 2 but the factor
        # of npoints is already accounted for in the IFFT normalization
        cov_2n = s2 * self.psd_cls.fs / 2.0

        if self.method == 'tapered':
            # Approximately solve the linear system C_oo x = eps
            x = solve(z_o)
        elif self.method == 'PCG':
            # Precompute solver if necessary
            if solve is None:
                # self.compute_preconditioner(r)
                raise ValueError("Please provide preconditionning operator")
            # First guess
            x0 = np.zeros(len(self.ind_obs))
            # Solve the linear system C_oo x = eps
            x, _ = matrixalgebra.pcg_solve(self.ind_obs, self.mask, cov_2n,
                                           z_o, x0,
                                           self.tol, self.n_it_max,
                                           solve,
                                          'scipy')
        elif self.method == 'woodbury':

            epsilon_masked = np.zeros(self.n)
            epsilon_masked[self.ind_obs] = z_o
            # Apply inverse sigma
            v_ = fastoeplitz.toepltiz_inverse_jain(epsilon_masked, 
                                                   self.lambda_n, 
                                                   self.a)
            y_ = np.zeros(self.n)
            y_[self.ind_mis] = self.sig_inv_mm_inv.dot(v_[self.ind_mis])
            e_ = v_ - fastoeplitz.toepltiz_inverse_jain(y_, self.lambda_n, 
                                                        self.a)
            x = e_[self.ind_obs]
            
        else:
            raise ValueError("Unknown imputation method.")
            
        return x


    def imputation(self, y, r, s2, solve=None, draw=True):
        """

        Impute the missing data using a conditional draw.

        Parameters
        ----------
        y : array_like
            masked residuals (size n_data)
        r : array_like
            autocovariance function until lag N_max
        s2 : array_like
            values of the noise one-sided PSD calculated on a Fourier grid of size
            2 N_max. WARNING: it used to be the noise spectrum S fs / 2
        solve : linear operator
            preconditionner
        draw : bool, optional
            if True (default), the missing data are drawn from their 
            conditional distribution. If False, their conditional expectation 
            is returned.
            

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
            else:
                c = None

            if draw:
                results = [self.single_imputation(y[indj], 
                                                self.mask[indj], 
                                                c,
                                                r,
                                                s2) 
                            for indj in self.indices]
            else:
                results = [self.single_conditional_mean(y[indj], 
                                                        self.mask[indj], 
                                                        c, r, s2) 
                           for indj in self.indices]
            # else:
            #     # If the number of points inside the gaps is too large, use a
            #     # FFT-based method
            #     results = [self.single_imputation_fast(y[indj],
            #                                            self.mask[indj],
            #                                            r,
            #                                            s2)
            #                for indj in self.indices]
            y_mis = np.concatenate(results)
            
        else:

            if draw:                
                # For missing data draw:
                e = np.real(generate_noise_from_psd(s2, self.psd_cls.fs)[0:self.n])
                u = self.apply_coo_inv(y[self.ind_obs] - e[self.ind_obs], s2, 
                                    solve=solve)
                # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
                y_mis = e[self.ind_mis] + matrixalgebra.mat_vect_prod(u, 
                                                                    self.ind_obs, 
                                                                    self.ind_mis, 
                                                                    self.mask, 
                                                                    s2)
            else:
                # For conditional mean computation:
                # Compute u = C_oo^{-1} z_o
                u = self.apply_coo_inv(y[self.ind_obs], s2, solve=solve)
                # Compute the missing data conditional mean via z|o = Cmo u
                y_mis = matrixalgebra.mat_vect_prod(u, self.ind_obs, self.ind_mis,
                                                    self.mask, s2)
            
        # elif self.method == 'tapered':
        #     # Approximately solve the linear system C_oo x = eps
        #     u = solve(y[self.ind_obs])
        #     # Compute the missing data conditional mean via z | o = Cmo Coo^-1 z_o
        #     y_mis = matrixalgebra.mat_vect_prod(u, self.ind_obs, self.ind_mis,
        #                                         self.mask, s2)
        # elif self.method == 'PCG':
        #     # Precompute solver if necessary
        #     if solve is None:
        #         # self.compute_preconditioner(r)
        #         raise ValueError("Please provide preconditionning operator")
        #     # First guess
        #     x0 = np.zeros(len(self.ind_obs))
        #     # Solve the linear system C_oo x = eps
        #     u, _ = matrixalgebra.pcg_solve(self.ind_obs, self.mask, s2,
        #                                    y[self.ind_obs], x0,
        #                                    self.tol, self.n_it_max,
        #                                    solve,
        #                                   'scipy')
        #     # Compute the missing data conditional mean z | o = Cmo Coo^-1 z_o
        #     y_mis = matrixalgebra.mat_vect_prod(u, self.ind_obs, self.ind_mis,
        #                                         self.mask, s2)
            
        # elif self.method == 'woodbury':

        #     epsilon_masked = self.mask * y
        #     # Apply inverse sigma
        #     v = fastoeplitz.toepltiz_inverse_jain(epsilon_masked, 
        #                                           self.lambda_n, 
        #                                           self.a)
        #     y_ = np.zeros(self.mask.shape[0])
        #     y_[self.ind_mis] = self.sig_inv_mm_inv.dot(v[self.ind_mis])
        #     e_ = v - fastoeplitz.toepltiz_inverse_jain(y_, 
        #                                                self.lambda_n, 
        #                                                self.a)
        #     # Apply Sigma after multiplying by the mask
        #     eps_given_o = fastoeplitz.toeplitz_multiplication(self.mask * e_, 
        #                                                       r, r)  
        #     y_mis = eps_given_o[self.ind_mis]

        return y_mis
            

    def single_imputation(self, yj, maskj, c, r, psd_2n, threshold=2000):
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
        r : ndarray
            autocovariance computed until lag n_max
        psd_2n : ndarray
            One-sided PSD computed on a Fourier grid of size 2nj
        threshold : int, optional
            Threshold for the size of the neighbooring segments, above which
            the methods switches from matrix-based to FFT-based.

        Returns
        -------
        eps : ndarray
            imputed missing data, of size len(np.where(maskj == 0)[0])

        """
        # Compute the DFT covariances from the one-sided PSD
        # The actual covariance is npoints x S(f) * fs / 2 but the factor
        # of npoints is already accounted for in the IFFT normalization
        cov_2n = psd_2n * self.psd_cls.fs / 2.0

        # Local indices of missing and observed data
        ind_obsj = np.where(maskj == 1)[0]
        ind_misj = np.where(maskj == 0)[0]

        # Compute the size of the neighbooring observed points + gap size
        segment_size = np.int(self.na + self.nb + len(ind_misj))
        
        # If the size is below some threshold, apply full-matrix method:
        if segment_size <= threshold:
        
            c_mo = c[np.ix_(ind_misj, ind_obsj)]
            #C_mm = C[np.ix_(ind_misj,ind_misj)]
            c_oo_inv = linalg.inv(c[np.ix_(ind_obsj, ind_obsj)])
            # out = self.conditional_draw(yj[ind_obsj], psd_2n, c_oo_inv, c_mo,
            #                             ind_obsj, ind_misj, maskj, c)
            e = np.random.multivariate_normal(
                np.zeros(maskj.shape[0]), 
                c[0:maskj.shape[0], 0:maskj.shape[0]])

            # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
            eps = e[ind_misj] + c_mo.dot(c_oo_inv.dot(yj[ind_obsj] - e[ind_obsj]))
        
        # Otherwise, use FFT-based method:
        else:
            # Covariance of observed data and its inverse
            c_oo = toeplitz(r, ind_obsj)
            c_oo_inv = linalg.inv(c_oo)
            # Covariance missing / observed data : matrix operator
            c_mo = lambda v: matrixalgebra.mat_vect_prod(v, ind_obsj, ind_misj,
                                                         maskj, cov_2n)

            # eps = self.conditional_draw_fast(yj[ind_obsj], psd_2n, c_oo_inv, 
            #                                  c_mo, ind_obsj, ind_misj, maskj)
            e = np.real(generate_noise_from_psd(psd_2n, self.psd_cls.fs)[0:maskj.shape[0]])

            # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
            eps = e[ind_misj] + c_mo(c_oo_inv.dot(yj[ind_obsj] - e[ind_obsj]))
            
        return eps
    
    def single_conditional_mean(self, yj, maskj, c, r, psd_2n, threshold=2000):
        """
        Compute the conditional expectation of missing data given the observed
        data, using direct brute-force computation 
        (to be used on short segments with the nearest-neighboor method.)

        Parameters
        ----------
        yj : ndarray
            segment of masked data
        maskj : ndarray
            local mask
        c : ndarray
            covariance matrix of sized nj x nj
        r : ndarray
            autocovariance computed until lag n_max
        psd_2n : ndarray
            One-sided PSD computed on a Fourier grid of size 2nj
        threshold : int, optional
            Threshold for the size of the neighbooring segments, above which
            the methods switches from matrix-based to FFT-based.

        Returns
        -------
        mu_mis_j : ndarray
            conditional expectation of missing data, 
            of size len(np.where(maskj == 0)[0])

        """

        # Compute the DFT covariances from the one-sided PSD
        # The actual covariance is npoints x S(f) * fs / 2 but the factor
        # of npoints is already accounted for in the IFFT normalization
        cov_2n = psd_2n * self.psd_cls.fs / 2.0

        # Local indices of missing and observed data
        ind_obsj = np.where(maskj == 1)[0]
        ind_misj = np.where(maskj == 0)[0]

        # Compute the size of the neighbooring observed points + gap size
        segment_size = np.int(self.na + self.nb + len(ind_misj))
        
        # If the size is below some threshold, apply full-matrix method:
        if segment_size <= threshold:
        
            c_mo = c[np.ix_(ind_misj, ind_obsj)]
            c_oo_inv = linalg.inv(c[np.ix_(ind_obsj, ind_obsj)])
            mu_mis_j = c_mo.dot(c_oo_inv.dot(yj[ind_obsj]))
        
        # Otherwise, use FFT-based method:
        else:
            # Covariance of observed data and its inverse
            c_oo = toeplitz(r, ind_obsj)
            c_oo_inv = linalg.inv(c_oo)
            # Covariance missing / observed data : matrix operator
            c_mo = lambda v: matrixalgebra.mat_vect_prod(v, ind_obsj, ind_misj,
                                                         maskj, cov_2n)

            mu_mis_j = c_mo(c_oo_inv.dot(yj[ind_obsj]))
            
        return mu_mis_j


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
        
        
