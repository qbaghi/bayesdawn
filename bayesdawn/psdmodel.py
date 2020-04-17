#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# This code provides routines for PSD estimation using a peace-continuous model
# that assumes that the logarithm of the PSD is linear per peaces.
import copy

import numpy as np
# FTT modules
import pyfftw
import tdi
from scipy import interpolate
from scipy import linalg as la
from scipy import optimize

pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft


# TODO: rewrite spline interpolation with LSQUnivariateSpline from scipy.interpolate

# ==============================================================================
# SPLINES
# ==============================================================================

def least_squares(mat, y):
    return la.pinv(mat.conjugate().transpose().dot(mat)).dot(
        mat.conjugate().transpose().dot(y))


def find_closest_points(f_target, f):
    """

    Find frequencies in f closest to the frequencies in f_target

    """

    inds = [np.argmin(np.abs(f - f0)) for f0 in f_target]

    return inds


def spline_loglike(beta, per, a_mat):
    """

    Whittle log-likelihood with spline PSD model

    Parameters
    ----------
    a_mat
    per : array_like
        vector of periodogram calculated at log-frequencies
    spl : instance of PSDSpline
        spline object


    Returns
    -------
    ll : scalar float
        value of the log-likelihood


    """

    psdmodel = a_mat.dot(beta)

    return - 0.5 * np.sum(np.log(psdmodel) + per / psdmodel)


def spline_loglike_grad(beta, per, A):
    """

    Gradient of the Whittle log-likelihood with spline PSD model

    Parameters
    ----------
    per : array_like
        vector of periodogram
    spl : instance of PSDSpline
        spline object


    Returns
    -------
    grad_ll : 1d numpy array
        gradient of the log-likelihood


    """

    psdmodel = np.dot(A, beta)

    grad_ll = - 0.5 * np.dot(A.T, 1 / psdmodel * (1 - per / psdmodel))

    return grad_ll


def spline_loglike_hessian(beta, per, A):
    """

    Hessian matrix of the Whittle log-likelihood with spline model for the PSD

    Parameters
    ----------
    A : array_like
        design matrix
    x : array_like
        vector of log-frequencies taken into account in the likelihood
    per : array_like
        vector of periodogram calculated at log-frequencies minus C0
    spl : instance of PSDSpline
        spline object


    Returns
    -------
    grad_ll : 1d numpy array
        gradient of the log-likelihood


    """

    psdmodel = np.dot(A, beta)

    E = 1 / psdmodel ** 2 * (-1 + 2 * per / psdmodel)

    AE = np.array([A[:, j] * E for j in range(A.shape[1])]).T

    hessian = - 0.5 * np.dot(A.T, AE)

    # grad_ll = np.array([np.sum( spl.derivatives() )])

    return hessian


def newton_raphson(beta_0, grad_func, hess_func, maxiter=1000, tol=1e-4):
    """

    Newton-Raphson algorithm to compute the maximum likelihood

    """

    eps = 1.0
    i = 0
    beta_old = beta_0

    while (i < maxiter) & (eps > tol):
        beta = beta_old - la.inv(hess_func(beta_old)).dot(grad_func(beta_old))
        eps = la.norm(beta - beta_old) / la.norm(beta_old)
        beta_old = copy.deepcopy(beta)
        i = i + 1

    print("Criterium at the end: " + str(eps))
    print("Number of iterations: " + str(i))

    return beta


def spline_fisher(a_mat):
    """

    Compute the Fisher matrix for the spline PSD model parameters.


    """

    return 0.5 * a_mat.conj().T.dot(a_mat)


def spline_matrix(x, knots, D):
    """

    Matrix of partial derivatives of the spline with respect to its parameters

    with:

    y_k = sum_d beta_d x_k^d + sum_j b_j (x_k - xi_j )_+

    where

    y_k = log(S(fk))
    x_k = log(fk)


    Parameters
    ----------
    x : numpy array of size n_data
        abscisse points where to compute the spline
    knots : numpy array of size J-1
        knots of the spline (asbisse of segments nodes)
    D : scalar integer
        degree of the spline

    Returns
    -------

    a_mat : 2d numpy array
        spline design matrix

    """

    # Size of the matrix
    # K = D + 1 + len(knots)

    # Polynomial
    A = [x ** d for d in range(D + 1)]

    # Truncated polynomial
    A2 = [np.concatenate((np.zeros(len(x[x < xi])), (x[x >= xi] - xi) ** D))
          for xi in knots]
    A.extend(A2)

    A_mat = np.array(A).T
    #
    # mu = np.mean(A_mat,axis = 0)
    # for j in range(A_mat.shape[1]):
    #     A_mat[:,j] = A_mat[:,j]/np.mean(A_mat[:,j])

    return A_mat


# =============================================================================
# General PSD CLASS
# =============================================================================
class PSD(object):

    def __init__(self, n_data, fs, fmin=None, fmax=None):

        # Sampling frequency
        self.fs = fs
        # Size of the sample
        self.N = n_data
        self.f = np.fft.fftfreq(n_data) * fs
        self.n = np.int((n_data - 1) / 2.)

        if fmin is None:
            self.fmin = fs / n_data
        else:
            self.fmin = fmin
        if fmax is None:
            self.fmax = fs / 2
        else:
            self.fmax = fmax

        # Flexible interpolation of the estimated PSD
        self.log_psd_fn = None

    def periodogram(self, y_fft, k2=None):
        """
        Simple periodogram with no windowing.
        Given as one-sided PSD [A / Hz]

        Parameters
        ----------
        y_fft : ndarray
            Fourier-transformed data, possibly pre-windowed.
            If so, the normalization factor K2 should be provided to
            account for the windowing.
        k2 : float (optional)
            If None, assume that no windowing has been applied to the data.
            Else, should be equal to sum(wd**2) where wd is the window vector.

        Returns
        -------
        per : ndarray
            periodogram scaled in Units / Hz. Consisten with one-sided
            power spectral density.

        """
        if k2 is None:
            per = np.abs(y_fft) ** 2 / len(y_fft)
        else:
            per = np.abs(y_fft) ** 2 / k2

        return per * 2 / self.fs

    def psd_fn(self, x):
        return np.exp(self.log_psd_fn(np.log(x)))

    def calculate(self, arg):
        """

        Calculate the power spectral density at an arbitrary frequency
        from the estimation.

        Parameters
        ----------
        arg : ndarray or int
            frequency array where to compute the PSD, or data size N.
            If N is given, computes the PSD on the Fourier grid of size N,
            for both positive and negative frequencies.

        Returns
        -------
        spectr_sym : ndarray
            one-sided power spectral density expressed in [Units / Hz]
            WE DROPPEP THE FACTOR OF fs / 2!

        """

        if (type(arg) == np.int) | (type(arg) == np.int64):
            n_data = arg
            # Symmetrize the estimates
            if n_data % 2 == 0:  # if n_data is even
                # Compute PSD from f=0 to f = fs/2
                if n_data == self.N:
                    n = self.n
                    f_tot = np.abs(np.concatenate(([self.f[1]],
                                                   self.f[1:n + 2])))
                else:
                    f = np.fft.fftfreq(n_data) * self.fs
                    n = np.int((n_data - 1) / 2.)
                    f_tot = np.abs(np.concatenate(([f[1]], f[1:n + 2])))

                spectr = self.psd_fn(f_tot)
                spectr_sym = np.concatenate((spectr[0:n + 1],
                                             spectr[1:n + 2][::-1]))

            else:  # if n_data is odd
                if n_data == self.N:
                    n = self.n
                    f_tot = np.abs(np.concatenate(([self.f[1]],
                                                   self.f[1:n + 1])))
                else:
                    f = np.fft.fftfreq(n_data) * self.fs
                    n = np.int((n_data - 1) / 2.)
                    f_tot = np.abs(np.concatenate(([f[1]], f[1:n + 1])))

                spectr = self.psd_fn(f_tot)
                spectr_sym = np.concatenate((spectr[0:n + 1],
                                             spectr[1:n + 1][::-1]))

        elif type(arg) == np.ndarray:

            f = arg[:]
            spectr_sym = self.psd_fn(f)

        else:

            raise TypeError("Argument must be integer or ndarray")

        return spectr_sym

    def calculate_autocorr(self, N):
        """
        Compute the autocovariance function from the PSD.

        """

        return np.real(ifft(self.calculate(2 * N))[0:N])


# ==============================================================================
# Spline PSD model
# ==============================================================================
class PSDSpline(PSD):

    def __init__(self, n_data, fs, J=30, D=3,
                 fmin=None, fmax=None, f_knots=None, ext=3):

        PSD.__init__(self, n_data, fs, fmin=fmin, fmax=fmax)

        # Number of knots for the log-PSD spline model
        self.J = J
        # Create a dictionary corresponding to each data length
        self.logf = {n_data: np.log(self.f[1:self.n + 1])}
        # Set the knot grid
        if f_knots is None:
            self.f_knots = self.choose_knots()
            self.f_min_est = self.f[1]
            self.f_max_est = self.f[self.n]
        else:
            self.f_knots = f_knots
            self.J = len(self.f_knots)

            self.f_min_est = copy.deepcopy(self.fmin)
            self.f_max_est = copy.deepcopy(self.fmax)

        self.logf_knots = np.log(self.f_knots)
        # Spline order
        self.D = D
        self.C0 = -0.57721
        # Spline coefficient vector
        self.beta = []
        # PSD at positive Fourier frequencies
        self.logS = []
        # Control frequencies
        self.logfc = np.concatenate(
            (np.log(self.f_knots), [np.log(self.fs / 2)]))
        self.logSc = []
        # Spline extension
        self.ext = ext
        # # Variance function values at control frequencies
        # self.varlogSc = np.array(
        #     [3.60807571e-01, 8.90158814e-02, 1.45631966e-02, 3.55646693e-03,
        #      1.09926717e-03, 4.15894275e-04, 1.86984136e-04, 9.73883423e-05,
        #      5.74981099e-05, 3.77721249e-05, 2.71731280e-05, 2.11167300e-05,
        #      1.75209167e-05, 1.53672320e-05, 1.41269765e-05, 1.35137347e-05,
        #      1.33692054e-05, 1.36074455e-05, 1.41863625e-05, 1.50926724e-05,
        #      1.63338849e-05, 1.79341767e-05, 1.99325803e-05, 2.23827563e-05,
        #      2.53543168e-05, 2.89370991e-05, 3.32545462e-05, 3.85055177e-05,
        #      4.50144967e-05, 5.26798764e-05, 4.86680827e-04])

        # # Spline estimator of the variance of the log-PSD estimate
        # self.logvar_fn = interpolate.interp1d(self.logfc[1:],
        #                                       self.varlogSc[1:],
        #                                       kind='cubic',
        #                                       fill_value="const")

    def set_knots(self, f_knots):

        self.f_knots = f_knots
        self.logf_knots = np.log(self.f_knots)
        self.logfc = np.concatenate(
            (np.log(self.f_knots), [np.log(self.fs / 2)]))

    def choose_knots(self):
        """

        Choose frequency knots such that

        f_knots = 10^-n_knots

        where the difference
        n_knots[j+1] - n_knots[j] = dn[j]

        is a geometric series.

        Parameters
        ----------
        J : scalar integer
            number of knots
        fmin : scalar float
            minimum frequency knot
        fmax : scalar float
            maximum frequency knot


        """

        base = 10
        # base = np.exp(1)
        ns = - np.log(self.fmax) / np.log(base)
        n0 = - np.log(self.fmin) / np.log(base)
        jvect = np.arange(0, self.J)
        alpha_guess = 0.8

        targetfunc = lambda x: n0 - (1 - x ** (self.J)) / (1 - x) - ns
        result = optimize.fsolve(targetfunc, alpha_guess)
        alpha = result[0]
        n_knots = n0 - (1 - alpha ** jvect) / (1 - alpha)
        f_knots = base ** (-n_knots)

        return f_knots

    def estimate(self, y, wind='hanning'):
        """

        Estimate the log-PSD using spline model by least-square method

        Parameters
        ----------
        y : array_like
            data (typically model residuals) in the time domain


        """

        if type(wind) == np.ndarray:
            w = wind[:]
        elif wind == 'hanning':
            w = np.hanning(len(y))

        k2 = np.sum(w ** 2)
        per = self.periodogram(fft(y * w), k2=k2)

        # Compute the spline parameter vector for the log-PSD model
        self.estimate_from_periodogram(per)

    def estimate_from_freq(self, y_fft, k2=None):
        """

        Estimate the log-PSD using spline model by least-square method from the
        discrete Fourier transformed data. This function is useful to avoid
        to compute FFTs multiple times.


        """

        # If there is only one periodogram
        if type(y_fft) == np.ndarray:
            per = self.periodogram(y_fft, k2=k2)
        # Otherwise calculate the periodogram for each data set:
        elif type(y_fft) == list:
            per = [self.periodogram(y_fft[i], k2=k2[i]) for i in
                   range(len(y_fft))]

        self.estimate_from_periodogram(per)

    def estimate_from_periodogram(self, per):
        """

        Estimate PSD from the periodogram

        """

        # If there is only one periodogram
        if type(per) == np.ndarray:
            self.log_psd_fn = self.spline_lsqr(per)
            self.beta = self.log_psd_fn.get_coeffs()
        elif type(per) == list:
            # If there are several periodograms, average the estimates
            spl_list = [self.spline_lsqr(I0) for I0 in per if
                        self.fs / len(I0) < self.f_knots[0]]
            self.beta = sum([spl.get_coeffs for spl in spl_list]) / len(per)
            self.log_psd_fn = interpolate.BSpline(spl_list[0].get_knots(),
                                                  self.beta, self.D)

        # Estimate psd at positive Fourier log-frequencies
        self.logS = self.log_psd_fn(self.logf[self.N])

        # # Spline estimator of the variance of the log-PSD estimate
        # self.logvar_fn = interpolate.LSQUnivariateSpline(self.logf[self.n_data],
        #                                                  (np.log(I[1:self.n+1]) - self.C0 - self.logS)**2,
        #                                                  self.logf_knots, k=1, ext='const')
        # Update PSD control values (for Bayesian estimation)
        # self.varlogSc = self.logvar_fn(self.logfc)
        self.logSc = self.log_psd_fn(self.logfc)

    def spline_lsqr(self, per):
        """

        Fit a spline to the log periodogram using least-squares

        Parameters
        ----------
        per : ndarray
            periodogram
        ext : extint or str, optional
            Controls the extrapolation mode for elements not in the interval defined by the knot sequence.
                if ext=0 or ‘extrapolate’, return the extrapolated value.
                if ext=1 or ‘zeros’, return 0
                if ext=2 or ‘raise’, raise a ValueError
                if ext=3 of ‘const’, return the boundary value
            The default value is 3.


        """

        NI = len(per)

        if NI not in list(self.logf.keys()):
            f = np.fft.fftfreq(NI) * self.fs
            self.logf[NI] = np.log(f[f > 0])
        else:
            f = np.concatenate(([0], np.exp(self.logf[NI])))

        n = np.int((NI - 1) / 2.)
        z = per[1:n + 1]
        v = np.log(z) - self.C0

        # Spline estimator of the log-PSD
        inds_est = np.where((self.f_min_est <= f[1:self.n + 1]) & (
                    f[1:self.n + 1] <= self.f_max_est))[0]
        spl = interpolate.LSQUnivariateSpline(self.logf[NI][inds_est],
                                              v[inds_est],
                                              self.logf_knots,
                                              k=self.D,
                                              ext=self.ext)

        return spl


# ======================================================================================================================
# Power-law PSD model
# ======================================================================================================================
class PSDPowerLaw(PSD):

    def __init__(self, n, fs, f_knots=None, fmin=None, fmax=None):

        PSD.__init__(self, n, fs, fmin=fmin, fmax=fmax)

        # Number of knots for the log-PSD spline model
        self.n_knots = len(self.d) + 1
        # Set the knot grid
        if f_knots is None:
            # self.f_knots = self.choose_knots()
            self.f_knots = self.fmin * (self.fmax / self.fmin) ** (
                        np.arange(0, self.n_knots) / (self.n_knots - 1))
        else:
            self.f_knots = f_knots
        # Find the corresponding Fourier indices
        # self.ind_knots = np.unique(np.array(self.f_knots / (self.fs / self.n)).round().astype(np.int))
        self.ind_knots = [np.where(f0 <= self.f < f0)[0] for f0 in
                          self.f_knots]
        # Redefine exact knots
        # self.f_knots = self.f[self.ind_knots]
        # self.f_knots[self.f_knots == 0] = self.f[1]
        # Spline order
        self.C0 = -0.57721
        # Create a dictionary corresponding to each data length
        self.logf = {n: np.log(self.f[1:self.n + 1])}
        # Control frequencies
        self.logfc = np.concatenate(
            (np.log(self.f_knots), [np.log(self.fs / 2)]))
        self.logSc = []
        # Prepare design matrix
        self.mat_list = self.build_matrix(self.f_knots)

    def build_matrix(self, x):
        return np.hstack([np.array([x ** i]).T for i in self.d])

    def choose_knots(self):
        """

        Choose frequency knots such that

        f_knots = 10^-n_knots

        where the difference
        n_knots[j+1] - n_knots[j] = dn[j]

        is a geometric series.



        """

        base = 10
        # base = np.exp(1)
        ns = - np.log(self.fmax) / np.log(base)
        n0 = - np.log(self.fmin) / np.log(base)
        jvect = np.arange(0, self.n_knots)
        alpha_guess = 0.8

        targetfunc = lambda x: n0 - (1 - x ** (self.n_knots)) / (1 - x) - ns
        result = optimize.fsolve(targetfunc, alpha_guess)
        alpha = result[0]
        n_knots = n0 - (1 - alpha ** jvect) / (1 - alpha)
        f_knots = base ** (-n_knots)

        return f_knots

    def estimate(self, y, wind='hanning'):
        """

        Estimate the log-PSD using spline model by least-square method

        Parameters
        ----------
        y : array_like
            data (typically model residuals) in the time domain


        """

        if type(wind) == np.ndarray:
            w = wind[:]
        elif wind == 'hanning':
            w = np.hanning(len(y))
        per = np.abs(fft(y * w)) ** 2 / np.sum(w ** 2)

        # Compute the spline parameter vector for the log-PSD model
        self.estimate_from_periodogram(per)

    def estimate_from_freq(self, y_fft, k2=None):
        """

        Estimate the log-PSD using spline model by least-square method from the
        discrete Fourier transformed data. This function is useful to avoid
        to compute FFTs multiple times.


        """

        # If there is only one periodogram
        if type(y_fft) == np.ndarray:
            per = self.periodogram(y_fft, k2=k2)
        # Otherwise calculate the periodogram for each data set:
        elif type(y_fft) == list:
            per = [self.periodogram(y_fft[i], k2=k2[i]) for i in
                   range(len(y_fft))]

        self.estimate_from_periodogram(per)

    def fit_lsqr(self, per):
        """

        Fit a spline to the log periodogram using least-squares

        Parameters
        ----------
        per : ndarray
            periodogram

        """

        NI = len(per)
        if NI not in list(self.logf.keys()):
            f = np.fft.fftfreq(NI) * self.fs
            self.logf[NI] = np.log(f[f > 0])

        # n = np.int((NI-1)/2.)
        # z = per[1:n + 1]
        z_list = [per[inds] for inds in self.ind_knots]
        v_list = [np.log(z) - self.C0 for z in z_list]
        # beta = la.pinv(self.mat.conjugate().transpose().dot(self.mat)).dot(self.mat.conjugate().transpose().dot(z))
        beta = [least_squares(self.mat[self.ind_knots[i], i], v_list[i]) for i
                in np.arange(len(v_list))]

        return np.array(beta)

    def estimate_from_periodogram(self, per):
        """

        Estimate PSD from the periodogram

        """

        # If there is only one periodogram
        if type(per) == np.ndarray:
            self.beta = self.fit_lsqr(per)
        elif type(per) == list:
            # If there are several periodograms, average the estimates
            self.beta = [self.fit_lsqr(I0) for I0 in per if
                         self.fs / len(I0) < self.f_knots[0]]

        # Estimate psd at positive Fourier log-frequencies
        # self.logS = self.log_psd_fn(self.logf[self.n_data])
        self.logSc = np.log(self.mat.dot(self.beta))

    def psd_fn(self, x):

        mat = self.build_matrix(x)

        return mat.dot(self.beta)

    def log_psd_fn(self, x):

        return np.log(self.psd_fn(np.log(x)))


def scaled_gamma_distribution(mu, var):
    """
    Compute the degree-of-freedom parameter nu and the scale parameter s2 of
    a scaled_gamma_distribution with mean mu and variance var

    Parameters
    ----------
    mu : array_like
        distribution mean
    var : array_like
        distribution variance

    Returns
    -------
    nu : array_like
        degree-of-freedom parameter
    s2 : array_like
        scale parameter

    """

    nu = 4 + 2 * mu ** 2 / var
    s2 = (nu - 2) / nu * mu

    return nu, s2


def log_normal_distribution(mu_X, var_X):
    """

    Compute the mean and variance of the log-normal distribution given the
    mean and variance of the underlying normal distribution

    Y = exp( X )

    """
    # Log-normal distribution mean and variance
    mu_Y = np.exp(mu_X + 0.5 * var_X)
    var_Y = np.exp(2 * mu_X + var_X) * (np.exp(var_X) - 1)

    return mu_Y, var_Y


def theoretical_spectrum_func(channel, scale=1.0):
    """

    Parameters
    ----------
    channel : str
        channel in {A, E, T}
    scale : float
        scale factor applied to the data, such that
        data_rescaled = data * scale

    Returns
    -------
    psd_fn : callable
        PSD function in the requested channel
        [rescaled Fractional Frequency / Hz]

    """

    if channel == 'a_mat':
        psd_fn = lambda x: tdi.noisepsd_AE(x, model='SciRDv1') * scale ** 2
    elif channel == 'E':
        psd_fn = lambda x: tdi.noisepsd_AE(x, model='SciRDv1') * scale ** 2
    elif channel == 'T':
        psd_fn = lambda x: tdi.noisepsd_T(x, model='SciRDv1') * scale ** 2

    return psd_fn


class PSDTheoretical(PSD):
    """
    Power spectral density class providing methods to compute the theoretical
    PSD of TDI data streams
    """

    def __init__(self, n_data, fs, channel, scale=1.0, J=30, D=3,
                 fmin=None, fmax=None, f_knots=None, ext=3):

        PSD.__init__(self, n_data, fs, fmin=fmin, fmax=fmax)
        self.channel = channel
        self.scale = scale
        self.log_psd_fn = lambda x: np.log(self.psd_fn(np.exp(x)))

    def psd_fn(self, x):

        if self.channel == 'A':
            return tdi.noisepsd_AE(x, model='SciRDv1') / self.scale ** 2
        elif self.channel == 'E':
            return tdi.noisepsd_AE(x, model='SciRDv1') / self.scale ** 2
        elif self.channel == 'T':
            return tdi.noisepsd_T(x, model='SciRDv1') / self.scale ** 2

    def sample(self, npsd):
        sampling_result = self.sample_psd(npsd)
        sample_list = [samp[0] for samp in sampling_result]
        logp_values_list = [samp[1] for samp in sampling_result]

        return sample_list, logp_values_list
