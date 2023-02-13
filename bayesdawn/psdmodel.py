# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# This code provides routines for PSD estimation using a peace-continuous model
# that assumes that the logarithm of the PSD is linear per peaces.
import copy
import numpy as np
# FTT modules
import pyfftw
from scipy import interpolate
from scipy import linalg as la
from scipy import optimize
from pyfftw.interfaces.numpy_fft import fft, ifft
try:
    import tdi
except:
    print("MLDC modules could not be loaded.")

pyfftw.interfaces.cache.enable()


# TODO: rewrite spline interpolation with LSQUnivariateSpline from scipy.interpolate
class MyLSQUnivariateSpline(interpolate.LSQUnivariateSpline):
    
    def __init__(self, *args, **kwargs):
        
        interpolate.LSQUnivariateSpline.__init__(self, *args, **kwargs)
        
    def set_coeffs(self, coeffs):
        """Set spline coefficients."""
        data = self._data
        k, n = data[5], data[7]
        data[9][:n-k-1] = coeffs
        self._data = data

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
    knots : numpy array of size n_knots-1
        knots of the spline (asbisse of segments nodes)
    D : scalar integer
        degree of the spline

    Returns
    -------

    a_mat : 2d numpy array
        spline design matrix

    """

    # Size of the matrix
    # K = d + 1 + len(knots)

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


def choose_frequency_knots(n_knots, freq_min=1e-5, freq_max=1.0, base=10):
    # Choose the frequency knots
    #JGB: Note. This function wasn't working in a sensible way, (alpha was<-1)
    #so I fixed it as I believe must have been intended.
    ns = np.log(freq_min) / np.log(base)
    n0 = np.log(freq_max) / np.log(base)
    jvect = np.arange(0, n_knots)
    alpha_guess = 1 + 1/(ns-n0) #Soln for k=inf
    targetfunc = lambda x: n0 - (1 - x ** (n_knots)) / (1 - x) - ns
    result = optimize.fsolve(targetfunc, alpha_guess)
    alpha = result[0]
    #print('alpha',alpha)
    n_knots = ns + (1 - alpha ** jvect) / (1 - alpha)
    f_knots = base ** (n_knots)
    #print('knots before trim:',f_knots)
    f_knots = f_knots[(f_knots >= freq_min) & (f_knots <= freq_max)]
    #print(len(f_knots))
    
    return np.unique(np.sort(f_knots))


# =============================================================================
# General PSD CLASS
# =============================================================================
class PSD(object):

    def __init__(self, n_data, fs, fmin=None, fmax=None):
        """Instantiate the PSD estimator class.

        Parameters
        ----------
        n_data : int
            size of analyzed data
        fs : float
            sampling frequency of analyzed data
        fmin : float
            minimum frequency where to estimate the PSD
        fmax : float
            maximum frequency where to estimate the PSD
        """

        # Sampling frequency
        self.fs = fs
        # Size of the sample
        self.n_data = n_data
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
            periodogram scaled in Units / Hz. Consistant with one-sided
            power spectral density.

        """
        if k2 is None:
            per = np.abs(y_fft) ** 2 / len(y_fft)
        else:
            per = np.abs(y_fft) ** 2 / k2

        return per * 2 / self.fs

    def psd_fn(self, x):
        """
        
        Returns the value of the PSD estimate at frequency x

        Parameters
        ----------
        x : array_like
            frequency

        Returns
        -------
        psd : ndarray
            one-sided PSD values in A^2 / Hz, where A is the unit of 
            the time series
            
        """
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
                if n_data == self.n_data:
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
                if n_data == self.n_data:
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

        return np.real(ifft(self.calculate(2 * N))[0:N]) * self.fs / 2


# ==============================================================================
# Spline PSD model
# ==============================================================================
class PSDSpline(PSD):

    def __init__(self, n_data, fs, n_knots=30, d=3,
                 fmin=None, fmax=None, f_knots=None, ext=3):
        """

        Parameters
        ----------
        n_data : int
            size of analyzed data
        fs : float
            sampling frequency of analyzed data
        n_knots : int
            number of spline knots
        d : int
            degree of the splines
        fmin : float
            minimum frequency where to estimate the PSD
        fmax : float
            maximum frequency where to estimate the PSD
        f_knots : ndarray
            knot frequencies (if provided, then discard n_knots)
        ext : extint or str, optional
            Controls the extrapolation mode for elements not in the interval
            defined by the knot sequence.
                if ext=0 or ‘extrapolate’, return the extrapolated value.
                if ext=1 or ‘zeros’, return 0
                if ext=2 or ‘raise’, raise a ValueError
                if ext=3 of ‘const’, return the boundary value
            The default value is 3.
        """

        PSD.__init__(self, n_data, fs, fmin=fmin, fmax=fmax)

        # Number of knots for the log-PSD spline model
        self.n_knots = n_knots
        # Create a dictionary corresponding to each data length
        self.logf = {n_data: np.log(self.f[1:self.n + 1])}
        # Set the knot grid
        if f_knots is None:
            self.f_knots = self.choose_knots()
            self.f_min_est = self.f[1]
            self.f_max_est = self.f[self.n]
        else:
            self.f_knots = f_knots
            self.n_knots = len(self.f_knots)

            self.f_min_est = copy.deepcopy(self.fmin)
            self.f_max_est = copy.deepcopy(self.fmax)

        self.logf_knots = np.log(self.f_knots)
        # Spline order
        self.D = d
        self.C0 = -0.57721
        # Spline coefficient vector
        self.beta = []
        # PSD at positive Fourier frequencies
        self.logs = []
        # Control frequencies
        self.logfc = np.concatenate(
            (np.log(self.f_knots), [np.log(self.fs / 2)]))
        self.logsc = []
        # Spline extension
        self.ext = ext
        # Variance function values at control frequencies
        self.varlogsc = np.pi**3 / 6 * np.ones(self.n_knots + 1)
        # self.varlogsc = np.array(
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
        #                                       self.varlogsc[1:],
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
        n_knots : scalar integer
            number of knots
        fmin : scalar float
            minimum frequency knot
        fmax : scalar float
            maximum frequency knot


        """

        f_knots = choose_frequency_knots(self.n_knots, freq_min=self.fmin, 
                                         freq_max=self.fmax, base=10)
        f_knots = f_knots[1:-1] #don't include the boundary points
        
        return f_knots
    
    def get_spline_control_points(self):
        """
        Outputs the spline parameters (values of the model PSD at frequency
        knots).
        """
        
        return self.log_psd_fn.get_coeffs()
        
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
            self.beta = sum([spl.get_coeffs() for spl in spl_list]) / len(per)
            self.log_psd_fn = interpolate.BSpline(spl_list[0].get_knots(),
                                                  self.beta, self.D)

        # Estimate psd at positive Fourier log-frequencies
        self.logs = self.log_psd_fn(self.logf[self.n_data])

        # # Spline estimator of the variance of the log-PSD estimate
        # self.logvar_fn = interpolate.LSQUnivariateSpline(self.logf[self.n_data],
        #                                                  (np.log(I[1:self.n+1]) - self.C0 - self.logs)**2,
        #                                                  self.logf_knots, k=1, ext='const')
        # Update PSD control values (for Bayesian estimation)
        # self.varlogsc = self.logvar_fn(self.logfc)
        self.logsc = self.log_psd_fn(self.logfc)

    def spline_lsqr(self, per, freq=None):
        """

        Fit a spline to the log periodogram using least-squares

        Parameters
        ----------
        per : ndarray
            periodogram
        freq :


        """

        if freq is None:
            # If the frequencies where per is computed are not given
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

        else:
            # If the frequencies are given
            v = np.log(per) - self.C0
            # Spline estimator of the log-PSD
            inds_est = np.where((self.f_min_est <= freq)
                                & (freq <= self.f_max_est))[0]
            spl = interpolate.LSQUnivariateSpline(np.log(freq)[inds_est],
                                                  v[inds_est],
                                                  self.logf_knots,
                                                  k=self.D,
                                                  ext=self.ext)

        return spl


# ==============================================================================
# Spline PSD model
# ==============================================================================
class PSDEstimator(object):

    def __init__(self, f_knots, d=3, ext=3, cross=False):
        """
        PSD estimator without any assumption on the data size / sampling.

        Parameters
        ----------
        f_knots : ndarray
            spline knot frequencies
        d : int
            degree of the splines
        ext : extint or str, optional
            Controls the extrapolation mode for elements not in the interval
            defined by the knot sequence.
                if ext=0 or ‘extrapolate’, return the extrapolated value.
                if ext=1 or ‘zeros’, return 0
                if ext=2 or ‘raise’, raise a ValueError
                if ext=3 of ‘const’, return the boundary value
            The default value is 3.
        cross : bool
            if True, we assume that the spectrum is a complex cross-spectrum
        """


        # Set the knot grid
        self.f_knots = f_knots
        # Number of knots for the log-PSD spline model
        self.n_knots = len(f_knots)
        # Spline order
        self.d = d
        if d == 1:
            self.kind = "linear"
        elif d == 2:
            self.kind = "quadratic"
        elif d == 3:
            self.kind = "cubic"
        # Whether the class is for real spectrum or complex 
        # cross-spectrum estimation
        self.cross = cross
        # Euler constant
        self.c0 = -0.57721
        # Control frequencies
        self.logf_knots = np.log(self.f_knots)
        # PSD amplitude values at control frequencies
        self.logs_knots = []
        # Spline extension
        self.ext = ext
        # Variance function values at control frequencies
        self.varlogsc = np.pi**3 / 6 * np.ones(len(f_knots))
        # Spline interpolation object of the estimated log PSD
        # Function of the log-frequency that outputs the log-PSD
        self.log_psd_fn = None
        self.psd_fn = None
        
    def estimate(self, freq, per):
        """
        Estimate the spline coefficients from the periodogram 
        or cross-periodogram.

        Parameters
        ----------
        freq : ndarray
            frequency vector
        per : ndarray
            periodogram computed at frequencies freq
        complex : bool
            if True, per is assumed to be a complex cross-periodogram. 
            Thus, its phase is estimated along with its amplitude. 
        """
        if not self.cross:
            # If the frequencies are given
            v = np.log(per.real) - self.c0
            # Spline estimator of the log-PSD
            self.log_psd_fn = interpolate.LSQUnivariateSpline(np.log(freq),
                                                              v,
                                                              self.logf_knots,
                                                              k=self.d,
                                                              ext=self.ext)
            # self.log_psd_fn = MyLSQUnivariateSpline(np.log(freq), v,
            #                                         self.logf_knots,
            #                                         k=self.d,
            #                                         ext=self.ext)
            # Save the values of the log-PSD at the frequency knots
            self.logs_knots = self.log_psd_fn(self.logf_knots)
            
        else:
            # # If the frequencies are given
            # v_amp = np.log(np.abs(per)) - self.c0
            # v_ang = np.angle(per)
            # # Spline estimator of the log-PSD
            # spl_amp = interpolate.LSQUnivariateSpline(np.log(freq),
            #                                           v_amp,
            #                                           self.logf_knots,
            #                                           k=self.d,
            #                                           ext=self.ext)

            # spl_ang = interpolate.LSQUnivariateSpline(freq,
            #                                           v_ang,
            #                                           self.f_knots,
            #                                           k=self.d,
            #                                           ext=self.ext)
            # self.log_psd_fn = lambda x: spl_amp(x) + 1j * spl_ang(np.exp(x))
            # self.psd_fn = lambda x: np.exp(spl_amp(np.log(x)) + 1j * spl_ang(x))
            
            # Spline estimator of real part of the cross-spectrum
            spl_real = interpolate.LSQUnivariateSpline(freq,
                                                       per.real,
                                                       self.f_knots,
                                                       k=self.d,
                                                       ext=self.ext)
            # Spline estimator of real part of the cross-spectrum
            spl_imag = interpolate.LSQUnivariateSpline(freq,
                                                       per.imag,
                                                       self.f_knots,
                                                       k=self.d,
                                                       ext=self.ext)
            self.psd_fn = lambda x: spl_real(x) + 1j * spl_imag(x)
            

            
    def calculate(self, x):
        """
        
        Returns the value of the PSD estimate at frequency x

        Parameters
        ----------
        x : array_like
            frequency

        Returns
        -------
        psd : ndarray
            PSD values
            
        """
        if not self.cross:    
            return np.exp(self.log_psd_fn(np.log(x)))
        else:
            return self.psd_fn(x)
    
    def set_params(self, x):
        """
        Set the Spline model coefficient values.

        Parameters
        ----------
        x : ndarray
            if the spectrum is real positive, values of the log-PSD at the knot 
            frequencies
            if the spectrum is crossed (complex), values of the cross-PSD at 
            the knot frequencies 
        """
        
        if not self.cross:
            # self.log_psd_fn.set_coeffs(x)
            self.log_psd_fn = interpolate.interp1d(self.logf_knots, x, 
                                                   kind=self.kind, 
                                                   fill_value="extrapolate")
        else:
            # # For the log-amplitude
            # logsfunc = interpolate.interp1d(self.logf_knots, 
            #                                 x.real, 
            #                                 kind=self.kind, 
            #                                 fill_value="extrapolate")
            # # For the phase
            # phasefunc = interpolate.interp1d(self.f_knots, 
            #                                  x.imag, 
            #                                  kind=self.kind, 
            #                                  fill_value="extrapolate")
            # # Update the log-PSD function of the log-frequency
            # self.log_psd_fn = lambda x: logsfunc(x) + 1j * phasefunc(np.exp(x))        
            # For the real part
            s_real_func = interpolate.interp1d(self.f_knots, 
                                               x.real, 
                                               kind=self.kind, 
                                               fill_value="extrapolate")
            # For the phase
            s_imag_func = interpolate.interp1d(self.f_knots, 
                                               x.imag, 
                                               kind=self.kind, 
                                               fill_value="extrapolate")
            # Update the log-PSD function of the log-frequency
            self.psd_fn = lambda x: s_real_func(x) + 1j * s_imag_func(x)   
    
    def likelihood(self, x, logfr, per):
        """

        Compute log-likelihood for the PSD update

        Parameters
        ----------
        x: array_like
            vector of log-PSD values at specific frequencies
        logfr : array_like
            logarithm of frequency vector
        per : array_like
            periodogram computed at frequencies fr
            
        Returns
        -------
        ll : scalar float
            value of the log-likelihood

        """
        
        # Update log PSD function with new parameters
        self.set_params(x)
        
        # If only one segment of data is analyzed
        if type(per) == np.ndarray:
            logs = self.log_psd_fn(logfr)
            ll = np.real(-0.5*np.sum(logs + per * np.exp(-logs)))
            
        # If several segments of different lengths are considered:
        elif type(per) == list:
            logs_list = [self.log_psd_fn(logfj) for logfj in logfr]
            ll = sum([np.real(-0.5*np.sum(logs_list[j] 
                                          + per[j] * np.exp(-logs_list[j])))
                              for j in range(len(per))])
            
        return ll
    
    def prior(self, x):
        """

        Compute the log-prior probability density for the PSD parameters

        """

        return -0.5 * np.sum(np.abs(x - self.logs_knots)**2 / (2*self.varlogsc))

    def posterior(self, x_psd, logfr, per):
        """
        Compute the log-posterior probability density for the PSD parameters

        """

        return self.likelihood(x_psd, logfr, per) + self.prior(x_psd)


# =============================================================================
# Power-law PSD model
# =============================================================================
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
        self.logsc = []
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
        # self.logs = self.log_psd_fn(self.logf[self.n_data])
        self.logsc = np.log(self.mat.dot(self.beta))

    def psd_fn(self, x):

        mat = self.build_matrix(x)

        return mat.dot(self.beta)


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

    def __init__(self, n_data, fs, channel, scale=1.0, n_knots=30, D=3,
                 fmin=None, fmax=None, f_knots=None, ext=3):

        PSD.__init__(self, n_data, fs, fmin=fmin, fmax=fmax)
        self.channel = channel
        self.scale = scale
        self.log_psd_fn = lambda x: np.log(self.psd_fn(np.exp(x)))

    def psd_fn(self, x):

        if self.channel == 'A':
            return tdi.noisepsd_AE(x, model='SciRDv1') * self.scale ** 2
        elif self.channel == 'E':
            return tdi.noisepsd_AE(x, model='SciRDv1') * self.scale ** 2
        elif self.channel == 'T':
            return tdi.noisepsd_T(x, model='SciRDv1') * self.scale ** 2

    def sample(self, npsd):
        sampling_result = self.sample_psd(npsd)
        sample_list = [samp[0] for samp in sampling_result]
        logp_values_list = [samp[1] for samp in sampling_result]

        return sample_list, logp_values_list







# =============================================================================
# Multipurpose PSD model
# =============================================================================
class ModelFDDataPSD(PSD):
    '''
    Specialization of the bayesdawn psd model class which provides a bayesdawn PSD model 
    build on a possible combination of analytic (LDC) noise models possibly with smoothing 
    and/or additional data fitting.  
    
    Parameters
    ----------
    data : numpy rec-array (or dict)
        Multichannel Fourier data set [named columns 'f','A','E',...
    channel : str
        Tag specifying the channel to use for this instance, eg 'A', 'E'
    fit_type :  str = 'poly' 
        Tag for the type of model fitting to apply, or None to use the LDC model.
        Supported ['poly', 'log_poly', 'spline', ..., 'logratio_spline']
    fit_dof : int [default 4]
        Number of degrees of freedom in the fit.
    fit_logx : bool [default True]
        If True use log(f) as the coordinate for fitting functions.
    noise_model : str [default 'spritz']
        Tag for the LDC base model to use for ratio, or None for no ratio.
    smooth_df: float [default None]
        Freq width scale to set number of points for smoothing
    fmin: float [default 1e-5]
        Minimum frequency for result
    fmax: float [default None]
        Maximum frequency for result
    offset_log_fit: bool [default True]
        Correct the PSD bias caused by fitting in log space. Only set this to False to demonstrate the bias.
        
    Returns
    -------
    bayesdawn.psfmodel.PSD object
    
    For the specified channel analytic PSD model will be generated. If noise_model is not None,
    then the model will be generated by a ratio against the specified analytic noise model.
    
    If smooth_df is given, then the analytic noise model will be smoothed, by convolution with a
    Hanning window. 
    
    The type of fitting is specifed by fit_type, with 'poly' providing a fit using np.polyfit with 
    polynomial degree fit_dof-1. A variation on this is 'log_poly' which fits the ratio of the log 
    of the two quantities, scaled by the sum of their max values. The 'logratio_poly' option fits
    the log of the ratio vs the model function.  The 'spline', 'log_spline' and 'logratio_spline' 
    options work analogously, but fit the data to a B-spline, with the number of knots guided by the
    fit_dof.  
    
    If fit_type is None, then the analytic model (possibly smoothed) is applied directly and only 'f' 
    will be used from the data.

    
    
    Notes 
    ----
    
    This derived class doesn't yer support a MCMC sampled spline fit for the PSD, though that should
    not be too difficult.

    This class assumes, and builds off Fourier data, though it would not be difficult to also allow
    time-domain data directly and then to compute an FT for it.
    
    '''

    def __init__(self, data, channel, fit_type='logratio_spline',fit_dof=10, fit_logx=True, noise_model='spritz', smooth_df=None, fmin=1e-5, fmax=None, offset_log_fit=True):
    
        self.channel = channel
        self.fit_type = fit_type
        self.chdata=data[channel]

        f=data['f']
        df=(f[-1]-f[0])/(len(f)-1) 
        self.df=df
        
        fs=f[-1]*2
        ndata=(len(f)-1)*2
        self.ndata=ndata
        
        if fmax is not None:
            self.chdata=self.chdata[f<=fmax]
            f = f[f<=fmax]
        if fmin is not None:
            self.chdata=self.chdata[f>=fmin]
            f = f[f>=fmin]
            
        self.fin=f.copy()
        
        PSD.__init__(self, ndata, fs, fmin=fmin, fmax=fmax)
        
        self.scalefac=2*self.df
        
        if noise_model is not None:
            from ldc.lisa.noise import get_noise_model
            Nmodel=get_noise_model(noise_model, f)
            Sinit = Nmodel.psd(tdi2=True, option=self.channel, freq=f)
            if smooth_df is not None:
                nsmooth=int(smooth_df/df)
                w=np.hanning(nsmooth)
                smooth=lambda s:np.convolve(w/w.sum(),s,mode='same')
                Sinit=smooth(Sinit)
        else: 
            Sinit=None
            if fit_type is None: raise ValueError("Must specify at least fit_type or noise_model")
        self.Sinit=None
        if Sinit is not None:
            self.logSinit=interpolate.interp1d(np.log(f),np.log(Sinit),fill_value="extrapolate")
            self.Sinit=lambda x:np.exp(self.logSinit(np.log(x)))
        
        self.fit=None
        if fit_type is None: return
        self.fit_dof=fit_dof
        
        self.fit_logx=fit_logx
        if fit_logx: x=np.log(f)
        else: x=f

        # We internally hold info about the fit type in the combination of
        #     fit_scale:  in ['linear', 'log', 'logratio']
        #  and fit_func:  in ['poly', 'spline']
        if not fit_type.startswith('log'):
            fit_scale='linear'
            fit_func=fit_type
            #print('Fitting on linear-scale')
        elif fit_type.startswith('log_'):
            fit_scale='log'
            fit_func=fit_type[4:]
            #print('Fitting on log-scale')
        elif fit_type.startswith('logratio_'):
            fit_scale='logratio'
            fit_func=fit_type[9:]
            #print('Fitting on log-scale')
        else: raise ValueError('Fit scale not understood for fit_type'+fit_type)
        # Now a check:
        if not fit_func in ['poly','spline']: raise ValueError("Couldn't process fit_type '"+fit_type+"', fit_func='"+fit_func+"'")        
        self.fit_scale=fit_scale
        self.fit_func=fit_func

        # prepare the data for fitting
        y=np.abs(self.chdata)**2*self.scalefac                
        if not fit_scale=='log' and self.Sinit is not None:
            zero=2*(max(y)+max(Sinit))
        else: zero=1
        self.fit_zero=zero

        if self.Sinit is not None:
            if fit_scale=='linear':
                y=y/self.Sinit(self.fin)
            elif fit_scale=='log':
                y=np.log(y/zero)/np.log(self.Sinit(self.fin)/zero)
            elif fit_scale=='logratio':
                y=np.log(y/self.Sinit(self.fin))
            else: raise ValueError('Unknown fit_scale '+fit_scale)
        else:
            if fit_scale=='linear':
                pass
            elif fit_scale=='log':
                y=np.log(y/zero)
            elif fit_scale=='logratio':
                y=np.log(y)
            else: raise ValueError('Unknown fit_scale '+fit_scale)

        # Offset the data when the fit is in log space
        # Because the mean of log is lower than the log of the mean by euler_gamma
        self.offset_log_fit = offset_log_fit and fit_scale in ['log','logratio']
        
        #perform the fit
        if fit_func=='poly':
            pf = np.polyfit(x,y,fit_dof+1,w=1/f)
            #print('poly fit:',pf)
            fitpoly=np.poly1d(pf)
            self.fit = lambda x: fitpoly(x)
        elif fit_func=='spline':
            # I tried using the functionality in psdmodel.py but couldn't get it working.
            # to stay close to the existing implementation in psdmodel.py                    
            print(f[0],'< f < ',f[-1],'fmin/fmax=',fmin,fmax)
            n_knots=fit_dof-2 #dof=n_knots+2 (maybe, sort of guessing)                    
            knots=self.choose_knots(f) #This function needs f, not x
            xknots=knots
            print('lowest f knots:',knots[:5])
            if fit_logx: xknots=np.log(knots) #transform to x-space if needed
            try:
                fitspline=interpolate.LSQUnivariateSpline(x, y, xknots[1:-1], w=1/f, k=3, ext=3, check_finite=False)
            except ValueError:
                
                print('Problem creating spline:')
                t= xknots[1:-1]
                print('inputs:\n  x=',x,'\n  y=',y,'\n  t=', t,'\n  w=',1/f)
                # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
                xb = x[0]
                xe = x[-1]
                k=3
                t = np.concatenate(([xb]*(k+1), t, [xe]*(k+1)))
                n = len(t)
                print('internal form of t',t)
                if not np.all(t[k+1:n-k]-t[k:n-k-1] > 0, axis=0):
                    raise ValueError('Interior knots t must satisfy '
                                     'Schoenberg-Whitney conditions')
                print('1) k+1 <= n-k-1 <= m:',k+1,n-k-1,len(x))
                print('2a) t(1) <= t(2) <= ... <= t(k+1)',t[:k+1])
                print('2b) t(n-k) <= t(n-k+1) <= ... <= t(n)',t[n-k-1:])
                print('3) t(k+1) < t(k+2) < ... < t(n-k): min(t[x+1]-t[x]) over interior range:',min(t[k+1:n-k]-t[k:n-k-1]))
                print('4) t(k+1) <= x(i) <= t(n-k): t[k],min(x),max(x),t[n-k-1]', t[k],min(x),max(x),t[n-k-1])
                print('5): Schoenberg and Whitney condition')
                #Now we check the Schoenberg-Whitney condition
                j=0
                k=3
                iscan=0
                while j+k+1<len(t):
                    nmin=1#minimum number of points in the segment
                    #First scan to the relevant part of the data
                    while x[iscan]<=t[j] and iscan<len(x):iscan+=1
                    #check whether the next point exists and is in the target range
                    if iscan>=len(x) or x[iscan]>=t[j+k+1]: 
                        #fails the condition. To fix, we remove an intermediate knot and try again
                        jkill=j+k//2+1
                        print('Fails for knot at ',t[jkill],'because no data in',t[j],'< t <',t[j+k+1])
                        print('nearby data:',x[iscan-2:iscan+4])
                    j+=1
                
                import scipy.interpolate.dfitpack as dfitpack
                if not dfitpack.fpchec(x, t, k) == 0:
                    print('Failed')
                scipy.interpolate.LSQUnivariateSpline(x, y, xknots[1:-1], w=1/f, k=3, ext=3, check_finite=False)
                    
            self.knots=knots
            self.fit=lambda x:fitspline(x)
                
        else:    
            raise ValueError('fit_func '+str(fit_func)+' not recognized.')
            

            
    def choose_knots(self,x,verbose=False):
        '''
        Arguments:
                  x : Numpy array
                      Data coordinates for which the spline will be applied. These are used for setting the 
                      spline range and for checking the suitability of the knots.
                 
        Returns:
             Numpy array with the ordered knots
             
        For spline data we want to set the location of the knots. There are diverse ways to do this
        of course.  Here we follow the scheme in bayesdawn.psdmodel (fixed).  For B-splines to work
        the knots have to meet a constraint, dependent on the data.  From the scipy documentation
        for LSQUnivariateSpline:
           "Knots t must satisfy the Schoenberg-Whitney conditions, i.e., there must be a subset of 
            data points x[j] such that t[j] < x[j] < t[j+k+1], for j=0, 1,...,n-k-2."
        If the conditions aren't met, knots will be dropped until the condition is met, this may result
        in an actual number of knots less than desired to realize the value of fit_dof. 
        [We could add a interative loop to try to improve that.]
        '''
        if verbose: print('Finding knots in ',x[0],'< x <',x[-1])
        nknots=self.fit_dof #Approx guess of the relation of knots to dof
        minf=x[0]
        maxf=x[-1]
        base=(maxf/minf)**(1/nknots)
        print('base=',base)
        # We use this choose_frequency_knots function, decreasing the 'base' value when
        # there are more knots desired to ensure that the benefit of more knots also
        # shows at lower frequencies. It is not clear whether this treatment is any
        # better than just even logarithmic sampling.
        t=choose_frequency_knots(nknots+2, freq_min=x[0], freq_max=x[-1],base=base)
        t=np.concatenate(([x[0]],t,[x[-1]])) #we add on points at the boundary to ensure that the check condition check is thorough
        if verbose: print(len(t),'knots before checking')
        i=0
        tlast=x[0]
        while i<len(t):
            if t[i]<=tlast: #not strictly ordered
                if verbose: print('Dropping knot at ',t[i],'because not beyond last knot at',tlast)
                t=np.delete(t,i)
            else:
                tlast=t[i]
                i+=1
                
        
        #Now we check the Schoenberg-Whitney condition
        j=0
        k=1 #we use 1 instead of 3 to be conservative, and to hopefully avoid issues that seem to happen near ends
        iscan=0
        while j+k+1<len(t):
            nmin=1#minimum number of points in the segment
            #First scan to the relevant part of the data
            while x[iscan]<=t[j] and iscan<len(x):iscan+=1
            #check whether the next point exists and is in the target range
            if iscan>=len(x) or x[iscan]>=t[j+k+1]: 
            #if iscan>=len(x) or x[iscan]>=t[j+k]: #Artificially tighten condition
                #fails the condition. To fix, we remove an intermediate knot and try again
                jkill=j+k//2+1
                if verbose: print('Dropping knot at ',t[jkill],'because no data in',t[j],'< t <',t[j+k+1])
                t=np.delete(t,jkill)
            else:
                j+=1
        if verbose: print(len(t),'knots after ensuring condition.')
        t=t[1:-1] #drop the boundaries (scipy will put them in)
        return t
    
    def plot(self,show_fit=False, tag=None, ref=None):
        '''
        Interactively how a plot illustrating how the PSD model compares with the data.
        Arguments:
           show_fit: bool
             Include a second plot illustrating directly how the fit (if any) works
           tag: string
             A label to include in the plot title
           ref: psdmodel
             Another PSD model to serve as a reference in the plot
        '''        
        
        if tag is None: 
            tag=self.fit_type
        if tag is None:
            tag='no fit'

        import matplotlib.pyplot as plt
        plt.loglog(self.fin,np.abs(self.chdata*np.sqrt(self.scalefac)),label='data ',alpha=0.3)
        if ref is not None:
            plt.loglog(self.fin,np.sqrt(ref.psd_fn(self.fin)),label='ref')
        plt.loglog(self.fin,np.sqrt(self.psd_fn(self.fin)),label='model ('+tag+')')
            
        plt.legend()                
            
        if show_fit and self.fit is not None and self.Sinit is not None:
            if self.fit_logx: x=np.log(self.fin)
            else: x=self.fin

            if ref is not None:
                y0=ref.psd_fn(self.fin)
            else:
                y0=1
                
            y=np.abs(self.chdata)**2*self.scalefac
            fit_scale=self.fit_scale
            if fit_scale=='linear':
                if self.Sinit is not None:
                    #print('Sinit min/max',min(Sinit),max(Sinit))
                    fac=self.Sinit(self.fin)
                    y=y/fac
                    y0=y0/fac
            elif fit_scale=='log':
                zero=self.fit_zero
                y=np.log(y/zero)
                y0=np.log(y0/zero)
                if self.Sinit is not None:
                    fac=(self.logSinit(np.log(self.fin))-np.log(zero))
                    y=y/fac
                    y0=y0/fac
            elif fit_scale=='logratio':
                y=np.log(y)
                y0=np.log(y0)
                if self.Sinit is not None:
                    #off=self.logSinit(np.log(self.fin))
                    off=np.log(self.Sinit(self.fin))
                    y=y-off
                    y0=y0-off
            else: raise ValueError("fit_scale unknown")

            spline=self.fit_func=='spline'
            if False:
                print('x',x[:5])
                print('y',y[:5])
                print('fit',y[:5])
                print('y v y0 min,max:',min(y),min(y0),max(y),max(y0))

            plt.show()

            pltfunc=plt.plot
            pltfunc(self.fin,(y),label='data ratio ')
            pltfunc(self.fin,(y0),label='ref ratio ')
            pltfunc(self.fin,self.fit(x),label='fit ('+tag+')')
            if spline:
                xknots=self.knots
                if self.fit_logx: xknots=np.log(xknots)
                
                pltfunc(self.knots,self.fit(xknots),'*',label='knots ('+tag+')')
            plt.legend()
            plt.show() 
            if False:
                plt.loglog(self.fin,(y),label='data ratio ('+tag+')')
                plt.loglog(self.fin,self.fit(x),label='fit ('+tag+')')
                plt.legend()
                plt.show() 

    def psd_fn(self, x):
        # returns the psd function defined earlier           

        if self.fit is not None:
            xx=x.copy()
            if self.fit_logx:
                xx=np.log(x)
            if self.fit_scale=='linear':
                dm = np.abs(self.fit(xx))            
                if self.Sinit is not None:
                    dm = dm*self.Sinit(x)
            elif self.fit_scale=='log':
                logdm = self.fit(xx)
                zero=self.fit_zero
                if self.Sinit is not None:
                    logdm*=(self.logSinit(np.log(x))-np.log(zero))
                dm=zero*np.exp(logdm)
            elif self.fit_scale=='logratio':
                logdm = self.fit(xx)
                if self.Sinit is not None:
                    logdm+=self.logSinit(np.log(x))
                dm=np.exp(logdm)
            if self.offset_log_fit:
                dm=dm*np.exp(np.euler_gamma)
        else:
            dm = self.Sinit(x)


        return dm
