#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
# This code provides routines for PSD estimation using a peace-continuous model
# that assumes that the logarithm of the PSD is linear per peaces.
import numpy as np
from scipy import linalg as LA
from scipy import interpolate
from scipy import optimize
import patsy
import copy

# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft

# TODO: rewrite spline interpolation with LSQUnivariateSpline from scipy.interpolate

# ==============================================================================
# SPLINES
# ==============================================================================

def find_closest_points(f_target, f):
    """

    Find frequencies in f closest to the frequencies in f_target

    """

    inds = [np.argmin(np.abs(f-f0)) for f0 in f_target]

    return inds

def spline_loglike(beta, per, A):
    """

    Whittle log-likelihood with spline PSD model

    Parameters
    ----------
    per : array_like
        vector of periodogram calculated at log-frequencies
    spl : instance of PSD_spline
        spline object


    Returns
    -------
    ll : scalar float
        value of the log-likelihood


    """

    psdmodel = A.dot(beta)

    return - 0.5*np.sum( np.log(psdmodel) + per/psdmodel )


def spline_loglike_grad(beta, per, A):
    """

    Gradient of the Whittle log-likelihood with spline PSD model

    Parameters
    ----------
    per : array_like
        vector of periodogram
    spl : instance of PSD_spline
        spline object


    Returns
    -------
    grad_ll : 1d numpy array
        gradient of the log-likelihood


    """

    psdmodel = np.dot(A,beta)

    grad_ll = - 0.5*np.dot( A.T , 1/psdmodel * ( 1  -  per/psdmodel ) )

    #grad_ll = np.array([np.sum( spl.derivatives() )])

    return grad_ll

def spline_loglike_hessian(beta, per, A):
    """

    Hessian matrix of the Whittle log-likelihood with spline model for the PSD

    Parameters
    ----------
    x : array_like
        vector of log-frequencies taken into account in the likelihood
    per : array_like
        vector of periodogram calculated at log-frequencies minus C0
    spl : instance of PSD_spline
        spline object


    Returns
    -------
    grad_ll : 1d numpy array
        gradient of the log-likelihood


    """

    psdmodel = np.dot(A,beta)

    E = 1/psdmodel**2 * ( -1 + 2 * per/psdmodel)

    AE = np.array([A[:,j]*E for j in range(A.shape[1])]).T

    hessian = - 0.5 * np.dot( A.T , AE )

    #grad_ll = np.array([np.sum( spl.derivatives() )])

    return hessian


def newton_raphson(beta_0, grad_func, hess_func, maxiter=1000, tol=1e-4):
    """

    Newton-Raphson algorithm to compute the maximum likelihood

    """

    eps = 1.0
    i = 0
    beta_old = beta_0

    while ((i<maxiter) & (eps > tol)):

        beta = beta_old - LA.inv(hess_func(beta_old)).dot( grad_func(beta_old) )
        eps = LA.norm(beta - beta_old)/LA.norm(beta_old)
        beta_old = copy.deepcopy(beta)
        i = i+1


    print("Criterium at the end: " + str(eps))
    print("Number of iterations: " + str(i))

    return beta

def spline_fisher(A):
    """

    Compute the Fisher matrix for the spline PSD model parameters.


    """

    return 0.5*A.conj().T.dot(A)


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
    x : numpy array of size N
        abscisse points where to compute the spline
    knots : numpy array of size J-1
        knots of the spline (asbisse of segments nodes)
    D : scalar integer
        degree of the spline

    Returns
    -------

    A : 2d numpy array
        spline design matrix

    """

    # Size of the matrix
    #K = D + 1 + len(knots)

    # Polynomial
    A = [x**d for d in range(D+1)]

    # Truncated polynomial
    A2 = [np.concatenate((np.zeros(len(x[x < xi])), (x[x >= xi]-xi)**D)) for xi in knots]
    A.extend(A2)

    A_mat = np.array(A).T
    #
    # mu = np.mean(A_mat,axis = 0)
    # for j in range(A_mat.shape[1]):
    #     A_mat[:,j] = A_mat[:,j]/np.mean(A_mat[:,j])

    return A_mat




# ==============================================================================
# PSD CLASS
# ==============================================================================

class PSD_spline(object):

    def __init__(self, N, fs, J=30, D=3, fmin=None, fmax=None):

        # Number of knots for the log-PSD spline model
        self.J = J
        self.fs = fs
        # Size of the sample
        self.N = N
        self.f = np.fft.fftfreq(N)*fs
        self.n = np.int((N-1)/2.)
        
        if fmin is None:
            fmin = fs/N
        if fmax is None:
            fmax = fs/2
            
        # Set the knot grid
        self.f_knots = self.choose_knots(J, fmin, fmax)
        self.logf_knots = np.log(self.f_knots)
        # Logarithm of positive Fourier frequencies for a grid of size N
        self.f = np.fft.fftfreq(N)*fs

        self.fmin = fmin
        self.fmax = fmax
        
        # Spline order
        self.D = D

        self.C0 = -0.57721

        # Create a dictionary corresponding to each data length
        self.logf = {N:np.log(self.f[1:self.n+1])}
        # Spline coefficient vector
        self.beta = []
        # PSD at positive Fourier frequencies
        self.logS = []

        # Flexible interpolation of the estimated PSD
        self.logPSD_fn = None
        # Spectrum calcualed on the Fourier frequency grid
        self.S = []

        # For Bayesian inference: the spline interpolation
        # Control frequencies
        self.logfc = np.concatenate((np.log(self.f_knots), [np.log(self.fs/2)]))
        # Variance function values at control frequencies
        self.varlogSc = np.array([3.60807571e-01, 8.90158814e-02, 1.45631966e-02, 3.55646693e-03,
                                  1.09926717e-03, 4.15894275e-04, 1.86984136e-04, 9.73883423e-05,
                                  5.74981099e-05, 3.77721249e-05, 2.71731280e-05, 2.11167300e-05,
                                  1.75209167e-05, 1.53672320e-05, 1.41269765e-05, 1.35137347e-05,
                                  1.33692054e-05, 1.36074455e-05, 1.41863625e-05, 1.50926724e-05,
                                  1.63338849e-05, 1.79341767e-05, 1.99325803e-05, 2.23827563e-05,
                                  2.53543168e-05, 2.89370991e-05, 3.32545462e-05, 3.85055177e-05,
                                  4.50144967e-05, 5.26798764e-05, 4.86680827e-04])

        # Spline estimator of the variance of the log-PSD estimate
        self.logvar_fn = interpolate.interp1d(self.logfc[1:], self.varlogSc[1:], kind='cubic', fill_value="extrapolate")

    def choose_knots(self, J, fmin, fmax):
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

        ns = - np.log(fmax)/np.log(10)
        n0 = - np.log(fmin)/np.log(10)
        jvect = np.arange(0,J)
        alpha_guess = 0.8

        targetfunc = lambda x : n0 - (1-x**(J))/(1-x) - ns
        result = optimize.fsolve(targetfunc, alpha_guess)
        alpha = result[0]
        n_knots = n0 - (1-alpha**jvect)/(1-alpha)
        f_knots = 10**(-n_knots)

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
        I = np.abs(fft(y * w))**2/np.sum(w**2)

        # Compute the spline parameter vector for the log-PSD model
        self.estimate_from_I(I)

    def periodogram(self, y_fft, K2=None):
        """
        Simple periodogram with no windowing
        """
        if K2 == None:
            I = np.abs(y_fft)**2/len(y_fft)
        else:
            I = np.abs(y_fft)**2/K2
        return I

    def estimate_from_freq(self, y_fft, K2=None):
        """

        Estimate the log-PSD using spline model by least-square method from the
        discrete Fourier transformed data. This function is useful to avoid
        to compute FFTs multiple times.


        """
        
        # If there is only one periodogram
        if type(y_fft) == np.ndarray:
            I = self.periodogram(y_fft, K2=K2)
            #self.beta = self.spline_lsqr(I)
        # Otherwise calculate the periodogram for each data set:
        elif (type(y_fft) == list):
            I = [self.periodogram(y_fft[i], K2=K2[i]) for i in range(len(y_fft))]
             
        self.estimate_from_I(I)
   

    def estimate_from_I(self, I):
        """

        Estimate PSD from the periodogram

        """

        # If there is only one periodogram
        if type(I) == np.ndarray:
            self.logPSD_fn = self.spline_lsqr(I)
            self.beta = self.logPSD_fn.get_coeffs()
        elif type(I) == list:
            # If there are several periodograms, average the estimates
            spl_list = [self.spline_lsqr(I0) for I0 in I if self.fs / len(I0) < self.f_knots[0]]
            self.beta = sum([spl.get_coeffs for spl in spl_list])/len(I)
            self.logPSD_fn = interpolate.BSpline(spl_list[0].get_knots(), self.beta, self.D)

        # Estimate psd at positive Fourier log-frequencies
        self.logS = self.logPSD_fn(self.logf[self.N])

        # # Spline estimator of the variance of the log-PSD estimate
        # self.logvar_fn = interpolate.LSQUnivariateSpline(self.logf[self.N],
        #                                                  (np.log(I[1:self.n+1]) - self.C0 - self.logS)**2,
        #                                                  self.logf_knots, k=1, ext='const')

        # Update PSD control values (for Bayesian estimation)
        self.logSc = self.logPSD_fn(self.logfc)
        # self.varlogSc = self.logvar_fn(self.logfc)


    def spline_lsqr(self, I):
        """

        Fit a spline to the log periodogram using least-squares

        """

        NI = len(I)
        
        if NI not in list(self.logf.keys()):
            f = np.fft.fftfreq(NI)*self.fs
            self.logf[NI] = np.log(f[f>0])
            
        n = np.int((NI-1)/2.)
        z = I[1:n+1]
        v = np.log(z) - self.C0

        # Spline estimator of the log-PSD
        spl = interpolate.LSQUnivariateSpline(self.logf[NI], v, self.logf_knots, k=self.D, ext="extrapolate")

        return spl

    def calculate(self, arg):
        """
        Calculate the PSD at frequencies x

        """

        if (type(arg) == np.int) | (type(arg) == np.int64):
            N = arg

            # Symmetrize the estimates
            if (N % 2 == 0): # if N is even
                # Compute PSD from f=0 to f = fs/2
                if N == self.N :
                    # f_compl = np.abs( np.array([f[1]/10.,f[n+1]]) )
                    # A_compl = self.compute_A(np.log(f_compl),self.logf_knots,self.D)
                    # A = np.vstack((A_compl[[0],:],self.A_log,A_compl[[1],:]))
                    # SN = np.exp( np.dot(A,self.beta) )
                    n = self.n
                    #f_tot = np.abs( np.concatenate(([self.f[1]/10.],self.f[1:n+2])) )
                    # for f = 0 we take S(0) = S(f[1])
                    f_tot = np.abs( np.concatenate(([self.f[1]],self.f[1:n+2])) )
                    SN = np.exp( self.logPSD_fn(np.log(f_tot)) )

                else:

                    f = np.fft.fftfreq(N)*self.fs
                    n = np.int( (N-1)/2. )
                    #f_tot = np.abs( np.concatenate(([f[1]/10.],f[1:n+2])) )
                    f_tot = np.abs( np.concatenate(([f[1]],f[1:n+2])) )
                    # A = self.compute_A(np.log(f_tot),
                    # self.logf_knots,
                    # self.D)
                    SN = np.exp( self.logPSD_fn(np.log(f_tot)) )

                #SN = self.cs(np.abs(f[0:n+2]))
                SN_sym = np.concatenate((SN[0:n+1],SN[1:n+2][::-1]))

            else: # if N is odd
                if N == self.N:
                    f_tot = np.abs( np.concatenate(([self.f[1]],self.f[1:self.n+1])) )
                    SN = np.exp(self.logPSD_fn(np.log(f_tot)))

                else:

                    f = np.fft.fftfreq(N)*self.fs
                    n = np.int( (N-1)/2. )
                    f_tot = np.abs( np.concatenate(([f[1]],f[1:n+1])) )
                    SN = np.exp( self.logPSD_fn(np.log(f_tot)) )

                SN_sym = np.concatenate((SN[0:n+1],SN[1:n+1][::-1]))

        elif type(arg) == np.ndarray:

            f = arg[:]
            SN_sym = np.exp(self.logPSD_fn(np.log(f)))

        else:

            raise TypeError("Argument must be integer or ndarray")

        return SN_sym

    def calculate_autocorr(self, N):
        """
        Compute the autocovariance function from the PSD.

        """

        return np.real(ifft(self.calculate(2*N))[0:N])



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

    nu = 4 + 2 * mu**2 / var
    s2 = (nu-2)/nu * mu

    return nu,s2


def log_normal_distribution(mu_X, var_X):
    """

    Compute the mean and variance of the log-normal distribution given the
    mean and variance of the underlying normal distribution

    Y = exp( X )

    """
    # Log-normal distribution mean and variance
    mu_Y = np.exp( mu_X + 0.5 * var_X )
    var_Y = np.exp( 2*mu_X + var_X ) * (np.exp(var_X) - 1 )

    return mu_Y,var_Y

