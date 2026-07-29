# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2026
"""Module to generate colored noise from a given PSD"""
import warnings
import numpy as np
from pyfftw.interfaces.numpy_fft import ifft


def generate_freq_noise_from_psd(psd, fs, myseed=None):
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

    n_fft = int((n_psd - 1) / 2)

    if psd.ndim == 1:
        psd_sqrt = np.sqrt(psd[0 : n_fft + 2])
        # Real part of the Noise fft : it is a gaussian random variable
        noise_tf_real = (
            np.sqrt(0.5)
            * psd_sqrt[0 : n_fft + 1]
            * np.random.normal(loc=0.0, scale=1.0, size=n_fft + 1)
        )
        # Imaginary part of the Noise fft :
        noise_tf_im = (
            np.sqrt(0.5)
            * psd_sqrt[0 : n_fft + 1]
            * np.random.normal(loc=0.0, scale=1.0, size=n_fft + 1)
        )
        # The Fourier transform must be real in f = 0
        noise_tf_im[0] = 0.0
        noise_tf_real[0] = noise_tf_real[0] * np.sqrt(2.0)
        # Create the NoiseTF complex numbers for positive frequencies
        noise_tf = noise_tf_real + 1j * noise_tf_im
    elif psd.ndim == 3:
        # Number of variables
        p = psd.shape[1]
        # Form the covariance matrices in the Fourier domain
        cov = psd[0 : n_fft + 1]
        # Perform Cholesky factorization of the spectrum matrices
        psd_sqrt = np.linalg.cholesky(cov)
        # Real part of the Noise fft : it is a gaussian random variable
        w_real = np.sqrt(0.5) * np.random.multivariate_normal(
            np.zeros(p), np.eye(p), size=n_fft + 1
        )
        w_imag = np.sqrt(0.5) * np.random.multivariate_normal(
            np.zeros(p), np.eye(p), size=n_fft + 1
        )
        # Generate the Z_m in the Fourier domain
        noise_tf = np.einsum("...jk, ...k", psd_sqrt, w_real + 1j * w_imag)
        # The Fourier transform must be real in f = 0
        noise_tf[0].imag = 0
        noise_tf[0].real = noise_tf[0].real * np.sqrt(2.0)

    # To get a real valued signal we must have NoiseTF(-f) = NoiseTF*
    if (n_psd % 2 == 0) & (psd.ndim == 1):
        # The TF at Nyquist frequency must be real in the case of an even number of data
        noise_sym0 = np.array([psd_sqrt[n_fft + 1] * np.random.normal(0, 1)])
        # Add the symmetric part corresponding to negative frequencies
        noise_tf = np.hstack(
            (noise_tf, noise_sym0, np.conj(noise_tf[1 : n_fft + 1])[::-1])
        )
    elif (n_psd % 2 != 0) & (psd.ndim == 1):
        noise_tf = np.hstack((noise_tf, np.conj(noise_tf[1 : n_fft + 1])[::-1]))

    elif (n_psd % 2 == 0) & (psd.ndim == 3):
        psd_sqrt_nyq = np.linalg.cholesky(psd[n_fft + 1])
        noise_sym0 = psd_sqrt_nyq @ np.random.multivariate_normal(
            np.zeros(p), np.eye(p)
        )
        noise_tf = np.concatenate(
            (
                noise_tf,
                noise_sym0[np.newaxis, :],
                np.conj(noise_tf[1 : n_fft + 1])[::-1],
            )
        )

    elif (n_psd % 2 != 0) & (psd.ndim == 3):
        noise_tf = np.concatenate((noise_tf, np.conj(noise_tf[1 : n_fft + 1])[::-1]))
    else:
        warnings.WarningMessage(
            "Invalid spectrum dimension", UserWarning, "invalid_dim", 149
        )

    return np.sqrt(n_psd * fs / 2.0) * noise_tf


def generate_noise_from_psd(psd, fs, myseed=None):
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

    return np.real(ifft(generate_freq_noise_from_psd(psd, fs, myseed=myseed), axis=0))
