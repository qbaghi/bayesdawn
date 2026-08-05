# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2026
"""Module to generate colored noise from a given PSD"""
import warnings
import numpy as np
from pyfftw.interfaces.numpy_fft import irfft
from scipy.signal import ShortTimeFFT


def generate_positive_freq_noise_from_psd(psd, myseed=None):
    """Generate the nonnegative-frequency half-spectrum for a real process."""

    n_psd = len(psd)
    np.random.seed(myseed)
    half_len = n_psd // 2 + 1

    if psd.ndim == 1:
        if n_psd % 2 == 0:
            psd_sqrt = np.sqrt(psd[:half_len])
            noise_tf_real = (
                np.sqrt(0.5)
                * psd_sqrt[: half_len - 1]
                * np.random.normal(loc=0.0, scale=1.0, size=half_len - 1)
            )
            noise_tf_im = (
                np.sqrt(0.5)
                * psd_sqrt[: half_len - 1]
                * np.random.normal(loc=0.0, scale=1.0, size=half_len - 1)
            )
            noise_tf_im[0] = 0.0
            noise_tf_real[0] = noise_tf_real[0] * np.sqrt(2.0)
            noise_tf = noise_tf_real + 1j * noise_tf_im
            noise_tf = np.concatenate((noise_tf, np.array([psd_sqrt[-1] * np.random.normal(0, 1)])))
        else:
            psd_sqrt = np.sqrt(psd[:half_len])
            noise_tf_real = (
                np.sqrt(0.5)
                * psd_sqrt
                * np.random.normal(loc=0.0, scale=1.0, size=half_len)
            )
            noise_tf_im = (
                np.sqrt(0.5)
                * psd_sqrt
                * np.random.normal(loc=0.0, scale=1.0, size=half_len)
            )
            noise_tf_im[0] = 0.0
            noise_tf_real[0] = noise_tf_real[0] * np.sqrt(2.0)
            noise_tf = noise_tf_real + 1j * noise_tf_im

    elif psd.ndim == 3:
        p = psd.shape[1]

        if n_psd % 2 == 0:
            cov = psd[: half_len - 1]
            psd_sqrt = np.linalg.cholesky(cov)
            w_real = np.sqrt(0.5) * np.random.multivariate_normal(
                np.zeros(p), np.eye(p), size=half_len - 1
            )
            w_imag = np.sqrt(0.5) * np.random.multivariate_normal(
                np.zeros(p), np.eye(p), size=half_len - 1
            )
            noise_tf = np.einsum("...jk, ...k", psd_sqrt, w_real + 1j * w_imag)
            noise_tf[0].imag = 0
            noise_tf[0].real = noise_tf[0].real * np.sqrt(2.0)

            psd_sqrt_nyq = np.linalg.cholesky(psd[-1])
            noise_sym0 = psd_sqrt_nyq @ np.random.multivariate_normal(
                np.zeros(p), np.eye(p)
            )
            noise_tf = np.concatenate((noise_tf, noise_sym0[np.newaxis, :]))
        else:
            cov = psd[:half_len]
            psd_sqrt = np.linalg.cholesky(cov)
            w_real = np.sqrt(0.5) * np.random.multivariate_normal(
                np.zeros(p), np.eye(p), size=half_len
            )
            w_imag = np.sqrt(0.5) * np.random.multivariate_normal(
                np.zeros(p), np.eye(p), size=half_len
            )
            noise_tf = np.einsum("...jk, ...k", psd_sqrt, w_real + 1j * w_imag)
            noise_tf[0].imag = 0
            noise_tf[0].real = noise_tf[0].real * np.sqrt(2.0)
    else:
        warnings.WarningMessage(
            "Invalid spectrum dimension", UserWarning, "invalid_dim", 149
        )
        return None

    return noise_tf


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

    n_psd = len(psd)
    noise_tf = generate_positive_freq_noise_from_psd(psd, myseed=myseed)

    if psd.ndim == 1:
        if n_psd % 2 == 0:
            noise_tf = np.hstack((noise_tf, np.conj(noise_tf[1:-1])[::-1]))
        else:
            noise_tf = np.hstack((noise_tf, np.conj(noise_tf[1:])[::-1]))
    elif psd.ndim == 3:
        if n_psd % 2 == 0:
            noise_tf = np.concatenate((noise_tf, np.conj(noise_tf[1:-1])[::-1]))
        else:
            noise_tf = np.concatenate((noise_tf, np.conj(noise_tf[1:])[::-1]))

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
    fs : scalar float
        sampling frequency
    myseed : scalar integer or None
        seed of the random number generator

    Returns
    -------
        b : numpy array
        time sample of the colored noise (size N)
    """

    noise_tf = generate_positive_freq_noise_from_psd(psd, myseed=myseed)
    return irfft(np.sqrt(len(psd) * fs / 2.0) * noise_tf, n=len(psd), axis=0)


def overlap_add(segments, win, hop, n_data):
    """
    Reconstruct a signal from overlapping time-domain segments.

    Parameters
    ----------
    segments : ndarray, shape (n_frames, L)
        Time-domain segments.
    win : ndarray, shape (L,)
        Synthesis window.
    hop : int
        Hop size.
    n_data : int
        Desired output length.

    Returns
    -------
    x : ndarray, shape (n_data,)
        Reconstructed signal.
    """
    L = len(win)

    # Use SciPy's own frame indexing
    SFT = ShortTimeFFT(win, hop=hop, fs=1.0)
    p = np.arange(SFT.p_min, SFT.p_max(n_data))

    if len(p) != len(segments):
        raise ValueError(
            f"Expected {len(p)} segments, got {len(segments)}."
        )

    x = np.zeros(n_data + L + (segments.shape[0]-1) * hop)
    wsum = np.zeros_like(x)

    offset = L

    for seg, pi in zip(segments, p):
        start = offset + pi * hop

        x[start:start+L] += seg * win
        wsum[start:start+L] += win**2

    # Normalize where windows overlap
    mask = wsum > 0
    x[mask] /= np.sqrt(wsum[mask])# Normalization to preserve the variance

    # Remove padding
    x = x[offset:offset+n_data]

    return x


def generate_time_noise_from_evolutionary_psd_function(psd_func, win, hop, fs, n_samples, 
                                                       myseeds=None, **kwargs):
    """
    Generate a Gaussian random field in the time domain assuming a locally stationary
    process for each time window.

    Parameters
    ----------
    psd_func : callable
        Function that takes frequency and time as input and returns the one-sided PSD at that 
        frequency and time. It can also return a 2d array of size (n_freq, n_chan, n_chan) for 
        multivariate processes.
    win : ndarray
        Tapper window.
    hop : int
        Hop size for the ShortTimeFFT. 
    fs : float
        Sampling frequency.
    n_samples : int
        Total number of samples to generate.
    myseeds : _type_, optional
        Array of Seeds for the random number generator, by default None.
    
    Returns
    -------
    noise_time : ndarray
        Generated time-domain noise of size (n_samples, n_chan) where n_chan is the number of
        channels.
    """

    # Instantiate the ShortTimeFFT class to get the time and frequency bins
    mfft = 2 * win.size  # Use twice the window size to avoid periodicity issues
    stft_cls = ShortTimeFFT(win, hop, fs, fft_mode='onesided', mfft=mfft)
    # Increase the desired signal length to avoid edge effects
    pad = win.size
    n_ext = n_samples + 2 * pad

    # Get the time and frequency bins for the STFT
    time_points = stft_cls.t(n_ext) - pad / fs / 2.0 # Adjust time points to account for padding

    # Full frequency array
    f_full = np.fft.fftfreq(mfft, d=1.0 / fs)
    # Generate the evolutionary PSD for each time window,  shape (n_freq, n_windows)
    psd_evolutionary = np.array([psd_func(f_full, t, **kwargs) for t in time_points]).T

    if myseeds is None:
        myseeds = [None] * psd_evolutionary.shape[1]

    noise_samples = np.asarray(
        [generate_noise_from_psd(psd_evolutionary[:, t], fs, myseed=myseeds[t])[0:win.size]
         for t in range(psd_evolutionary.shape[1])])
    noise_time = overlap_add(noise_samples, win, hop, n_ext)
    # Crop the time series by one hop on each side to avoid edge effects
    noise_time = noise_time[pad:-pad]
    return noise_time
