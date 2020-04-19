#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:11:56 2019

@author: qbaghi
"""
import numpy as np
from scipy import linalg as LA
from bayesdawn import gaps
from bayesdawn.utils import physics
# FTT modules
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
from bayesdawn.waveforms import lisaresp
pyfftw.interfaces.cache.enable()


def dft(x, w):
    """
    Discrete Fourier transform, which takes into account possible windowing,
    renormalized such that the amplitudes match those that would be obtained
    with Python FFT

    Parameters
    ----------
    x : ndarray or list
        time series
    w : ndarray
        time window

    Returns
    -------
    x_dft : ndarray or list
        DFT of the input time series

    """

    if type(x) == np.ndarray:
        return fft(x * w) * x.shape[0] / np.sum(w)
    elif type(x) == list:
        return [fft(xi * w) * xi.shape[0] / np.sum(w) for xi in x]


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

    return LA.pinv(mat_fft_weighed.conj().T.dot(mat_fft)).dot(
        mat_fft_weighed.conj().T.dot(y_fft))


class LikelihoodModel(object):
    """

    This class provide all the functions necessary to calcualte the likelihood
    of the gravitational wave data model

    """

    def __init__(self,
                 signal_model,
                 psd_model,
                 data_model,
                 names=[],
                 channels=['X1'],
                 fmin=1e-5,
                 fmax=0.5e-1,
                 nsources=1,
                 order=None,
                 reduced=False,
                 window='modified_hann',
                 n_wind=500, n_wind_psd=500000,
                 imputation=False,
                 psd_estimation=False,
                 normalized=False,
                 n_update=100):
        """

        Parameters
        ----------
        signal_model : instance of waveform.lisaresp.GWwaveform
            GW LISA response model class
        psd_model : instance of PSDSpline class
            class defining the noise model and methods necessary to update its
            parameters
        data_model : instance of the DataModel class
            class defining the way to perform missing data imputation
        names : list of strings
            parameter names
        channels : list of strings
            type of tdi channels to use
        fmin : scalar float
            mininum frequency where to compute the likelihood
        fmax : scalar float
            maximum frequency where to compute the likelihood
        nsources : scalar integer
            number of considered sources in the analysis
        order : list
            list of integers that specify an inequality constraint on the
            parameters, that will be reflected in the
            log-prior distribution.
        reduced : bool
            whether to use the full or the reduced likelihood model
        window : string
            Type of time windowing applied to the signal to prevent noise
            leakage for the GW parameter estimation.
            Default is "modified_hann", i.e. Tukey window
        n_wind : scalar integer
            smoothing parameter of the Tukey window applied to the signal to
            prevent noise leakage for the GW parameter estimation
        n_wind_psd : scalar integer
            smoothing parameter of the Tukey window applied to the residuals to
            prevent leakage for noise PSD estimation
        n_update : int
            frequency of auxiliary parameters update (PSD and missing data).
            Update will be done every n_update calls
            of the log_likelihood method

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
        # Waveform model
        self.signal_model = signal_model
        # Data model
        self.data_model = data_model
        # Noise model
        self.psd_model = psd_model
        # Update auxiliary parameters every n_update calls of the
        # log_likelihood function
        self.n_update = n_update
        # Counter of log_likelihood calls
        self.counter = 0
        # ======================================================================
        # Time windowing
        # ======================================================================
        # Type of time windowing
        self.window = window
        # Windowing smoothing parameter (optimized for signal estimation)
        self.n_wind = n_wind
        if n_wind > 0:
            self.w = gaps.gapgenerator.modified_hann(self.data_mode.n,
                                                     n_wind=self.n_wind)
        else:
            self.w = np.ones(self.data_model.n)
        # Normalization constant for signal amplitude
        self.k1 = np.sum(self.w)
        # Windowing smoothing parameter (optimized for noise psd estimation)
        self.n_wind_psd = n_wind_psd
        # Corresponding window function
        self.w_psd = gaps.gapgenerator.modified_hann(self.data_model.n,
                                                     n_wind=self.n_wind_psd)
        # If there are missing data, then compute a window that takes gaps into
        # account (smooth masking)
        if any(self.data_model.mask == 0):
            nd, nf = gaps.gapgenerator.findEnds(data_model.mask)
            self.w_psd_gaps = gaps.gapgenerator.windowing(
                nd, nf, self.data_model.n, window=self.window,
                n_wind=self.n_wind_psd)
        else:
            self.w_psd_gaps = self.w_psd[:]
        # Normalization constant for noise amplitude
        self.k1_psd = np.sum(self.w_psd)
        # Normalization constant for noise power spectrum
        self.k2_psd = np.sum(self.w_psd ** 2)

        # ======================================================================
        # Frequency bandwidth
        # ======================================================================
        self.fmin = fmin
        self.fmax = fmax
        # Frequency vector
        self.f = np.fft.fftfreq(self.data_model.n)*self.data_model.fs
        # Find corresponding indices in the frequency vector
        self.inds_pos = np.where((self.f >= fmin) & (self.f <= fmax))[0]
        self.inds_neg = self.data_model.n - self.inds_pos
        self.inds = np.concatenate((self.inds_pos, self.inds_neg))
        print("The estimation domain has size " + str(len(self.inds_pos)))
        # Number of positive frequencies considered
        self.npos = len(self.inds_pos)
        # Vector of analysed frequencies
        self.f_pos = self.f[self.inds_pos]

        # ======================================================================
        # Fourier-transformed data
        # ======================================================================
        # Windowed signal DFT optimized for PSD estimation
        self.y_fft_psd = dft(self.data_model.y, self.w_psd)
        # Windowed signal DFT optimized for signal estimation, renormalized to
        # have the same power as Python's FFT
        self.y_fft = dft(self.data_model.y, self.w)
        # Stack the channel data DFTs
        self.y_fft_stack = self.concatenate_data_pos(self.y_fft)

        # ======================================================================
        # Power spectral density
        # ======================================================================
        # Spectrum value (can be a numpy array or a list)
        self.spectrum = self.psd_model.calculate(self.data_model.n)
        self.psd_samples = []
        self.psd_logpvals = []
        self.psd_save = 1
        # Imputation flag
        self.imputation = imputation
        # PSD estimation flag
        self.psd_estimation = psd_estimation
        # Log-likelihood normalization flag
        self.normalized = normalized
        if self.normalized:
            self.log_norm = self.compute_log_norm()

    def concatenate_model(self, x):
        """
        Way of concatenating model depending on the number of analyzed channels
        Parameters
        ----------
        x

        Returns
        -------

        """
        if len(self.channels) == 1:
            return x
        elif len(self.channels) > 1:
            return np.concatenate(x)
        else:
            raise ValueError("Please indicate at least one channel")

    def concatenate_data_pos(self, x):
        """
        Way of concatenating data depending on the number of analyzed channels, using only positive frequencies

        Parameters
        ----------
        x

        Returns
        -------

        """
        if len(self.channels) == 1:
            return x[self.inds_pos]
        elif len(self.channels) > 1:
            return np.concatenate([x0[self.inds_pos] for x0 in x])
        else:
            raise ValueError("Please indicate at least one channel")

    def compute_log_norm(self):
        """
        Compute the logarithm of the likelihood's normalization constant

        Returns
        -------
        log_norm : float
            log normalization

        """

        if type(self.spectrum) == np.array:
            # Normalization constant (calculated once and for all)
            log_norm = np.real(-0.5 * (np.sum(np.log(self.spectrum))
                                       + self.data_model.n * np.log(2*np.pi)))
        elif type(self.spectrum) == list:
            # If the noise spectrum is a list of spectra corresponding to each
            # TDI channel, concatenate the spectra
            # in a single array
            # Restricted spectrum
            # spectrum_arr = np.concatenate([spect[self.posterior_cls.inds_pos]
            # for spect in self.spectrum])
            spectrum_arr = np.concatenate([spect for spect in self.spectrum])
            log_norm = np.real(-0.5 * (np.sum(np.log(spectrum_arr))) + len(self.spectrum) * self.data_model.n * np.log(2 * np.pi))

        return log_norm

    def update_missing_data(self, params):
        """

        Update missing data imputation

        Parameters
        ----------
        params : array_like
            vector of current parameter values

        Returns
        -------

        """

        # Inverse Fourier transform back in time domain (can be a numpy array
        # or a list of arrays)
        y_gw = self.compute_time_signal(params)
        # Draw the missing data (can be a numpy array or a list of arrays)
        y_rec = self.data_model.imputation(y_gw, self.psd_model.psd_list)
        # Calculate the DFT of the reconstructed data
        self.y_fft = dft(y_rec, self.w)

    def update_psd(self, pos0):
        """
        Update noise PSD parameters

        Parameters
        ----------
        pos0 : array_like
            vector of current parameter values
        npsd : integer
            number of draws during one MCMC psd update

        Returns
        -------

        """

        # PSD POSTERIOR STEP
        # Draw (or compute) the signal in the Fourier domain
        # (without any normalization, like Python's FFT)
        y_gw_fft = self.compute_frequency_signal(pos0)
        # Calculate the model residuals using the DFT of the data windowed with
        # the smoothest time window
        # Need to rescale the GW signal so account for this windowing
        if type(y_gw_fft) == np.ndarray:
            z_fft = self.y_fft_psd - self.k1_psd / self.data_model.n * y_gw_fft
        elif type(y_gw_fft) == list:
            z_fft = [self.y_fft_psd[i] - self.k1_psd / self.data_model.N * y_gw_fft[i]
                     for i in range(len(y_gw_fft))]
        # Update PSD estimate
        self.psd_model.estimate_from_freq(z_fft, k2=self.k2_psd)
        # Update new value of the spectrum for the posterior step
        self.spectrum = self.psd_model.calculate(self.data_model.n)
        # Update the normalizing constant
        self.log_norm = self.compute_log_norm()

    def matrix_model(self, params):
        """

        function of frequency f, source model parameters params describing the
        features of the analyzed data

        Parameters
        ----------
        params : bytearray
            vector of waveform parameters
        Returns
        -------
        mat : bytearray
            design matrix in frequency domain


        """

        return self.signal_model.design_matrix_freq(self.f_pos, params,
                                                    self.data_model.del_t,
                                                    self.data_model.tobs,
                                                    channel=self.channels,
                                                    complex=True)

    def compute_frequency_signal(self, params):

        y_gw_fft = np.zeros(self.data_model.n, dtype=np.complex128)

        y_gw_fft[self.inds_pos] = self.signal_model.compute_signal_freq(
            self.f_pos, params, self.data_model.del_t, self.data_model.tobs,
            channel=self.channels)
        y_gw_fft[self.inds_neg] = np.conj(y_gw_fft[self.inds_pos])

        return y_gw_fft

    def log_likelihood(self, params):
        """
        Logarithm of the likelihood, optimized for FREQUENCY domain
        computations, depending on all parameters

        Parameters
        ----------
        params : array_like
            vector of parameters

        Returns
        -------
        logL : scalar float
            logarithm of the likelihood

        """

        # For a full likelihood model
        if not self.reduced:
            # Stack the computed waveform DFTs
            s_fft_stack = self.concatenate_model(
                self.signal_model.compute_signal_freq(self.f_pos, params,
                                                      self.data_model.del_t,
                                                      self.data_model.tobs,
                                                      channel='TDIAET',
                                                      ldc=False,
                                                      tref=0))
        # For a reduced likelihood model
        else:
            # Compute design matrices
            mat_list = self.matrix_model(params)
            # Compute extrinsinc amplitudes for each channel
            amplitudes = [gls(mat_list[i], self.spectrum[i][self.inds_pos],
                              self.y_fft[i][self.inds_pos])
                          for i in range(len(mat_list))]

            # Stack the modeled signals
            s_fft_stack = self.concatenate_model(
                [mat_list[i].dot(amplitudes[i]) for i in range(len(mat_list))])

        # Compute periodogram for relevant frequencies
        per_inds = np.abs(self.y_fft_stack - s_fft_stack)**2/self.data_model.n

        # Increment the counter
        self.counter += 1
        # If it is a multiple of n_update, update auxiliary parameters
        if (self.counter % self.n_update == 0) & (self.counter != 0):
            # Missing data imputation step
            if self.imputation:
                print("Update missing data at likelihood evaluation number "
                      + str(self.counter))
                self.update_missing_data(params)
            # PSD parameter posterior step
            if self.psd_estimation:
                print("Update PSD estimate at likelihood evaluation number "
                      + str(self.counter))
                self.update_psd(params)
                self.log_norm = self.compute_log_norm()

        # Update reduced likelihood
        return np.real(-0.5 * np.sum(
            per_inds / self.concatenate_data_pos(self.spectrum)))+self.log_norm

    def compute_frequency_residuals(self, y_gw_fft):
        """
        Compute the residuals used for the PSD estimation
        Parameters
        ----------
        y_gw_fft : numpy.ndarray or list of arrays
            GW signal in the Fourier domain

        Returns
        -------
        z_fft : numpy.ndarray or list
            residuals in Fourier domain

        """

        if type(y_gw_fft) == np.ndarray:
            z_fft = self.y_fft_psd - self.k1_psd / self.data_model.n * y_gw_fft
        elif type(y_gw_fft) == list:
            z_fft = [self.y_fft_psd[i] - self.k1_psd / self.data_model.n * y_gw_fft[i]
                     for i in range(len(y_gw_fft))]

        return z_fft

    def draw_single_freq_signal_on_bw(self, params, psd, y_fft):
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
            y_gw_fft = self.draw_single_freq_signal_on_bw(params, psd, y_fft)
        else:
            y_gw_fft = np.zeros(self.npos, dtype=np.complex128)

        # Return the frequency domain GW signal
        return y_gw_fft

    def draw_frequency_signal(self, params, psd, y_fft):
        """
        Compute deterministic signal model on the full Fourier grid
        """

        y_gw_fft = np.zeros(self.data_model.n, dtype=np.complex128)

        y_gw_fft[self.inds_pos] = self.draw_frequency_signal_onBW(params,
                                                                  psd,
                                                                  y_fft)
        y_gw_fft[self.inds_neg] = np.conj(y_gw_fft[self.inds_pos])

        return y_gw_fft

    def compute_matrices(self, params, s):
        """

        Compute the design matrix, its weighted version, and the inverse normal
        matrix

        """
        if self.nsources == 1:
            # Update frequency model of the waveform
            mat_freq = self.matrix_model(params)

        elif self.nsources > 1:
            mat_freq = np.hstack([self.matrix_model(
                params[i*self.ndim:(i+1)*self.ndim])
                                  for i in range(self.nsources)])

        # Weight matrix columns by the inverse of the PSD
        s2 = np.concatenate((s[self.inds_pos], s[self.inds_pos]))
        mat_freq_w = np.array([mat_freq[:, j]/s2
                               for j in range(mat_freq.shape[1])]).T

        # Inverse normal matrix
        ZI = LA.pinv(np.dot(np.transpose(mat_freq).conj(), mat_freq_w))

        return mat_freq, mat_freq_w, ZI

    def ls_estimate_beta(self, ZI, ApS, data_fft):
        """
        Generalized least-square estimate of the extrinsic parameters
        """

        data_real = np.concatenate((data_fft[self.inds_pos].real,
                                    data_fft[self.inds_pos].imag))
        return np.dot(ZI, np.dot(np.transpose(ApS).conj(), data_real))

    def draw_beta(self, ZI, ApS, data_fft):
        """
        Draw the GW amplitude parameter vector beta from its conditional
        distribution

        Parameters
        ----------
        ZI : 2d numpy array
            inverse weighted normal matrix (a_mat^* C^-1 a_mat )^-1
        ApS : 2d numpy array
            weighted design matrix
        data_fft : array_like
            discrete fourier transform of data

        Returns
        -------
        beta : 1d numpy array
            random drawn of the conditional distribution of beta

        """
        return np.random.multivariate_normal(
            self.ls_estimate_beta(ZI, ApS, data_fft), ZI)

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
        #y_gw_fft = np.zeros(self.data_model.n, dtype = np.complex128)
        #y_gw_fft[self.inds] = y_gw_fft_inds

        return np.real(ifft(y_gw_fft))

    def reset_counter(self):

        self.counter = 0


class LogLike(object):

    def __init__(self, data, sn, inds, tobs, del_t, normalized=False,
                 t_offset=52.657, channels=None, scale=1.0,
                 model_cls=None, psd_cls=None, wd=None, wd_full=None):
        """

        Parameters
        ----------
        data : list of ndarrays
            TDI data a_mat, E, T in the time domain, without any smooth
            windowing (except binary masking)
        sn : list of ndarrays
            noise PSD computed at freq for each channel
        inds: array_like
            indices of the frequencies to consider in the Fourier frequency
            array
        tobs : float
            observation time
        del_t : float
            data sampling cadence
        normalized : boolean
            whether to apply normalization constant for the log-likelihood
        t_offset : float
            offset time for the waveform start
        channels : list of ints
            TDI channels to consider
        scale : float
            rescaling factor for waveforms
        model_cls : bayesdawn.datamodel.GaussianStationaryModel instance
            data model used for missing data imputation
        psd_cls : list bayesdawn.psdmodel.PSD instance
            noise PSD model


        """

        self.data = data
        self.scale = scale
        self.sn = sn
        self.tobs = tobs
        self.del_t = del_t
        # Full data size
        self.n_data = self.data[0].shape[0]
        self.nf = len(inds)
        self.t_offset = t_offset
        self.df = 1 / (self.n_data * self.del_t)
        self.f = np.fft.fftfreq(self.n_data) / self.del_t
        # Fourier bin indices where we restrict the analysis
        # intersect, self.inds, comm2 = np.intersect1d(self.f, freq)
        self.inds = inds

        # Time windowing for gapped data
        if wd is None:
            self.wd = np.ones(data[0].shape[0])
        else:
            self.wd = wd
        # Time windowing for full data
        if wd_full is None:
            self.wd_full = np.ones(data[0].shape[0])
        else:
            self.wd_full = wd_full
        # Rescaling factor to compensate for the windowing, so that waveform
        # template has the same power
        self.resc = self.n_data / np.sum(self.wd)
        self.resc_full = self.n_data / np.sum(self.wd_full)
        # Convert time data to frequency domain and restrict it to the band of
        # interest
        self.data_dft = [fft(self.wd * dat)[self.inds] * self.del_t * self.resc
                         for dat in self.data]

        if normalized:
            self.ll_norm = self.log_norm()
        else:
            self.ll_norm = 0
        # The full set of parameters is
        # m1, m2, chi1, chi2, tc, dist, incl, phi0, lam, bet, psi
        # Indices of extrinsic parameters
        self.i_ext = [5, 6, 7, 10]
        # Indices of intrinsic parameters m1, m2, chi1, chi2, tc, bet, psi
        self.i_intr = [0, 1, 2, 3, 4, 8, 9]

        if channels is None:
            self.channels = [1, 2]
        else:
            self.channels = channels

        # For missing data imputation
        self.model = model_cls
        # For noise PSD estimation
        self.psd_list = psd_cls

    def update_psd(self, y_gw_list):
        """

        Parameters
        ----------
        y_gw_list : list
            list of waveform in the time domain for each channel.
            Should *not* have windowing of any kind.

        Returns
        -------

        """

        # Estimate PSD parameters from the residuals in the time domain,
        # in the first TDI channel.
        [self.psd_list[i].estimate(self.data[i] - y_gw_list[i], wind='hanning')
         for i in range(len(self.data))]
        # Calculate the spectrum in the estimation band
        self.sn = [psd.calculate(self.f[self.inds]) for psd in self.psd_list]
        # It is currently x fs / 2. Should correct for that.

    def update_missing_data(self, y_gw_list):
        """
        Update the value of missing data

        Parameters
        ----------
        y_gw_list : list
            list of waveform in the time domain for each channel

        Returns
        -------
        Nothing, just update the attribute self.data_dft

        """

        # Impute missing data
        y_imp = self.model.impute(y_gw_list, self.psd_list)
        # Transform back to Fourier domain, applying the windowing for complete
        # time series, with re-scaling
        resc = self.del_t * self.resc_full
        self.data_dft = [fft(y_imp0 * self.wd_full)[self.inds] * resc
                         for y_imp0 in y_imp]

    def update_auxiliary_params(self, par, reduced=True):

        if (self.psd_list is not None) | (self.model is not None):
            # Compute waveform template in the frequency domain
            if reduced:
                at, et = self.compute_signal_reduced(par)
            else:
                at, et = self.compute_signal(par)

            # y_gw_fft_list = [at, et]
            # Convert the signal in the time domain (factor 1 / del_t incluced)
            y_gw_list = [self.frequency_to_time(y_gw_fft_pos)
                         for y_gw_fft_pos in [at, et]]
            # Update PSD if requested
            if self.psd_list is not None:
                self.update_psd(y_gw_list)
            if self.model is not None:
                self.update_missing_data(y_gw_list)

        else:
            pass

    def log_norm(self):
        """
        Compute normalizing constant for the log-likelihood

        Returns
        -------
        ll_norm : float
            normalization constant for the log-likelihood

        """

        ll_norm = - self.nf/2 * np.log(2 * np.pi * 2 * self.del_t)
        ll_norm += sum([- 0.5 * np.sum(np.log(self.sn[i]))
                        - 0.5 * np.sum(np.abs(self.data_dft[i]) ** 2 / self.sn)
                        for i in range(len(self.data_dft))])

        return ll_norm

    def compute_signal(self, par):
        """
        Compute the GW signal in the frequency domain

        Parameters
        ----------
        par : array_like
            array of sampling parameters

        Returns
        -------

        """

        # Convert likelihood parameters into waveform-compatible parameters
        params = physics.like_to_waveform(par)

        # Compute waveform template
        ch = lisaresp.lisabeta_template(params, self.f[self.inds], self.tobs,
                                        tref=0, t_offset=self.t_offset,
                                        channels=self.channels)

        return [ch_i * self.scale for ch_i in ch]

    def frequency_to_time(self, y_gw_fft_pos):
        """
        Compute the waveform in the time domain from the waveform values in the
        frequency domain evaluated at positive
        Fourier frequencies
        """

        y_gw_fft = np.zeros(self.n_data, dtype=np.complex128)
        y_gw_fft[self.inds] = y_gw_fft_pos
        y_gw_fft[self.n_data - self.inds] = np.conj(y_gw_fft_pos)

        return np.real(ifft(y_gw_fft))/self.del_t

    def log_likelihood(self, par):
        """

        Parameters
        ----------
        par : array_like
            vector of waveform parameters in the following order:
            [Mc, q, tc, chi1, chi2, logDL, ci, sb, lam, psi, phi0]


        Returns
        -------

        """

        # Compute waveform template
        at, et = self.compute_signal(par)

        # (h | y)
        sna = np.sum(np.real(self.data_dft[0] * np.conjugate(at)) / self.sn[0])
        sne = np.sum(np.real(self.data_dft[1] * np.conjugate(et)) / self.sn[1])

        # (h | h)
        aa = np.sum(np.abs(at) ** 2 / self.sn[0])
        ee = np.sum(np.abs(et) ** 2 / self.sn[1])

        # (h | y) - 1/2 (h | h)
        ll_a = 4.0 * self.df * (sna - 0.5*aa)
        ll_e = 4.0 * self.df * (sne - 0.5*ee)

        return ll_a + ll_e + self.ll_norm

    def compute_signal_reduced(self, par_intr):
        """

        Parameters
        ----------
        par_intr

        Returns
        -------

        """

        # Transform parameters into waveform-compatible ones
        params_intr = physics.like_to_waveform_intr(par_intr)

        # Design matrices for each channel
        mat_list = lisaresp.design_matrix(params_intr, self.f[self.inds],
                                          self.tobs,
                                          tref=0, t_offset=self.t_offset,
                                          channels=self.channels)
        # Weighted design matrices
        mat_list_weighted = [mat_list[i] / np.array([self.sn[i]]).T
                             for i in range(len(self.channels))]
        # Compute amplitudes
        amps = [LA.pinv(np.dot(mat_list_weighted[i].conj().T,
                               mat_list[i])).dot(
                                   np.dot(mat_list_weighted[i].conj().T,
                                          self.data_dft[i]))
                for i in range(len(self.channels))]
        # amps = [LA.pinv(np.dot(mat_list[i].conj().T, mat_list[i])).dot(np.dot(mat_list[i].conj().T, self.data[i]))
        #         for i in range(len(self.channels))]
        # aet_rec = [np.dot(mat_list[i], amps[i]) for i in range(len(aet))]
        # at = np.dot(mat_list[0], amps[0])
        # et = np.dot(mat_list[1], amps[1])

        return [np.dot(mat_list[i-1], amps[i-1]) for i in self.channels]

    def log_likelihood_reduced(self, par_intr):
        """

        Parameters
        ----------
        par_intr : array_like
            vector of intrinsic waveform parameters in the following order:
            [Mc, q, tc, chi1, chi2, sb, lam]

        Returns
        -------

        """

        # Compute the signal in the frequency domain
        at, et = self.compute_signal_reduced(par_intr)

        # (h | y)
        sna = np.sum(np.real(self.data_dft[0] * np.conjugate(at)) / self.sn[0])
        sne = np.sum(np.real(self.data_dft[1] * np.conjugate(et)) / self.sn[1])

        # (h | h)
        aa = np.sum(np.abs(at) ** 2 / self.sn[0])
        ee = np.sum(np.abs(et) ** 2 / self.sn[1])

        # (h | y) - 1/2 (h | h)
        ll_a = 4.0 * self.df * (sna - 0.5 * aa)
        ll_e = 4.0 * self.df * (sne - 0.5 * ee)

        return ll_a + ll_e + self.ll_norm
