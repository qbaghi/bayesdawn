import unittest
import numpy as np

# FTT modules
import fftwisdom
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft
import configparser
from bayesdawn.utils import loadings, physics, preprocess
from bayesdawn import likelihoodmodel
import tdi
import lisabeta.lisa.ldctools as ldctools
# Plotting modules
import matplotlib.pyplot as plt
import time


def rms_error(x, x_ref,  relative=True):

    rms = np.sqrt(np.sum(np.abs(x - x_ref) ** 2))

    if relative:
        rms = rms / np.sqrt(np.sum(np.abs(x_ref) ** 2))

    return rms


class TestWaveform(unittest.TestCase):

    def test_waveform_freq(self, config_file="../configs/config_ldc.ini", plot=True):

        config = configparser.ConfigParser()
        config.read(config_file)
        fftwisdom.load_wisdom()

        # Unpacking the hdf5 file and getting data and source parameters
        p, td = loadings.load_ldc_data(config["InputData"]["FilePath"])

        # Pre-processing data: anti-aliasing and filtering
        tm, xd, yd, zd, q, t_offset, tobs, del_t, p_sampl = preprocess.preprocess_ldc_data(p, td, config)

        # Frequencies
        freq_d = np.fft.fftfreq(len(tm), del_t * q)
        # Restrict the frequency band to high SNR region
        inds = np.where((float(config['Model']['MinimumFrequency']) <= freq_d)
                        & (freq_d <= float(config['Model']['MaximumFrequency'])))[0]
        # Theoretical PSD
        sa = tdi.noisepsd_AE(freq_d[inds], model='Proposal', includewd=None)

        # Convert Michelson TDI to A, E, T (time domain)
        ad, ed, td = ldctools.convert_XYZ_to_AET(xd, yd, zd)

        # Transform to frequency domain
        wd = np.ones(ad.shape[0])
        wd_full = wd[:]
        mask = wd[:]
        a_df, e_df, t_df = preprocess.time_to_frequency(ad, ed, td, wd, del_t, q, compensate_window=True)

        # Instantiate likelihood class
        ll_cls = likelihoodmodel.LogLike([mask * ad, mask * ed], sa, inds, tobs, del_t * q,
                                         normalized=config['Model'].getboolean('normalized'),
                                         t_offset=t_offset, channels=[1, 2],
                                         scale=config["InputData"].getfloat("rescale"),
                                         model_cls=None, psd_cls=None,
                                         wd=wd,
                                         wd_full=wd_full)

        t1 = time.time()
        if config['Model'].getboolean('reduced'):
            i_sampl_intr = [0, 1, 2, 3, 4, 7, 8]
            aft, eft = ll_cls.compute_signal_reduced(p_sampl[i_sampl_intr])
        else:
            aft, eft = ll_cls.compute_signal(p_sampl)
        t2 = time.time()
        print('Waveform Calculated in ' + str(t2 - t1) + ' seconds.')

        rms = rms_error(aft, a_df[inds], relative=True)
        print("Cumulative relative error is " + str(rms))

        self.assertLess(rms, 1e-3, "Cumulative relative error sould be less than 0.02 (2 percents)")

        # Plotting
        if plot:
            from plottools import presets
            presets.plotconfig(ctype='time', lbsize=16, lgsize=14)

            # Frequency plot
            fig1, ax1 = plt.subplots(nrows=2, sharex=True, sharey=True)
            ax1[0].semilogx(freq_d[inds], np.real(a_df[inds]))
            ax1[0].semilogx(freq_d[inds], np.real(aft), '--')
            ax1[0].set_ylabel("Fractional frequency")
            # ax1[0].legend()
            ax1[1].semilogx(freq_d[inds], np.imag(a_df[inds]))
            ax1[1].semilogx(freq_d[inds], np.imag(aft), '--')
            ax1[1].set_xlabel("Frequency [Hz]")
            ax1[1].set_ylabel("Fractional frequency")
            # ax1[1].legend()
            plt.show()

    def test_waveform_time(self, config_file="../configs/config_ldc.ini", plot=True):

        config = configparser.ConfigParser()
        config.read(config_file)
        fftwisdom.load_wisdom()

        # Unpacking the hdf5 file and getting data and source parameters
        p, td = loadings.load_ldc_data(config["InputData"]["FilePath"])

        # Pre-processing data: anti-aliasing and filtering
        tm, xd, yd, zd, q, t_offset, tobs, del_t, p_sampl = preprocess.preprocess_ldc_data(p, td, config)

        # Frequencies
        freq_d = np.fft.fftfreq(len(tm), del_t * q)
        # Restrict the frequency band to high SNR region
        inds = np.where((float(config['Model']['MinimumFrequency']) <= freq_d)
                        & (freq_d <= float(config['Model']['MaximumFrequency'])))[0]
        # Theoretical PSD
        sa = tdi.noisepsd_AE(freq_d[inds], model='Proposal', includewd=None)

        # Convert Michelson TDI to A, E, T (time domain)
        ad, ed, td = ldctools.convert_XYZ_to_AET(xd, yd, zd)

        # Transform to frequency domain
        wd = np.ones(ad.shape[0])
        wd_full = wd[:]
        mask = wd[:]
        a_df, e_df, t_df = preprocess.time_to_frequency(ad, ed, td, wd, del_t, q, compensate_window=True)

        # Instantiate likelihood class
        ll_cls = likelihoodmodel.LogLike([mask * ad, mask * ed], sa, inds, tobs, del_t * q,
                                         normalized=config['Model'].getboolean('normalized'),
                                         t_offset=t_offset, channels=[1, 2],
                                         scale=config["InputData"].getfloat("rescale"),
                                         model_cls=None, psd_cls=None,
                                         wd=wd,
                                         wd_full=wd_full)

        t1 = time.time()
        if config['Model'].getboolean('reduced'):
            i_sampl_intr = [0, 1, 2, 3, 4, 7, 8]
            aft, eft = ll_cls.compute_signal_reduced(p_sampl[i_sampl_intr])
        else:
            aft, eft = ll_cls.compute_signal(p_sampl)
        t2 = time.time()
        print('Waveform Calculated in ' + str(t2 - t1) + ' seconds.')

        # Convert to time-domain waveform
        aft_time = ll_cls.frequency_to_time(aft)

        rms = rms_error(aft_time, ad, relative=True)
        print("Cumulative relative error is " + str(rms))
        self.assertLess(rms, 1e-2, "Cumulative relative error sould be less than 0.02 (2 percents)")

        if plot:
            # Time plot
            fig0, ax0 = plt.subplots(nrows=1, sharex=True, sharey=True)
            ax0.plot(tm, ad, 'k')
            ax0.plot(tm, aft_time, 'r')
            plt.show()


if __name__ == '__main__':

    unittest.main()
