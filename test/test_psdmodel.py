import unittest
import numpy as np

# FTT modules
import fftwisdom
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft
import configparser
from bayesdawn.utils import loadings, physics, preprocess
from bayesdawn import likelihoodmodel, psdmodel
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


class TestPSDModel(unittest.TestCase):

    def test_psd_estimation(self, config_file="../configs/config_ldc_full.ini", plot=True):

        config = configparser.ConfigParser()
        config.read(config_file)
        fftwisdom.load_wisdom()

        # Unpacking the hdf5 file and getting data and source parameters
        p, td = loadings.load_ldc_data(config["InputData"]["FilePath"])

        # Pre-processing data: anti-aliasing and filtering
        tm, xd, yd, zd, q, t_offset, tobs, del_t, p_sampl = preprocess.preprocess_ldc_data(p, td, config)

        # Introducing gaps if requested
        wd, wd_full, mask = loadings.load_gaps(config, tm)
        print("Ideal decay number: "
              + str(np.int((config["InputData"].getfloat("EndTime") - p_sampl[2]) / (2 * del_t))))

        # Now we get extract the data and transform it to frequency domain
        freq_d = np.fft.fftfreq(len(tm), del_t * q)
        # Convert Michelson TDI to A, E, T (time domain)
        ad, ed, td = ldctools.convert_XYZ_to_AET(xd, yd, zd)

        # Restrict the frequency band to high SNR region, and exclude distorted frequencies due to gaps
        if (config["TimeWindowing"].getboolean('gaps')) & (not config["Imputation"].getboolean('imputation')):
            f1, f2 = physics.find_distorted_interval(mask, p_sampl, t_offset, del_t, margin=0.4)
            f1 = np.max([f1, 0])
            f2 = np.min([f2, 1 / (2 * del_t)])
            inds = np.where((float(config['Model']['MinimumFrequency']) <= freq_d)
                            & (freq_d <= float(config['Model']['MaximumFrequency']))
                            & ((freq_d <= f1) | (freq_d >= f2)))[0]
        else:
            inds = np.where((float(config['Model']['MinimumFrequency']) <= freq_d)
                            & (freq_d <= float(config['Model']['MaximumFrequency'])))[0]

        # Restriction of sampling parameters to instrinsic ones Mc, q, tc, chi1, chi2, np.sin(bet), lam
        i_sampl_intr = [0, 1, 2, 3, 4, 7, 8]
        print("=================================================================")
        fftwisdom.save_wisdom()

        # Auxiliary parameter classes
        print("PSD estimation enabled.")
        psd_cls = psdmodel.PSDSpline(tm.shape[0], 1 / del_t, J=config["PSD"].getint("knotNumber"),
                                     D=config["PSD"].getint("SplineOrder"),
                                     fmin=1 / (del_t * tm.shape[0]) * 1.05,
                                     fmax=1 / (del_t * 2))
        psd_cls.estimate(ad)
        sa = psd_cls.calculate(freq_d[inds])

        data_cls = None

        # Instantiate likelihood class
        ll_cls = likelihoodmodel.LogLike([mask * ad, mask * ed], sa, inds, tobs, del_t * q,
                                         normalized=config['Model'].getboolean('normalized'),
                                         t_offset=t_offset, channels=[1, 2],
                                         scale=config["InputData"].getfloat("rescale"),
                                         model_cls=data_cls, psd_cls=psd_cls,
                                         wd=wd,
                                         wd_full=wd_full)

        # Test PSD estimation from the likelihood class
        ll_cls.update_auxiliary_params(p_sampl[i_sampl_intr], reduced=True)

        sa2 = ll_cls.psd.calculate(freq_d[inds])

        # Plotting
        if plot:
            from plottools import presets
            presets.plotconfig(ctype='time', lbsize=16, lgsize=14)

            # Frequency plot
            fig0, ax0 = plt.subplots(nrows=1)
            ax0.semilogx(freq_d[inds], np.sqrt(sa), 'g', label='First estimate.')
            ax0.semilogx(freq_d[inds], np.sqrt(sa2), 'r', label='Second estimate.')
            plt.legend()
            plt.show()

        rms = rms_error(sa2, sa, relative=True)

        print("Cumulative relative error is " + str(rms))
        self.assertLess(rms, 1e-2, "Cumulative relative error sould be less than 0.05 (5 percents)")


if __name__ == '__main__':

    unittest.main()
