from scipy import signal
import numpy as np
from bayesdawn.utils import physics
import lisabeta.lisa.ldctools as ldctools
# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft


def filter(dat, fc, del_t):
    
    b, a = signal.butter(5, fc, 'low', analog=False, fs=1/del_t)
    xd = signal.filtfilt(b, a, dat)

    return xd

def preprocess_all(config, td, i1, i2, scale=1.0):

    del_t = td[1, 0] - td[0, 0]

    if config['InputData'].getboolean('decimation'):
        fc = config['InputData'].getfloat('filterFrequency')
        b, a = signal.butter(5, fc, 'low', analog=False, fs=1/del_t)
        Xd = signal.filtfilt(b, a, td[:, 1] * scale)
        Yd = signal.filtfilt(b, a, td[:, 2] * scale)
        Zd = signal.filtfilt(b, a, td[:, 3] * scale)
        # Downsampling
        q = config['InputData'].getint('decimationFactor')
        tm = td[i1:i2:q, 0]
        Xd = Xd[i1:i2:q]
        Yd = Yd[i1:i2:q]
        Zd = Zd[i1:i2:q]

    else:
        q = 1
        tm = td[i1:i2, 0]
        Xd = td[i1:i2, 1] * scale
        Yd = td[i1:i2, 2] * scale
        Zd = td[i1:i2, 3] * scale

    return tm, Xd, Yd, Zd, q


def preprocess_ldc_data(p, td, config):

    del_t = float(p.get("Cadence"))
    tobs = float(p.get("ObservationDuration"))

    # ==================================================================================================================
    # Get the parameters
    # ==================================================================================================================
    # Get parameters as an array from the hdf5 structure (table)
    p_sampl = physics.get_params(p, sampling=True)

    # ==================================================================================================================
    # Pre-processing data: anti-aliasing and filtering
    # ==================================================================================================================
    if config['InputData'].getboolean('trim'):
        i1 = np.int(config["InputData"].getfloat("StartTime") / del_t)
        i2 = np.int(config["InputData"].getfloat("EndTime") / del_t)
        t_offset = 52.657 + config["InputData"].getfloat("StartTime")
        tobs = (i2 - i1) * del_t
    else:
        i1 = 0
        i2 = np.int(td.shape[0])
        t_offset = 52.657

    scale = config["InputData"].getfloat("rescale")
    tm, xd, yd, zd, q = preprocess_all(config, td, i1, i2,
                                   scale=scale)

    return tm, xd, yd, zd, q, t_offset, tobs, del_t, p_sampl


def time_to_frequency(ad, ed, td, wd, del_t, q, compensate_window=True):
    """
    Convert time domain data to discrete Fourier transformed data,
    with normalization by del_t x q x n / k1
    Parameters
    ----------
    ad
    ed
    td
    wd
    del_t
    q

    Returns
    -------

    """

    # ==================================================================================================================
    # Now we get extract the data and transform it to frequency domain
    # ==================================================================================================================
    if compensate_window:
        resc = ad.shape[0] / np.sum(wd)
    else:
        resc = 1.0
    a_df = fft(wd * ad) * del_t * q * resc
    e_df = fft(wd * ed) * del_t * q * resc
    t_df = fft(wd * td) * del_t * q * resc

    return a_df, e_df, t_df


# def determine_frequency_bounds(config, freq_d):
#
#     # Restrict the frequency band to high SNR region, and exclude distorted frequencies due to gaps
#     if (config["TimeWindowing"].getboolean('gaps')) & (not config["Imputation"].getboolean('imputation')):
#
#         f1, f2 = physics.find_distorted_interval(mask, p_sampl, t_offset, del_t, margin=1e-4)
#
#         inds = np.where((float(config['Model']['MinimumFrequency']) <= freq_d)
#                         & (freq_d <= float(config['Model']['MaximumFrequency']))
#                         & (freq_d >= f1) & (freq_d <= f2))[0]
#     inds = np.where((float(config['Model']['MinimumFrequency']) <= freq_d)
#                     & (freq_d <= float(config['Model']['MaximumFrequency'])))[0]
