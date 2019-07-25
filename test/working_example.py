# Import mecm and other useful packages
import bayesdawn
import numpy as np
import random
from scipy import signal
# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft


if __name__=='__main__':

    # Choose size of data
    N = 2 ** 14
    # Generate Gaussian white noise
    noise = np.random.normal(loc=0.0, scale=1.0, size=N)
    # Apply filtering to turn it into colored noise
    r = 0.01
    b, a = signal.butter(3, 0.1 / 0.5, btype='high', analog=False)
    n = signal.lfilter(b, a, noise, axis=-1, zi=None) + noise * r

    # Generate sinusoidal signal
    t = np.arange(0, N)
    f0 = 1e-2
    a0 = 5e-3
    s = a0 * np.sin(2 * np.pi * f0 * t)

    mask = np.ones(N)
    Ngaps = 30
    gapstarts = (N * np.random.random(Ngaps)).astype(int)
    gaplength = 10
    gapends = (gapstarts + gaplength).astype(int)
    for k in range(Ngaps):
        mask[gapstarts[k]:gapends[k]] = 0

    # Observed data
    y = mask*(s + n)
    # Observed signal
    s_mask = mask * s

    # Crude estimation of noise psd from masked data
    psd_cls = bayesdawn.psd.PSDSpline(N, 1.0)
    psd_csl.estimate(y - s_mask)

    # Imputation of missing data
    # instantiate imputation class
    imp_cls = bayesdawn.imputation.nearestNeighboor(mask, Na=150, Nb=150)
    y_rec = imp_cls.draw_missing_data(y, s_mask, psd_csl)





