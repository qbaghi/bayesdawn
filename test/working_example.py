# Import mecm and other useful packages
from bayesdawn import imputation
from bayesdawn.psd import psdspline
import numpy as np
import random
from scipy import signal
# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft
from matplotlib import pyplot as plt
from scipy import linalg as LA

if __name__=='__main__':

    # Choose size of data
    N = 2 ** 14
    # Choose sampling frequency
    fs = 1.0
    # Generate Gaussian white noise
    noise = np.random.normal(loc=0.0, scale=1.0, size=N)
    # Apply filtering to turn it into colored noise
    r = 0.01
    b, a = signal.butter(3, 0.1 / 0.5, btype='high', analog=False)
    n = signal.lfilter(b, a, noise, axis=-1, zi=None) + noise * r

    # Generate sinusoidal signal
    t = np.arange(0, N) / fs
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
    y = s + n
    y_mask = mask*(s + n)
    # Observed signal
    s_mask = mask * s

    # Crude estimation of noise psd from masked data
    psd_cls = psdspline.PSDSpline(N, fs, J=10, D=2, fmin=fs/N, fmax=fs/2)
    # psd_cls.set_knots(np.array([1e-2, 5e-2, 1e-1]))
    psd_cls.estimate(y - s_mask)
    psd = psd_cls.calculate(N)


    # instantiate imputation class
    imp_cls = imputation.nearestNeighboor(mask, Na=50, Nb=50)
    # Imputation of missing data
    y_rec = imp_cls.draw_missing_data(y_mask, s, psd_cls)


    # Observed effect on time series
    f = np.fft.fftfreq(N)*fs
    wind = np.hanning(N)
    K2 = np.sum(wind**2)
    y_fft = fft(wind * y)
    y_mask_fft = fft(wind * y_mask)
    y_rec_fft = fft(wind * y_rec)

    # Computing periodograms
    py = np.abs(y_fft)**2 / (K2 * fs)
    py_mask = np.abs(y_mask_fft) ** 2 / (K2 * fs)
    py_rec = np.abs(y_rec_fft)**2 / (K2 * fs)

    # # Other estimate
    # powers = [-1, 0, 1, 2, 3, 4]
    # fpos = f[f > 0]
    # pypos = py[f > 0]
    # mat = np.array([np.log(fpos) ** p for p in powers]).T
    # beta = LA.pinv(mat.T.dot(mat)).dot(mat.T.dot(np.log(pypos)))
    # psd_pow = np.exp(np.dot(mat, beta))

    plt.loglog(f, np.sqrt(py), label='complete')
    plt.loglog(f, np.sqrt(py_mask), label='masked')
    plt.loglog(f, np.sqrt(py_rec), label='imputed')
    plt.loglog(f, np.sqrt(psd/2), label='PSD estimate (splines)')
    # plt.loglog(fpos, np.sqrt(psd_pow), label='PSD estimate (power law)')
    plt.legend()
    plt.show()









