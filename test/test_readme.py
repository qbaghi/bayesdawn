# -*- coding: utf-8 -*-
"""
Quick start example in the README.

"""
 


if __name__ == '__main__':
    # Import mecm and other useful packages
    from bayesdawn import datamodel, psdmodel
    import numpy as np
    import random
    from scipy import signal
    from matplotlib import pyplot as plt
    # Choose size of data
    n_data = 2**14
    # Set sampling frequency
    fs = 1.0
    # Generate Gaussian white noise
    noise = np.random.normal(loc=0.0, scale=1.0, size = n_data)
    # Apply filtering to turn it into colored noise
    r = 0.01
    b, a = signal.butter(3, 0.1/0.5, btype='high', analog=False)
    n = signal.lfilter(b,a, noise, axis=-1, zi=None) + noise*r

    t = np.arange(0, n_data) / fs
    f0 = 1e-2
    a0 = 5e-3
    s = a0 * np.sin(2 * np.pi * f0 * t)

    mask = np.ones(n_data)
    n_gaps = 30
    gapstarts = (n_data * np.random.random(n_gaps)).astype(int)
    gaplength = 10
    gapends = (gapstarts+gaplength).astype(int)
    for k in range(n_gaps): mask[gapstarts[k]:gapends[k]]= 0

    y = s + n
    y_masked = mask * y


    # Fit PSD with a spline of degree 2 and 10 knots
    psd_cls = psdmodel.PSDSpline(n_data, fs, n_knots=10, d=2, fmin=fs/n_data, fmax=fs/2)
    psd_cls.estimate(y - s)
    psd = psd_cls.calculate(n_data)

    # instantiate imputation class
    imp_cls = datamodel.GaussianStationaryProcess(s, mask, psd_cls, na=50, nb=50)
    # perform offline computations
    imp_cls.compute_offline()
    # Imputation of missing data
    y_rec = imp_cls.draw_missing_data(y_masked)

    # Transform results in the Fourier domain
    f = np.fft.fftfreq(n_data) * fs
    # Fourier transforms
    y_fft = np.fft.fft(y)
    y_masked_fft = np.fft.fft(y_masked)
    y_rec_fft = np.fft.fft(y_rec)

    # Plot results
    fig, ax = plt.subplots()
    ax.set_title(r"Noise FFT")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel(r"PSD [1/Hz]") 
    ax.loglog(f[f>0], np.abs(y_fft[f>0])/np.sqrt(n_data*fs), label="Full")
    ax.loglog(f[f>0], np.abs(y_masked_fft[f>0])/np.sqrt(n_data*fs), label="Gapped")
    ax.loglog(f[f>0], np.abs(y_rec_fft[f>0])/np.sqrt(n_data*fs), label="Imputed")
    plt.legend()
    plt.show()