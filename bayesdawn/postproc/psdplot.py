import numpy as np
from scipy import optimize
import scipy.signal
import numpy.fft

# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft

# Monkey patch fftpack with pyfftw.interfaces.scipy_fftpack
scipy.fftpack = pyfftw.interfaces.scipy_fftpack
numpy.fft = pyfftw.interfaces.numpy_fft
from scipy import signal

# Plot modules
import myplots

def compute_periodogram(x, fs=1.0, wind='tukey'):
    """

    Parameters
    ----------
    x : numpy array
        time series data
    fs : scalar float
        sampling frequency
    wind : basestring
        type of time windowing

    Returns
    -------
    freq : numpy array
        frequency vector
    per : numpy array
        periodogram of the time series expressed in A / Hz where A is the unit of x

    """

    n = x.shape[0]

    freq = np.fft.fftfreq(n) * fs

    if wind == 'tukey':
        w = signal.tukey(n)
    elif wind == 'hanning':
        w = np.hanning(n)
    elif wind == 'blackman':
        w = np.blackman(n)
    elif wind == 'rectangular':
        w = np.ones(n)

    k2 = np.sum(w**2)

    return freq, np.abs(fft(x * w))**2 / (k2 * fs)


def plot_periodogram(x, fs=1.0, wind='tukey', xlabel='Frequency [Hz]', ylabel='Periodogram', sqr=True, colors=None,
                     linewidths=None, labels=None, linestyles='solid'):

    if type(x) == list:
        data_list = [compute_periodogram(xi, fs=fs, wind=wind) for xi in x]
    else:
        data_list = [compute_periodogram(x, fs=fs, wind=wind)]


    fp = myplots.fplot(plotconf='frequency')
    fp.xscale = 'log'
    fp.yscale = 'log'
    fp.draw_frame = True
    fp.ylabel = r'Fractional frequency'
    fp.legendloc = 'upper left'
    fp.xlabel = xlabel
    fp.ylabel = ylabel

    if labels is None:
        labels = [None for dat in data_list]
    if linestyles=='solid':
        linestyles = ['solid' for dat in data_list]
    if linewidths is None:
        linewidths = np.ones(7)
    if colors is None:
        colors = ['k', 'r', 'b', 'g', 'm', 'gray', 'o']

    colors = [colors[i] for i in range(len(data_list))]

    x_list = [dat[0][dat[0] > 0] for dat in data_list]
    if sqr:
        y_list = [np.sqrt(dat[1][dat[0] > 0]) for dat in data_list]
    else:
        y_list  = [dat[1][dat[0] > 0] for dat in data_list]

    fig1, ax1 = fp.plot(x_list, y_list, colors, linewidths, linestyles=linestyles, labels=labels)

    return fig1, ax1


def compute_psd_map(chain_psd, logp_psd, fs, N):
    """
    Compute MAP estimator of PSD from posterior distribution and log-posterior probability values

    Parameters
    ----------
    chain_psd : 2d numpy array
        nsamples x ndim matrix containing the posterior samples
    logp_psd : 1d numpy array
        vector of size nsamples containing the log-probability values


    Returns
    -------
    S_samples : 2d numpy array
        samples transformed in PSD(f) values where f are the knots of the spline


    """
    J = 30
    fmin = fs / N
    fmax = fs / 2
    ns = - np.log(fmax) / np.log(10)
    n0 = - np.log(fmin) / np.log(10)  # 6
    jvect = np.arange(0, J)
    alpha_guess = 0.8
    targetfunc = lambda x: n0 - (1 - x ** (J)) / (1 - x) - ns
    result = optimize.fsolve(targetfunc, alpha_guess)
    alpha = result[0]
    n_knots = n0 - (1 - alpha ** jvect) / (1 - alpha)
    f_knots = 10 ** (-n_knots)

    fc = np.concatenate((f_knots, [fs / 2]))

    S_samples = [np.exp(chain_psd[i, :]) for i in range(chain_psd.shape[0])]

    logS_mean = np.mean(chain_psd, axis=0)
    # S_mean = np.exp(logS_mean)
    logS_map = chain_psd[np.where(logp_psd == np.max(logp_psd))[0][0], :]
    S_map = np.exp(logS_map)
    logS_sig = np.std(chain_psd, axis=0)
    S_map_low = np.exp(logS_map - logS_sig * 3)
    S_map_up = np.exp(logS_map + logS_sig * 3)

    return fc, S_samples, S_map, S_map_low, S_map_up


