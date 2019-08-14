import load_mcmc_config

import gapgenerator as gg
import pyFLR
import alwaves

import os
import numpy as np
from scipy import optimize
import scipy.signal
import numpy.fft
import h5py
import time

import tdi

# FTT modules
import pyfftw
import fftwisdom
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft

# Monkey patch fftpack with pyfftw.interfaces.scipy_fftpack
scipy.fftpack = pyfftw.interfaces.scipy_fftpack
numpy.fft = pyfftw.interfaces.numpy_fft
from scipy import signal


# Plot modules
import myplots
import seaborn as sns
from matplotlib import pyplot as plt

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

if __name__ == '__main__':

    import resanalysis
    # # For antenna gaps and f0 = 1e-4 Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/1e-4Hz/"
    # filepaths = [base, base, base]
    # #    #2019-02-12_18h20-32 #2019-02-25_19h32-49 2019-02-26_13h57-54
    # # prefixes = ["2019-02-27_16h50-41","2019-02-26_13h58-04","2019-02-27_16h50-40"]
    # prefixes = ["2019-03-04_23h27-50", "2019-03-04_23h38-27", "2019-03-05_13h18-54"]
    # maskname = 'periodic'
    # f0 = "1e-4Hz"

    # For random gaps and f0 = 1e-4 Hz
    base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/1e-4Hz/"
    #    filepaths = [base + "complete_1e-4/",base + "randgaps_1e-4/",
    #                 base + "randgaps_1e-4_imp/"]
    filepaths = [base, base, base]
    #    prefixes = ["2019-02-15_17h00-34","2019-02-17_16h27-29","2019-02-15_17h45-22"]
    #    prefixes = ["2019-02-25_19h32-49","2019-02-26_10h59-46","2019-02-26_11h39-42"]
    #    prefixes = ["2019-02-27_16h50-41","2019-02-27_17h03-46","2019-02-27_17h25-49"]
    prefixes = ["2019-03-04_23h27-50", "2019-03-04_23h01-13", "2019-03-05_13h28-01"]
    maskname = 'random'
    f0 = "1e-4Hz"

    # # For antenna gaps and f0 = 2e-4 Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/2e-4Hz/"
    # filepaths = [base,
    #              base,
    #              base]
    # #prefixes = ["2019-02-26_11h30-35","2019-02-27_19h03-46","2019-02-26_11h31-43"]
    # prefixes = ["2019-03-05_01h23-32", "2019-03-04_23h35-50", "2019-03-04_23h46-03"]
    # maskname = 'periodic'

    # # For random gaps and f0 = 2e-4 Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/2e-4Hz/"
    # filepaths = [base,base,base]
    # # prefixes = ["2019-02-26_11h30-35","2019-02-26_16h26-10","2019-02-27_14h47-08"]
    # prefixes = ["2019-03-05_01h23-32", "2019-03-04_23h39-12", "2019-03-05_13h26-35"]
    # maskname = 'random'
    # f0="2e-4Hz"

    # # For antenna gaps and f0 = 5e-4 Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/5e-4Hz/"
    # filepaths = [base,
    #             base,
    #             base]
    # prefixes = ["2019-03-04_23h53-12", "2019-03-04_23h53-48", "2019-03-05_13h30-15"]
    # maskname = 'periodic'

    # # For random gaps and f0 = 5e-4 Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/5e-4Hz/"
    # filepaths = [base,
    #              base,
    #              base]
    # #prefixes = ["2019-02-22_10h54-28","2019-02-22_10h55-01","2019-02-22_10h55-17"]
    # #prefixes = ["2019-02-27_20h53-37","2019-02-27_21h31-18","2019-02-27_21h31-57"]
    # #prefixes = ["2019-03-04_11h50-00","2019-03-04_12h42-39","2019-03-04_12h49-19"]
    # prefixes = ["2019-03-04_23h53-12", "2019-03-04_23h57-01", "2019-03-05_13h36-46"]
    # maskname = 'random'


    # ==========================================================================
    # Load configuration file
    # ==========================================================================

    filenames = [pre + '_chain_full2.hdf5' for pre in prefixes]
    noisenames = [pre + '__psd.hdf5' for pre in prefixes]

    # signal_name = "chains/chain/"
    # signal_name = "chain"
    signal_name = "chains/chain1/"

    # config_file = '/Users/qbaghi/Codes/python/inference/configs/wgmcmc_config_lowSNR_mono_2sources.txt'
    # config_file = '/Users/qbaghi/Codes/python/inference/configs/ptemcee_config_2e-4Hz_gaps.txt'
    config_file = filepaths[0] + prefixes[0] + '_config.txt'
    conf = load_mcmc_config.loadconfig(config_file)

    # ==========================================================================
    # Load configuration file
    # ==========================================================================

    filenames = [pre + '_chain_full2.hdf5' for pre in prefixes]
    noisenames = [pre + '__psd.hdf5' for pre in prefixes]

    # signal_name = "chains/chain/"
    # signal_name = "chain"
    signal_name = "chains/chain1/"

    # config_file = '/Users/qbaghi/Codes/python/inference/configs/wgmcmc_config_lowSNR_mono_2sources.txt'
    # config_file = '/Users/qbaghi/Codes/python/inference/configs/ptemcee_config_2e-4Hz_gaps.txt'
    config_file = filepaths[0] + prefixes[0] + '_config.txt'


    # ==========================================================================
    # Load TDI data
    # ==========================================================================
    conf = load_mcmc_config.loadconfig(config_file)
    hdf5_name = '/Users/qbaghi/Codes/data/simulations/' + os.path.basename(conf.hdf5_name)
    dTDI, p, truths, labels, ts, Tobs = resanalysis.loadtdi(hdf5_name)



    # =========================================================================
    # For the mask
    # =========================================================================

    N = np.int(Tobs / ts)
    fs = 1 / ts

    if maskname == 'random':
        maskfile = '/Users/qbaghi/Codes/data/masks/2018-11-14_15h48-51_mask_micrometeorites.hdf5'
        if np.abs(truths[2] - 1e-4) < 1e-8:
            n_wind = 2310
        elif np.abs(truths[2] - 2e-4) < 1e-8:
            n_wind = 1860
        elif np.abs(truths[2] - 5e-4) < 1e-8:
            n_wind = 810
    elif maskname == 'periodic':
        maskfile = '/Users/qbaghi/Codes/data/masks/2018-10-29_15h22-12_mask_antenna.hdf5'
        if np.abs(truths[2] - 1e-4) < 1e-8:
            n_wind = 1500
        elif np.abs(truths[2] - 2e-4) < 1e-8:
            n_wind = 750
        elif np.abs(truths[2] - 5e-4) < 1e-8:
            n_wind = 300

    maskfh5 = h5py.File(maskfile, 'r')
    M = maskfh5['mask'][()]
    maskfh5.close()

    maskfh5 = h5py.File('/Users/qbaghi/Codes/data/masks/2018-11-14_15h48-51_mask_micrometeorites.hdf5', 'r')
    M2 = maskfh5['mask'][()]
    maskfh5.close()

    M_binary = np.ones(N)
    M_binary[M == 0] = 0
    Nd, Nf = gg.findEnds(M_binary)
    M = gg.windowing(Nd, Nf, N, window='modified_hann', n_wind=n_wind)

    n_wd = conf.n_wind
    # n_wd = 500000
    wd = gg.modified_hann(N, n_wind=n_wd)

    # =========================================================================
    # Prepare data to be plotted
    # =========================================================================
    fftwisdom.load_wisdom()
    data_full = (dTDI[0:N, 1] - np.mean(dTDI[0:N, 1])) / conf.scale
    # data = M_binary * data_full

    data_full_fft = fft(wd * dTDI[0:N, 1] / conf.scale)
    data_mask_fft = fft(M * dTDI[0:N, 1] / conf.scale)
    # data_full_fft = fft(data_full)

    # Per = np.abs(fft(dTDI[0:N,1]/conf.scale))**2 / (fs*N)


    # Averaged periodograms

    K = 8
    N0 = np.int(N/K)
    wd0 = gg.modified_hann(N0, n_wind=n_wd)
    Per = 0
    Per_mask = 0
    for k in range(K):

        data_full_fft = fft(wd0 * dTDI[k*N0:(k+1)*N0, 1] / conf.scale)
        data_mask_fft = fft(M[k*N0:(k+1)*N0] * dTDI[k*N0:(k+1)*N0, 1] / conf.scale)

        Per += np.abs(data_full_fft) ** 2 / (fs * np.sum(wd0 ** 2))
        Per_mask += np.abs(data_mask_fft) ** 2 / (fs * np.sum(M[k*N0:(k+1)*N0] ** 2))

    fr = np.fft.fftfreq(N0) * fs
    Per = Per[fr > 0]/K
    Per_mask = Per_mask[fr > 0]/K
    freq = fr[fr > 0]



    # # Average the periodograms
    # print("Start welch periodogram calculation")
    # t1 = time.time()
    # # f_per, Per = scipy.signal.welch(data_full, fs=fs, window='hanning', nperseg=2**10)
    # # f_per, Per_mask = scipy.signal.welch(data, fs=fs, window='hanning', nperseg=2**10)*N/np.sum(M_binary)
    # (Per, freq), spect = spectrum.WelchPeriodogram(data_full, NFFT=2**20, sampling=fs)
    # (Per_mask, freq), spect_mask = spectrum.WelchPeriodogram(data, NFFT=2**20, sampling=fs)
    # Per = Per/2
    # Per_mask = Per_mask*N/np.sum(M_binary)/2
    # t2 = time.time()
    # print("Welch periodograms computed in " + str(t2-t1))

    # =========================================================================
    # FOr the PSD samples
    # =========================================================================
    N = np.int(Tobs / ts)
    f = np.fft.fftfreq(N) / ts
    f_pos = f[f > 0]

    jn = 2
    # For noise parameters
    chain_psd = resanalysis.load_chain([filepaths[jn] + noisenames[jn]], data_name="psd/samples")
    logp_psd = resanalysis.load_chain([filepaths[jn] + noisenames[jn]], data_name="psd/logpvals")

    fc, S_samples, S_map, S_map_low, S_map_up = compute_psd_map(chain_psd, logp_psd, fs, N)

    jn = 1
    # For noise parameters
    chain_psd = resanalysis.load_chain([filepaths[jn] + noisenames[jn]], data_name="psd/samples")
    logp_psd = resanalysis.load_chain([filepaths[jn] + noisenames[jn]], data_name="psd/logpvals")

    fc, S_samples_wind, S_map_wind, S_map_low_wind, S_map_up_wind = compute_psd_map(chain_psd, logp_psd, fs, N)

    # PSD = psdspline2.PSD_spline(N,1/ts,J = 30, D = 3, fmin = None,fmax=None)
    # fc = np.exp(PSD.logfc)



    # =========================================================================
    # GW signals at other frequencies
    # =========================================================================
    simu_dir = "/Users/qbaghi/Codes/data/simulations/"
    simu_paths = [simu_dir + "2019-02-15_15h40-19_monof0=1e-4Hz_1year_A=15e-20_ts=10s_mynoise.hdf5",
                  simu_dir + "2019-02-05_14h25-42_monof0=2e-4Hz_1year_A=2e-20_ts=10s_mynoise.hdf5",
                  simu_dir + "2019-02-15_15h41-18_monof0=5e-4Hz_1year_A=2e-21_ts=10s_mynoise.hdf5"]


    simu_params = [resanalysis.loadtdi(hdf5_name) for hdf5_name in simu_paths]
    # dTDI, p, truths, labels, ts, Tobs = resanalysis.loadtdi(hdf5_name)


    dTDI_signal_list = []
    for i in range(len(simu_params)):

        p = simu_params[i][1]

        # Sampling time, observation duration, number of data points
        ts = p.get('Cadence')
        Tobs = p.get('ObservationDuration')
        N = np.int(Tobs/ts)
        t = np.arange(0, N)*ts
        # ==========================================================================
        # Generate the GW TDI signal with linear model
        # ==========================================================================
        fastresp = pyFLR.fastResponse(alwaves.Phi_GB_func, pyFLR.beta_GB, 0, low_freq=True)
        t1 = time.time()
        tdi_channels = ['X1','Y1','Z1']
        # Same format as in MLDC
        dTDI_signal = np.empty((N,4),dtype = np.float64)
        for i in range(3):
            tm, dTDI_signal[:, i+1] = fastresp.evaluate_tdi(p, channel=tdi_channels[i])
        dTDI_signal[:, 0] = t
        t2 = time.time()
        print("Fast approximation: " + str(t2-t1) + " sec")

        dTDI_signal_list.append(dTDI_signal)


    signal_fft_list = [fft(wd * dTDI_signal[0:N, 1] / conf.scale) for dTDI_signal in dTDI_signal_list]
    Per_signal_list = [np.abs(signal_fft) ** 2 / (fs * np.sum(wd ** 2)) for signal_fft in signal_fft_list]


    # =========================================================================
    # True PSD
    # =========================================================================

    SnX = tdi.noisepsd_X(fc, model='SciRDv1', includewd=None)

    # f2N = np.fft.fftfreq(2*N) / ts
    # f2N[0] = f2N[1]
    # SnX = tdi.noisepsd_X(np.abs(f2N), model='SciRDv1', includewd=None)
    # RnX = ifft(SnX)[0:N]
    #
    # n = 200000
    # r1 = np.arange(n)*ts
    # r0 = np.zeros(n)
    # y = RnX[0:n]/RnX[0]
    # ipos = np.where(y>=0)[0]
    # ymax = np.zeros(n)
    # ymax[ipos] = y[ipos]
    # ymin = np.copy(y)
    # ymin[ipos] = np.zeros(len(ipos))
    # plt.vlines(r1, ymin, ymax)
    # # plt.plot(r1, y)
    # plt.xscale('linear')
    # plt.yscale('linear')
    # # plt.yscale('log')
    # # plt.xscale('log')
    # plt.show()

    # ==========================================================================
    # Plots
    # ==========================================================================


    #    X = [f[1:n+1],f[1:n+1],f[1:n+1]]
    #    Y = [np.sqrt(z/fs)*conf.scale,np.sqrt(0.5*S[1:n+1])*conf.scale,np.sqrt(S_mean/fs)*conf.scale]
    #    linewidths = [1,2,2]
    #    linestyles = ['solid','solid','dashed']
    #    colors = ['k','g','r']
    #    labels = ['Periodogram','True PSD','Posterior mean']
    #    X = [fc]
    #    Y = [np.sqrt(S_mean/fs)*conf.scale]
    # X = [f[f > 0], f[f > 0], fc, fc]
    #     # Y = [np.sqrt(Per_mask[f > 0]) * conf.scale, np.sqrt(Per[f > 0]) * conf.scale,
    #     #      np.sqrt(SnX / 2), np.sqrt(S_map / fs) * conf.scale]
    #     # linewidths = [1, 1, 2, 2]
    #     # linestyles = ['solid', 'solid', 'solid', 'dashed']
    #     # colors = ['gray', 'black', 'green', 'blue']
    #     # labels = ['Masked data', 'Complete data', 'True PSD', 'PSD estimate']


    # freq = f[f > 0]

    X = [freq, freq]# , fc, fc, fc]
    Y = [np.sqrt(Per_mask) * conf.scale, np.sqrt(Per) * conf.scale]#,
         # np.sqrt(SnX / 2), np.sqrt(S_map_wind / fs) * conf.scale, np.sqrt(S_map / fs) * conf.scale]
    linewidths = [1, 1] #, 3, 2, 2]
    linestyles = ['solid', 'solid'] # , 'solid', 'dotted', 'dashed']
    colors = ['gray', 'black'] # , 'green', 'red', 'dodgerblue']
    labels = ['Masked data', 'Complete data'] # , 'True PSD', 'PSD estimate (wind.)', 'PSD estimate (DA)']


    # signal_colors = ["navajowhite", "orange", "darkgoldenrod"]
    # signal_labels = [r"GW signal, $f_0 = 0.1$ mHz", r"GW signal, $f_0 = 0.2$ mHz", r"GW signal, $f_0 = 0.5$ mHz"]
    #
    # for i in range(len(Per_signal_list)):
    #
    #     X.append(f[f > 0])
    #     Y.append(np.sqrt(Per_signal_list[i][f > 0]) * conf.scale)
    #     linewidths.append(1)
    #     linestyles.append("solid")
    #     colors.append(signal_colors[i])
    #     labels.append(signal_labels[i])

    fp = myplots.fplot(plotconf='frequency')
    fp.lgsize = 12
    fp.xscale = 'log'
    fp.yscale = 'log'
    fp.ylabel = r'$\sqrt{\rm PSD}$ [$\rm Hz^{-1/2}$]'
    fp.legendloc = 'upper right'
    # fp.xlims = [1e-7, fs / 2]
    # fp.ylims = [1e-22, 1e-16]
    fp.xlims = [1e-6, fs / 2]
    fp.ylims = [1e-22, 1e-18]
    fig, ax1 = fp.plot(X, Y, colors, linewidths, labels, linestyles=linestyles, zorders=[1, 2])
    # myplots.confidenceIntervals(ax1, fc, np.sqrt(S_map_low / fs) * conf.scale,
    #                             np.sqrt(S_map_up / fs) * conf.scale, 'dodgerblue', 0.5)

    plt.legend(ncol=2, frameon=False)#, handleheight=2.4, labelspacing=0.05)
    # plt.savefig('/Users/qbaghi/Documents/articles/papers/papers/gaps/figures/psd_estimation/'+ maskname + '_' + f0 + '.pdf')
    plt.savefig('/Users/qbaghi/Documents/conferences/LISA_consortium_Gainesville/'+ maskname + '_' + f0 + '.png')
    plt.draw()
    plt.show()
