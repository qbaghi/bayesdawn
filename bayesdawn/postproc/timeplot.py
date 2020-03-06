#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import load_mcmc_config
# from . import resanalysis
import resanalysis
import gapgenerator as gg
import psdplot
from bayesdawn import imputation
from bayesdawn import gwsampler
from bayesdawn import psdsampler
import alfreq

import os
import numpy as np
import h5py
import time

# FTT modules
import pyfftw
import fftwisdom
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft

# Plot modules
import myplots
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

from scipy import linalg as LA

import pyFLR
import alwaves

if __name__=='__main__':

    fftwisdom.load_wisdom()
    # For antenna gaps and f0 = 1e-4 Hz
    base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/1e-4Hz/"
    filepaths = [base, base, base]
    #    #2019-02-12_18h20-32 #2019-02-25_19h32-49 2019-02-26_13h57-54
    # prefixes = ["2019-02-27_16h50-41","2019-02-26_13h58-04","2019-02-27_16h50-40"]
    prefixes = ["2019-03-04_23h27-50", "2019-03-04_23h38-27", "2019-03-05_13h18-54"]
    maskname = 'periodic'
    f0 = "1e-4Hz"

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

    # ==========================================================================
    # Compute injected GW signal
    # ==========================================================================
    # Sampling time, observation duration, number of data points
    ts = p.get('Cadence')
    Tobs = p.get('ObservationDuration')
    N = np.int(Tobs / ts)
    t = np.arange(0, N) * ts
    fastresp = pyFLR.fastResponse(alwaves.Phi_GB_func, pyFLR.beta_GB, 0, low_freq=True)
    t1 = time.time()
    tdi_channels = ['X1', 'Y1', 'Z1']
    # Same format as in MLDC
    dTDI_gw = np.empty((N, 4), dtype=np.float64)
    for i in range(3):
        tm, dTDI_gw[:, i + 1] = fastresp.evaluate_tdi(p, channel=tdi_channels[i])
    dTDI_gw[:, 0] = t
    t2 = time.time()
    print("Fast approximation: " + str(t2 - t1) + " sec")

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
    t = np.arange(0, N)/fs

    # ==========================================================================
    # Set the bounds of the parameters
    # ==========================================================================
    # Frequency domain to consider for the regression:
    f_maximum = conf.f0
    delta_f = 1e-7
    f1 = f_maximum - conf.B / 2
    f2 = f_maximum + conf.B / 2
    f_dot_min = 0.1 * delta_f / Tobs
    f_dot_max = 10 * delta_f / Tobs
    nc = alfreq.optimal_order(np.pi / 2, p.get('Frequency'))

    if conf.matmodel == 'chirping_GB_2p':
        bounds = [[f_maximum - 2 * delta_f, f_maximum + 2 * delta_f], [f_dot_min, f_dot_max]]
        names = ['f_0', 'f_dot']
        distribs = ['symbeta', 'uniform']

    elif conf.matmodel == 'chirping_GB_4p':

        bounds = [[0, np.pi],
                  [0, 2 * np.pi],
                  [f_maximum - 2 * delta_f, f_maximum + 2 * delta_f],
                  [f_dot_min, f_dot_max]]
        names = ['theta', 'phi', 'f_0', 'f_dot']
        distribs = ['uniform', 'uniform', 'uniform', 'uniform']

    elif conf.matmodel == 'mono_GB_3p':
        bounds = [[0, np.pi],
                  [0, 2 * np.pi],
                  [f_maximum - 2 * delta_f, f_maximum + 2 * delta_f]]
        names = ['theta', 'phi', 'f_0']
        distribs = ['uniform', 'uniform', 'symbeta']

    # =========================================================================
    # Set GW signal model
    # =========================================================================
    t1 = time.time()

    gwm = gwsampler.GWModel(names=names,
                            bounds=bounds,
                            distribs=distribs,
                            matmodeltype=conf.matmodel,
                            timevect=np.copy(t),
                            tdi='X1',
                            nc=nc,
                            fmin=f1,
                            fmax=f2)

    t2 = time.time()
    print("Loading Bayesian model took " + str(t2 - t1) + " seconds.")




    # =========================================================================
    # For the PSD samples
    # =========================================================================
    N = np.int(Tobs / ts)
    f = np.fft.fftfreq(N) / ts
    f_pos = f[f > 0]

    j0=2
    # For noise parameters
    chain_psd = resanalysis.load_chain([filepaths[j0] + noisenames[j0]], data_name="psd/samples")
    logp_psd = resanalysis.load_chain([filepaths[j0] + noisenames[j0]], data_name="psd/logpvals")
    fc, S_samples, S_map, S_map_low, S_map_up = psdplot.compute_psd_map(chain_psd, logp_psd, fs, N)

    # Create PSD class
    psd = psdsampler.PSDSampler(N, fs, J = 30, D = 3, fmin = None, fmax=None)
    psd.update_psd_func(np.log(S_map))
    S = psd.calculate(N)

    # ==========================================================================
    # Load MCMC GW parameter results
    # ==========================================================================
    sample_list, logp_list, logl_list, bayes_list, bayes_std_list, f0, gaps, prefix, sufix = \
        resanalysis.load_mcmc_results(base, filepaths, filenames, noisenames, prefixes, maskname, conf)

    # MAP estimate
    if logp_list[j0].shape[1] == 2:
        imap = np.where(logp_list[j0][:, 1] == np.max(logp_list[j0][:, 1]))[0][0]
    elif logp_list[j0].shape[1] == 1:
        imap = np.where(logp_list[j0][:, 0] == np.max(logp_list[j0][:, 0]))[0][0]



    # ==========================================================================
    # Generate a few missing data draws
    # ==========================================================================

    y = (dTDI[0:N, 1] - np.mean(dTDI[0:N, 1]))/conf.scale
    y_mask = M_binary * y
    y_wind = M * y

    # MAP estimate of signal parameters
    param_map = sample_list[j0][imap]
    rescale = [1, 1, 1e-3]

    imp = imputation.nearestNeighboor(M_binary, Na=150, Nb=150)
    # Draw the signal
    y_gw_fft = gwm.draw_frequency_signal(param_map * rescale, S, fft(y_wind)*N/np.sum(M))
    # Inverse Fourier transform back in time domain
    y_gw = np.real(ifft(y_gw_fft))
    # Draw the missing data
    y_rec = imp.draw_missing_data(y_mask, y_gw, psd)
    # Draw the signal again
    y_gw_fft = gwm.draw_frequency_signal(param_map * rescale, S, y_rec)
    # Inverse Fourier transform back in time domain
    y_gw = np.real(ifft(y_gw_fft))
    # Draw the missing data
    y_rec = imp.draw_missing_data(y_mask, y_gw, psd)

    # ==========================================================================
    # Restrict to a short segment of data with one single gap
    # ==========================================================================
    i1 = np.int(971000 * fs)
    i2 = np.int(981500 * fs)
    Mseg = M[i1:i2]
    Mseg2 = M2[i1:i2]
    Mplot = M_binary[i1:i2] * 1e-19 -2.6e-20

    R = psd.calculate_autocorr(imp.N)[0:imp.N_max]
    S2 = psd.calculate(2 * imp.N_max)
    C = LA.toeplitz(R)
    # Local indices of missing and observed data
    ind_obsj = np.where(M_binary[i1:i2] == 1)[0]
    ind_misj = np.where(M_binary[i1:i2] == 0)[0]
    C_mo = C[np.ix_(ind_misj, ind_obsj)]
    C_mm = C[np.ix_(ind_misj, ind_misj)]
    C_oo = C[np.ix_(ind_obsj, ind_obsj)]
    CooI = LA.pinv(C_oo)

    # gw signal
    s_gw = y_gw[i1:i2]
    # Offset data
    y_seg = y[i1:i2] - y[i1:i2][ind_obsj].mean()
    # Conditional mean
    mu_given_o = s_gw[ind_misj] + C_mo.dot(CooI.dot(y_seg[ind_obsj] - s_gw[ind_obsj]))
    # # Conditional variance
    # var_given_o = np.diag(C_mm -  C_mo.dot( CooI.dot( C_mo.T ) ))


    # Create a matrix of samples of size N_samples x N_mis
    y_mis_samples = np.array([imp.single_imputation(y_seg - s_gw, M_binary[i1:i2], C, S2) + s_gw[ind_misj] for k in range(1000)])
    # Conditional sample mean
    # mu_given_o = np.mean(y_mis_samples, axis=0)
    # Conditional sample variance
    var_given_o = np.var(y_mis_samples, axis=0)

    # ==========================================================================
    # Plots in the time domain
    # ==========================================================================
    #y_mask2 = M2 * (dTDI[0:N, 1] - np.mean(dTDI[0:N, 1]))
    y_comp = (1 - M_binary) * y
    y_comp[M_binary > 0] = None
    y_comp2 = (1 - M2) * y
    y_comp2[M2 > 0] = None

    y_rec_m = (1 - M_binary) * y_rec
    y_rec_m[M_binary > 0] = None









    day = 3600 * 24


    # i1 = 90000
    # i2 = 188000 # np.int(30 * 24 * 3600 * fs)
    # Mseg = M[i1:i2]
    # Mseg2 = M2[i1:i2]
    #
    # # X = [t[i1:i2], t[i1:i2], t[i1:i2], t[i1:i2]]
    # # Y = [y[i1:i2], y_comp[i1:i2], Mseg * conf.scale, Mseg2 * conf.scale]
    # day = 3600*24
    # X = [t[i1:i2]/day, t[i1:i2]/day, t[i1:i2]/day]
    # Y = [y[i1:i2], y_comp[i1:i2], y_comp2[i1:i2]]
    # # colors = ['black', 'gray', 'red', 'orange']
    # # linewidths = [1, 1, 1, 1]
    # # labels = ['Observed data', 'Missing data', 'Mask']
    # # linestyles = ['solid', 'solid', 'dashed', 'solid']
    # colors = ['black', '0.7', 'red']
    # linewidths = [1, 1, 1]
    # labels = ['Observed data', 'Five-day period gaps', 'Daily random gaps']
    # linestyles = ['solid', 'solid', 'solid']
    #
    # fp = myplots.fplot(plotconf='time')
    # fp.xscale = 'linear'
    # fp.yscale = 'linear'
    # fp.ylabel = r'Fractional frequency'
    # fp.legendloc = 'upper right'
    # fp.xlabel = 'Time [days]'
    # fp.ylabel = 'TDI X [fractional frequency]'
    # fp.ylims = [-2.5e-20, 5e-20]
    # fp.xlims = [i1/fs/day,i2/fs/day]
    # fig2, ax2 = fp.plot(X, Y, colors, linewidths, labels, linestyles=linestyles)
    # ax2.minorticks_on()
    # fig2.savefig('/Users/qbaghi/Documents/studies/gaps/time_series/time_series_gap.png')
    # plt.draw()
    # plt.show()




    ymin, ymax = -2.6e-20, 9e-20

    fp = myplots.fplot(plotconf='time')
    fp.xscale = 'linear'
    fp.yscale = 'linear'
    fp.draw_frame = True
    fp.ylabel = r'Fractional frequency'
    fp.legendloc = 'upper left'
    fp.xlabel = 'Time [days]'
    fp.ylabel = 'TDI X [fractional frequency]'
    fp.ylims = [-1.5e-20, 7e-20]
    fp.xlims = [i1/fs/day, i2/fs/day]
    # mpl.rcParams['figure.autolayout'] = False
    #fig2, ax2 = fp.plot(X, Y, colors, linewidths, labels, linestyles=linestyles, zorders=[1, 3, 2, 5, 4])
    # fig2, ax2 = fp.plot(X, Y, colors, linewidths, labels, linestyles=linestyles, zorders=[1, 3, 2])
    #



    # fig2.savefig('/Users/qbaghi/Documents/articles/papers/papers/gaps/figures/time_series/time_series_gap_rec2.pdf')

    # PLOT THE LONGEST TIME SERIES (between i1 and i2)
    fig2, ax2 = plt.subplots(ncols=1, figsize=[9, 7])

    X2 = list([t[i1:i2]/day, t[i1:i2]/day, t[i1:i2]/day])
    Y2 = list([y[i1:i2]*conf.scale, M_binary[i1:i2] * 2e-19 - 3e-20, y_rec_m[i1:i2]*conf.scale])
    colors = ['black', 'red', 'blue']
    linewidths = [1, 1, 1]
    labels = ['Observed data', None, 'Imputation']
    linestyles = ['solid', 'solid', 'solid']

    # fig2, ax2 = fp.plot(X2, Y2, colors, linewidths, labels, linestyles=linestyles, zorders=[1, 3, 2])

    # plt.show()

    for i in range(len(X2)):
        ax2.plot(X2[i], Y2[i], color=colors[i], label=labels[i], linewidth=linewidths[i], linestyle=linestyles[i])
    ax2.set_xlim(i1/fs/day, i2/fs/day)
    ax2.set_ylim(ymin, ymax)
    ax2.minorticks_on()

    ax2.legend(loc="upper right")



    # X2 = [t[i1:i2]/day, t[i1:i2]/day, t[i1:i2][ind_misj]/day, t[i1:i2][ind_misj]/day, t[i1:i2]/day]
    # # Y = [y_seg*conf.scale, Mplot, y_rec_m[i1:i2]*conf.scale, mu_given_o*conf.scale, dTDI_gw[i1:i2, 1]]
    # Y2 = [y_seg * conf.scale, Mplot, y_mis_samples[0] * conf.scale, mu_given_o * conf.scale, dTDI_gw[i1:i2, 1]]
    # colors = ['black', 'red', 'blue', 'orange', 'g']
    # linewidths = [1, 1, 1, 2, 2]
    # labels = ['Observed data', None, 'Imputation', 'Conditional mean', 'True GW signal']
    # linestyles = ['solid', 'solid', 'solid', 'dashed', 'solid']
    # for i in range(len(X2)):
    #     axins2.plot(X2[i], Y2[i], label=labels[i], linewidth=linewidths[i], linestyle=linestyles[i])

    t3 = 11.2650 * day
    t4 = 11.2670 * day
    i3 = np.int(t3/fs)
    i4 = np.int(t4/fs)
    inds = np.where( (t[i1:i2] < t4) & (t[i1:i2] >= t3))[0]
    inds2 = np.where( (t[i1:i2][ind_misj] < t4) & (t[i1:i2][ind_misj] >= t3))[0]

    Xin = [t[i1:i2][inds]/day, t[i1:i2][ind_misj][inds2]/day, t[i1:i2][inds]/day, t[i1:i2][ind_misj][inds2]/day]
    Yin = [y_seg[inds] * conf.scale, y_mis_samples[0][inds2] * conf.scale, dTDI_gw[i1:i2, 1][inds],
           mu_given_o[inds2] * conf.scale]
    colors = ['black', 'blue', 'g', 'chocolate']
    linewidths = [1,  1, 2, 2]
    labels = [None, None, 'True GW signal', 'Conditional mean']
    linestyles = ['solid', 'solid', 'solid', 'dashed']
    markers = ['.', '.', None, None]



    # PLOT Inset
    # Second subplot, showing an image with an inset zoom
    # and a marked inset
    # axins2 = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
    # x1, x2, y1, y2 = t3 / day, t4 / day, -1e-20, 2e-20
    x1, x2, y1, y2 = t[i1:i2][inds][0]/day, t[i1:i2][inds][-1]/day, -2e-20, 3e-20

    axins2 = inset_axes(ax2, 3, 2, loc='upper center', axes_kwargs={"xlim": [x1, x2], "ylim": [y1, y2]})
    #, bbox_to_anchor=(0.2, 0.55), bbox_transform=ax2.figure.transFigure)
    # axins2 = zoomed_inset_axes(ax2, 4, loc='upper center', axes_kwargs={"xlim":[x1, x2], "ylim": [y1, y2]})  # zoom = 6

    for i in range(len(Xin)):
        axins2.plot(Xin[i], Yin[i], color=colors[i], label=labels[i], linewidth=linewidths[i], linestyle=linestyles[i],
                    marker=markers[i])

    # Confidence interval
    myplots.confidenceIntervals(axins2, t[i1:i2][ind_misj][inds2]/day,
                                (mu_given_o[inds2] - 2.575*np.sqrt(np.abs(var_given_o[inds2]))) * conf.scale,
                                (mu_given_o[inds2] + 2.575*np.sqrt(np.abs(var_given_o[inds2]))) * conf.scale,
                                'orange', 0.2)

    # sub region of the original time series

    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.legend(fontsize=10, loc="upper right", frameon=False)



    # fix the number of ticks on the inset axes
    # axins2.yaxis.get_major_locator().set_params(nbins=7)
    # axins2.xaxis.get_major_locator().set_params(nbins=7)

    ax2.set_xlabel(fp.xlabel, fontsize=19)
    ax2.set_ylabel(fp.ylabel, fontsize=19)

    # plt.setp(axins2.get_xticklabels(), visible=False)
    # plt.setp(axins2.get_yticklabels(), visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax2, axins2, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.vlines(x=t[i1:i2][ind_misj[0]] / day, ymin=ymin, ymax=ymax, linewidth=1, color='r')

    fig2.savefig('/Users/qbaghi/Documents/articles/papers/papers/gaps/figures/time_series/time_series_gap_rec_inset.pdf')
    # plt.draw()
    plt.show()
