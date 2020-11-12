# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:22:49 2019

@author: qbaghi
"""

import h5py
import numpy as np
import corner
from ptemcee import util
from scipy.integrate import simps
# import load_mcmc_config
# from LISAhdf5 import LISAhdf5
from bayesdawn import psdsampler

from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
import pandas

# from ligo.skymap import Clustered2Plus1DSkyKDE, Clustered2DSkyKDE
# from ligo.skymap import io, kde, bayestar, version, plot, postprocess
# from astropy.coordinates import SkyCoord
# from astropy.time import Time
# from astropy import units as u

import seaborn as sns


def log_evidence_estimate(logl, betas, fburnin=0.1):
    """
    Thermodynamic integration estimate of the evidence for the sampler.
    :param fburnin: (optional)
        The fraction of the chain to discard as burnin samples; only the
        final ``1-fburnin`` fraction of the samples will be used to
        compute the evidence; the default is ``fburnin = 0.1``.
    :return ``(logZ, dlogZ)``: Returns an estimate of the
        log-evidence and the error associated with the finite
        number of temperatures at which the posterior has been
        sampled.
    For details, see ``thermodynamic_integration_log_evidence``.
    """

    istart = int(logl.shape[0] * fburnin + 0.5)
    mean_logls = np.mean(np.mean(logl, axis=2)[istart:, :], axis=0)
    return util.thermodynamic_integration_log_evidence(betas[-1], mean_logls)


def cornerplot(s_list, truths_vect, offset, rscales, labels,
               colors=['k', 'gray', 'b'], limits=None, fontsize=16,
               bins=50, truth_color='cadetblue', figsize=(9, 8.5),
               linewidth=1, smooth=1.0, smooth1d=1.0,
               plot_datapoints=False, alpha=1):
    """

    Parameters
    ----------
    s_list : list of 2d numpy arrays
        list of posterior samples
    truths_vect : array_like
        vector of truth value of parameters
    offset : array_like
        vector of offsets to re-center each parameter distribution
    rscales : array_like
        vector of scale factors to re-center each parameter distribution
    labels : list of strings
        list of labels describing the samples
    colors : list of strings
        specifies colors associated with the labels
    limits : array_like
        set the limits of the distribution corner plots
    fontsize : scalar integer
        set the size of the x-label and y-label fonts
    bins : scalar integers
        histogram number of bins
    truth_color : string
        specifies the vertical bar colors referring the true value of parameters

    Returns
    -------
    fig : figure handle
        output figure
    axes : array of axes
        output axis array

    """

    if truths_vect is not None:
        truths_res = (truths_vect[0:s_list[0].shape[1]]-offset)*rscales
    else:
        truths_res = None

    ndim = s_list[0].shape[1]

    fig, axes = plt.subplots(ndim, ndim, figsize=figsize)

    fig = corner.corner((s_list[0]-offset)*rscales, 
                        truths=truths_res,
                        labels=labels, 
                        range=limits,
                        color=colors[0],
                        plot_datapoints=plot_datapoints, 
                        fill_contours=True, 
                        bins=bins,
                        smooth=smooth,
                        smooth1d=smooth1d,
                        label_kwargs={"fontsize": fontsize},
                        hist_kwargs={"linewidth": linewidth, "alpha": alpha},
                        truth_color=truth_color, fig=fig, use_math_text=True)

    # fig = plt.figure(figsize = (8,8))

    if len(s_list) > 1:
        for j in range(0, len(s_list)):
            fig = corner.corner((s_list[j]-offset)*rscales, 
                                truths=truths_res,
                                labels=labels, 
                                range=limits,
                                color=colors[j],
                                plot_datapoints=plot_datapoints, 
                                fill_contours=True, 
                                bins=bins,
                                smooth=smooth, 
                                smooth1d=smooth1d,
                                label_kwargs={"fontsize": fontsize}, 
                                hist_kwargs={"linewidth": linewidth, 
                                             "alpha": alpha},
                                truth_color=truth_color, 
                                fig=fig, 
                                use_math_text=True)


    return fig, axes


def jointplot(s_list, truths_vect, labels, legend_labels, kind="kde", 
              colors=['k', 'gray', 'b'], levels=4,
              linestyles = ['solid', 'dotted', 'dashed'], limits=None, 
              fontsize=16, linewidth=3, alpha=0.7, shade=False):
    """

    Parameters
    ----------
    s_list : list of 2d numpy arrays
        list of posterior samples
    truths_vect : array_like
        vector of truth value of parameters
    offset : array_like
        vector of offsets to re-center each parameter distribution
    rscales : array_like
        vector of scale factors to re-center each parameter distribution
    kind : string
        kind of histogram
    labels : list of strings
        list of labels describing the samples
    colors : list of strings
        specifies colors associated with the labels


    Returns
    -------

    """

    sns.set(style="white", palette="muted", color_codes=True)

    # fig, axes = plt.subplots(1, 1, figsize=figsize)

    # cmaps = []
    # for col in colors:
    #     if (col == 'blue') | (col == 'b'):
    #         cmaps.append(plt.cm.Blues)
    #     elif (col == 'gray') | (col == 'grey'):
    #         cmaps.append(plt.cm.bone)
    #     elif (col == 'black') | (col == 'k'):
    #         cmaps.append(plt.cm.Greys)
    #     elif (col == 'red') | (col == 'r'):
    #         cmaps.append(plt.cm.Reds)
    #     elif (col == 'orange') | (col == 'o'):
    #         cmaps.append(plt.cm.Oranges)



    # >> > iris = sns.load_dataset("iris")
    # >> > g = sns.jointplot("sepal_width", "petal_length", data=iris,
    #                        ...
    # kind = "kde", space = 0, color = "g")

    if limits == None:
        limits = [(None, None) for s in s_list]


    # For one category of data
    i=0

    # Combine data into DataFrame
    df = pandas.DataFrame({labels[0]: s_list[i][:, 0], labels[1]: s_list[i][:, 1]})

    # graph = sns.jointplot(x=labels[0], y=labels[1], data=df, kind=kind, stat_func=None,
    #                       color=colors[i], height=6, ratio=5, space=0.0,
    #                       dropna=True, xlim=limits[i][0], ylim=limits[i][1],
    #                       joint_kws={"label": legend_labels[i]},
    #                       marginal_kws=None, annot_kws=None)#, fontsize=fontsize)

    # marginal_kws = {"color": "black", "lw": 0.5}, joint_kws = {"colors": "black", "cmap": None, "linewidths": 0.5}

    graph = sns.JointGrid(labels[0], labels[1], data=df, space=0.0, ratio=5)

    graph = graph.plot_joint(sns.kdeplot, color=colors[i], linestyles=linestyles[i], label=legend_labels[i],
                             shade=shade, levels=levels, shade_lowest=False)
    graph = graph.plot_marginals(sns.distplot, hist=False, kde=True, color=colors[i],
                                 kde_kws={"shade": True, "linewidth": linewidth, "linestyle": linestyles[i]})
                                          #"alpha": alpha})

    # plt.sca("axis_name")
    # if len(s_list) > 1:
    for i in range(0, len(s_list)):
        graph.x = s_list[i][:, 0]
        graph.y = s_list[i][:, 1]
        # graph.plot_joint(plt.scatter, kde=True)
        # graph.plot_marginals(sns.distplot, kde=False, color=colors[i])

        graph = graph.plot_joint(sns.kdeplot, color=colors[i],
                                 linestyles=linestyles[i], label=legend_labels[i],
                                 shade=shade, levels=levels, shade_lowest=False)#, xlim=limits[i][0], ylim=limits[i][1])#  cmap="Blues_d")
        graph = graph.plot_marginals(sns.distplot, hist=False, kde=True, color=colors[i],
                                     kde_kws={"shade": True, "linewidth": linewidth, "linestyle": linestyles[i]})
                                              #"alpha": alpha})

    graph.ax_joint.set_xlim(limits[0])
    graph.ax_joint.set_ylim(limits[1])
    graph.ax_marg_x.set_xlim(limits[0])
    graph.ax_marg_y.set_ylim(limits[1])

    graph.ax_joint.xaxis.label.set_fontsize(fontsize)
    graph.ax_joint.yaxis.label.set_fontsize(fontsize)

    # graph.ax_joint.legend(loc=4, fontsize=10)

            # graph.plot_joint(s_list[i][:, 0], s_list[i][:, 1], data=None, kind=kind, stat_func=None, color=None,
            #                  height=6, ratio=5, space=0.0, dropna=True, xlim=limits[0][0], ylim=limits[0][1],
            #                  joint_kws=None, marginal_kws=None, annot_kws=None)
            # sns.jointplot(s_list[i][:, 0], s_list[i][:, 1], data=None, kind=kind, stat_func=None,
            #               color=None, height=6, ratio=5, space=0.0,
            #               dropna=True, xlim=limits[0][0], ylim=limits[0][1], joint_kws=None,
            #               marginal_kws=None, annot_kws=None)


    # axes.set_xlabel(labels[0], fontsize=fontsize)
    # axes.set_ylabel(labels[1], fontsize=fontsize)


    #plt.xlabel(labels[0], fontsize=fontsize)
    #plt.xlabel(labels[1], fontsize=fontsize)
    # axes = graph.axes
    # axes.set_xlim(limits[0])
    # axes.set_ylim(limits[1])

    graph.ax_joint.axvline(truths_vect[0], color='green', linewidth=1.5)
    graph.ax_joint.axhline(truths_vect[1], color='green', linewidth=1.5)
    # #[axes[i].axvline(truths_res[i], color='cadetblue', linewidth=1.5) for i in range(len(limits))]
    # [axes[i].axvline(truths_res[i], color='green', linewidth=1.5) for i in range(len(limits))]

    # sns.plt.ylim(limits[1])
    # plt.legend()

    return graph.ax_joint


def create_sample_file(sample_data, file_name):

    fd = h5py.File(file_name, 'w')
    dset = fd.create_dataset('overall_post', sample_data.shape)
    dset.attrs['right_ascension'] = sample_data[:, 0] - np.pi / 2
    dset.attrs['declination'] = sample_data[:, 1] + np.pi
    dset.attrs['frequency'] = sample_data[:, 2]

    fd.close()


def load_volume_map(input_name, nside=None, interpolate='nearest', contour=[90], simplify=True):
    """

    Load volume posterior map from HEALPix probability map

    Parameters
    ----------
    input_name

    Returns
    -------

    """

    return 0


def compute_log_evidence(logpvals, beta_ladder, deg=5):
    """

    Parameters
    ----------
    logpvals : numpy array
        ntemps x nsamples matrix containing all the samples at different temperatures.
        from temperature T=1 to temperature T=inf  (beta = 1 to beta=0)
    beta_ladder : numpy array
        inverse temperature ladder


    Returns
    -------
    loge: scalar float
        logarithm of the evidence
    loge_std : scalar float
        estimated standard deviation of the log-evidence


    References
    ----------
    Lartillot, Nicolas and Philippe, Herve, Computing Bayes Factors Using Thermodynamic Integration, 2006


    """

    # Sample average of log(p|theta)
    eu = np.mean(logpvals, axis=1)
    # Number of samples
    K = logpvals.shape[1]
    # Sample variance of log(p|theta)
    vu = np.var(logpvals, axis=1)/K

    # Integral over beta
    beta_max = np.max(beta_ladder)
    beta_min = np.min(beta_ladder)
    ntemps = len(beta_ladder)
    # loge = (beta_max - beta_min) / ntemps * np.sum(eu)
    loge_std = (beta_max - beta_min) / ntemps * np.sqrt(np.sum(vu))
    # loge_var = - (vu[0] + vu[ntemps-1])/(4*K**2) + (eu[0] - eu[ntemps-1])/K

    # Can use a cubic spline fit
    # spl = UnivariateSpline(beta_ladder[::-1], eu[::-1])
    # Convert to integral
    # loge = spl.integral(0, 1)
    loge = simps(eu[::-1], beta_ladder[::-1])

    return loge, loge_std #np.sqrt(loge_var)





def load_chain(chain_names, data_name="chains/chain/"):
    N_chains = len(chain_names)

    # Initialization if first chain
    fd5 = h5py.File(chain_names[0], 'r')
    chaindata = fd5[data_name]
    chain = chaindata[:]
    fd5.close()

    if N_chains > 1:
        for i in range(N_chains):
            fd5 = h5py.File(chain_names[i], 'r')
            chaindata = fd5[data_name]
            chain = np.hstack((chain, chaindata))
            fd5.close()

    return chain


def load_data(hdf5_name):

    LH = LISAhdf5(hdf5_name)
    dTDI = LH.getPreProcessTDI()
    GWs = LH.getSourcesName()
    if len(GWs) == 1:
        p = LH.getSourceParameters(GWs[0])
        p.display()
    else:
        p = [LH.getSourceParameters(GW) for GW in GWs]

    return dTDI, p


def loadtdi(hdf5_name):
    dTDI, p = load_data(hdf5_name)
    if type(p) == list:
        truths = []
        ts = p[0].get("Cadence")
        Tobs = p[0].get("ObservationDuration")
        for p0 in p:
            p0.display()
            truths.append([p0.get("EclipticLatitude") + np.pi / 2,
                           p0.get("EclipticLongitude") - np.pi,
                           p0.get('Frequency')])
        truths = np.concatenate(truths)

        labels = np.array([r'$\hat{\theta}_1$ [rad]', r'$\hat{\phi}_1$ [rad]', r'$\hat{f}_{1}-f_1$ [nHz]',
                           r'$\hat{\theta}_2$ [rad]', r'$\hat{\phi}_2$ [rad]', r'$\hat{f}_{2}-f_2$ [nHz]'])

    else:
        p.display()
        ts = p.get("Cadence")
        Tobs = p.get("ObservationDuration")
        truths = np.array([p.get("EclipticLatitude") + np.pi / 2,
                           p.get("EclipticLongitude") - np.pi,
                           p.get('Frequency')])

        labels = np.array([r'$\hat{\theta}$ [rad]', r'$\hat{\phi}$ [rad]', r'$\hat{f}_{0}-f_{0}$ [nHz]'])

    return dTDI, p, truths, labels, ts, Tobs


def load_mcmc_results(base, filepaths, filenames, noisenames, prefixes, 
                      maskname, conf, signal_name = "chains/chain1/"):

    # ==========================================================================
    # Prefixes for figure saving
    # ==========================================================================

    if '1e-4Hz' in base:
        f0 = '1e-4Hz'
    elif ('2e-4Hz' in base) | ('2sources' in base):
        f0 = '2e-4Hz'
    elif '5e-4Hz' in base:
        f0 = '5e-4Hz'
    if maskname == 'periodic':
        gaps = 'antgaps'
    elif maskname == 'random':
        gaps = 'randgaps'
    if '2sources' in base:
        if ('df=1e-7Hz' in base) | ('df1e-7Hz' in base):
            prefix = '2sources_df=1e-7Hz_'
        if ('df=1e-8Hz' in base) | ('df1e-8Hz' in base):
            prefix = '2sources_df=1e-8Hz_'
        elif ('df=1e-9Hz' in base) | ('df1e-9Hz' in base):
            prefix = '2sources_df=1e-9Hz_'
        elif ('df=1e-10Hz' in base) | ('df1e-10Hz' in base):
            prefix = '2sources_df=1e-10Hz_'
            # hdf5_name = '/Users/qbaghi/Codes/data/simulations/' + '2019-03-18_14h33-04_monof0=2e-4Hz_2sources_df=1e-10_1year_ts=10s_mynoise.hdf5'
        elif ('df=1e-6Hz' in base) | ('df1e-6Hz' in base):
            prefix = '2sources_df=1e-6Hz_'
            # hdf5_name = '/Users/qbaghi/Codes/data/simulations/' + '2019-03-18_10h22-46_monof0=2e-4Hz_2sources_df=1e-6_1year_ts=10s_mynoise.hdf5'
        else:
            prefix = '2sources_df=1e-7Hz_'
    else:
        prefix = ''
    if 'single_model' in base:
        sufix = '_single_model'
    else:
        sufix = ''


    # ==========================================================================
    # Load MCMC samples
    # ==========================================================================
    # Dimension of parameter space
    ndim = 3
    if 'double_model' in base:
        ndim = 6

    # Number of last steps to retain for the analysis
    nsteps = 1000
    nwalkers = 12
    #nsteps = 15000
    #nsteps = 500
    # nburn = 1000
    # Apply extra thinning
    # thin = 10
    thin = 1


    sample_list = []
    logp_list = []
    logl_list = []
    beta_list = []
    bayes_list = []
    bayes_std_list = []
    # load PSD mcmc samples
    # For noise parameters
    logSinds_list = []
    fs = 0.1
    psd = psdsampler.PSDSampler(2 ** 22, fs, n_knots=30, d=3, fmin=None, fmax=None)
    f = np.fft.fftfreq(psd.N)*fs
    f1 = conf.f0 - conf.B / 2
    f2 = conf.f0 + conf.B / 2
    finds = f[(f >= f1) & (f <= f2)]

    def calculate_logpsd(logSc, psd_cls):
        psd_cls.update_psd_func(logSc)
        return np.log(psd_cls.calculate(finds)*conf.scale**2)

    print("Length of prefixes is " + str(len(prefixes)))

    for j in range(len(prefixes)):


        # For GW parameters
        chain = load_chain([filepaths[j] + filenames[j]], data_name=signal_name)
        nburn = np.int(chain.shape[2] - nsteps)

        print("Chain shape for j=" + str(j) + " is " + str(chain.shape))

        if len(chain.shape) == 4:
            # Size ntemps x nwalkers x nstep x ndim
            samples = chain[0, :nwalkers, nburn::thin, 0:ndim].reshape((-1, ndim))

            if chain.shape[3] > ndim :
                ll = chain[:, :nwalkers, nburn::thin, ndim]
                loglike = ll.reshape((ll.shape[0], np.int(ll.shape[1]*ll.shape[2])))
                # size ntemps x nwalkers x nsteps x 2
                logp = chain[0, :nwalkers, nburn::thin, ndim:].reshape((-1, 2))
            elif chain.shape[3] == ndim:
                ll = load_chain([filepaths[j] + filenames[j]], data_name="probas/logl/")
                loglike = ll.reshape((ll.shape[0], np.int(ll.shape[1] * ll.shape[2])))
                # lp = load_chain([filepaths[j] + filenames[j]], data_name="probas/logp/")
                logp = ll[0, :, nburn::thin].reshape((-1, 1))

        elif len(chain.shape) == 2:
            samples = chain[nburn::thin, 0:ndim]
            logp = chain[nburn::thin, ndim:]

        else:
            print("Unknown chain shape for case j="+str(j))

        # Rescale frequency values to same dimension if necessary
        pre_scale = np.ones(samples.shape[1])
        mus = np.mean(samples, axis=0)

        if mus[2] > 1e-3:
            pre_scale[2] = 1e-3
        if samples.shape[1] == 6:
            if mus[5] > 1e-3:
                pre_scale[5] = 1e-3


        sample_list.append(samples*pre_scale)
        logp_list.append(logp)
        logl_list.append(loglike)


        # # FOR PSD
        # psd_chain = load_chain([filepaths[j] + noisenames[j]], data_name="psd/samples")
        # ndim_psd = psd_chain.shape[1]
        #
        # # Create a matrix with the same size as the GW parameter sample. size nwalkers x nsteps x ndim
        # # psd_samples = np.zeros((chain.shape[1], chain.shape[2], ndim_psd))
        # # Covariance constant term in the likelihood
        # psd_constants = np.zeros((chain.shape[2]))
        # # Fill it
        # for k in range(np.int(chain.shape[2]/conf.nupdate)):
        #     # this is a nupdate x ndim matrix
        #     # psd_samples[q, k*conf.nupdate:(k+1)*conf.nupdate, :] = np.tile(psd_chain[[k], :].T, conf.nupdate).T
        #     psd_constants[k * conf.nupdate:(k + 1) * conf.nupdate] = - 0.5*np.sum(calculate_logpsd(psd_chain[k, :], psd))
        # # Throw away burn-in
        # consts = psd_constants[nburn::thin]

        # Resize it as the GW parameter samples
        # psd_sample_list.append(psd_samples[:, nburn::thin, :].reshape((-1, ndim_psd)))

        fd5 = h5py.File(filepaths[j] + filenames[j], 'r')

        try:
            beta_hist = fd5["temperatures/beta_hist"][()]
            fd5.close()
            beta_list.append(beta_hist[:, 0])

            # loge, loge_std = compute_log_evidence(loglike, beta_hist[:, 0])
            # loge, loge_std = log_evidence_estimate(ll, beta_hist[:, 0], fburnin=0)
            # Mean log-likelihood at all temperatures ntemps x nwalkers x nsteps

            # # Average over all walkers to get a (ntemps x nsteps) matrix
            # m1 = np.mean(ll, axis=1)
            # mean_logls = np.zeros((m1.shape[0]))
            # # For each temperature r
            # for r in range(m1.shape[0]):
            #     # average over all steps, adding the covariance term to the log-likelihood
            #     mean_logls[r] = np.mean(m1[r, :] + consts)

            mean_logls = np.mean(np.mean(ll, axis=2), axis=1)
            betas = beta_hist[:, 0]
            loge, loge_std = util.thermodynamic_integration_log_evidence(betas, mean_logls)
            bayes_list.append(loge)
            bayes_std_list.append(loge_std)

        except KeyError:
            print("No temperature ladder history found")

            fd5.close()


    return sample_list, logp_list, logl_list, bayes_list, bayes_std_list, f0, gaps, prefix, sufix



if __name__ == '__main__':

    import os
    import tdi


    # FTT modules
    import pyfftw
    import fftwisdom
    import myplots
    import seaborn as sns

    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import fft, ifft

    # ==========================================================================
    # Specify file paths
    # ==========================================================================

    # # For antenna gaps and f0 = 1e-4 Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/1e-4Hz/"
    # filepaths = [base, base, base]
    # #    #2019-02-12_18h20-32 #2019-02-25_19h32-49 2019-02-26_13h57-54
    # # prefixes = ["2019-02-27_16h50-41","2019-02-26_13h58-04","2019-02-27_16h50-40"]
    # prefixes = ["2019-03-04_23h27-50", "2019-03-04_23h38-27", "2019-03-05_13h18-54"]
    # maskname = 'periodic'

    # # For random gaps and f0 = 1e-4 Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/1e-4Hz/"
    # #    filepaths = [base + "complete_1e-4/",base + "randgaps_1e-4/",
    # #                 base + "randgaps_1e-4_imp/"]
    # filepaths = [base, base, base]
    # #    prefixes = ["2019-02-15_17h00-34","2019-02-17_16h27-29","2019-02-15_17h45-22"]
    # #    prefixes = ["2019-02-25_19h32-49","2019-02-26_10h59-46","2019-02-26_11h39-42"]
    # #    prefixes = ["2019-02-27_16h50-41","2019-02-27_17h03-46","2019-02-27_17h25-49"]
    # prefixes = ["2019-03-04_23h27-50", "2019-03-04_23h01-13", "2019-03-05_13h28-01"]
    # maskname = 'random'

    # # For random gaps and f0 = 1e-4 Hz WITH FILTERING
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/1e-4Hz/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-08_16h23-02"]
    # maskname = 'random'

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









    # # For 2 sources and randgaps and df = 1e-6Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-6Hz/double_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-18_16h32-18", "2019-03-18_16h34-17", "2019-03-18_17h37-41"]
    # prefixes = ["2019-03-19_02h12-38", "2019-03-19_02h13-25", "2019-03-19_02h26-55"]
    # maskname = 'random'

    # # For 2 sources and randgaps and df = 1e-6Hz with SINGLE MODEL
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-6Hz/single_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-19_00h27-38", "2019-03-19_00h40-19", "2019-03-19_01h10-08"]
    # maskname = 'random'



    # For 2 sources and randgaps and df = 1e-7Hz
    #base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/df1e-7Hz/double_model/"
    # prefixes = ["2019-03-14_14h15-47", "2019-03-14_14h09-48", "2019-03-14_14h38-30"]
    # prefixes = ["2019-03-19_17h24-21", "2019-03-19_17h35-36", "2019-03-19_18h06-19"]
    # # With 100 temperatures
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_100T/df1e-7Hz/double_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-04-18_15h41-16", "2019-04-18_15h39-53", "2019-04-18_15h46-34"]
    # With more frequent imputation
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_frequent_imputation/df1e-7Hz/double_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-19_17h24-21", "2019-03-19_17h35-36", "2019-05-06_18h45-34"]
    # maskname = 'random'


    # For 2 sources and random gaps and df = 1e-7Hz with SINGLE-SOURCE MODEL
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-7Hz/single_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/df1e-7Hz/single_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_100T/df1e-7Hz/single_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-12_13h34-37", "2019-03-12_13h25-45", "2019-03-12_13h51-06"]
    # prefixes = ["2019-03-14_01h22-11", "2019-03-14_01h52-57", "2019-03-14_02h15-31"]
    # prefixes = ["2019-03-14_01h22-11", "2019-03-14_01h52-57", "2019-03-14_02h15-31"]
    # prefixes = ["2019-03-19_15h33-05", "2019-03-19_15h30-59", "2019-03-19_15h49-14"]
    # With 100 temperatures
    # prefixes = ["2019-04-18_13h34-47", "2019-04-18_13h31-31", "2019-04-18_13h37-48"]
    # With more frequent imputation
    base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_frequent_imputation/df1e-7Hz/single_model/"
    filepaths = [base, base, base]
    prefixes = ["2019-03-19_15h33-05", "2019-03-19_15h30-59", "2019-05-06_16h24-04"]
    maskname = 'random'





    # # For 2 sources and randgaps and df = 1e-8Hz
    # # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-8Hz/double_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/df1e-8Hz/double_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-14_14h24-29", "2019-03-14_14h15-35", "2019-03-14_14h36-25"]
    # prefixes = ["2019-03-19_17h58-34", "2019-03-19_17h39-16", "2019-03-19_18h09-06"]
    # maskname = 'random'

    # # For 2 sources and randgaps and df = 1e-8Hz and SINGLE  MODEL
    # #base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-8Hz/single_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/df1e-8Hz/single_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-14_14h24-29", "2019-03-14_14h15-35", "2019-03-14_14h36-25"]
    # # prefixes = ["2019-03-14_21h07-19", "2019-03-14_21h00-18", "2019-03-14_21h21-45"]
    # prefixes = ["2019-03-19_15h41-45", "2019-03-19_15h30-34", "2019-03-19_15h54-12"]
    # maskname = 'random'




    # # For 2 sources and randgaps and df = 1e-9Hz
    # # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-9Hz/double_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/df1e-9Hz/double_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-14_22h29-38", "2019-03-14_22h18-57", "2019-03-14_22h45-12"]
    # prefixes = ["2019-03-19_17h51-26", "2019-03-19_17h45-19", "2019-03-19_18h05-29"]
    # maskname = 'random'

    # # For 2 sources and randgaps and df = 1e-9Hz with SINGLE MODEL
    # # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-9Hz/single_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/df1e-9Hz/single_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-14_20h19-17", "2019-03-14_20h15-51", "2019-03-14_20h40-10"]
    # prefixes = ["2019-03-19_15h45-34", "2019-03-19_15h35-53", "2019-03-19_15h53-26"]
    # maskname = 'random'



    # # For 2 sources and randgaps and df = 1e-10Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/df1e-10Hz/double_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-18_21h13-30", "2019-03-18_21h28-34", "2019-03-18_22h21-52"]
    # prefixes = ["2019-03-19_17h52-58", "2019-03-19_17h53-06", "2019-03-19_18h09-38"]
    # maskname = 'random'

    # # For 2 sources and randgaps and df = 1e-10Hz with SINGLE MODEL
    # # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-10Hz/single_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/df1e-10Hz/single_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-18_19h51-43", "2019-03-18_19h45-04", "2019-03-18_19h22-25"]
    # prefixes = ["2019-03-19_15h41-44", "2019-03-19_19h00-21", "2019-03-19_19h24-37"]
    # maskname = 'random'



    # ANTENNA GAPS

    # # For 2 sources and antenna gaps and df = 1e-7Hz
    # # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-7Hz/double_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/antenna_gaps/df1e-7Hz/double_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-14_14h15-47", "2019-03-14_21h59-30", "2019-03-14_22h14-59"]
    # prefixes = ["2019-03-19_17h24-21", "2019-03-20_17h10-18", "2019-03-20_17h45-36"]
    # maskname = 'periodic'

    # # # For 2 sources and antenna gaps and df = 1e-7Hz with SINGLE-SOURCE MODEL
    # # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources/df=1e-7Hz/single_model/"
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/antenna_gaps/df1e-7Hz/single_model/"
    # filepaths = [base, base, base]
    # # prefixes = ["2019-03-12_13h34-37", "2019-03-12_17h26-02", "2019-03-12_17h37-23"]
    # # prefixes = ["2019-03-14_01h22-11", "2019-03-15_14h45-01", "2019-03-15_15h49-58"]
    # prefixes = ["2019-03-19_15h33-05", "2019-03-20_14h26-06", "2019-03-20_14h45-21"]
    # maskname = 'periodic'

    # # For 2 sources and antenna gaps and df = 1e-8Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/antenna_gaps/df1e-8Hz/double_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-19_17h58-34", "2019-03-20_16h59-22", "2019-03-20_17h04-10"]
    # maskname = 'periodic'

    # # # For 2 sources and antenna gaps and df = 1e-8Hz with SINGLE-SOURCE MODEL
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/antenna_gaps/df1e-8Hz/single_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-19_15h41-45", "2019-03-20_15h17-57", "2019-03-20_15h28-20"]
    # maskname = 'periodic'

    # # For 2 sources and antenna gaps and df = 1e-9Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/antenna_gaps/df1e-9Hz/double_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-19_17h51-26", "2019-03-20_16h50-16", "2019-03-20_18h01-56"]
    # maskname = 'periodic'

    # # # For 2 sources and antenna gaps and df = 1e-9Hz with SINGLE-SOURCE MODEL
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/antenna_gaps/df1e-9Hz/single_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-19_15h45-34", "2019-03-20_16h02-34", "2019-03-20_16h29-59"]
    # maskname = 'periodic'

    # # For 2 sources and antenna gaps and df = 1e-10Hz
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/antenna_gaps/df1e-10Hz/double_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-19_17h52-58", "2019-03-20_18h36-45", "2019-03-20_18h44-31"]
    # maskname = 'periodic'

    # # For 2 sources and antenna gaps and df = 1e-10Hz with SINGLE-SOURCE MODEL
    # base = "/Users/qbaghi/Codes/data/results_ptemcee/discover/results_2sources_same_amp/antenna_gaps/df1e-10Hz/single_model/"
    # filepaths = [base, base, base]
    # prefixes = ["2019-03-19_15h41-44", "2019-03-20_17h16-28", "2019-03-20_17h26-47"]
    # maskname = 'periodic'


    # ==========================================================================
    # Load configuration file
    # ==========================================================================

    filenames = [pre + '_chain_full2.hdf5' for pre in prefixes]
    noisenames = [pre + '__psd.hdf5' for pre in prefixes]

    # signal_name = "chains/chain/"
    # signal_name = "chain"


    # config_file = '/Users/qbaghi/Codes/python/inference/configs/wgmcmc_config_lowSNR_mono_2sources.txt'
    # config_file = '/Users/qbaghi/Codes/python/inference/configs/ptemcee_config_2e-4Hz_gaps.txt'
    config_file = filepaths[0] + prefixes[0] + '_config.txt'


    # ==========================================================================
    # Load TDI data
    # ==========================================================================
    conf = load_mcmc_config.loadconfig(config_file)
    hdf5_name = '/Users/qbaghi/Codes/data/simulations/' + os.path.basename(conf.hdf5_name)
    dTDI, p, truths, labels, ts, Tobs = loadtdi(hdf5_name)

    # ==========================================================================
    # Load MCMC results
    # ==========================================================================
    sample_list, logp_list, logl_list, bayes_list, bayes_std_list, f0, gaps, prefix, sufix = \
        load_mcmc_results(base, filepaths, filenames, noisenames, prefixes, maskname, conf)
    ndim = sample_list[0].shape[1]

    # Number of standard deviation fold to plot
    k_sig = 5

    # To center the posterior distribution on zero
    frequency_offset = True







    # =========================================================================
    # PLot data
    # =========================================================================
    # import seaborn as sns
    # sns.set(style="white", palette="muted", color_codes=True)
    mpl.rcdefaults()
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.minor.width'] = 1
    # mpl.rcParams['lines.markeredgewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1
    # mpl.rcParams['legend.handletextpad'] = 0.3
    mpl.rcParams['legend.fontsize'] = 14
    # # mpl.rcParams['figure.figsize'] = 8, 6
    # # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['figure.autolayout'] = True

    # mpl.rcParams['axes.formatter.useoffset'] = False
    # mpl.rcParams['axes.formatter.limits'] = [0, 0]
    #    mpl.rcParams['axes.formatter.min_exponent'] = 0,
    #    mpl.rcParams['axes.formatter.offset_threshold'] = 4,
    #    mpl.rcParams['axes.formatter.use_locale'] = False,
    #    mpl.rcParams['axes.formatter.use_mathtext'] = False,
    #    mpl.rcParams['axes.formatter.useoffset'] = False,

    if len(prefixes) > 1:
        j0 = 0
    else:
        j0 = 0

    if ndim == 3:
        nsig = k_sig * np.ones(ndim)
        # nsig = 15 * np.ones(ndim)
        #rscales = [1.0, 1.0, 1.0e3]
        rscales = [1.0, 1.0, 1.0e9]
        offset = np.zeros(ndim)

        if frequency_offset:
            offset[2] = truths[2]

    elif ndim == 6:
        # nsig = [0.5,0.5,0.5,1,1,0.1]
        # nsig = [3,3,3,1.5,1.5,1.5]
        # nsig = [4, 4, 4, 4, 4, 4]
        nsig = k_sig * np.ones(ndim)
        rscales = [1.0, 1.0, 1.0e9, 1.0, 1.0, 1.0e9]
        offset = np.zeros(ndim)

        if frequency_offset:
            offset[2] = truths[2]
            offset[5] = truths[5]

    # MAP estimate
    if logp_list[j0].shape[1] == 2:
        imap = np.where(logp_list[j0][:, 1] == np.max(logp_list[j0][:, 1]))[0][0]
    elif logp_list[j0].shape[1] == 1:
        imap = np.where(logp_list[j0][:, 0] == np.max(logp_list[j0][:, 0]))[0][0]

    sigmas = [np.std(sample_list[j0][:, i]) for i in range(ndim)]
    # limits = [(truths[i]-nsig*sigmas[i],truths[i]+nsig*sigmas[i]) for i in range(ndim)]
    limits = [(sample_list[j0][imap, i] - nsig[i] * sigmas[i],
               sample_list[j0][imap, i] + nsig[i] * sigmas[i]) for i in range(ndim)]

    limits = [((limits[i][0]-offset[i])*rscales[i], (limits[i][1]-offset[i])*rscales[i]) for i in range(len(limits))]

    # limits = None
    colors = ['black', 'gray', 'blue']
    cases = ['Complete data', 'Gaps, window method', 'Gaps, DA method']
    caselabels = ['complete', 'window', 'da']


    # ==================================================================================================================
    # SKYMAP generation
    # ==================================================================================================================
    # t1 = time.time()
    # i0 = 2
    # write_healpix_map(sample_list[i0][:, 0:2], outdir="/Users/qbaghi/Codes/data/results_ptemcee/discover/"+f0+"/skylocmaps/",
    #                   enable_multiresolution=False, fitsoutname='skymap_'+caselabels[i0]+'_'+maskname+'.fits.gz',
    #                   nest=True)
    # t2 = time.time()
    # print("Skymap construction took " + str((t2-t1)/60.) + " minutes.")
    # ==================================================================================================================

    # load_healpix_map("/Users/qbaghi/Codes/data/results_ptemcee/posteriors/",
    #                  fitsoutname='skymap.fits.gz',
    #                  annotate=False,
    #                  radecs=[(truths[1] + np.pi)*180/np.pi, (truths[0] - np.pi/2)*180/np.pi],
    #                  contour=[90], inset=True)

    #mpl.rcParams['text.usetex'] = True
    # load_healpix_map("/Users/qbaghi/Codes/data/results_ptemcee/posteriors/", fitsoutname='skymap_multires.fits.gz',
    #                  radecs=[],
    #                  contour=[90], annotate=False)

    # create_sample_file(sample_list[0], "/Users/qbaghi/Codes/data/results_ptemcee/posteriors/SAMPLES.hdf5")


    # # ==================================================================================================================
    # # 2D Joint posterior plot using seaborn
    # # ==================================================================================================================
    #
    # # If we make a subselection of parameters
    # degscales = 180 / np.pi
    # subinds = np.array([0, 1]).astype(int)
    #
    # # sub_sample_list = []
    # # for i in range(len(sample_list)):
    # #     # Keep only parameters within limits
    # #     inds_list = [np.where(limits[p][0] <= sample_list[i][:, p] <= limits[p][1]) for p in subinds]
    # #
    # #     sub_sample = sample_list[i][:, subinds]
    # #     sub_sample_list.append(sub_sample)
    # # limits=[limits[sub] for sub in subinds]
    # sub_sample_list = [sample[:, subinds]*degscales for sample in sample_list]
    #
    # truths_vect = truths[subinds] * degscales
    # deglimits = [(limits[sub][0]*degscales, limits[sub][1]*degscales) for sub in subinds]
    # deglimits = [(105, 130), (45, 70)]
    # # deglimits = None
    #
    # axes = jointplot(sub_sample_list, truths_vect, kind="kde", colors=['black', 'gray', 'blue'], limits=deglimits, shade=False,
    #                  labels=[r"$\theta$ [deg]", r"$\phi$ [deg]"], legend_labels=cases, fontsize=20, linewidth=3,
    #                  levels=4)
    #
    #
    # # plt.savefig('/Users/qbaghi/Documents/articles/papers/papers/gaps/figures/mcmc/skylocs_' + f0 + '_' + gaps + '.pdf')
    #
    # plt.show()


    # ==================================================================================================================
    # Corner plot using corner.py
    # ==================================================================================================================

    fig, axes = cornerplot(sample_list, truths, offset, rscales, labels, colors=colors, limits=limits,
                           fontsize=16, bins=50, truth_color='cadetblue', figsize=(8, 7.5), linewidth=1.5)

    # plt.savefig('/Users/qbaghi/Documents/studies/gaps/mcmc_superimposed/' + prefix + f0 + '_' + gaps + sufix + '.pdf')
    #
    # plt.savefig('/Users/qbaghi/Documents/studies/gaps/mcmc_superimposed/' + prefix + 'frequency_' + f0 + '_'
    #             + gaps + sufix + '.pdf')



    # =========================================================================
    # Grid of corner plots using sns
    # =========================================================================
    #    #,figsize=(10, 8))#,sharex=True)
    #    import seaborn as sns
    #    import pandas
    #    #mpl.use('PS')
    #    mpl.rcParams['text.usetex']=True
    #    mpl.rcParams['text.latex.unicode']=True
    #    #plt.switch_backend('PS')
    #
    #    sns.set(font_scale=1.5, rc={'text.usetex' : True},
    #            style="white", palette="muted", color_codes=True)
    #    sns.set_style("ticks")
    ##    df_complete = pandas.DataFrame(data=sample_list[0],
    ##                                   columns = [r"theta",r"phi",r"f"])
    #    Ns = np.int(2**15)
    #    df_complete = pandas.DataFrame(data={r"\theta":sample_list[0][0:Ns,0],
    #                                         r"\phi":sample_list[0][0:Ns,1],
    #                                         r"f_0":sample_list[0][0:Ns,2] })
    ##    df_antgaps = pandas.DataFrame(data={"theta":sample_list2[1][0:,0],
    ##                                         "phi":sample_list2[1][0:Ns,1],
    ##                                         "f":sample_list2[1][0:Ns,2] })
    #
    #    f, axes = plt.subplots(1, 2)
    #    #fig,axes = plt.subplots(1,1)
    #    g = sns.PairGrid(df_complete,despine = True)
    #    g.map_diag(plt.hist, histtype="step",bins = 50, linewidth=1)#sns.kdeplot)
    #    #g.map_offdiag(sns.kdeplot, n_levels=6)
    #    g.map_upper(plt.scatter, s = 5)
    #    g.map_lower(sns.kdeplot)
    #    # plt.show()
    ##    g2 = sns.PairGrid(df_complete,despine = True, ax=axes[1])
    ##    g2.map_diag(plt.hist, histtype="step",bins = 50, linewidth=1)#sns.kdeplot)
    ##    #g.map_offdiag(sns.kdeplot, n_levels=6)
    ##    g2.map_upper(plt.scatter, s = 5)
    ##    g2.map_lower(sns.kdeplot)

    # # =========================================================================
    # # Single histograms
    # # =========================================================================
    #
    # truths_res = (truths[0:sample_list[0].shape[1]] - offset) * rscales
    # cases = ['Complete data', 'Gaps, window method', 'Gaps, DA method']
    # linestyles = ['solid', 'dotted', 'dashed']
    #
    # #f, axes = plt.subplots(2, 3,figsize=(9, 6.5))#,sharex=True)
    # # f, axes = plt.subplots(1, 3, figsize=(9, 3))  # ,sharex=True)
    # f, axes = plt.subplots(1, 1, figsize=(5, 5))
    # #sns.set(style="white", palette="muted", color_codes=True)
    # hist = False
    # #alphas = [0.3, 0.3, 0.3]
    # alphas = [1, 1, 1]
    # zorder = [0, 2, 1]
    # hist_kws = [{"histtype": "step", "linewidth": 2, "alpha":  alphas[i]} for i in range(len(alphas))]
    # kde = True
    # #kde_kws = [{"shade": True, "linewidth": 3, "linestyle": linestyles[i], "alpha": alphas[i]} for i in range(len(linestyles))]
    # kde_kws = [{"shade": True, "linewidth": 3, "linestyle": linestyles[i]} for i in range(len(linestyles))]
    # bins = None
    # norm_hist = True
    #
    # for i in zorder:
    # #
    # #     # Theta
    # #     sns.distplot((sample_list[i][:, 0]-offset[0])*rscales[0], hist=hist, kde=kde, color=colors[i],
    # #                  kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist, ax=axes[0])
    # #
    # #     # Phi
    # #     sns.distplot((sample_list[i][:, 1]-offset[1])*rscales[1], hist=hist, kde=kde, color=colors[i],
    # #                  kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist, ax=axes[1])
    # #
    # #     # f_0
    # #     sns.distplot((sample_list[i][:, 2]-offset[2])*rscales[2], hist=hist, kde=kde, color=colors[i],
    # #                  kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes[2])
    #
    #     # f_0
    #     sns.distplot((sample_list[i][:, 2]-offset[2])*rscales[2], hist=hist, kde=kde, color=colors[i],
    #                  kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes)
    #
    #
    #
    #     # # For antenna gaps
    #     # sns.distplot(sample_list2[i][:,0], hist=hist, kde=kde, color=colors[i],
    #     #             kde_kws={"shade": shade},hist_kws = hist_kws, bins = bins,
    #     #             norm_hist = norm_hist, ax=axes[0,0])
    #     # sns.distplot(sample_list2[i][:,1], hist=hist, kde=kde, color=colors[i],
    #     #             kde_kws={"shade": shade},hist_kws = hist_kws, bins = bins,
    #     #             norm_hist = norm_hist, ax=axes[0,1])
    #     # sns.distplot(sample_list2[i][:,2], hist=hist, kde=kde, color=colors[i],
    #     #             kde_kws={"shade": shade},hist_kws = hist_kws, bins = bins,
    #     #             norm_hist = norm_hist,label = cases[i], ax=axes[0,2])
    # #
    # #
    # #        # For random gaps
    # #        sns.distplot(sample_list[i][:,0], hist=hist, kde=kde, color=colors[i],
    # #                     kde_kws={"shade": shade},hist_kws = hist_kws, bins = bins,
    # #                     norm_hist = norm_hist, ax=axes[1,0])
    # #        sns.distplot(sample_list[i][:,1], hist=hist, kde=kde, color=colors[i],
    # #                     kde_kws={"shade": shade},hist_kws = hist_kws, bins = bins,
    # #                     norm_hist = norm_hist, ax=axes[1,1])
    # #        sns.distplot(sample_list[i][:,2], hist=hist, kde=kde, color=colors[i],
    # #                     kde_kws={"shade": shade},  bins = bins,
    # #                     norm_hist = norm_hist, hist_kws = hist_kws,
    # #                     ax=axes[1,2])
    # #
    # #
    # #
    # #
    # #
    # #    sns.distplot(sample_list[1][:,2], hist=hist, kde=kde, color="gray",
    # #                  kde_kws={"shade": shade}, label = 'Gapped data, window method',
    # #                  ax=axes[0, 2])
    # #    sns.distplot(sample_list[2][:,2], hist=hist, kde=kde, color="blue",
    # #                  kde_kws={"shade": shade}, label = 'Gapped data, DA method',
    # #                  ax=axes[0, 2])
    #    #plt.xlabel(r'Source frequency [Hz]',fontsize=16)
    #    #plt.xlabel(r'Arbitrary units',fontsize=16)
    #    # axes[0,0].set_ylabel('Gap pattern a_mat',fontsize=18)
    #    # axes[1,0].set_ylabel('Gap pattern B',fontsize=18)
    #    # [axes[0,i].set_xlabel(labels[i],fontsize=18) for i in range(len(labels))]
    #    # [axes[1,i].set_xlabel(labels[i],fontsize=18) for i in range(len(labels))]
    #    # [axes[1,i].set_xlim(limits[i]) for i in range(len(limits))]
    #    # [axes[0,i].set_xlim(limits[i]) for i in range(len(limits))]
    #
    # # axes[2].legend(axes[2].lines, cases, loc='lower center', frameon=False, bbox_to_anchor=(0.5, -0.25))
    #
    #
    # axes[0].set_ylabel('Gap pattern a_mat', fontsize=18)
    # [axes[i].set_xlabel(labels[i], fontsize=18) for i in range(len(labels))]
    # [axes[i].set_xlim(limits[i]) for i in range(len(limits))]
    # #[axes[i].axvline(truths_res[i], color='cadetblue', linewidth=1.5) for i in range(len(limits))]
    # [axes[i].axvline(truths_res[i], color='green', linewidth=1.5) for i in range(len(limits))]

    #    [axes[1,i].set_ylim(limits[i]) for i in range(len(limits))]
    #    [axes[0,i].set_ylim(limits[i]) for i in range(len(limits))]
       #[axes[i].set_xticks(fontsize=14) for i in range(len(labels))]
       #plt.yticks(fontsize=14)
    #axes[0].legend(loc=4, fontsize=10)
       #plt.tight_layout()
       #plt.axvline(x=truths[2], color = 'green',linewidth = 2, linestyle = 'dashed')#, linestyle=(0, (1, 3)))

    # axes.set_xlabel(labels[2], fontsize=20)
    # # axes.set_xlim(limits[2])
    # axes.set_xlim([-1.5, 1.5])
    # axes.axvline(truths_res[2], color='green', linewidth=1.5)
    #
    # plt.setp(axes, yticks=[])
    # plt.tight_layout()
    # # plt.savefig('/Users/qbaghi/Documents/studies/gaps/mcmc_superimposed/histograms_'+f0+'_'+gaps+'.pdf')
    # plt.savefig('/Users/qbaghi/Documents/studies/gaps/mcmc_superimposed/frequency_histogram_' + f0 + '_' + gaps + '.pdf')
    # plt.show()



    # # =========================================================================
    # # Single histograms FOR 2 SOURCES
    # # =========================================================================
    #
    # truths_res = (truths[0:sample_list[0].shape[1]] - offset) * rscales
    # # cases = ['Complete data', 'Gaps, window method', 'Gaps, DA method']
    # cases = [None, None, None]
    # linestyles = ['solid', 'dotted', 'dashed']
    #
    # #f, axes = plt.subplots(2, 3,figsize=(9, 6.5))#,sharex=True)
    # # f, axes = plt.subplots(1, 3, figsize=(9, 3))  # ,sharex=True)
    #
    # #sns.set(style="white", palette="muted", color_codes=True)
    # hist = False
    # #alphas = [0.3, 0.3, 0.3]
    # alphas = [1, 1, 1]
    # zorder = [0, 2, 1]
    # hist_kws = [{"histtype": "step", "linewidth": 2, "alpha":  alphas[i]} for i in range(len(alphas))]
    # # hist_kws = [None for i in range(len(alphas))]
    # kde = True
    # #kde_kws = [{"shade": True, "linewidth": 3, "linestyle": linestyles[i], "alpha": alphas[i]} for i in range(len(linestyles))]
    # kde_kws = [{"shade": True, "linewidth": 3, "linestyle": linestyles[i]} for i in range(len(linestyles))]
    # bins = None
    # norm_hist = True
    #
    #
    #
    # f, axes = plt.subplots(1, 1, figsize=(6, 5))
    #
    # for i in zorder:
    #
    #     # # f_1
    #     # sns.distplot((sample_list[i][:, 2]-offset[2])*rscales[2], hist=hist, kde=kde, color=colors[i],
    #     #              kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes)
    #     # # f_2
    #     # sns.distplot((sample_list[i][:, 5]-offset[5])*rscales[5], hist=hist, kde=kde, color=colors[i],
    #     #              kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes)
    #
    #
    #     # # f_1
    #     # sns.distplot(sample_list[i][:, 2], hist=hist, kde=kde, color=colors[i],
    #     #              kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes)
    #     # # f_2
    #     # sns.distplot(sample_list[i][:, 5], hist=hist, kde=kde, color=colors[i],
    #     #              kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes)
    #
    #     # f_1 - f_0
    #     sns.distplot((sample_list[i][:, 2]-offset[2])*rscales[2], hist=hist, kde=kde, color=colors[i],
    #                  kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes,
    #                  label='_nolegend_')
    #
    #     if sample_list[i].shape[1] > 3:
    #         # f_2 - f_0
    #         sns.distplot((sample_list[i][:, 5]-offset[2])*rscales[2], hist=hist, kde=kde, color=colors[i],
    #                      kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes, label=cases[i])
    #
    #     # # f_2 - f_1
    #     # sns.distplot(sample_list[i][:, 5] - sample_list[i][:, 2], hist=hist, kde=kde, color=colors[i],
    #     #              kde_kws=kde_kws[i], hist_kws=hist_kws[i], bins=bins, norm_hist=norm_hist,  ax=axes)#, label=cases[i])
    #
    # axes.set_xlabel(r"$\hat{f} - f_1$ [nHz]", fontsize=20)
    # #axes.set_xlabel(r"$\Delta \hat{f}$ [Hz]", fontsize=20)
    # limits = [(sample_list[j0][imap, i] - nsig[i] * sigmas[i],
    #            sample_list[j0][imap, i] + nsig[i] * sigmas[i]) for i in range(ndim)]
    # # axes.set_xlim([limits[2][0], limits[5][1]])
    #
    # #
    # # axes.axvline(truths_res[2], color='green', linewidth=1.5)
    # # axes.axvline(truths_res[5], color='green', linewidth=1.5)
    # axes.axvline((truths[5]-offset[2])*rscales[2], color='green', linewidth=1.5)
    # axes.axvline((truths[2]-offset[2])*rscales[2], color='green', linewidth=1.5)
    #
    #
    # axes.minorticks_on()
    # # axes.set_xlim([limits[2][0], limits[5][1]])
    # # axes.set_xlim([(limits[2][0]-offset[2])*rscales[2], (limits[5][1]-offset[2])*rscales[2]])
    # axes.set_xlim([-6, 16])
    # #axes.set_xlim([1.99985e-4, 2.00113e-4])
    # # axes.set_xlim([limits[5][0] - limits[2][0], limits[5][1] - limits[2][1]])
    # axes.legend(loc="upper center", fontsize=20, frameon=False)
    # # axes.set_xscale('log')
    #
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.tick_params(axis='both', which='minor', labelsize=14)
    # plt.setp(axes, yticks=[])
    # plt.tight_layout()
    # # plt.savefig('/Users/qbaghi/Documents/studies/gaps/mcmc_superimposed/histograms_'+f0+'_'+gaps+'.pdf')
    # # plt.savefig('/Users/qbaghi/Documents/studies/gaps/mcmc_superimposed/'+prefix+'frequency_hist_' + f0 + '_' + gaps + '.pdf')
    # plt.show()








    # # ==========================================================================
    # # Barplots for Bayes factors
    # # ==========================================================================

    # delta_f = np.array([1e-7, 1e-8, 1e-9])
    # ind = np.arange(len(delta_f))
    # X = [ind, ind, ind]
    # # For Complete data
    # logB_complete = np.array([104.00453313, 38.62348088, 24.73178091])
    # logB_window = np.array([-0.49605142, -0.43551874, 2.64928399])
    # logB_DA = np.array([106.71811731, 40.0273956, 28.16463517])
    #
    # Y = [logB_complete, logB_window, logB_DA]
    # linewidths = [0.05, 0.05, 0.05]
    # linestyles = ['solid', 'solid', 'solid']
    # colors = ['black', 'gray', 'blue']
    # labels = ['Complete data', 'Gapped data, windowing', 'Gapped data, DA method']
    # fp = myplots.fplot(plotconf='time')
    # #fp.xscale = 'log'
    # #fp.yscale = 'linear'
    # #fp.yscale = 'log'
    # fp.xlabel = r'$\Delta f$ [Hz]'
    # fp.ylabel = r'$\log B_{21}$'
    # # fp.legendloc = 'upper right'
    # # fp.xlims = [1e-7, fs / 2]
    # # fp.ylims = [1e-24, 1e-15]
    # fig, ax1 = fp.plot(X, Y, colors, linewidths, labels, linestyles=linestyles, plot_type='bars', zorders=[3,2,1])
    # plt.xticks(ind, (r'$10^{-7}$', r'$10^{-8}$', r'$10^{-9}$'))
    # plt.draw()
    # plt.show()

    # df_bfactors = pandas.DataFrame(data={"Complete data": logB_complete,
    #                                     "Gapped data, windowing": logB_window,
    #                                     "Gapped data, DA method": logB_DA,
    #                                      "Df": [r'$10^{-7}$', r'$10^{-8}$', r'$10^{-9}$']})





    #
    # sns.barplot(x='Df', y=["Complete data", "Gapped data, windowing", "Gapped data, DA method"], data=df_bfactors)
# "Df": [r'$10^{-7}$', r'$10^{-8}$', r'$10^{-9}$'
    # df_bars = pandas.DataFrame(data=np.hstack(Y).T,
    #                            columns=['Complete data', 'Gapped data, windowing', 'Gapped data, DA method'],
    #                            index=delta_f)
