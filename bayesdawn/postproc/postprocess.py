# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
from bayesdawn.postproc import resanalysis
from scipy import stats
import configparser
import os
from ..utils import loadings, physics


def postprocess(chain, lnprob, names, par0, n_burn=500, n_thin=1, n_bins=40, k=4):
    """

    Parameters
    ----------
    chain : ndarray
        numpy array containing the posterior samples
    names : list of str
        list of parameter names
    par0 : ndarray
        full vector of true parameter values
    n_burn : int
        number of samples to discard at the beginning of the chains
    k : int
        multiple of standard deviation up to which the cornerplots are restricted

    Returns
    -------
    fig : matplotlib.pyplot.figure  instance
        figure where the cornerplot is drawn
    axes :  matplotlib.pyplot.axes instance
        axes of fig

    """

    chain_eff = chain[:, :, chain[0, 0, :, 0] != 0, :]
    print("Shape of non-zero sample array: " + str(chain_eff.shape))
    chain_eff = chain_eff[0, :, n_burn::n_thin, :]
    offset = 0
    scales = 1

    print("Shape of effective chain: " + str(chain_eff.shape))
    chain_flatten = chain_eff.reshape((-1, chain_eff.shape[2]))
    medians = np.median(chain_flatten, axis=0)
    # stds = np.std(chain_flatten, axis=0)
    stds = stats.median_absolute_deviation(chain_flatten, axis=0)
    limits = [[medians[i] - k * stds[i], medians[i] + k * stds[i]] for i in range(chain_flatten.shape[1])]
    print("Shape of flattened chain: " + str(chain_flatten.shape))
    print("Length of parameter vector: " + str(len(par0)))

    fig, axes = resanalysis.cornerplot([chain_flatten],
                                       par0, offset, scales, names,
                                       colors=['k', 'gray', 'blue'],
                                       limits=limits, fontsize=16,
                                       bins=n_bins, truth_color='red', figsize=(9, 8.5), linewidth=1,
                                       plot_datapoints=False, smooth=1.0, smooth1d=2.0)

    plt.show()

    return fig, axes


def get_simu_parameters(config_path):

    # Load config file
    config = configparser.ConfigParser()
    prefix = os.path.basename(config_path)[0:19]
    config.read(config_path)
    # Determinte which sampler was used
    try:
        sampler_type = config["Sampler"]["Type"]
    except KeyError:
        print("No configuration file found.")
        config = None
        sampler_type = 'dynesty'

    # Load the corresponding simulation
    names_full = np.array([key for key in config['ParametersLowerBounds']])
    simu_name = os.path.basename(config["InputData"]["FilePath"])
    names_math = [r'$M_c$', r'$q$', r'$t_c$', r'$\chi_1$', r'$\chi_2$',
                  r'$\log D_L$', r'$\cos \i$',
                  r'$\cos \beta$', r'$\lambda$', r'$\psi$', r'$\phi_0$']

    if 'LDC' in simu_name:
        simu_path = '/Users/qbaghi/Codes/data/LDC/'
        p, td = loadings.load_ldc_data(simu_path + '/' + simu_name)
        # Convert waveform parameters to actually sampled parameters
        par = physics.get_params(p, sampling=True)
        i_intr = [0, 1, 2, 3, 4, 7, 8]
    else:
        simu_path = '/Users/qbaghi/Codes/data/simulations/mbhb/'
        time_vect, signal_list, noise_list, par = loadings.load_simulation(simu_name)
        i_intr = [0, 1, 2, 3, 4, 8, 9]

    if sampler_type == 'ptemcee':
        # Load the MCMC samples
        chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_chain.p')
        lnprob = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_lnprob.p')

        if not config["Model"].getboolean('reduced'):
            # # Keep all parameters
            # names = np.array(names_math)
            # par0 = np.array(par)
            # Restrict to instrinsic parameters
            names = np.array(names_math)[i_intr]
            par0 = np.array(par)[i_intr]
            print("Shape of loaded sample array: " + str(chain.shape))
            chain0 = chain[:, :, :, i_intr]
            inds = np.where(chain0[0, 0, :, 0] != 0)[0]
            print("Number of non-zero entries: " + str(len(inds)))
            chain0 = chain0[:, :, inds, :]
            # chain0 = chain[:, :, inds, :]
            print("Shape of non-zero sample array: " + str(chain0.shape))
        else:
            # Restrict to instrinsic parameters
            names = np.array(names_math)[i_intr]
            par0 = np.array(par)[i_intr]
            print("Shape of loaded sample array: " + str(chain.shape))
            chain0 = chain[:, :, chain[0, 0, :, 0] != 0, :]
            print("Shape of non-zero sample array: " + str(chain0.shape))

    elif sampler_type == 'dynesty':
        # fig, axes = dyplot.runplot(chain)  # summary (run) plot
        try:
            chain0 = loadings.load_samples(os.path.dirname(config_path) 
                                           + '/' + prefix + '_final_save.p')
        except ValueError:
            print("No final dinesty result, loading the initial file.")
            chain0 = loadings.load_samples(os.path.dirname(config_path) 
                                           + '/' + prefix + '_initial_save.p')

    return names, par0, chain0, lnprob, sampler_type