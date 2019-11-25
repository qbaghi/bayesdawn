import os
import configparser
from bayesdawn.postproc import resanalysis
from scipy import stats
from chainconsumer import ChainConsumer


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
    names_math = [r'$M_c$', r'$q$', r'$t_c$', r'$\chi_1$', r'$\chi_2$', r'$\log D_L$', r'$\cos \i$',
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

    # Restrict to instrinsic parameters
    names = np.array(names_math)[i_intr]
    par0 = np.array(par)[i_intr]

    if sampler_type == 'ptemcee':
        # Load the MCMC samples
        chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_chain.p')
        lnprob = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_lnprob.p')

        if not config["Model"].getboolean('reduced'):
            print("Shape of loaded sample array: " + str(chain.shape))
            chain0 = chain[:, :, chain[0, 0, :, 0] != 0, i_intr]
            print("Shape of non-zero sample array: " + str(chain0.shape))
        else:
            print("Shape of loaded sample array: " + str(chain.shape))
            chain0 = chain[:, :, chain[0, 0, :, 0] != 0, :]
            print("Shape of non-zero sample array: " + str(chain0.shape))

    elif sampler_type == 'dynesty':
        # fig, axes = dyplot.runplot(chain)  # summary (run) plot
        try:
            chain0 = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_final_save.p')
        except ValueError:
            print("No final dinesty result, loading the initial file.")
            chain0 = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_initial_save.p')

    return names, par0, chain0, lnprob, sampler_type


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import numpy as np
    from dynesty import plotting as dyplot
    from bayesdawn.postproc import postprocess
    from bayesdawn.utils import loadings, physics

    # Path for the analysis configuration
    analysis_config_path = "/Users/qbaghi/Codes/python/bayesdawn/bayesdawn/configs/config_analysis.ini"

    # Load config file
    config_a = configparser.ConfigParser()
    config_a.read(analysis_config_path)

    # Path for the simulation configuration files
    i_max = 2
    config_paths = [config_a["InputData"]["configPath" + str(i)] for i in range(1, i_max)]

    # names, par0, chain0, lnprob = get_simu_parameters(config_paths[0])
    simu_params = [get_simu_parameters(config_path) for config_path in config_paths]

    # Corner plot
    n_burn = config_a["Analysis"].getint("burnin")
    n_thin = config_a["Analysis"].getint("thin")
    n_samples = config_a["Analysis"].getint("samples")
    k = config_a["Plot"].getint("sigmas")
    n_bins = config_a["Plot"].getint("bins")

    offset = 0
    scales = 1

    flattened_chains = []

    for i in range(len(simu_params)):

        names, par0, chain, lnprob, sampler_type = simu_params[i]
        print("Shape of effective chain: " + str(chain.shape))
        chain_flatten = chain[0, :, n_burn:, :].reshape((-1, chain.shape[3]))
        medians = np.median(chain_flatten, axis=0)
        # stds = np.std(chain_flatten, axis=0)
        stds = stats.median_absolute_deviation(chain_flatten, axis=0)
        limits = [[medians[i] - k * stds[i], medians[i] + k * stds[i]] for i in range(chain_flatten.shape[1])]
        print("Shape of flattened chain: " + str(chain_flatten.shape))
        print("Length of parameter vector: " + str(len(par0)))

        flattened_chains.append(chain_flatten[0:n_samples, :])

    if sampler_type == 'ptemcee':

        # fig, axes = resanalysis.cornerplot(flattened_chains,
        #                                    par0, offset, scales, names,
        #                                    colors=['k', 'gray', 'blue'],
        #                                    limits=limits, fontsize=16,
        #                                    bins=n_bins, truth_color='red', figsize=(9, 8.5), linewidth=1,
        #                                    plot_datapoints=False, smooth=1.0, smooth1d=2.0)
        #
        # plt.show()

        c = ChainConsumer()
        colors = ['#0A6E2B', '#646464', '#0868AC']
        labels = ['Complete data', 'Gapped data, window', 'Gapped data, DA']
        for i in range(len(flattened_chains)):
            c.add_chain(flattened_chains[i], parameters=list(names), name=labels[i], color=colors[i])

        fig = c.plotter.plot(truth=par0, filename=config_a["OutputData"]["outputPath"])
        plt.show()
        # # Try to use dynesty plotting
        # chain_flat = chain[0, :, n_burn:, :].reshape((-1, chain.shape[3]))
        # results = {'samples': chain_flat,
        #            'weights': np.ones(chain_flat.shape[0])}
        # lnprob[0, :, n_burn:].reshape((-1))
        # fig, axes = plt.subplots(ndim, ndim, figsize=(9, 8.5))
        # fg, ax = dyplot.cornerplot(results, truths=par, truth_color='red', use_math_text=True, verbose=True,
        #                            labels=np.array(names_math)[i_intr])#, fig=(fig, axes))

        # fig, axes = plt.subplots(len(i_intr), len(i_intr), figsize=(9, 8.5))

    elif sampler_type == 'dynesty':

        fig, axes = plt.subplots(len(par0), len(par0), figsize=(9, 8.5))
        fg, ax = dyplot.cornerplot(chain, # dims=i_intr,
                                   span=None, quantiles=None, color='black',
                                   smooth=0.02, quantiles_2d=None, hist_kwargs=None, hist2d_kwargs=None,
                                   labels=names,
                                   label_kwargs=None, show_titles=False, title_fmt='.2f', title_kwargs=None,
                                   truths=par0, truth_color='red', truth_kwargs=None, max_n_ticks=5, top_ticks=False,
                                   use_math_text=False, verbose=False, fig=(fig, axes))
            # mc, q, tc, chi1, chi2, np.log10(dl), np.cos(incl), np.sin(bet), lam, psi, phi0
            # fig = corner.corner(chain.samples[1500:, i_intr], bins=40, range=None, weights=None, color='k',
            #                     hist_bin_factor=1, smooth=None, smooth1d=None, truths=par[i_intr], labels=names)

            # fig, axes = resanalysis.cornerplot([chain.samples[n_burn:, i_intr]],
            #                                    params[i_intr], 0, 1, names[i_intr],
            #                                    colors=['k', 'gray', 'blue'],
            #                                    limits=None, fontsize=16,
            #                                    bins=40, truth_color='red', figsize=(9, 8.5), linewidth=1)
        plt.show()
