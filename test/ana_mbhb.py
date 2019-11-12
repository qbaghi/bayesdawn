import os
import configparser
from bayesdawn.postproc import resanalysis
import corner

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import numpy as np
    from dynesty import plotting as dyplot
    from bayesdawn.postproc import postprocess
    from bayesdawn.utils import loadings, physics

    # Pathf for the simulation
    # config_path = '/Users/qbaghi/Codes/data/results_dynesty/mbhb/2019-11-01_14h48-14_config.ini'
    # config_path = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-11-05_19h22-55_config.ini'
    # With gaps
    config_path = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-11-11_23h42-15_config.ini'
    # config_path = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-11-04_18h55-04_config.ini'

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
    # Corner plot
    n_burn = 300
    ndim = len(i_intr)
    if sampler_type == 'ptemcee':
        # Load the MCMC samples
        chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_chain.p')
        lnprob = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_lnprob.p')

        if not config["Model"].getboolean('reduced'):
            chain0 = chain[:, :, :, i_intr]
            names = np.array(names_math)[i_intr]
        else:
            chain0 = chain[:]
            names = np.array(names_math)

        fig, ax = postprocess.postprocess(chain0, lnprob, names, par0, n_burn=n_burn, n_thin=1, n_bins=40, k=4)
        # # Try to use dynesty plotting
        # chain_flat = chain[0, :, n_burn:, :].reshape((-1, chain.shape[3]))
        # results = {'samples': chain_flat,
        #            'weights': np.ones(chain_flat.shape[0])}
        # lnprob[0, :, n_burn:].reshape((-1))
        # fig, axes = plt.subplots(ndim, ndim, figsize=(9, 8.5))
        # fg, ax = dyplot.cornerplot(results, truths=par, truth_color='red', use_math_text=True, verbose=True,
        #                            labels=np.array(names_math)[i_intr])#, fig=(fig, axes))

    elif sampler_type == 'dynesty':
        # fig, axes = dyplot.runplot(chain)  # summary (run) plot
        try:
            chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_final_save.p')
        except ValueError:
            print("No final dinesty result, loading the initial file.")
            chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_initial_save.p')

        # fig, axes = plt.subplots(len(i_intr), len(i_intr), figsize=(9, 8.5))
        fig, axes = plt.subplots(len(par), len(par), figsize=(9, 8.5))
        fg, ax = dyplot.cornerplot(chain, # dims=i_intr,
                                   span=None, quantiles=None, color='black',
                                   smooth=0.02, quantiles_2d=None, hist_kwargs=None, hist2d_kwargs=None,
                                   labels=names_math,
                                   label_kwargs=None, show_titles=False, title_fmt='.2f', title_kwargs=None,
                                   truths=par, truth_color='red', truth_kwargs=None, max_n_ticks=5, top_ticks=False,
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
