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
    # config_path = '/Users/qbaghi/Codes/data/results_dynesty/mbhb/2019-10-24_17h28-32_config.ini'
    # config_path = '/Users/qbaghi/Codes/data/results_dynesty/mbhb/2019-11-01_00h58-47_config.ini'
    config_path = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-11-04_18h55-04_config.ini'
    # hdf5_name = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-10-29_14h17-48_chain.p'
    # hdf5_name_simu = '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5'
    # hdf5_name = '/Users/qbaghi/Codes/data/results_dynesty/mbhb/2019-10-24_17h28-32_initial_save.p'

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
    simu_name = os.path.basename(config["InputData"]["FilePath"])
    if 'LDC' in simu_name:
        simu_path = '/Users/qbaghi/Codes/data/LDC/'
        p, td = loadings.load_ldc_data(simu_path + '/' + simu_name)
        params = np.array(physics.get_params(p, sampling=True))
        i_intr = [0, 1, 2, 3, 4, 7, 8]
    else:
        simu_path = '/Users/qbaghi/Codes/data/simulations/mbhb/'
        time_vect, signal_list, noise_list, params = loadings.load_simulation(simu_name)
        i_intr = [0, 1, 2, 3, 4, 8, 9]

    # Corner plot
    n_burn = 2000
    if sampler_type == 'ptemcee':
        # Load the MCMC samples
        chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_chain.p')
        lnprob = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_lnprob.p')
        fig, ax = postprocess.postprocess(chain, lnprob, config, params, n_burn=n_burn, n_thin=1, n_bins=40, k=4)
    elif sampler_type == 'dynesty':
        # fig, axes = dyplot.runplot(chain)  # summary (run) plot
        try:
            chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_final_save.p')
        except:
            print("No final dinesty result, loading the initial file.")
            chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_initial_save.p')
        # fg, ax = dyplot.cornerpoints(chain, cmap='plasma', truths=params, kde=False)
        par = physics.waveform_to_like(params)
        names = np.array([key for key in config['ParametersLowerBounds']])
        fig, axes = plt.subplots(len(i_intr), len(i_intr), figsize=(9, 8.5))
        fg, ax = dyplot.cornerplot(chain, dims=i_intr, span=None, quantiles=None, color='black',
                                   smooth=0.02, quantiles_2d=None, hist_kwargs=None, hist2d_kwargs=None, labels=names,
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