import os
import configparser
from bayesdawn.postproc import resanalysis


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import numpy as np
    from dynesty import plotting as dyplot
    from bayesdawn.postproc import postprocess
    from bayesdawn.utils import loadings, physics

    # Pathf for the simulation
    # config_path = '/Users/qbaghi/Codes/data/results_dynesty/mbhb/2019-10-24_17h28-32_config.ini'
    config_path = '/Users/qbaghi/Codes/data/results_dynesty/mbhb/2019-10-30_21h38-52_config.ini'
    # config_path = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-10-29_14h17-48_config.ini'
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
    n_burn = 10000
    if sampler_type == 'ptemcee':
        # Load the MCMC samples
        chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_chain.p')
        lnprob = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_lnprob.p')
        fig, ax = postprocess.postprocess(chain, lnprob, config, params, n_burn=n_burn, n_thin=1, n_bins=40, k=5)
    elif sampler_type == 'dynesty':
        # fig, axes = dyplot.runplot(chain)  # summary (run) plot
        try:
            chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_final_save.p')
        except:
            print("No final dinesty result, loading the initial file.")
            chain = loadings.load_samples(os.path.dirname(config_path) + '/' + prefix + '_initial_save.p')
        # fg, ax = dyplot.cornerpoints(chain, cmap='plasma', truths=params, kde=False)
        # fg, ax = dyplot.cornerplot(chain, truths=params, dims=i_intr)
        names = np.array([key for key in config['ParametersLowerBounds']])
        fig, axes = resanalysis.cornerplot([chain.samples[n_burn:, i_intr]],
                                           params[i_intr], 0, 1, names[i_intr],
                                           colors=['k', 'gray', 'blue'],
                                           limits=None, fontsize=16,
                                           bins=40, truth_color='red', figsize=(9, 8.5), linewidth=1)

        plt.show()