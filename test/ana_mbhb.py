import os
import configparser


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import numpy as np
    from dynesty import plotting as dyplot
    from bayesdawn.postproc import postprocess
    from bayesdawn.utils import loadings

    # File name of the simulation
    hdf5_name_simu = '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5'
    hdf5_name = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-10-29_14h17-48_chain.p'
    # hdf5_name_simu = '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5'
    # hdf5_name = '/Users/qbaghi/Codes/data/results_dynesty/mbhb/2019-10-24_17h28-32_initial_save.p'

    # Load config file
    config = configparser.ConfigParser()
    prefix = os.path.basename(hdf5_name)[0:19]
    config_path = os.path.dirname(hdf5_name) + '/' + prefix + '_config.ini'
    config.read(config_path)
    # Load the MCMC samples
    chain = loadings.load_samples(hdf5_name)
    lnprob = loadings.load_samples(os.path.dirname(hdf5_name) + '/' + prefix + '_lnprob.p')

    # Load the corresponding simulation
    time_vect, signal_list, noise_list, params = loadings.load_simulation(hdf5_name_simu)

    # Corner plot
    try:
        sampler_type = config["Sampler"]["Type"]
    except KeyError:
        print("No configuration file found.")
        config = None
        sampler_type = 'dynesty'

    if sampler_type == 'ptemcee':
        fig, ax = postprocess.postprocess(chain, lnprob, config, params, n_burn=1000, n_thin=1, n_bins=40, k=5)
    else:
        # fig, axes = dyplot.runplot(chain)  # summary (run) plot
        fg, ax = dyplot.cornerpoints(chain, cmap='plasma', truths=params, kde=False)