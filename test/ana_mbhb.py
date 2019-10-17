
if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import numpy as np
    import h5py
    from bayesdawn.postproc import resanalysis
    from bayesdawn.utils import loadings
    from bayesdawn.waveforms import lisaresp
    import os
    import configparser
    import pandas as pd

    # File name of the simulation
    hdf5_name_simu = '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5'
    # File name where the PTMCMC result are stored
    hdf5_name = '/Users/qbaghi/Codes/data/results_dynesty/mbhb/2019-10-11_15h52-41_chains_temp.hdf5'
    # hdf5_name = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-10-08_13h56-32_chains.hdf5'
    # Extract prefix
    prefix = os.path.basename(hdf5_name)[:-12]
    # Get info from config file
    config = configparser.ConfigParser()
    config_file = os.path.dirname(hdf5_name) + '/' + prefix + '_config.ini'
    config.read(config_file)

    # Load data
    chain = pd.read_hdf(hdf5_name, key='chain', mode='r').to_numpy()
    # chain_eff = chain[chain != 0]
    # Load data
    # fh5 = h5py.File(hdf5_name, 'r')
    # chain = fh5["chains/chain"][()]
    # chain = fh5["chain"][()]
    # fh5.close()

    if config["Sampler"]["Type"] == 'ptemcee':
        chain_eff = chain[0, :, 500:, :].reshape((-1, chain.shape[3]))
    elif config["Sampler"]["Type"] == 'dynesty':
        chain_eff = chain[500:, :]

    # Load simulation parameters
    # names = chain.columns.values
    names_full = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi']
    time_vect, signal_list, noise_list, params = loadings.load_simulation(hdf5_name_simu)
    # names = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi']
    bounds_full = [[0.1e6, 1e7], [0.1e6, 1e7], [0, 1], [0, 1], [2000000.0, 2162000.0], [100000, 500000],
                   [0, np.pi], [0, 2 * np.pi], [0, np.pi], [0, 2 * np.pi], [0, 2 * np.pi]]
    # truths_vect = params[:]
    # offset = np.array([bound[0] for bound in bounds])
    signal_cls = lisaresp.MBHBWaveform()

    # Get true parameter values
    if chain.shape[-1] < len(params):
        names = np.array(names_full)[signal_cls.i_intr]
        params0 = np.array(params)[signal_cls.i_intr]
        lo = np.array([bounds_full[i][0] for i in signal_cls.i_intr])
        hi = np.array([bounds_full[i][1] for i in signal_cls.i_intr])
    elif chain.shape[-1] == len(params):
        names = names_full[:]
        params0 = np.array(params)
        lo = np.array([bound[0] for bound in bounds_full])
        hi = np.array([bound[1] for bound in bounds_full])

    # Convert them in the interval [0, 1]
    params0_u = (params0 - lo) / (hi - lo)
    # offset = lo
    # scales = hi - lo
    offset = 0
    scales = 1
    # chain_eff = chain_eff * (hi - lo) + lo
    if config["Sampler"]["Type"] == 'dynesty':
        # Rescale between [0, 1]
        chain_eff = (chain_eff - lo) / (hi - lo)

    medians = np.median(chain_eff, axis=0)
    stds = np.std(chain_eff, axis=0)
    k = 6
    limits = [[medians[i] - k * stds[i], medians[i] + k * stds[i]] for i in range(chain_eff.shape[1])]

    fig, axes = resanalysis.cornerplot([chain_eff], params0_u, offset, scales, names,
                                       colors=['r', 'b', 'g', 'm', 'o', 'k', 'y', 'gray', 'brown', 'royalblue', 'violet'],
                                       limits=limits, fontsize=16,
                                       bins=50, truth_color='cadetblue', figsize=(9, 8.5), linewidth=1)

    plt.show()