
if __name__ == '__main__':

    import numpy as np
    import h5py
    from bayesdawn.postproc import resanalysis
    from bayesdawn.utils import loadings

    # File name of the simulation
    hdf5_name_simu = '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5'
    # File name where the PTMCMC result are stored
    hdf5_name = '/Users/qbaghi/Codes/data/results_ptemcee/mbhb/2019-08-16_13h56-19_chains_temp.hdf5'

    # Load data
    fh5 = h5py.File(hdf5_name, 'r')
    # params = [fh5["parameters/" + par][()]for par in param_keylist]
    chain = fh5["chain"][()]
    fh5.close()

    # Load simulation parameters
    time_vect, signal_list, noise_list, params = loadings.load_simulation(hdf5_name_simu)
    names = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi']
    bounds = [[0.5e6, 5e6], [0.5e6, 5e6], [0, 1], [0, 1], [2000000.0, 2162000.0], [100000, 500000],
              [0, np.pi], [0, 2 * np.pi], [0, np.pi], [0, 2 * np.pi], [0, 2 * np.pi]]
    truths_vect = params[:]
    offset = np.array([bound[0] for bound in bounds])

    # fig, axes = resanalysis.cornerplot([chain[0, :, 50:, :]], truths_vect, offset, rscales, labels,
    #                                    colors=['r', 'b', 'g', 'm', 'o', 'k', 'y', 'gray', 'brown', 'royalblue', 'violet'],
    #                                    limits=None, fontsize=16,
    #                                    bins=50, truth_color='cadetblue', figsize=(9, 8.5), linewidth=1)