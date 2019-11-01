# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2018
import h5py
import configparser
import pickle
from LISAhdf5 import LISAhdf5
import re



def load_samples(hdf5_name):
    """

    Parameters
    ----------
    hdf5_name : str
        path of sampling result file

    Returns
    -------
    chain : ndarray
        numpy array containing the posterior samples

    """

    # Load data
    # chain = pd.read_hdf(hdf5_name, key='chain', mode='r').to_numpy()
    # chain_eff = chain[chain != 0]
    # Load data
    chain_file = open(hdf5_name, 'rb')
    chain = pickle.load(chain_file)
    chain_file.close()

    return chain


def load_simulation(hdf5_name,
                    param_keylist=['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi'],
                    signal_keylist=['tdi_a', 'tdi_e', 'tdi_ts']):

    # Load data
    fh5 = h5py.File(hdf5_name, 'r')
    params = [fh5["parameters/" + par][()]for par in param_keylist]
    time_vect = fh5["signal/time"][()]
    signal_list = [fh5["signal/" + sig][()] for sig in signal_keylist]
    noise_list = [fh5["noise/" + sig][()] for sig in signal_keylist]
    fh5.close()

    return time_vect, signal_list, noise_list, params


def create_config_file(file_name='example.ini'):

    config = configparser.ConfigParser()
    config['InputData'] = {'FilePath': '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5',
                           'StartTime': 850880,
                           'EndTime': 2161600}

    config['Model'] = {'Likelihood': 'full',
                       'MinimumFrequency': 1e-5,
                       'MaximumFrequency': 1e-2}

    config['Sampler'] = {'Type': 'dynesty',
                         'WalkerNumber': 44,
                         'TemperatureNumber': 10,
                         'MaximumIterationNumber': 100000,
                         'ThinningNumber': 1,
                         'AuxiliaryParameterUpdateNumber': 100,
                         'SavingNumber': 100,
                         'PSDEstimation': False,
                         'MissingDataImputation': False}

    config['OutputData'] = {'DirectoryPath': '/Users/qbaghi/Codes/data/results_dynesty/local/',
                            'FileSuffix': 'chains.hdf5'}

    with open(file_name, 'w') as configfile:
        config.write(configfile)


def load_ldc_data(hdf5_name):

    print("Loading data...")
    fd5 = LISAhdf5(hdf5_name)
    n_src = fd5.getSourcesNum()
    gws = fd5.getSourcesName()
    print("Found %d GW sources: " % n_src, gws)
    if not re.search('MBHB', gws[0]):
        raise NotImplementedError
    p = fd5.getSourceParameters(gws[0])
    td = fd5.getPreProcessTDI()
    p.display()
    print("Data loaded.")

    return p, td
