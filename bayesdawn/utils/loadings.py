# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2018
import h5py
import configparser
import pickle
import re
import numpy as np
from bayesdawn.gaps import gapgenerator
try:
    from LISAhdf5 import LISAhdf5
except:
    print("MLDC modules could not be loaded.")


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
                    param_keylist=["m1", "m2", "chi1", "chi2", "Deltat", 
                                   "dist", "inc", "phi", "lambda", "beta", 
                                   "psi"],
                    signal_keylist=['tdi_a', 'tdi_e', 'tdi_t']):
    """
    Load home-made simulation.

    Parameters
    ----------
    hdf5_name : str
        name of input hdf5 file
    param_keylist : list, optional
        GW parameters keys, by default ["m1", "m2", "chi1", "chi2", "Deltat", 
        "dist", "inc", "phi", "lambda", "beta", "psi"]
    signal_keylist : list, optional
        Channel keys, by default ['tdi_a', 'tdi_e', 'tdi_t']

    Returns
    -------
    signal_list : list
        List of signal time series for each channel
    noise_list : list
        List of noise time series for each channel
    params : ndarray
        GW parameters
    tstart : float
        Start time of observation [s]
    del_t : float
        Sampling time [s]
    tobs : float
        Observation time [s]
    """

    # Load data
    fh5 = h5py.File(hdf5_name, 'r')
    params = [fh5["parameters/" + par][()]for par in param_keylist]
    # time_vect = fh5["signal/time"][()]
    signal_list = [fh5["signal/" + sig][()] for sig in signal_keylist]
    noise_list = [fh5["noise/" + sig][()] for sig in signal_keylist]
    tstart = fh5["time/start_time"][()]
    del_t = fh5["time/del_t"][()]
    tobs = fh5["time/tobs"][()]
    tvect = fh5["time/time_stamps"][()]
    fh5.close()

    return tvect, signal_list, noise_list, params, tstart, del_t, tobs


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
    # p.display()
    print("Data loaded.")

    return p, td


def load_gaps(config, tm):
    """

    Parameters
    ----------
    config : parseconfig instance
        configuration of the simulation
    tm : ndarray
        time vector

    Returns
    -------
    wd : ndarray
        smooth time window including gaps
    wd_full : ndarray
        smooth time window for full time series
    mask : ndarray
        binary mask (sharp gap window)

    """

    del_t = tm[1] - tm[0]

    if config["TimeWindowing"].getboolean("Gaps"):

        if config["TimeWindowing"]["GapType"] == 'single':
            nd = [np.int(config["TimeWindowing"].getfloat("GapStartTime")/del_t)]
            nf = [np.int(config["TimeWindowing"].getfloat("GapEndTime")/del_t)]

        elif config["TimeWindowing"]["GapType"] == 'file':
            mask = np.load(config["TimeWindowing"]["MaskFilePath"])
            nd, nf = gapgenerator.find_ends(mask)

        else:
            nd, nf = gapgenerator.generategaps(
                tm.shape[0], 1/del_t, 
                config["TimeWindowing"].getint("GapNumber"),
                config["TimeWindowing"].getfloat("GapDuration"),
                gap_type=config["TimeWindowing"]["GapType"],
                f_gaps=config["TimeWindowing"].getfloat("GapFrequency"),
                wind_type='rect', std_loc=0, std_dur=0)

        wd = gapgenerator.windowing(
            nd, nf, tm.shape[0], 
            window=config["TimeWindowing"]["WindowType"],
            n_wind=config["TimeWindowing"].getint("DecayNumberGaps"))
        wd_full = gapgenerator.modified_hann(
            tm.shape[0], 
            n_wind=config["TimeWindowing"].getint("DecayNumberFull"))
        mask = gapgenerator.windowing(nd, nf, tm.shape[0], window='rect')
        # wd_full = gapgenerator.modified_hann(tm.shape[0],
        # n_wind=np.int((config["InputData"].getfloat("EndTime") - tc) / (2*del_t)))

    else:

        wd = gapgenerator.modified_hann(
            tm.shape[0], 
            n_wind=config["TimeWindowing"].getint("DecayNumberFull"))

        # wd = gapgenerator.modified_hann(tm.shape[0],
        #                                 n_wind=np.int((config["InputData"].getfloat("EndTime") - tc) / (2*del_t)))
        wd_full = wd[:]

        mask = np.ones(tm.shape[0])
        # wd = signal.tukey(Xd.shape[0], alpha=(config["InputData"].getfloat("EndTime") - tc) / tobs, sym=True)

    return wd, wd_full, mask
