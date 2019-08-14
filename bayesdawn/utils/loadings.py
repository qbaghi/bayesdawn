# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2018
import h5py

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