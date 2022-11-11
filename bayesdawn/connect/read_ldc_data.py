# Gap filling investigation support for LISA inference with lisabeta (maybe more generally)
# 
# set of functions to import LDC 2 data and convert them in more convenient format 
# Eleonora Castelli NASA-GSFC 2022

import numpy as np
import h5py

def load_tdi_timeseries(fname, 
                        import_datasets = ['obs','clean','sky','noisefree'], 
                        generate_additional_datasets = True,
                        additional_datasets = ['clean_gapped', 'noise_gapped', 'sky_gapped']):
    """
    Loads LDC-2 TDI time-series from HDF5 file and packs them in a dictionary.
    Dictionary keys are the HDF5 dataset groups.
    Each key items are the corresponding TDI time-series in a numpy rec-array with fields ['t', 'X', 'Y', 'Z'].
    
    Parameters:
        fname: string
            LDC-2 HDF5 filename with .h5 extension
        import_datasets: list
            list of HDF5 groups containing /tdi datasets
        generate_additional_datasets: bool
            flag to generate additional datasets
        additional_datasets: list
            list of desired additional datasets
    
    Returns:
        tdi: dict
            dict with keys corresponding to HDF5 groups (import_datasets + additional_datasets)
            each key item is a numpy rec-array with fields ['t', 'X', 'Y', 'Z']
    """
    
    fid = h5py.File(fname)
    tdi={}
    for ds in import_datasets:
        tdi[ds] = fid[ds+'/tdi'][()].squeeze()
    # generate noise-only dataset
    tdi['noise'] = np.copy(tdi['clean'])
    for comb in tdi['clean'].dtype.names[1:]:
        tdi['noise'][comb] = tdi['clean'][comb] - tdi['sky'][comb]
    if generate_additional_datasets:
        dt = tdi['obs']['t'][1]-tdi['obs']['t'][0]
        gaps = tdi['obs']['t'][np.isnan(tdi['obs']['X'])]
        # find start times of gaps by taking first point and the point for which 
        # the difference between two subsequent samples is more than dt
        gapsstart = np.hstack([np.array([gaps[0]]), (gaps[1:])[gaps[1:]-gaps[:-1]>dt]])
        # same thing with endpoints, but reversed
        gapsend = np.hstack([(gaps[:-1])[gaps[1:]-gaps[:-1]>dt], np.array([gaps[-1]])])
        # get the indexes for the gaps
        gaps_indices = np.vstack([gapsstart, gapsend])
        # generate gapped dataset
        for ds, dsgap in zip(['clean','noise','sky'], additional_datasets):
            tdi[dsgap] = np.copy(tdi[ds])
            for comb in tdi['obs'].dtype.names[1:]:
                tdi[dsgap][comb][np.isnan(tdi['obs']['X'])] = 0
       
    return tdi

def build_orthogonal_tdi(tdi_xyz, skip = 100):
    """
    Builds orthogonal TDi combinations and packs them in a recarray with fields ['t', A', 'E', 'T'].
    
    Parameters:
        tdi_xyz: dict or numpy rec-array
            LDC imported data. Format can be either dict containing multiple numpy recarrays 
            or a single numpy rec-array with fields ['t', 'X', 'Y', 'Z']
        skip: integer
            number of samples to skip to remove margin effects
    
    Returns:
        data: dict or numpy rec-array
            either dict containing multiple numpy recarrays 
            or a single numpy rec-array with fields ['t', A', 'E', 'T'].
    """
    if type(tdi_xyz) is dict:
        data = {}
        for k in tdi_xyz.keys():
            # load tdi A, E, T
            A = (tdi_xyz[k]['Z'][skip:] -   tdi_xyz[k]['X'][skip:])/np.sqrt(2)
            E = (tdi_xyz[k]['X'][skip:] - 2*tdi_xyz[k]['Y'][skip:] + tdi_xyz[k]['Z'][skip:])/np.sqrt(6)
            T = (tdi_xyz[k]['X'][skip:] +   tdi_xyz[k]['Y'][skip:] + tdi_xyz[k]['Z'][skip:])*float(1./np.sqrt(3))

            data[k] = np.rec.fromarrays([tdi_xyz[k]['t'][skip:], A, E, T], names = ['t', 'A', 'E', 'T'])
    
    elif type(tdi_xyz) is np.ndarray:
        # load tdi A, E, T
        A = (tdi_xyz['Z'][skip:] -   tdi_xyz['X'][skip:])/np.sqrt(2)
        E = (tdi_xyz['X'][skip:] - 2*tdi_xyz['Y'][skip:] + tdi_xyz['Z'][skip:])/np.sqrt(6)
        T = (tdi_xyz['X'][skip:] +   tdi_xyz['Y'][skip:] + tdi_xyz['Z'][skip:])*float(1./np.sqrt(3))

        data = np.rec.fromarrays([tdi_xyz['t'][skip:], A, E, T], names = ['t', 'A', 'E', 'T'])
    
    return data
