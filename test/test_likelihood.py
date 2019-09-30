

if __name__ == '__main__':

    import numpy as np
    import configparser
    import datetime
    # import myplots
    import time
    import h5py
    # from scipy import signal
    from bayesdawn.waveforms import lisaresp
    from bayesdawn import gwmodel, dasampler, datamodel, psdmodel
    from bayesdawn.utils import loadings
    # FTT modules
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import fft, ifft

    # ==================================================================================================================
    config = configparser.ConfigParser()
    config_file = "../configs/config_dynesty.ini"
    config.read(config_file)

    # Input file name
    hdf5_name = config["InputData"]["FilePath"]
    # Load simulation data
    time_vect, signal_list, noise_list, params = loadings.load_simulation(hdf5_name)
    params = np.array(params)
    # i1 = np.int(216160 - 2**17)
    # i2 = 216160
    t_sig = time_vect[:]
    del_t = t_sig[1] - t_sig[0]
    scale = 1.0
    # i1 = np.int(np.float(config["InputData"]["StartTime"]) / del_t)
    # i2 = np.int(np.float(config["InputData"]["EndTime"]) / del_t)
    i1 = 0
    i2 = signal_list[0].shape[0]
    y_list = [(signal_list[j][i1:i2] + noise_list[j][i1:i2])/scale for j in range(len(signal_list))]
    y_fft_list = [fft(y0) for y0 in y_list]
    n = len(y_list[0])
    fs = 1 / del_t
    tobs = n * del_t
    # No missing data
    mask = np.ones(n)
    # ==================================================================================================================
    # Choose way to compute likelihood
    # ==================================================================================================================
    reduced = True

    # ==================================================================================================================
    # Instantiate waveform class
    # ==================================================================================================================
    signal_cls = lisaresp.MBHBWaveform()

    # ==================================================================================================================
    # # Create data analysis GW model with instrinsic parameters only
    if reduced:
        names = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'lam', 'beta']
        bounds = [[0.1e6, 1e7], [0.1e6, 1e7], [0, 1], [0, 1], [2000000.0, 2162000.0], [0, np.pi], [0, 2*np.pi]]
        params0 = params[signal_cls.i_intr]
    else:
        # Create data analysis GW model with the full parameter vector
        names = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi']
        bounds = [[0.5e6, 5e6], [0.5e6, 5e6], [0, 1], [0, 1], [2000000.0, 2162000.0], [100000, 500000],
                  [0, np.pi], [0, 2*np.pi], [0, np.pi], [0, 2 * np.pi], [0, 2 * np.pi]]
        params0 = params[:]

    distribs = ['uniform' for name in names]
    channels = ['A', 'E', 'T']
    # Instantiate data analysis model class
    model_cls = gwmodel.GWModel(signal_cls,
                                tobs,
                                del_t,
                                names=names,
                                bounds=bounds,
                                distribs=distribs,
                                channels=channels,
                                fmin=float(config['Model']['MinimumFrequency']),
                                fmax=float(config['Model']['MaximumFrequency']),
                                nsources=1,
                                reduced=reduced)

    # ==================================================================================================================
    # Creation of PSD class
    # ==================================================================================================================
    psd_cls = psdmodel.PSDTheoretical(n, fs, channels=channels, scale=scale)
    spectrum_list = psd_cls.calculate(n)

    # ==================================================================================================================
    # Test of likelihood computation
    # ==================================================================================================================
    t1 = time.time()
    ll = model_cls.log_likelihood(params0, spectrum_list, y_list)
    t2 = time.time()
    print("Computation time: " + str(t2-t1) + " seconds.")
    print("Likelihood value: " + str(ll))