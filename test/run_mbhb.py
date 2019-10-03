import numpy as np
import datetime
# import myplots
import time
import h5py
# from scipy import signal
from bayesdawn.waveforms import lisaresp
from bayesdawn import gwmodel, dasampler, datamodel, psdmodel, posteriormodel
from bayesdawn.utils import loadings
from bayesdawn import samplers
import tdi
from dynesty import NestedSampler

# FTT modules
import pyfftw

pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft


if __name__ == '__main__':

    import configparser
    from optparse import OptionParser

    # ==================================================================================================================
    parser = OptionParser(usage="usage: %prog [options] YYY.txt", version="08.02.2018, Quentin Baghi")
    # ### Options  ###
    # parser.add_option("-o", "--out_dir",
    #                   type="string", dest="out_dir", default="",
    #                   help="Path to the directory where the results are written [default ]")
    #
    # parser.add_option("-i", "--in_dir",
    #                   type="string", dest="in_dir", default="",
    #                   help="Path to the directory where the simulation file is stored [default ]")

    (options, args) = parser.parse_args()
    if args == []:
        config_file = "../configs/config_ptemcee.ini"
    else:
        config_file = args[0]
    # ==================================================================================================================
    config = configparser.ConfigParser()
    # config.read("../configs/config_dynesty.ini")
    config.read(config_file)
    # ==================================================================================================================
    # Input file name
    hdf5_name = config["InputData"]["FilePath"]
    # Load simulation data
    time_vect, signal_list, noise_list, params = loadings.load_simulation(hdf5_name)
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
    # Instantiate waveform class
    # ==================================================================================================================
    signal_cls = lisaresp.MBHBWaveform()

    # ==================================================================================================================
    # Instantiate GW model class
    # ==================================================================================================================
    if config['Model'].getboolean('reduced'):
        # Create data analysis GW model with instrinsic parameters only
        names = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'lam', 'beta']
        bounds = [[0.1e6, 1e7], [0.1e6, 1e7], [0, 1], [0, 1], [2000000.0, 25000000.0], [0, np.pi], [0, 2*np.pi]]
        params0 = np.array(params)[signal_cls.i_intr]
    else:
        # Create data analysis GW model with the full parameter vector
        names = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi']
        bounds = [[0.5e6, 5e6], [0.5e6, 5e6], [0, 1], [0, 1], [2000000.0, 2162000.0], [100000, 500000],
                  [0, np.pi], [0, 2*np.pi], [0, np.pi], [0, 2 * np.pi], [0, 2 * np.pi]]
        params0 = np.array(params)

    distribs = ['uniform' for name in names]
    channels = ['A', 'E', 'T']
    # Instantiate data analysis model class
    model_cls = gwmodel.GWModel(signal_cls,
                                tobs,
                                del_t,
                                names=names,
                                channels=channels,
                                fmin=float(config['Model']['MinimumFrequency']),
                                fmax=float(config['Model']['MaximumFrequency']),
                                nsources=1,
                                reduced=config['Model'].getboolean('reduced'))


    # ==================================================================================================================
    # Creation of PSD class
    # ==================================================================================================================
    psd_cls = psdmodel.PSDTheoretical(n, fs, channels=channels, scale=scale)
    spectrum_list = psd_cls.calculate(n)

    # ==================================================================================================================
    # Instanciate posterior model class
    # ==================================================================================================================
    posterior_cls = posteriormodel.PosteriorModel(model_cls, bounds, rescaled=config['Model'].getboolean('rescaled'))
    if config['Model'].getboolean('normalized'):
        posterior_cls.compute_log_norm(spectrum_list)

    # ==================================================================================================================
    # Creation of data class instance
    # ==================================================================================================================
    dat_cls = datamodel.GaussianStationaryProcess(y_list, mask)

    # ==================================================================================================================
    # Creation of sampler class instance
    # ==================================================================================================================
    print("Chosen sampler: " + config["Sampler"]["Type"])
    if config["Sampler"]["Type"] == 'dynesty':
        nlive = np.int(config["Sampler"]["WalkerNumber"]) # model_cls.ndim_tot * (model_cls.ndim_tot + 1) // 2
        sampler_cls = samplers.extended_nested_sampler(posterior_cls.log_likelihood,
                                                       posterior_cls.uniform2param,
                                                       model_cls.ndim_tot,
                                                       nlive=nlive,
                                                       logl_args=(spectrum_list, y_fft_list))
    elif config["Sampler"]["Type"] == 'ptemcee':
        sampler_cls = samplers.ExtendedPTMCMC(int(config["Sampler"]["WalkerNumber"]),
                                              model_cls.ndim_tot,
                                              posterior_cls.log_likelihood,
                                              posterior_cls.logp,
                                              ntemps=int(config["Sampler"]["TemperatureNumber"]),
                                              loglargs=(spectrum_list, y_fft_list),
                                              logpargs=(posterior_cls.lo_rescaled, posterior_cls.hi_rescaled))


    # ==================================================================================================================
    # Creation of data augmentation sampling class instance
    # ==================================================================================================================
    das = dasampler.FullModel(posterior_cls, psd_cls, dat_cls, sampler_cls,
                              outdir=config["OutputData"]["DirectoryPath"],
                              prefix='samples',
                              n_wind=500,
                              n_wind_psd=50000,
                              imputation=config['Sampler'].getboolean('MissingDataImputation'),
                              psd_estimation=config['Sampler'].getboolean('PSDEstimation'),
                              normalized=config['Model'].getboolean('normalized'))

    # ==================================================================================================================
    # Test of likelihood calculation
    # ==================================================================================================================
    # Full parameter vector
    t1 = time.time()
    test0 = model_cls.log_likelihood(params0, spectrum_list, y_fft_list) + posterior_cls.log_norm
    if posterior_cls.rescaled:
        test1 = posterior_cls.log_likelihood(posterior_cls.param2uniform(params0), spectrum_list, y_fft_list)
    else:
        test1 = posterior_cls.log_likelihood(params0, spectrum_list, y_fft_list)
    t2 = time.time()
    print("Test with set of parameters 1")
    print("loglike value: " + str(test1) + " should be equal to " + str(test0))
    print("Likelihood computation time: " + str(t2 - t1))
    # params2 = np.array(posterior_cls.lo)
    # t1 = time.time()
    # test2 = posterior_cls.log_likelihood(posterior_cls.param2uniform(params2), spectrum_list, y_fft_list)
    # t2 = time.time()
    # print("Test with set of parameters 2")
    # print("loglike value: " + str(test2))
    # print("Likelihood computation time: " + str(t2 - t1))

    # ==================================================================================================================
    # Run the sampling
    # ==================================================================================================================
    now = datetime.datetime.now()
    prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_")
    out_dir = config["OutputData"]["DirectoryPath"]
    print("start sampling...")

    # sampler_cls0 = NestedSampler(posterior_cls.log_likelihood, posterior_cls.uniform2param,
    #                              model_cls.ndim_tot, nlive=np.int(config["Sampler"]["WalkerNumber"]),
    #                              logl_args=(spectrum_list, y_fft_list))
    # sampler_cls0.run_nested(maxiter=int(config['Sampler']['MaximumIterationNumber']))
    # # das.sampler_cls = sampler_cls0

    das.run(n_it=int(config['Sampler']['MaximumIterationNumber']),
            n_update=int(config['Sampler']['AuxiliaryParameterUpdateNumber']),
            n_thin=int(config['Sampler']['ThinningNumber']),
            n_save=int(config['Sampler']['SavingNumber']),
            save_path=out_dir + prefix + 'chains_temp.hdf5')

    print("done.")

    fh5 = h5py.File(out_dir + prefix + config["OutputData"]["FileSuffix"], 'w')
    fh5.create_dataset("chains/chain", data=das.sampler_cls.get_chain())
    if config["Sampler"]["Type"] == 'ptemcee':
        fh5.create_dataset("temperatures/beta_hist", data=das.sampler_cls._beta_history)
    fh5.close()





