import numpy as np
import datetime
import pandas as pd
# import myplots
import time
import h5py
import pickle
# from scipy import signal
from bayesdawn.waveforms import lisaresp
from bayesdawn import likelihoodmodel, dasampler, datamodel, psdmodel, posteriormodel
from bayesdawn.utils import loadings
from bayesdawn import samplers
import tdi
import dynesty
from dynesty.dynamicsampler import stopping_function, weight_function
import ptemcee
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
    # No missing data
    mask = np.ones(n)

    # ==================================================================================================================
    # Instantiate waveform class
    # ==================================================================================================================
    signal_cls = lisaresp.MBHBWaveform()

    # ==================================================================================================================
    # Instantiate GW model class
    # ==================================================================================================================
    # Get all parameter name keys
    names = [key for key in config['ParametersLowerBounds']]
    # Get prior bound values
    bounds = [[float(config['ParametersLowerBounds'][name]), float(config['ParametersUpperBounds'][name])]
              for name in names]

    if config['Model'].getboolean('reduced'):
        # Create data analysis GW model with instrinsic parameters only
        names = np.array(names)[signal_cls.i_intr]
        bounds = np.array(bounds)[signal_cls.i_intr]
        params0 = np.array(params)[signal_cls.i_intr]
        periodic = [5]
    else:
        # Create data analysis GW model with the full parameter vector
        params0 = np.array(params)
        periodic = [7, 8, 10]

    distribs = ['uniform' for name in names]
    channels = ['A', 'E', 'T']

    # Upper bounds
    lo = np.array([bound[0] for bound in bounds])
    # Lower bounds
    hi = np.array([bound[1] for bound in bounds])

    # ==================================================================================================================
    # Creation of data class instance
    # ==================================================================================================================
    dat_cls = datamodel.GaussianStationaryProcess(time_vect, y_list, mask)

    # ==================================================================================================================
    # Creation of PSD class
    # ==================================================================================================================
    psd_cls = psdmodel.PSDTheoretical(n, fs, channels=channels, scale=scale)
    spectrum_list = psd_cls.calculate(n)


    # ==================================================================================================================
    # Instanciate likelihood class
    # ==================================================================================================================
    model_cls = likelihoodmodel.LikelihoodModel(signal_cls,
                                                psd_cls,
                                                dat_cls,
                                                names=names,
                                                channels=channels,
                                                fmin=float(config['Model']['MinimumFrequency']),
                                                fmax=float(config['Model']['MaximumFrequency']),
                                                nsources=1,
                                                reduced=config['Model'].getboolean('reduced'),
                                                window = 'modified_hann',
                                                n_wind=int(config["TimeWindowing"]["DecayNumber"]),
                                                n_wind_psd=int(config["TimeWindowing"]["DecayNumberPSD"]),
                                                imputation=config['Sampler'].getboolean('MissingDataImputation'),
                                                psd_estimation=config['Sampler'].getboolean('PSDEstimation'),
                                                normalized=config['Model'].getboolean('normalized'),
                                                n_update=int(config['Sampler']['AuxiliaryParameterUpdateNumber']))


    # ==================================================================================================================
    # Test of likelihood calculation
    # ==================================================================================================================
    # Full parameter vector
    t1 = time.time()
    test0 = model_cls.log_likelihood(params0) + model_cls.log_norm
    t2 = time.time()
    print("Likelihood computation time: " + str(t2 - t1))

    # ==================================================================================================================
    # Creation of sampler class instance and start the Monte-Carlo sampling
    # ==================================================================================================================
    print("Chosen sampler: " + config["Sampler"]["Type"])
    np.random.seed(int(config["Sampler"]["RandomSeed"]))
    n_save = int(config['Sampler']['SavingNumber'])

    now = datetime.datetime.now()
    prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_")
    out_dir = config["OutputData"]["DirectoryPath"]
    print("start sampling...")

    def stop_and_save(results, args=None, rstate=None, M=None, return_vals=False):

        print("Save data after reaching " + str(results.samples.shape[0]) + " samples.")
        if (results.samples.shape[0] % n_save == 0) & (results.samples.shape[0] != 0):
            df = pd.DataFrame(results.samples[-n_save:, :], columns=names)
            df.to_hdf(out_dir + prefix + 'chains_temp.hdf5', 'chain', append=True, mode='a', format='table')

        return dynesty.dynamicsampler.stopping_function(results, args=args, rstate=rstate, M=M, return_vals=return_vals)

    if config["Sampler"]["Type"] == 'dynesty':
        nlive = np.int(config["Sampler"]["WalkerNumber"])

        def uniform2param(u):
            return posteriormodel.unwrap(u, lo, hi)

        # Instantiate the sampler
        # sampler = dynesty.DynamicNestedSampler(model_cls.log_likelihood,
        #                                        uniform2param,
        #                                        model_cls.ndim_tot,
        #                                        bound='multi',
        #                                        sample='slice',
        #                                        periodic=periodic)

        sampler = dynesty.NestedSampler(model_cls.log_likelihood, uniform2param, model_cls.ndim_tot,
                                        bound='multi', sample='slice', nlive=int(config["Sampler"]["WalkerNumber"]))

        # Instantiate the sampler
        # # Run the sampler
        # sampler.run_nested(nlive_init=int(config["Sampler"]["WalkerNumber"]), maxiter_init=None, maxcall_init=None,
        #                    dlogz_init=0.01, nlive_batch=500, wt_function=None, wt_kwargs={'pfrac': 1.0},
        #                    stop_kwargs={'pfrac': 1.0}, maxiter_batch=None, maxcall_batch=None,
        #                    maxiter=int(config["Sampler"]["MaximumIterationNumber"]),
        #                    print_progress=config['Sampler'].getboolean('printProgress'))

        # # Baseline run.
        # it = 0
        # it_max = int(int(config["Sampler"]["MaximumIterationNumber"])/2)
        # print("Initial sampling begins")
        # for results in sampler.sample_initial(nlive=500):
        #
        #     if (it % n_save == 0) & (n_save != 0):
        #         print("Saving results at iteration " + str(it))
        #         file_object = open(out_dir + prefix + "initial_save.p", "wb")
        #         pickle.dump(sampler.results, file_object)
        #         file_object.close()
        #
        #     it += 1
        #
        # # Add batches until we hit the stopping criterion.
        # it = 0
        # print("Batch sampling begins")
        # while True:
        #     stop = stopping_function(sampler.results)  # evaluate stop
        #     if not stop:
        #         logl_bounds = weight_function(sampler.results)  # derive bounds
        #         for results in sampler.sample_batch(logl_bounds=logl_bounds, maxiter=it_max):
        #             if (it % n_save == 0) & (n_save != 0):
        #                 print("Saving results at iteration " + str(it))
        #                 file_object = open(out_dir + prefix + "batch_save.p", "wb")
        #                 pickle.dump(sampler.results, file_object)
        #             it += 1
        #
        #         sampler.combine_runs()  # add new samples to previous results
        #     else:
        #         break

        # The main nested sampling loop.
        print("Main nested sampling loop starts...")
        for it, res in enumerate(sampler.sample(maxiter=int(config["Sampler"]["MaximumIterationNumber"]),
                                                save_samples=True)):
            # If it is a multiple of n_save, run the callback function
            if (it % n_save == 0) & (n_save != 0):
                print("Saving results at iteration " + str(it))
                file_object = open(out_dir + prefix + "initial_save.p", "wb")
                pickle.dump(sampler.results, file_object)
                file_object.close()
            else:
                pass

        # Adding the final set of live points.
        print("Adding the final set of live points...")
        for it_final, res in enumerate(sampler.add_live_points()):
            if (it_final % n_save == 0) & (n_save != 0):
                file_object = open(out_dir + prefix + "final_save.p", "wb")
                pickle.dump(sampler.results, file_object)
                file_object.close()
            else:
                pass

    elif config["Sampler"]["Type"] == 'ptemcee':

        # sampler = ptemcee.Sampler(int(config["Sampler"]["WalkerNumber"]),
        #                           model_cls.ndim_tot,
        #                           model_cls.log_likelihood,
        #                           posteriormodel.logp,
        #                           ntemps=int(config["Sampler"]["TemperatureNumber"]),
        #                           loglargs=(spectrum_list, y_fft_list),
        #                           logpargs=(lo, hi))
        sampler = samplers.ExtendedPTMCMC(int(config["Sampler"]["WalkerNumber"]),
                                          model_cls.ndim_tot,
                                          model_cls.log_likelihood,
                                          posteriormodel.logp,
                                          ntemps=int(config["Sampler"]["TemperatureNumber"]),
                                          logpargs=(lo, hi))

        result = sampler.run(int(config["Sampler"]["MaximumIterationNumber"]), n_save,
                             int(config["Sampler"]["thinningNumber"]), callback=None, pos0=None,
                             save_path=out_dir + prefix)
        # Run the sampler
        # result = sampler.run_mcmc(p0=None, iterations=int(config["Sampler"]["MaximumIterationNumber"]),
        #                           thin=int(config["Sampler"]["thinningNumber"]),
        #                           storechain=True, adapt=False, swap_ratios=False)
    # pos, lnlike0, lnprob0
    # ==================================================================================================================
    # Saving the data finally
    # ==================================================================================================================
    fh5 = h5py.File(out_dir + prefix + config["OutputData"]["FileSuffix"], 'w')
    if config["Sampler"]["Type"] == 'dynesty':
        fh5.create_dataset("chains/chain", data=sampler.results.samples)
        fh5.create_dataset("chains/logl", data=sampler.results.logl)
        fh5.create_dataset("chains/logwt", data=sampler.results.logwt)
        fh5.create_dataset("chains/logvol", data=sampler.results.logvol)
        fh5.create_dataset("chains/logz", data=sampler.results.logz)
        pickle.dump(sampler.results, open(out_dir + prefix + "save.p", "wb"))

    elif config["Sampler"]["Type"] == 'ptemcee':
        fh5.create_dataset("chains/chain", data=sampler.chain)
        fh5.create_dataset("chains/lnlike", data=result[1])
        fh5.create_dataset("chains/lnprob", data=result[2])
        fh5.create_dataset("temperatures/beta_hist", data=sampler._beta_history)

    fh5.close()






