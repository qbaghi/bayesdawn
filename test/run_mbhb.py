# Standard useful python module
import time
import numpy as np
from bayesdawn import likelihoodmodel
# LDC modules
# MC Sampler modules
from bayesdawn.utils import physics


def prior_transform(theta_u, lower_bound, upper_bound):
    # order in theta
    # 0  1  2   3     4    5     6  7    8    9    10
    # [Mc, q, tc, chi1, chi2, logDL, ci, sb, lam, psi, phi0]
    theta = lower_bound + (upper_bound - lower_bound) * theta_u

    return theta


if __name__ == '__main__':

    # Bayesdawn modules
    from bayesdawn import datamodel, psdmodel, samplers, posteriormodel
    from bayesdawn.utils import loadings, preprocess
    from bayesdawn.postproc import postprocess
    # For parallel computing
    # from multiprocessing import Pool, Queue
    import ptemceeg
    # LDC tools
    import lisabeta.lisa.ldctools as ldctools
    # Loading modules
    import datetime
    import configparser
    from optparse import OptionParser
    import os
    # FTT modules
    import fftwisdom
    import pyfftw
    pyfftw.interfaces.cache.enable()

    # =========================================================================
    # Load configuration file
    # =========================================================================
    parser = OptionParser(usage="usage: %prog [options] YYY.txt",
                          version="10.28.2018, Quentin Baghi")
    (options, args) = parser.parse_args()
    if not args:
        config_file = "../configs/config_ldc.ini"
    else:
        config_file = args[0]
    # =========================================================================
    config = configparser.ConfigParser()
    config.read(config_file)
    fftwisdom.load_wisdom()

    # =========================================================================
    # Unpacking the hdf5 file and getting data and source parameters
    # =========================================================================
    p, tdi_data = loadings.load_ldc_data(config["InputData"]["FilePath"])

    # =========================================================================
    # Pre-processing data: anti-aliasing, filtering and rescaling
    # =========================================================================
    preproc_data = preprocess.preprocess_ldc_data(p, tdi_data, config)
    tm, xd, yd, zd, q, t_offset, tobs, del_t, p_sampl = preproc_data

    # =========================================================================
    # Introducing gaps if requested
    # =========================================================================
    wd, wd_full, mask = loadings.load_gaps(config, tm)
    print("Ideal decay number: "
          + str(np.int((config["InputData"].getfloat("EndTime")
                        - p_sampl[2]) / (2 * del_t))))

    # =========================================================================
    # Now we get extract the data and transform it to frequency domain
    # =========================================================================
    freq_d = np.fft.fftfreq(len(tm), del_t * q)
    # Convert Michelson TDI to a_mat, E, T (time domain)
    ad, ed, td = ldctools.convert_XYZ_to_AET(xd, yd, zd)

    # Restrict the frequency band to high SNR region, and exclude distorted
    # frequencies due to gaps
    f_minimum = config['Model'].getfloat('MinimumFrequency')
    f_maximum = config['Model'].getfloat('MaximumFrequency')
    
    include_gaps = config["TimeWindowing"].getboolean('gaps')
    imputation = config["Imputation"].getboolean('imputation')
    frequency_windowing = config["Model"].getboolean('accountForDistortions')

    if (include_gaps) & (not imputation) & frequency_windowing:
        f1, f2 = physics.find_distorted_interval(mask, p_sampl, t_offset,
                                                 del_t, margin=0.5)
        f1 = np.max([f1, 0])
        f2 = np.min([f2, 1 / (2 * del_t)])
        # inds = np.where((f_minimum <= freq_d)
        #                 & (freq_d <= f_maximum)
        #                 & ((freq_d <= f1) | (freq_d >= f2)))[0]
        inds = np.where((f_minimum <= freq_d) & (freq_d <= f_maximum) 
                        & (freq_d >= f2))[0]
    else:
        inds = np.where((f_minimum <= freq_d) & (freq_d <= f_maximum))[0]

    # Restriction of sampling parameters to instrinsic ones
    # Mc, q, tc, chi1, chi2, np.sin(bet), lam
    i_sampl_intr = [0, 1, 2, 3, 4, 7, 8]
    print("=================================================================")

    # And in time domain, including mask
    data_ae_time = [mask * ad, mask * ed]

    # =========================================================================
    # Auxiliary parameter classes
    # =========================================================================
    scale = config["InputData"].getfloat("rescale")
    psd_estimation = config["PSD"].getboolean("estimation")
    psd_model = config["PSD"].get("model")
    imputation = config["Imputation"].getboolean("imputation")
    if psd_estimation | (psd_model == 'spline'):
        print("PSD estimation enabled.")
        f_knots = np.array([1e-5, 5e-4, 5e-3, 1e-2, 3e-2, 4e-2, 
                            4.5e-2, 4.7e-2, 4.8e-2, 4.9e-2, 4.97e-2, 4.99e-2])
        # n_knots = config["PSD"].getint("knotNumber")
        n_knots = f_knots.shape[0]
        psd_cls = [psdmodel.PSDSpline(tm.shape[0], 1 / del_t,
                                      n_knots=n_knots,
                                      d=config["PSD"].getint("SplineOrder"),
                                      fmin=1 / (del_t * tm.shape[0]) * 1.05,
                                      fmax=1 / (del_t * 2),
                                      f_knots=f_knots,
                                      ext=0)
                   for dat in data_ae_time]
        [psd_cls[i].estimate(data_ae_time[i]) for i in range(len(psd_cls))]
        sn = [psd.calculate(freq_d[inds]) for psd in psd_cls]

    else:
        # Compute the spectra once for all
        psd_cls = [psdmodel.PSDTheoretical(tm.shape[0], 1 / del_t, ch,
                                           scale=scale, fmin=None, fmax=None)
                   for ch in ['A', 'E']]
        sn = [psd.calculate(freq_d[inds]) for psd in psd_cls]
        # Then set the PSD class to None to prevent its update
        # psd_cls = None

        # psd_cls = None
        # # One-sided PSD
        # sa = tdi.noisepsd_AE(freq_d[inds], model='Proposal', includewd=None)
        # sn = [sa, sa]
        # Select a chunk of data free of signal
        # t_end = config["InputData"].getfloat("EndTime")
        # ind_noise = np.where(tdi_data[t_end > tdi_data[:, 0], 0])[0]
        # ad_noise, ed_noise, td_noise = ldctools.convert_XYZ_to_AET(
        #     tdi_data[ind_noise, 1],
        #     tdi_data[ind_noise, 2],
        #     tdi_data[ind_noise, 3])
        # data_ae_noise = [ad_noise, ed_noise]
        # del ad_noise, ed_noise, td_noise
        #
        # psd_cls = [psdmodel.PSDSpline(data_ae_noise[0].shape[0], 1 / del_t,
        #                               n_knots=config["PSD"].getint("knotNumber"),
        #                               d=config["PSD"].getint("SplineOrder"),
        #                               fmin=1 / (del_t * tm.shape[0]) * 1.05,
        #                               fmax=1 / (del_t * 2))
        #            for dat in data_ae_time]
        # [psd_cls[i].estimate(data_ae_noise[i]) for i in range(len(psd_cls))]
        # sn = [psd.calculate(freq_d[inds]) for psd in psd_cls]

    if imputation:
        print("Missing data imputation enabled.")
        data_mean = [np.zeros(dat.shape[0]) for dat in data_ae_time]
        data_cls = datamodel.GaussianStationaryProcess(
            data_mean, mask, psd_cls,
            method=config["Imputation"]['method'],
            na=150, nb=150, p=config["Imputation"].getint("precondOrder"),
            tol=config["Imputation"].getfloat("tolerance"),
            n_it_max=config["Imputation"].getint("maximumIterationNumber"))
        # Pre-compute quantities that do not get updated very often
        data_cls.compute_offline()
        if config["Imputation"]['method'] == 'PCG':
            data_cls.compute_preconditioner()

    else:
        data_cls = None

    # =========================================================================
    # Instantiate likelihood class
    # =========================================================================
    normalized = config['Model'].getboolean('normalized')

    ll_cls = likelihoodmodel.LogLike(data_ae_time, sn, inds, tobs, del_t * q,
                                     normalized=normalized,
                                     t_offset=t_offset,
                                     channels=[1, 2],
                                     scale=scale,
                                     model_cls=data_cls,
                                     psd_cls=psd_cls,
                                     wd=wd,
                                     wd_full=wd_full)

    # =========================================================================
    # Testing likelihood
    # =========================================================================
    par_aux0 = np.concatenate(ll_cls.data_dft + sn)
    t1 = time.time()
    if config['Model'].getboolean('reduced'):
        aft, eft = ll_cls.compute_signal_reduced(p_sampl[i_sampl_intr],
                                                 ll_cls.data_dft,
                                                 sn)
    else:
        aft, eft = ll_cls.compute_signal(p_sampl)
    t2 = time.time()
    print('Waveform Calculated in ' + str(t2 - t1) + ' seconds.')

    # =========================================================================
    # Get parameter names and bounds
    # =========================================================================
    # Get all parameter name keys
    names = [key for key in config['ParametersLowerBounds']]
    # Get prior bound values
    bounds = [[config['ParametersLowerBounds'].getfloat(name),
               config['ParametersUpperBounds'].getfloat(name)]
              for name in names]
    lower_bounds = np.array([bound[0] for bound in bounds])
    upper_bounds = np.array([bound[1] for bound in bounds])
    # Print it
    [print("Bounds for parameter " + names[i] + ": " + str(bounds[i]))
     for i in range(len(names))]
    # If reduced likelihood is chosen, sample only intrinsic parameters
    reduced = config['Model'].getboolean('reduced')
    if reduced:
        names = np.array(names)[i_sampl_intr]
        lower_bounds = lower_bounds[i_sampl_intr]
        upper_bounds = upper_bounds[i_sampl_intr]
        log_likelihood = ll_cls.log_likelihood_reduced
        print("Reduced likelihood chosen, with intrinsic parameters: ")
        print(names)
        # Mc, q, tc, chi1, chi2, np.sin(bet), lam
    else:
        log_likelihood = ll_cls.log_likelihood
        # Mc, q, tc, chi1, chi2,
        # np.log10(DL), np.cos(incl), np.sin(bet), lam, psi, phi0
        periodic = [6]

    fftwisdom.save_wisdom()
    
    # =======================================================================
    # Prepare data to save during sampling
    # =======================================================================
    # Append current date and prefix to file names
    now = datetime.datetime.now()
    prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_")
    out_dir = config["OutputData"]["DirectoryPath"]
    print("Chosen saving path: " + out_dir)
    # Save the configuration file used for the run
    with open(out_dir + prefix + "config.ini", 'w') as configfile:
        config.write(configfile)

    # =========================================================================
    # Start sampling
    # =========================================================================
    if not config["Sampler"].getboolean("numpyParallel"):
        os.environ["OMP_NUM_THREADS"] = "1"
    threads = config["Sampler"].getint("threadNumber")
    # Set seed
    np.random.seed(int(config["Sampler"]["RandomSeed"]))
    # Multiprocessing pool
    # pool = Pool(4)

    if config["Sampler"]["Type"] == 'dynesty':
        # Instantiate sampler
        if config['Sampler'].getboolean('dynamic'):
            sampler = dynesty.DynamicNestedSampler(log_likelihood,
                                                   prior_transform,
                                                   ndim=len(names),
                                                   bound='multi',
                                                   sample='slice',
                                                   periodic=periodic,
                                                   ptform_args=(lower_bounds,
                                                                upper_bounds),
                                                   ptform_kwargs=None)
                                                   # pool=pool, queue_size=4)
        else:
            # Instantiate sampler
            nwalkers = config["Sampler"].getint("WalkerNumber")
            sampler = dynesty.NestedSampler(log_likelihood, prior_transform,
                                            ndim=len(names),
                                            bound='multi', sample='slice',
                                            periodic=periodic,
                                            ptform_args=(lower_bounds,
                                                         upper_bounds),
                                            nlive=nwalkers)

        print("n_live_init = " + config["Sampler"]["WalkerNumber"])
        print("Save samples every " + config['Sampler']['SavingNumber']
              + " iterations.")

        nlive = config["Sampler"].getint("WalkerNumber")
        n_save = config['Sampler'].getint('SavingNumber')
        n_iter = config["Sampler"].getint("MaximumIterationNumber")
        dynamic = config['Sampler'].getboolean('dynamic')
        samplers.run_and_save(sampler,
                              nlive=nlive,
                              n_save=n_save,
                              n_iter=n_iter,
                              file_path=out_dir + prefix,
                              dynamic=dynamic)

    elif config["Sampler"]["Type"] == 'ptemcee':

        initialization = config["Sampler"].get("Initialization")
        nwalkers = config["Sampler"].getint("WalkerNumber")
        ntemps = config["Sampler"].getint("TemperatureNumber")
        n_callback = config['Sampler'].getint('AuxiliaryParameterUpdate')
        n_start_callback = config['Sampler'].getint('AuxiliaryParameterStart')
        n_iter = config["Sampler"].getint("MaximumIterationNumber")
        n_save = config['Sampler'].getint('SavingNumber')
        n_thin = config["Sampler"].getint("thinningNumber")
        gibbsargs = []
        gibbskwargs = {'reduced': reduced, 
                       'update_mis': imputation,
                       'update_psd': psd_estimation}
        
        # =====================================================================
        # Initialization of parameter state
        # =====================================================================
        if initialization == 'prior':
            pos0 = np.random.uniform(lower_bounds, upper_bounds,
                                     size=(ntemps, nwalkers, 
                                           len(upper_bounds)))
        elif initialization == 'file':
            run_config_path = config["InputData"].get("initialRunPath")
            n_burn = config["Sampler"].getint("burnin")
            names, par0, chain0, lnprob, sampler_type = postprocess.get_simu_parameters(
                run_config_path, intrinsic=False)
            i_map = np.where(lnprob[0, :, n_burn:] == np.max(lnprob[0, :, n_burn:]))
            # p_map = chain0[0, :, n_burn:, :][i_map[0][0], i_map[1][0]]
            # pos0 = chain0[:, :, -1, :]
            pos0 = chain0[:, :, i_map[1][0], :]
            # Deleting useless variables
            del chain0, lnprob
        
        # Choosing parallelization process
        multiproc = config["Sampler"].get("multiproc")
        if multiproc == 'ray':
            from ray.util.multiprocessing.pool import Pool
            pool = Pool(threads)
        else:
            pool = None

        if (not psd_estimation) & (not imputation):

            sampler = samplers.ExtendedPTMCMC(nwalkers, 
                                              len(names),
                                              log_likelihood,
                                              posteriormodel.logp,
                                              ntemps=ntemps,
                                              threads=threads,
                                              pool=pool,
                                              loglargs=[par_aux0],
                                              logpargs=(lower_bounds,
                                                        upper_bounds))
            t1 = time.time()
            result = sampler.run(int(config["Sampler"]["MaximumIterationNumber"]),
                                 config['Sampler'].getint('SavingNumber'),
                                 int(config["Sampler"]["thinningNumber"]),
                                 callback=None,
                                 n_callback=n_callback,
                                 n_start_callback=n_start_callback,
                                 pos0=pos0,
                                 save_path=out_dir + prefix)
            t2 = time.time()
            
        else:
            
            # Define the function updating auxiliary parameters
            def gibbs(x, x2, **kwargs):
                return ll_cls.update_auxiliary_params(x, x2, **kwargs)
        
            sampler = ptemceeg.Sampler(
                nwalkers, len(names), 
                log_likelihood, 
                posteriormodel.logp,
                ntemps=ntemps,
                gibbs=gibbs,
                dim2=par_aux0.shape[0],
                threads=threads,
                pool=pool,
                loglargs=[],
                loglkwargs={},
                logpargs=(lower_bounds, upper_bounds),
                gibbsargs=gibbsargs,
                gibbskwargs=gibbskwargs)
            
            print("Start MC sampling...")
            t1 = time.time()
            aux0 = np.full((nwalkers, par_aux0.shape[0]), par_aux0, 
                           dtype=np.complex128)
            result = sampler.run(n_iter, n_save, n_thin,
                                 n_update=n_callback,
                                 n_start_update=n_start_callback,
                                 pos0=pos0,
                                 aux0=aux0,
                                 save_path=out_dir + prefix,
                                 verbose=2)
            t2 = time.time()
        print("MC completed in " + str(t2 - t1) + " seconds.")
