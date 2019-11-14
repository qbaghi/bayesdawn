# Standard useful python module
import numpy as np
import time
from scipy import linalg
# LDC modules
import tdi
# MC Sampler modules
import dynesty
from bayesdawn.utils import physics
from bayesdawn.waveforms import lisaresp
from bayesdawn import likelihoodmodel


def prior_transform(theta_u, lower_bound, upper_bound):

    # order in theta
     # 0   1   2   3     4      5     6  7    8    9    10
    # [Mc, q, tc, chi1, chi2, logDL, ci, sb, lam, psi, phi0]
    theta = lower_bound + (upper_bound - lower_bound) * theta_u

    return theta


if __name__ == '__main__':

    # Standard modules
    from scipy import signal
    import datetime
    import pickle
    from bayesdawn import samplers, posteriormodel, datamodel, psdmodel
    from bayesdawn.utils import loadings, preprocess
    # For parallel computing
    # from multiprocessing import Pool, Queue
    # LDC tools
    import lisabeta.lisa.ldctools as ldctools
    # Plotting modules
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # FTT modules
    import fftwisdom
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import fft, ifft
    import configparser
    from optparse import OptionParser
    from bayesdawn.gaps import gapgenerator

    # ==================================================================================================================
    # Load configuration file
    # ==================================================================================================================
    parser = OptionParser(usage="usage: %prog [options] YYY.txt", version="10.28.2018, Quentin Baghi")
    (options, args) = parser.parse_args()
    if not args:
        config_file = "../configs/config_ldc.ini"
    else:
        config_file = args[0]
    # ==================================================================================================================
    config = configparser.ConfigParser()
    config.read(config_file)
    fftwisdom.load_wisdom()

    # ==================================================================================================================
    # Unpacking the hdf5 file and getting data and source parameters
    # ==================================================================================================================
    p, td = loadings.load_ldc_data(config["InputData"]["FilePath"])
    del_t = float(p.get("Cadence"))
    tobs = float(p.get("ObservationDuration"))

    # ==================================================================================================================
    # Get the parameters
    # ==================================================================================================================
    # Get parameters as an array from the hdf5 structure (table)
    p_sampl = physics.get_params(p, sampling=True)
    tc = p_sampl[2]

    # ==================================================================================================================
    # Pre-processing data: anti-aliasing and filtering
    # ==================================================================================================================
    if config['InputData'].getboolean('trim'):
        i1 = np.int(config["InputData"].getfloat("StartTime") / del_t)
        i2 = np.int(config["InputData"].getfloat("EndTime") / del_t)
        t_offset = 52.657 + config["InputData"].getfloat("StartTime")
        tobs = (i2 - i1) * del_t
    else:
        i1 = 0
        i2 = np.int(td.shape[0])
        t_offset = 52.657

    tm, Xd, Yd, Zd, q = preprocess.preprocess(config, td, i1, i2, scale=config["InputData"].getfloat("rescale"))

    # ==================================================================================================================
    # Introducing gaps if requested
    # ==================================================================================================================
    if config["TimeWindowing"].getboolean("Gaps"):

        if config["TimeWindowing"]["GapType"] == 'single':
            nd = [np.int(config["TimeWindowing"].getfloat("GapStartTime")/del_t)]
            nf = [np.int(config["TimeWindowing"].getfloat("GapEndTime")/del_t)]

        else:
            nd, nf = gapgenerator.generategaps(tm.shape[0], 1/del_t, config["TimeWindowing"].getint("GapNumber"),
                                               config["TimeWindowing"].getfloat("GapDuration"),
                                               gap_type=config["TimeWindowing"]["GapType"],
                                               f_gaps=config["TimeWindowing"].getfloat("GapFrequency"),
                                               wind_type='rect', std_loc=0, std_dur=0)

        wd = gapgenerator.windowing(nd, nf, tm.shape[0], window=config["TimeWindowing"]["WindowType"],
                                    n_wind=config["TimeWindowing"].getint("DecayNumber"))
        mask = gapgenerator.windowing(nd, nf, tm.shape[0], window='rect')
        wd_full = gapgenerator.modified_hann(tm.shape[0],
                                             n_wind=np.int((config["InputData"].getfloat("EndTime") - tc) / (2*del_t)))
    else:
        wd = gapgenerator.modified_hann(tm.shape[0],
                                        n_wind=np.int((config["InputData"].getfloat("EndTime") - tc) / (2*del_t)))
        wd_full = wd[:]
        # wd = signal.tukey(Xd.shape[0], alpha=(config["InputData"].getfloat("EndTime") - tc) / tobs, sym=True)

    # ==================================================================================================================
    # Now we get extract the data and transform it to frequency domain
    # ==================================================================================================================
    resc = Xd.shape[0]/np.sum(wd)
    XDf = fft(wd * Xd) * del_t * q * resc
    YDf = fft(wd * Yd) * del_t * q * resc
    ZDf = fft(wd * Zd) * del_t * q * resc

    freqD = np.fft.fftfreq(len(tm), del_t * q)
    # Convert Michelson TDI to A, E, T (freq. domain)
    ADf, EDf, TDf = ldctools.convert_XYZ_to_AET(XDf, YDf, ZDf)
    # Convert Michelson TDI to A, E, T (time domain)
    ad, ed, td = ldctools.convert_XYZ_to_AET(Xd, Yd, Zd)

    # Restrict the frequency band to high SNR region
    inds = np.where((float(config['Model']['MinimumFrequency']) <= freqD)
                    & (freqD <= float(config['Model']['MaximumFrequency'])))[0]
    aet = [ADf[inds], EDf[inds], TDf[inds]]
    # Restriction of sampling parameters to instrinsic ones Mc, q, tc, chi1, chi2, np.sin(bet), lam
    i_sampl_intr = [0, 1, 2, 3, 4, 7, 8]
    print("=================================================================")

    # Consider only A and E TDI data in frequency domain
    dataAE = [ADf[inds], EDf[inds]]
    # And in time domain
    data_ae_time = [ad, ed]
    fftwisdom.save_wisdom()

    # ==================================================================================================================
    # Auxiliary parameter classes
    # ==================================================================================================================
    if config["PSD"].getboolean("estimation"):
        print("PSD estimation enabled.")
        psd_cls = psdmodel.PSDSpline(tm.shape[0], 1 / del_t, J=config["PSD"].getint("knotNumber"),
                                     D=config["PSD"].getint("SplineOrder"),
                                     fmin=1 / (del_t * tm.shape[0]) * 1.05,
                                     fmax=1 / (del_t * 2))
        psd_cls.estimate(ad)
        sa = psd_cls.calculate(freqD[inds])
    else:
        psd_cls = None
        # One-sided PSD
        sa = tdi.noisepsd_AE(freqD[inds], model='Proposal', includewd=None)

    if config["Imputation"].getboolean("imputation"):
        print("Missing data imputation enabled.")
        data_cls = datamodel.GaussianStationaryProcess(data_ae_time, mask, method=config["Imputation"]['method'],
                                                       na=150, nb=150, p=config["Imputation"].getint("precondOrder"),
                                                       tol=config["Imputation"].getfloat("tolerance"),
                                                       n_it_max=config["Imputation"].getint("maximumIterationNumber"))

        data_cls.compute_preconditioner(psd_cls.calculate_autocorr(data_cls.n)[0:data_cls.n_max])

    else:
        data_cls = None


    # ==================================================================================================================
    # Instantiate likelihood class
    # ==================================================================================================================
    ll_cls = likelihoodmodel.LogLike(data_ae_time, sa, freqD[inds], tobs, del_t * q,
                                     normalized=config['Model'].getboolean('normalized'),
                                     t_offset=t_offset, channels=[1, 2],
                                     scale=config["InputData"].getfloat("rescale"),
                                     model_cls=data_cls, psd_cls=psd_cls,
                                     n_update=config['Sampler'].getint('AuxiliaryParameterUpdateNumber'),
                                     wd=wd,
                                     wd_full=wd_full)

    # ==================================================================================================================
    # Testing likelihood
    # ==================================================================================================================
    t1 = time.time()
    if config['Model'].getboolean('reduced'):
        aft, eft = ll_cls.compute_signal_reduced(p_sampl[i_sampl_intr])
    else:
        aft, eft = ll_cls.compute_signal(p_sampl)
    t2 = time.time()
    print('Waveform Calculated in ' + str(t2 - t1) + ' seconds.')

    # ==================================================================================================================
    # Get parameter names and bounds
    # ==================================================================================================================
    # Get all parameter name keys
    names = [key for key in config['ParametersLowerBounds']]
    # Get prior bound values
    bounds = [[config['ParametersLowerBounds'].getfloat(name), config['ParametersUpperBounds'].getfloat(name)]
              for name in names]
    lower_bounds = np.array([bound[0] for bound in bounds])
    upper_bounds = np.array([bound[1] for bound in bounds])
    # Print it
    [print("Bounds for parameter " + names[i] + ": " + str(bounds[i])) for i in range(len(names))]
    # If reduced likelihood is chosen, sample only intrinsic parameters
    if config['Model'].getboolean('reduced'):
        names = np.array(names)[i_sampl_intr]
        lower_bounds = lower_bounds[i_sampl_intr]
        upper_bounds = upper_bounds[i_sampl_intr]
        log_likelihood = ll_cls.log_likelihood_reduced
        print("Reduced likelihood chosen, with intrinsic parameters: ")
        print(names)
        # Mc, q, tc, chi1, chi2, np.sin(bet), lam
    else:
        log_likelihood = ll_cls.log_likelihood
        # Mc, q, tc, chi1, chi2, np.log10(DL), np.cos(incl), np.sin(bet), lam, psi, phi0
        periodic = [6]

    # ==================================================================================================================
    # Prepare data to save during sampling
    # ==================================================================================================================
    # Append current date and prefix to file names
    now = datetime.datetime.now()
    prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_")
    out_dir = config["OutputData"]["DirectoryPath"]
    print("Chosen saving path: " + out_dir)
    # Save the configuration file used for the run
    with open(out_dir + prefix + "config.ini", 'w') as configfile:
        config.write(configfile)

    # ==================================================================================================================
    # Start sampling
    # ==================================================================================================================
    # Set seed
    np.random.seed(int(config["Sampler"]["RandomSeed"]))
    # Multiprocessing pool
    # pool = Pool(4)

    if config["Sampler"]["Type"] == 'dynesty':
        # Instantiate sampler
        if config['Sampler'].getboolean('dynamic'):
            sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim=len(names),
                                                   bound='multi', sample='slice', periodic=periodic,
                                                   ptform_args=(lower_bounds, upper_bounds), ptform_kwargs=None)
                                                   # pool=pool, queue_size=4)
            # # Start run
            # dsampl.run_nested(dlogz_init=0.01, nlive_init=config["Sampler"].getint("WalkerNumber"),
            # nlive_batch=500, wt_kwargs={'pfrac': 1.0},
            #                   stop_kwargs={'pfrac': 1.0})
        else:
            # Instantiate sampler
            sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=len(names),
                                            bound='multi', sample='slice', periodic=periodic,
                                            ptform_args=(lower_bounds, upper_bounds),
                                            nlive=int(config["Sampler"]["WalkerNumber"]))

        print("n_live_init = " + config["Sampler"]["WalkerNumber"])
        print("Save samples every " + config['Sampler']['SavingNumber'] + " iterations.")

        samplers.run_and_save(sampler, nlive=config["Sampler"].getint("WalkerNumber"),
                              n_save=config['Sampler'].getint('SavingNumber'),
                              n_iter=config["Sampler"].getint("MaximumIterationNumber"),
                              file_path=out_dir + prefix, dynamic=config['Sampler'].getboolean('dynamic'))

    elif config["Sampler"]["Type"] == 'ptemcee':

        sampler = samplers.ExtendedPTMCMC(int(config["Sampler"]["WalkerNumber"]),
                                          len(names),
                                          log_likelihood,
                                          posteriormodel.logp,
                                          ntemps=int(config["Sampler"]["TemperatureNumber"]),
                                          logpargs=(lower_bounds, upper_bounds))

        result = sampler.run(int(config["Sampler"]["MaximumIterationNumber"]),
                             config['Sampler'].getint('SavingNumber'),
                             int(config["Sampler"]["thinningNumber"]),
                             callback=None, pos0=None,
                             save_path=out_dir + prefix)

    # # ==================================================================================================================
    # # Plotting
    # # ==================================================================================================================
    # from plottools import presets
    # presets.plotconfig(ctype='frequency', lbsize=16, lgsize=14)
    #
    # # Frequency plot
    # fig1, ax1 = plt.subplots(nrows=2, sharex=True, sharey=True)
    # ax1[0].semilogx(freqD, np.real(ADf))
    # ax1[0].semilogx(freqD[inds], np.real(aft), '--')
    # ax1[1].semilogx(freqD, np.imag(ADf))
    # ax1[1].semilogx(freqD[inds], np.imag(aft), '--')
    # plt.show()


