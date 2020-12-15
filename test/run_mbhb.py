# Standard useful python module
import time
import numpy as np
import copy
# Bayesdawn modules
from bayesdawn.algebra import matrixalgebra
# LISABeta and LDC tools
import lisabeta.lisa.ldctools as ldctools
import lisabeta.lisa.lisa as lisa
import lisabeta.tools.pyspline as pyspline


def shiftTime(f,Xf,delay):
    return Xf*np.exp(-2j*np.pi*f*delay)

def generate_lisa_signal(wftdi, freq=None, channels=[1, 2, 3]):
    '''
    Args:
      wftdi              # dictionary output from GenerateLISATDI
      freq               # numpy array of freqs on which to sample waveform (or None)
    Return:
      dictionary with output TDI channel data yielding numpy complex data arrays 
    '''

    fs = wftdi['freq']
    amp = wftdi['amp']
    phase = wftdi['phase']
    tr_list = [wftdi['transferL' + str(i)] for i in channels]

    if freq is not None:
        amp = pyspline.resample(freq, fs, amp)
        phase = pyspline.resample(freq, fs, phase)
        tr_list = [pyspline.resample(freq, fs, tr) for tr in tr_list]
    else:
        freq=fs

    h = amp*np.exp(1j*phase)
    signal={'ch'+str(i): h * tr_list[i-1] for i in channels}
    signal['freq']=freq
    
    return signal

# Creating a waveform generator from a vector of parameters
def lisabeta_waveform(params, freq, 
                      minf=1e-5, 
                      maxf=0.1, 
                      tshift=0,
                      channels=[1, 2],
                      tmin=None, 
                      tmax=None,
                      scale=1.0):
    # params = m1s, m2s, a1, a2, tc, DL, inc, phi0, lam, beta, psi
    # Convert vector to dictionary
    params_dic = ldctools.make_params_dict(params)
    # Compute waveform on coarse grid
    wftdi=lisa.GenerateLISATDI_SMBH(params_dic, 
                                    minf=minf, 
                                    maxf=maxf,
                                    tmin=tmin,
                                    tmax=tmax,
                                    TDI='TDIAET', 
                                    order_fresnel_stencil=0, 
                                    TDIrescaled=False, 
                                    approximant='IMRPhenomD')
    # Resample on finer grid
    sig = generate_lisa_signal(wftdi[(2,2)], freq=freq, channels=channels)
    return [shiftTime(freq, sig['ch' + str(i)], tshift).conj()*scale for i in channels]


def compute_signal(p_sampl, freq,
                   minf=1e-5, 
                   maxf=0.1,
                   t_offset=0.0,
                   channels=[1, 2],
                   scale=1.0):
    
    # Convert likelihood parameters into waveform-compatible parameters
    params = physics.like_to_waveform(p_sampl)
    # Waveform from LISABeta
    return lisabeta_waveform(params, freq,
                             minf=minf, 
                             maxf=maxf,
                             tshift=t_offset,
                             channels=channels,
                             scale=scale)
    

def design_matrix(params_intr, freq,
                  minf=1e-5, 
                  maxf=0.1, 
                  t_offset=0.0,
                  channels=[1, 2],
                  scale=1.0,
                  i_intr=[0, 1, 2, 3, 4, 8, 9],
                  i_dist=5,
                  i_inc=6,
                  i_phi0=7,
                  i_psi=10,
                  i_tc=4):
    """
    Design matrix of basis functions for F-statistics computation.

    Parameters
    ----------
    params_intr : ndarray
        intrinsic parameters m1, m2, chi1, chi2, tc, lambd, beta
    freq : ndarray
        frequency array where to compute the basis.
    tobs : float
        observation time [s]
    tref : int, optional
        [description], by default 0
    minf : [type], optional
        [description], by default 1e-5
    maxf : float, optional
        [description], by default 0.1
    t_offset : float, optional
        [description], by default 0.0
    channels : list, optional
        [description], by default [1, 2]
    i_intr : list, optional
        [description], by default [0, 1, 2, 3, 4, 8, 9]
    i_dist : int, optional
        [description], by default 5
    i_inc : int, optional
        [description], by default 6
    i_phi0 : int, optional
        [description], by default 7
    i_psi : int, optional
        [description], by default 10
    i_tc : int, optional
        [description], by default 4


    Returns
    -------
    mat_list : list
        list of design matrices for each channel
        
    Reference
    ---------
    Marsat, Sylvain and Baker, John G, Fourier-domain modulations and delays of 
    gravitational-wave signals, 2018
    Neil J. Cornish and Kevin Shuman, Black Hole Hunting with LISA, 2020
    
    """

    # First column
    # m1, m2, chi1, chi2, tc, dist, inc, phi, lambd, beta, psi = params
    # Building the F-statistics basis
    
    params_0 = np.zeros(11)
    # Save intrinsic parameters
    params_0[i_intr] = params_intr
    # Luminosity distance (Mpc)
    params_0[i_dist] = 1e4
    # Inclination
    params_0[i_inc] = 0.5 * np.pi
    # 4 elements
    params_list = [copy.deepcopy(params_0) for j in range(4)]
    # First element (phi_c, phi) = (0, 0)
    params_list[0][i_phi0] = 0
    params_list[0][i_psi] = 0
    # Second element (phi_c, phi) = (pi/2, pi/4)
    params_list[1][i_phi0] = np.pi / 2
    params_list[1][i_psi] = np.pi / 4
    # Third element (phi_c, phi) = (3pi/4, 0)
    params_list[2][i_phi0] = 3 * np.pi / 4
    params_list[2][i_psi] = 0
    # Fourth element (phi_c, phi) = (pi/4, pi/4)
    params_list[3][i_phi0] = np.pi / 4
    params_list[3][i_psi] = np.pi / 4
    # Compute all 4 elements for all channels
    elements = [lisabeta_waveform(params, freq, 
                                  minf=minf, 
                                  maxf=maxf, 
                                  tshift=t_offset, 
                                  channels=channels,
                                 scale=scale)
                for params in params_list]
    # Compute the design matrices for each channel
    mat_list = [np.vstack([el[i] for el in elements]).T 
                for i in range(len(elements[0]))]

    return mat_list


def compute_signal_reduced(par_intr, freq, data_dft, sn,
                           minf=1e-5, maxf=0.1, t_offset=0.0,
                           channels=[1, 2], scale=1.0):
    """

    Parameters
    ----------
    par_intr : array_like
        vector of intrinsic waveform parameters in the following order:
        [Mc, q, tc, chi1, chi2, sb, lam]
    data_dft :  list[ndarray]
        List of windowed frequency-domain data restricted to the band 
        of interest
    sn : list of ndarrays
        list of noise PSDs computed at freq for each channel

    Returns
    -------
    ch_list : list[ndarrays]
        list of complex GW strains for each channel

    """

    # Transform parameters into waveform-compatible ones
    params_intr = physics.like_to_waveform_intr(par_intr)
    # Compute design matrices for all channels
    mat_list = design_matrix(params_intr, freq,
                             minf=minf, 
                             maxf=maxf, 
                             t_offset=t_offset,
                             channels=channels,
                            scale=scale)
    # Compute amplitudes
    amps = [matrixalgebra.gls(data_dft[i], mat_list[i], sn[i]) 
            for i in range(len(channels))]
    # Compute estimated signals
    ch_list = [np.dot(mat_list[i], amps[i]) for i in range(len(channels))]
    
    return ch_list


if __name__ == '__main__':

    # Bayesdawn modules
    from bayesdawn import datamodel, psdmodel, samplers, posteriormodel
    from bayesdawn import likelihoodmodel
    from bayesdawn.utils import loadings, preprocess, postprocess, physics
    # For parallel computing
    # from multiprocessing import Pool, Queue
    import ptemceeg
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
        config_file = "../configs/config_template.ini"
    else:
        config_file = args[0]
    # =========================================================================
    config = configparser.ConfigParser()
    config.read(config_file)
    fftwisdom.load_wisdom()

    # =========================================================================
    # Load and pre-process input data
    # =========================================================================
    if 'LDC' in config["InputData"]["FilePath"]:
        # Unpacking the hdf5 file and getting data and source parameters
        p, tdi_data = loadings.load_ldc_data(config["InputData"]["FilePath"])

        # Pre-processing data: anti-aliasing, filtering and rescaling
        preproc_data = preprocess.preprocess_ldc_data(p, tdi_data, config)
        tm, xd, yd, zd, q, tstart, tobs, del_t, p_sampl = preproc_data

        # Convert Michelson TDI to a_mat, E, T (time domain)
        ad, ed, td = ldctools.convert_XYZ_to_AET(xd, yd, zd)

    else:
        # Unpacking the hdf5 file and getting data and source parameters
        tvect, signal_list, noise_list, params, tstart, del_t, tobs = loadings.load_simulation(
            config["InputData"]["FilePath"])
        # Form data = signal + noise
        data_list = [signal_list[i] + noise_list[i] 
                    for i in range(len(signal_list))]
        # Convert waveform parameters to sampling parameters
        p_sampl = physics.waveform_to_like(params)

        # Trim the data if needed
        if config['InputData'].getboolean('trim'):
            # i1 = np.int(config["InputData"].getfloat("StartTime") / del_t)
            # i2 = np.int(config["InputData"].getfloat("EndTime") / del_t)
            i1 = np.argmin(np.abs(config["InputData"].getfloat("StartTime") - tvect))
            i2 = np.argmin(np.abs(config["InputData"].getfloat("EndTime") - tvect))
            tobs = (i2 - i1) * del_t
            tstart = tvect[i1]
        else:
            i1 = 0
            i2 = np.int(tvect.shape[0])
            
        # Downsampling and filtering if needed
        scale = config["InputData"].getfloat("rescale")
        dec = config['InputData'].getboolean('decimation')

        if dec:
            fc = config['InputData'].getfloat('filterFrequency')
            q = config['InputData'].getint('decimationFactor')
            data_pre_list = [preprocess.filter(dat, fc, del_t)
                            for dat in data_list]
        else:
            data_pre_list = data_list[:]
            q = 1

        # Preprocessed data
        data_pre_list = [dat[i1:i2:q]*scale for dat in data_pre_list]
        tm = tvect[i1:i2:q]
        ad, ed, td = data_pre_list

    # =========================================================================
    # Load mask if needed
    # =========================================================================
    # Introducing gaps if requested
    wd, wd_full, mask = loadings.load_gaps(config, tm)
    print("Ideal decay number: "
        + str(np.int((config["InputData"].getfloat("EndTime")
                        - p_sampl[2]) / (2 * del_t))))
    # And in time domain, including mask
    data_ae_time = [mask * ad, mask * ed]

    # =========================================================================
    # Transform data to frequency domain
    # =========================================================================
    freq_d = np.fft.fftfreq(len(tm), del_t * q)
    # Restrict the frequency band to high SNR region, and exclude distorted
    # frequencies due to gaps
    f_minimum = config['Model'].getfloat('MinimumFrequency')
    f_maximum = config['Model'].getfloat('MaximumFrequency')
    
    include_gaps = config["TimeWindowing"].getboolean('gaps')
    imputation = config["Imputation"].getboolean('imputation')
    frequency_windowing = config["Model"].getboolean('accountForDistortions')

    if (include_gaps) & (not imputation) & frequency_windowing:
        f1, f2 = physics.find_distorted_interval(mask, p_sampl, tstart,
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

    if imputation:
        print("Missing data imputation enabled.")
        data_mean = [np.zeros(dat.shape[0]) for dat in data_ae_time]
        data_cls = datamodel.GaussianStationaryProcess(
            data_mean, mask, psd_cls,
            method=config["Imputation"]['method'],
            precond=config["Imputation"]['precond'],
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
    # Normalization of DFT to account for windowing
    normalized = config['Model'].getboolean('normalized')
    reduced = config['Model'].getboolean('reduced')
    # Whether to apply the gap distortion to the waveform
    gap_convolution = config['Model'].getboolean('gapConvolution')
    # Arguments of waveform generator
    signal_kwargs = {"minf": 1e-5, 
                    "maxf": 0.1,
                    "t_offset": tstart,
                    "channels": [1, 2],
                    "scale": scale}
    # Likelihood definition
    ll_cls = likelihoodmodel.LogLike(data_ae_time, sn, inds, tobs, del_t * q,
                                    compute_signal,
                                    compute_signal_reduced,
                                    signal_args=[],
                                    signal_kwargs=signal_kwargs,
                                    normalized=normalized,
                                    channels=[1, 2],
                                    model_cls=data_cls,
                                    psd_cls=psd_cls,
                                    wd=wd,
                                    wd_full=wd_full,
                                    gap_convolution=gap_convolution)

    # =========================================================================
    # Testing likelihood
    # =========================================================================
    par_aux0 = np.concatenate(ll_cls.data_dft + sn)
    t1 = time.time()
    if reduced:
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
    suffix = config["OutputData"].get("FileSuffix")
    prefix += suffix
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
    seed = config["Sampler"].get("RandomSeed")
    if seed != "None":
        np.random.seed(np.int(seed))

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
