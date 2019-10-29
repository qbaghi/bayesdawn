# Standard useful python module
import numpy as np
import sys, os, re
import time
import copy
# # Adding the LDC packages to the python path
# sys.path.append("/root/.local/lib/python3.6/site-packages/")
# LDC modules
import tdi
from LISAhdf5 import LISAhdf5, ParsUnits
import LISAConstants as LC

from scipy.interpolate import InterpolatedUnivariateSpline as spline
import Cosmology
import GenerateFD_SignalTDIs as GenTDIFD
import pyIMRPhenomD
import pyFDresponse as FD_Resp
import dynesty

import lisabeta.lisa.lisa as lisa
import lisabeta.lisa.ldctools as ldctools
import lisabeta.lisa.pyresponse as pyresponse
import lisabeta.pyconstants as pyconstants
import lisabeta.tools.pytools as pytools
import lisabeta.tools.pyspline as pyspline
from scipy.interpolate import interp1d


def get_params(p_gw):
    """
    returns array of parameters from hdf5 structure
    Parameters
    ----------
    p_gw

    Returns
    -------

    """
    # print (pGW.get('Mass1')*1.e-6, pGW.get('Mass2')*1.e-6)
    m1 = p_gw.get('Mass1') ### Assume masses redshifted
    m2 = p_gw.get('Mass2')
    tc = p_gw.get('CoalescenceTime')
    chi1 = p_gw.get('Spin1') * np.cos(p_gw.get('PolarAngleOfSpin1'))
    chi2 = p_gw.get('Spin2') * np.cos(p_gw.get('PolarAngleOfSpin2'))
    theL = p_gw.get('InitialPolarAngleL')
    phiL = p_gw.get('InitialAzimuthalAngleL')
    longt = p_gw.get('EclipticLongitude')
    lat = p_gw.get('EclipticLatitude')
    phi0 = p_gw.get('PhaseAtCoalescence')
    z = p_gw.get("Redshift")
    DL = Cosmology.DL(z, w=0)[0]
    dist = DL * 1.e6 * LC.pc
    print ("DL = ", DL*1.e-3, "Gpc")
    print ("Compare DL:", p_gw.getConvert('Distance', LC.convDistance, 'mpc'))

    bet, lam, incl, psi = ldctools.GetSkyAndOrientation(p_gw)

    return m1, m2, tc, chi1, chi2, dist, incl, bet, lam, psi, phi0, DL


def SimpleLogLik(data, template, Sn, df, tdi='XYZ'):

    if (tdi=='XYZ'):
        Xd = data[0]
        Yd = data[1]
        Zd = data[2]

        Xt = template[0]
        Yt = template[1]
        Zt = template[2]

        SNX = np.sum( np.real(Xd*np.conjugate(Xt))/Sn )
        SNY = np.sum( np.real(Yd*np.conjugate(Yt))/Sn )
        SNZ = np.sum( np.real(Zd*np.conjugate(Zt))/Sn )

        # print ('SN = ', 4.0*df*SNX, 4.0*df*SNY, 4.0*df*SNZ)

        XX = np.sum( np.abs(Xt)**2/Sn )
        YY = np.sum( np.abs(Yt)**2/Sn )
        ZZ = np.sum( np.abs(Zt)**2/Sn )

        # print ('hh = ', 4.0*df*XX, 4.0*df*YY, 4.0*df*ZZ)
        llX = 4.0*df*(SNX - 0.5*XX)
        llY = 4.0*df*(SNY - 0.5*YY)
        llZ = 4.0*df*(SNZ - 0.5*ZZ)

        return (llX, llY, llZ)

    else: ### I presume it is A, E
        Ad = data[0]
        Ed = data[1]

        At = template[0]
        Et = template[1]

        SNA = np.sum( np.real(Ad*np.conjugate(At))/Sn )
        SNE = np.sum( np.real(Ed*np.conjugate(Et))/Sn )

        # print ('SN = ', 4.0*df*SNA, 4.0*df*SNE)

        AA = np.sum( np.abs(At)**2/Sn )
        EE = np.sum( np.abs(Et)**2/Sn )

        # print ('hh:', 4.0*df*AA, 4.0*df*EE)

        llA = 4.0*df*(SNA - 0.5*AA)
        llE = 4.0*df*(SNE - 0.5*EE)

        return (llA, llE)


def compute_masses(mc, q):

    ma = (q / ((q + 1.) ** 2)) ** (-3.0 / 5.0)
    mb = ((q / ((q + 1.) ** 2)) ** (-6.0 / 5.0) - 4.0 * (q / ((q + 1.) ** 2)) ** (-1.0 / 5.0)) ** 0.5

    m1 = 0.5 * mc * (ma + mb)
    m2 = 0.5 * mc * (ma - mb)

    return m1, m2


def generate_lisa_signal(wftdi, freq=None, channels=None):
    """

    Parameters
    ----------
    wftdi : dict
        dictionary output from GenerateLISATDI
    freq : ndarray
        numpy array of freqs on which to sample waveform (or None)

    Returns
    -------
    dictionary with output TDI channel data yielding numpy complex data arrays

    """

    if channels is None:
        channels = [1, 2]


    fs = wftdi['freq']
    amp = wftdi['amp']
    phase = wftdi['phase']
    print("Length of initial frequency grid " + str(len(fs)))

    tr = [wftdi['transferL' + str(np.int(i))] for i in channels]

    if freq is not None:
        amp = pyspline.resample(freq, fs, amp)
        phase = pyspline.resample(freq, fs, phase)
        tr_int = [pyspline.resample(freq, fs, tri) for tri in tr]
        # amp = np.interp(freq, fs, amp)
        # phase = np.interp(freq, fs, phase)
        # tr_int = [np.interp(freq, fs, tri) for tri in tr]
    else:
        freq = fs

    h = amp * np.exp(1j * phase)

    signal = {'ch' + str(np.int(i)): h * tr_int[i - 1] for i in channels}
    signal['freq'] = freq

    return signal


def lisabeta_template(params, freq, tobs, tref=0, t_offset=52.657, channels=None):
    """

    Parameters
    ----------
    params
    freq
    tobs
    tref
    toffset

    Returns
    -------

    """

    if channels is None:
        channels = [1, 2, 3]

    # TDI response
    wftdi = lisa.GenerateLISATDI(params, tobs=tobs, minf=1e-5, maxf=1., tref=tref, torb=0., TDItag='TDIAET',
                                 acc=1e-4, order_fresnel_stencil=0, approximant='IMRPhenomD',
                                 responseapprox='full', frozenLISA=False, TDIrescaled=False)

    signal_freq = generate_lisa_signal(wftdi, freq, channels)

    # Phasor for the time shift
    z = np.exp(-2j * np.pi * freq * t_offset)
    # Get coalescence time
    ch_interp = [(signal_freq['ch' + str(np.int(i))] * z).conj() for i in channels]

    return ch_interp


def SimpleLogLik(data, template, Sn, df, tdi='XYZ'):
    """

    Parameters
    ----------
    data
    template
    Sn
    df
    tdi

    Returns
    -------

    """

    if tdi == 'XYZ':

        Xd = data[0]
        Yd = data[1]
        Zd = data[2]

        Xt = template[0]
        Yt = template[1]
        Zt = template[2]

        SNX = np.sum(np.real(Xd*np.conjugate(Xt))/Sn)
        SNY = np.sum(np.real(Yd*np.conjugate(Yt))/Sn)
        SNZ = np.sum(np.real(Zd*np.conjugate(Zt))/Sn)

        XX = np.sum(np.abs(Xt)**2/Sn)
        YY = np.sum(np.abs(Yt)**2/Sn)
        ZZ = np.sum(np.abs(Zt)**2/Sn)

        llX = 4.0*df*(SNX - 0.5*XX)
        llY = 4.0*df*(SNY - 0.5*YY)
        llZ = 4.0*df*(SNZ - 0.5*ZZ)

        return (llX, llY, llZ)

    else: ### I presume it is A, E
        Ad = data[0]
        Ed = data[1]

        At = template[0]
        Et = template[1]

        SNA = np.sum(np.real(Ad*np.conjugate(At))/Sn)
        SNE = np.sum(np.real(Ed*np.conjugate(Et))/Sn)

        # print ('SN = ', 4.0*df*SNA, 4.0*df*SNE)

        AA = np.sum(np.abs(At)**2/Sn)
        EE = np.sum(np.abs(Et)**2/Sn)

        # print ('hh:', 4.0*df*AA, 4.0*df*EE)

        llA = 4.0*df*(SNA - 0.5*AA)
        llE = 4.0*df*(SNE - 0.5*EE)

        return (llA, llE)


def like_to_waveform(par):
    """
    Convert likelihood parameters into waveform-compatible parameters
    Parameters
    ----------
    par : array_like
        parameter vector for sampling the posterior:
        Mc, q, tc, chi1, chi2, np.log10(DL), np.cos(incl), np.sin(bet), lam, psi, phi0

    Returns
    -------
    params : array_like
       parameter vector for LISABeta waveform: m1, m2, chi1, chi2, del_t, dist, incl, phi0, lam, bet, psi

    """

    # Explicit the vector of paramters
    mc, q, tc, chi1, chi2, logdl, ci, sb, lam, psi, phi0 = par

    # Convert chirp mass into individual masses
    m1, m2 = compute_masses(mc, q)

    # Convert base-10 logarithmic luminosity distance into lum. dist. in Mega parsecs
    dl = 10.0 ** logdl

    # Convert inclination
    incl = np.arccos(ci)
    # Convert Source latitude in SSB-frame
    bet = np.arcsin(sb)

    params = m1, m2, chi1, chi2, tc, dl, incl, phi0, lam, bet, psi

    return params


class LogLike(object):

    def __init__(self, data, sn, freq, tobs, del_t, normalized=False, t_offset=52.657):
        """

        Parameters
        ----------
        data : array_like
            DFT of TDI data A, E, T computed at frequencies freq
        sn : ndarray
            noise PSD computed at freq
        freq: ndarray
            frequency array
        tobs : float
            observation time
        del_t : float
            data sampling cadence
        ll_norm : float
            normalization constant for the log-likelihood
        """

        self.data = data
        self.sn = sn
        self.freq = freq
        self.tobs = tobs
        self.del_t = del_t
        self.nf = len(freq)
        self.t_offset = t_offset
        if normalized:
            self.ll_norm = self.log_norm()
        else:
            self.ll_norm = 0

    def log_norm(self):
        """
        Compute normalizing constant for the log-likelihood

        Returns
        -------
        ll_norm : float
            normalization constant for the log-likelihood

        """

        ll_norm = - self.nf/2 * np.log(2 * np.pi * 2 * self.del_t) - 0.5 * np.sum(np.log(self.sn))

        return ll_norm

    def log_likelihood(self, par):
        """

        Parameters
        ----------
        par : array_like
            vector of waveform parameters in the following order: [Mc, q, tc, chi1, chi2, logDL, ci, sb, lam, psi, phi0]
        data : array_like
            DFT of TDI data A, E, T computed at frequencies freq
        sn : ndarray
            noise PSD computed at freq
        freq: ndarray
            frequency array


        Returns
        -------

        """

        # Convert likelihood parameters into waveform-compatible parameters
        params = like_to_waveform(par)

        # Compute waveform template
        at, et = lisabeta_template(params, self.freq, tobs, tref=0, t_offset=self.t_offset, channels=[1, 2])

        # (h | y)
        sna = np.sum(np.real(self.data[0]*np.conjugate(at)) / self.sn)
        sne = np.sum(np.real(self.data[1]*np.conjugate(et)) / self.sn)

        # (h | h)
        aa = np.sum(np.abs(at) ** 2 / self.sn)
        ee = np.sum(np.abs(et) ** 2 / self.sn)

        # (h | y) - 1/2 (h | h)
        llA = 4.0*(self.freq[1] - self.freq[0])*(sna - 0.5*aa)
        llE = 4.0*(self.freq[1] - self.freq[0])*(sne - 0.5*ee)

        return llA + llE + self.ll_norm


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
    from bayesdawn import samplers, posteriormodel
    # For parallel computing
    # from multiprocessing import Pool, Queue
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
    print("Loading data...")
    fd5 = LISAhdf5(config["InputData"]["FilePath"])
    n_src = fd5.getSourcesNum()
    gws = fd5.getSourcesName()
    print("Found %d GW sources: " % n_src, gws)
    if not re.search('MBHB', gws[0]):
        raise NotImplementedError
    p = fd5.getSourceParameters(gws[0])
    td = fd5.getPreProcessTDI()
    del_t = float(p.get("Cadence"))
    tobs = float(p.get("ObservationDuration"))
    p.display()
    print("Cadence = ", del_t, "Observation time=", tobs)
    print("Data loaded.")

    # ==================================================================================================================
    # Get the parameters
    # ==================================================================================================================
    # Get parameters as an array from the hdf5 structure (table)
    m1, m2, tc, chi1, chi2, dist, incl, bet, lam, psi, phi0, DL = get_params(p)
    Mc = FD_Resp.funcMchirpofm1m2(m1, m2)
    q = m1 / m2
    parS = Mc, q, tc, chi1, chi2, dist, incl, bet, lam, psi, phi0, DL, m1, m2

    # transforming into sampling parameters
    pS_sampl = np.array([Mc, q, tc, chi1, chi2, np.log10(DL), np.cos(incl), np.sin(bet), lam, psi, phi0])

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
    if config['InputData'].getboolean('decimation'):
        fc = config['InputData'].getfloat('filterFrequency')
        b, a = signal.butter(5, 0.03, 'low', analog=False, fs=1/del_t)
        Xd = signal.filtfilt(b, a, td[:, 1])
        Yd = signal.filtfilt(b, a, td[:, 2])
        Zd = signal.filtfilt(b, a, td[:, 3])
        # Downsampling
        q = config['InputData'].getint('decimationFactor')
        tm = td[i1:i2:q, 0]
        Xd = Xd[i1:i2:q]
        Yd = Yd[i1:i2:q]
        Zd = Zd[i1:i2:q]

    else:
        q = 1
        tm = td[i1:i2, 0]
        Xd = td[i1:i2, 1]
        Yd = td[i1:i2, 2]
        Zd = td[i1:i2, 3]

    # ==================================================================================================================
    # Now we get extract the data and transform it to frequency domain
    # ==================================================================================================================

    XDf = fft(Xd) * del_t * q
    YDf = fft(Yd) * del_t * q
    ZDf = fft(Zd) * del_t * q

    freqD = np.fft.fftfreq(len(tm), del_t * q)
    freqD = freqD[:int(len(freqD) / 2)]

    Nf = len(freqD)
    XDf = XDf[:Nf]
    YDf = YDf[:Nf]
    ZDf = ZDf[:Nf]
    df = freqD[1] - freqD[0]

    # Restrict the frequency band to high SNR region
    inds = np.where((float(config['Model']['MinimumFrequency']) <= freqD)
                    & (freqD <= float(config['Model']['MaximumFrequency'])))[0]

    params = ldctools.get_params_from_LDC(p)
    t1 = time.time()
    Aft, Eft, Tft = lisabeta_template(params, freqD[inds], tobs, tref=0, t_offset=t_offset, channels=[1, 2, 3])
    t2 = time.time()
    print("=================================================================")
    print("LISABeta template computation time: " + str(t2 - t1))
    print("=================================================================")

    # Verification of parameters compatibility m1, m2, chi1, chi2, Deltat, dist, inc, phi, lambd, beta, psi
    params0 = like_to_waveform(pS_sampl)

    # Convert Michelson TDI to A, E, T
    ADf, EDf, TDf = ldctools.convert_XYZ_to_AET(XDf, YDf, ZDf)

    # # Testing the right offset
    # offsets = np.linspace(50.60, 53, 50)
    # results = [lisabeta_template(params, freqD, tobs, tref=0, toffset=toffset) for toffset in offsets]
    # rms = np.array([np.sum(np.abs(ADf - res[0])**2) for res in results])

    fftwisdom.save_wisdom()

    # ==================================================================================================================
    # Comparing log-likelihoood
    # ==================================================================================================================
    # One-sided PSD
    SA = tdi.noisepsd_AE(freqD[inds], model='Proposal', includewd=None)
    # Consider only A and E TDI data
    dataAE = [ADf[inds], EDf[inds]]
    templateAE = [Aft, Eft]
    llA1, llE1 = SimpleLogLik(dataAE, templateAE, SA, df, tdi='AET')
    llA2, llE2 = SimpleLogLik(dataAE, dataAE, SA, df, tdi='AET')
    llA3, llE3 = SimpleLogLik(templateAE, templateAE, SA, df, tdi='AET')
    print('compare A', llA1, llA2, llA3)
    print('compare E', llE1, llE2, llE3)
    print('total lloglik', llA1 + llE1, llA2 + llE2, llA3 + llE3)

    # Full computation of likelihood
    ll_cls = LogLike(dataAE, SA, freqD[inds], tobs, del_t * q, normalized=False, t_offset=t_offset)
    t1 = time.time()
    lltot = ll_cls.log_likelihood(pS_sampl)
    t2 = time.time()
    print('My total likelihood: ' + str(lltot))
    print('Calculated in ' + str(t2 - t1) + ' seconds.')
    # Include normalization
    ll_cls = LogLike(dataAE, SA, freqD[inds], tobs, del_t * q, normalized=config['Model'].getboolean('normalized'),
                     t_offset=t_offset)

    # ==================================================================================================================
    # Get parameter names and bounds
    # ==================================================================================================================
    # Get all parameter name keys
    names = [key for key in config['ParametersLowerBounds']]
    # Get prior bound values
    bounds = [[float(config['ParametersLowerBounds'][name]), float(config['ParametersUpperBounds'][name])]
              for name in names]
    lower_bounds = np.array([bound[0] for bound in bounds])
    upper_bounds = np.array([bound[1] for bound in bounds])
    # Print it
    [print("Bounds for parameter " + names[i] + ": " + str(bounds[i])) for i in range(len(names))]

    # ==================================================================================================================
    # Test prior transform consistency
    # ==================================================================================================================
    # Draw random numbers in [0, 1]
    theta_u = np.random.random(len(names))
    # Transform to physical parameters
    par_ll = prior_transform(theta_u, lower_bounds, upper_bounds)
    # Check that they lie within bounds
    print("Within bounds: "
          + str(np.all(np.array([lower_bounds[i] <= par_ll[i] <= upper_bounds[i] for i in range(len(par_ll))]))))
    print('random param loglik = ' + str(ll_cls.log_likelihood(par_ll)))

    # # ==================================================================================================================
    # # Prepare data to save during sampling
    # # ==================================================================================================================
    # # Append current date and prefix to file names
    # now = datetime.datetime.now()
    # prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_")
    # out_dir = config["OutputData"]["DirectoryPath"]
    # # Save the configuration file used for the run
    # with open(out_dir + prefix + "config.ini", 'w') as configfile:
    #     config.write(configfile)
    #
    # # ==================================================================================================================
    # # Start sampling
    # # ==================================================================================================================
    # # Set seed
    # np.random.seed(int(config["Sampler"]["RandomSeed"]))
    # # Multiprocessing pool
    # # pool = Pool(4)
    #
    # if config["Sampler"]["Type"] == 'dynesty':
    #     # Instantiate sampler
    #     dsampl = dynesty.DynamicNestedSampler(ll_cls.log_likelihood, prior_transform, ndim=len(names),
    #                                           bound='multi', sample='slice', periodic=[8, 9, 10],
    #                                           ptform_args=(lower_bounds, upper_bounds), ptform_kwargs=None)
    #                                           # pool=pool, queue_size=4)
    #     # # Start run
    #     # dsampl.run_nested(dlogz_init=0.01, nlive_init=config["Sampler"].getint("WalkerNumber"),
    #     # nlive_batch=500, wt_kwargs={'pfrac': 1.0},
    #     #                   stop_kwargs={'pfrac': 1.0})
    #
    #     print("n_live_init = " + config["Sampler"]["WalkerNumber"])
    #     print("Save samples every " + config['Sampler']['SavingNumber'] + " iterations.")
    #
    #     samplers.run_and_save(dsampl, nlive=config["Sampler"].getint("WalkerNumber"),
    #                           n_save=config['Sampler'].getint('SavingNumber'),
    #                           file_path=out_dir + prefix)
    #
    # elif config["Sampler"]["Type"] == 'ptemcee':
    #
    #     sampler = samplers.ExtendedPTMCMC(int(config["Sampler"]["WalkerNumber"]),
    #                                       len(names),
    #                                       ll_cls.log_likelihood,
    #                                       posteriormodel.logp,
    #                                       ntemps=int(config["Sampler"]["TemperatureNumber"]),
    #                                       logpargs=(lower_bounds, upper_bounds))
    #
    #     result = sampler.run(int(config["Sampler"]["MaximumIterationNumber"]),
    #                          config['Sampler'].getint('SavingNumber'),
    #                          int(config["Sampler"]["thinningNumber"]),
    #                          callback=None, pos0=None,
    #                          save_path=out_dir + prefix)

    # ==================================================================================================================
    # Plotting
    # ==================================================================================================================
    fig1, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    ax[0].semilogx(freqD, np.real(ADf))
    ax[0].semilogx(freqD[inds], np.real(Aft), '--')
    ax[1].semilogx(freqD, np.imag(ADf))
    ax[1].semilogx(freqD[inds], np.imag(Aft), '--')
    # ax[0].semilogx(freqD, np.real(XDf))
    # ax[0].semilogx(freqD, np.real(Xft), '--')
    # ax[1].semilogx(freqD, np.imag(XDf))
    # ax[1].semilogx(freqD, np.imag(Xft), '--')
    # plt.xlim([6.955e-3, 6.957e-3])
    # plt.ylim([-1e-18, 1e-18])
    plt.show()

