import numpy as np
from bayesdawn.bayesdawn import datamodel, psdmodel
# from dynesty import NestedSampler
# from dynesty import utils as dyfunc
# import ptemcee

class PSDTheoretical(object):

    def __init__(self, N, fs, fmin=None, fmax=None, channels=['A'], scale=1.0):
        self.channels = channels
        self.psd_list = [psdmodel.PSD(N, fs, fmin=fmin, fmax=fmax) for ch in channels]

        for i in range(len(channels)):
            self.psd_list[i].PSD_fn = theoretical_spectrum_func(fs, channels[i], scale=scale)
            self.psd_list[i].logPSD_fn = lambda x: np.log(self.psd_list[i].PSD_fn(np.exp(x)))

    def calculate(self, arg):
        return [psd.calculate(arg) for psd in self.psd_list]

    def set_periodogram(self, z_fft, K2):
        [psd.set_periodogram(z_fft, K2) for psd in self.psd_list]

    def sample(self, npsd):
        sampling_result = [psd.sample_psd(npsd) for psd in self.psd_list]
        sample_list = [samp[0] for samp in sampling_result]
        logp_values_list = [samp[1] for samp in sampling_result]

        return sample_list, logp_values_list



def theoretical_spectrum_func(f_sampling, channel, scale=1.0):

    if channel == 'A':
        PSD_fn = lambda x: tdi.noisepsd_AE(x, model='SciRDv1') * f_sampling / 2 / scale**2
    elif channel == 'E':
        PSD_fn = lambda x: tdi.noisepsd_AE(x, model='SciRDv1') * f_sampling / 2 / scale**2
    elif channel == 'T':
        PSD_fn = lambda x: tdi.noisepsd_T(x, model='SciRDv1') * f_sampling / 2 / scale**2

    return PSD_fn



if __name__ == '__main__':

    from matplotlib import pyplot as plt
    # import pywt
    # from gwavelets import gwplot, gwdenoise
    import datetime
    # import myplots
    import time
    import h5py
    # from scipy import signal
    from bayesdawn.bayesdawn.postproc import psdplot
    from bayesdawn.bayesdawn.waveforms import lisaresp
    from bayesdawn.bayesdawn import gwmodel, dasampler
    from bayesdawn.bayesdawn.utils import loadings
    from bayesdawn.bayesdawn import samplers
    import tdi

    # FTT modules
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import fft, ifft

    # ==================================================================================================================
    # Input file name
    hdf5_name = '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5'
    # Load simulation data
    time_vect, signal_list, noise_list, params = loadings.load_simulation(hdf5_name)

    i1 = np.int(216160 - 2**17)
    i2 = 216160
    # sig = signal_list[0]/1e-21# [200000:220000]
    t_sig = time_vect[:]# [200000:220000]
    del_t = t_sig[1] - t_sig[0]
    scale = 1.0# 1e-21

    y_list = [(signal_list[j][i1:i2] + noise_list[j][i1:i2])/scale for j in range(len(signal_list))]
    y_fft_list = [fft(y0) for y0 in y_list]

    n = len(y_list[0])
    fs = 1 / del_t
    tobs = n * del_t

    # No missing data
    mask = np.ones(n)

    # # Entire signal dct
    # sig_fft = fft(sig)
    #
    # # Cut signal dct
    # i0 = 199808
    # n2 = 2**14
    # i1 = np.int(i0 + n2)
    # sig_cut = sig[i0:i1]
    # sig_cut_fft = fft(sig_cut)
    #
    # # Plot with right normalization
    #
    # fig1, ax1 = psdplot.plot_periodogram([sig, sig_cut], fs=fs, wind='hanning', ylabel='TDI A', colors=['k', 'r'],
    #                          labels=['Original', 'Cut'], linestyles=['solid', 'dashed'])
    #
    # plt.show()

    # ==================================================================================================================
    # Instantiate waveform class
    signal_cls = lisaresp.MBHBWaveform()
    # PSD function
    f = np.fft.fftfreq(n) * fs
    f_pos = np.abs(f)
    f_pos[0] = f_pos[1]
    psd_ae = tdi.noisepsd_AE(f_pos, model='SciRDv1')
    s_ae = psd_ae * fs / 2
    s_t = tdi.noisepsd_T(f_pos, model='SciRDv1') * fs / 2
    s_list = [s_ae, s_ae, s_t]

    # ==================================================================================================================
    # # Create data analysis GW model with instrinsic parameters only
    # names = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'lam', 'beta']
    # bounds = [[0.1e6, 1e7], [0.1e6, 1e7], [0, 1], [0, 1], [2000000.0, 25000000.0], [0, np.pi], [0, 2*np.pi]]
    # Create data analysis GW model with the full parameter vector
    names = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi']
    bounds = [[0.5e6, 5e6], [0.5e6, 5e6], [0, 1], [0, 1], [2000000.0, 2162000.0], [100000, 500000],
              [0, np.pi], [0, 2*np.pi], [0, np.pi], [0, 2 * np.pi], [0, 2 * np.pi]]
    distribs = ['uniform' for name in names]
    channels = ['A', 'E', 'T']
    # Instantiate data analysis model class
    model_cls = gwmodel.GWModel(signal_cls, tobs, del_t, names=names, bounds=bounds, distribs=distribs,
                                channels=channels, fmin=1e-5, fmax=1e-2, nsources=1)

    # ==================================================================================================================
    # Creation of PSD class
    # ==================================================================================================================
    psd_cls = PSDTheoretical(n, fs, channels=channels, scale=scale)
    spectrum_list = psd_cls.calculate(n)

    # ==================================================================================================================
    # Creation of data class instance
    # ==================================================================================================================
    dat_cls = datamodel.GaussianStationaryProcess(y_list, mask)

    # ==================================================================================================================
    # Creation of sampler class instance
    # ==================================================================================================================

    nwalkers = 4 * model_cls.ndim_tot
    ntemps = 10
    # nlive = model_cls.ndim_tot * (model_cls.ndim_tot + 1) // 2
    # log_likelihood = lambda x: model_cls.log_likelihood(x, spectrum_list, y_fft_list)
    # sampler_cls = NestedSampler(log_likelihood, model_cls.ptform, model_cls.ndim_tot, nlive=nlive)
    # sampler_cls = ExtendedNestedSampler(log_likelihood, model_cls.ptform, signal_cls.ndim_tot, nlive=nlive)
    # sampler_cls = ptemcee.Sampler(nwalkers, model_cls.ndim_tot, model_cls.log_likelihood,
    #                               model_cls.logp, ntemps=ntemps, loglargs=(spectrum_list, y_fft_list),
    #                               logpargs=(model_cls.lo, model_cls.hi))
    sampler_cls = samplers.ExtendedPTMCMC(nwalkers, model_cls.ndim_tot, model_cls.log_likelihood, model_cls.logp,
                                          ntemps=ntemps, loglargs=(spectrum_list, y_fft_list), logpargs=(model_cls.lo, model_cls.hi))

    # ==================================================================================================================
    # Test of likelihood calculation
    # ==================================================================================================================
    # Intrinsic parameters
    # params_intr = np.array([params[i] for i in [0, 1, 2, 3, 4, 8, 9]])
    # model_cls.log_likelihood(params_intr, psd_ae, sig_fft)


    # ==================================================================================================================
    # Creation of sampling class instance
    das = dasampler.FullModel(model_cls, psd_cls, dat_cls, sampler_cls, outdir='./', prefix='samples', n_wind=500,
                              n_wind_psd=50000, imputation=False, psd_estimation=False, normalized=True)

    # Full parameter vector
    t1 = time.time()
    # model_cls.log_likelihood(params, psd_ae, sig_fft)
    test1 = das.log_likelihood(das.params2uniform(params), spectrum_list, y_fft_list)
    t2 = time.time()
    print("Test with set of parameters 1")
    print("loglike value: " + str(test1))
    print("Likelihood computation time: " + str(t2 - t1))

    params2 = np.array(model_cls.lo)
    t1 = time.time()
    test2 = das.log_likelihood(das.params2uniform(params2), spectrum_list, y_fft_list)
    t2 = time.time()
    print("Test with set of parameters 2")
    print("loglike value: " + str(test2))
    print("Likelihood computation time: " + str(t2 - t1))


    now = datetime.datetime.now()
    prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_")
    out_dir = '/Users/qbaghi/Codes/data/results_ptemcee/local/'
    print("start sampling...")
    das.run(n_it=100000, n_update=100, n_thin=1, n_save=10, save_path=out_dir + 'chains_temp.hdf5')
    print("done.")
    # res = das.run(n_it=100000, n_update=1000, n_psd=10)
    # Or:
    # das.sampler_cls.run_nested(maxiter=100000)


    fh5 = h5py.File(out_dir + prefix + 'chains.hdf5', 'w')
    fh5.create_dataset("chains/chain", data=das.sampler_cls.chain)
    fh5.create_dataset("temperatures/beta_hist", data=das.sampler_cls._beta_history)
    fh5.close()



    # # Instantiate waveform class
    # signal_cls = lisaresp.MBHBWaveform()
    # # Compute waveform in frequency domain
    # freq = np.fft.fftfreq(n) * fs
    # f_cut = freq[(freq >= 1e-5) & (freq <= 1e-2)]



    # t1 = time.time()
    # af, ef, tf = signal_cls.compute_signal_freq(f_cut, params, del_t, tobs, channel='TDIAET', real_imag=False, ldc=False)
    # t2 = time.time()
    # print(t2-t1)

    # wave = 'sym4'
    # sig_w = pywt.wavedec(sig, wave)
    #
    # # gwplot.coef_pyramid_plot(sig_w, len(sig), fs=0.1)
    #
    # sig_w_den, sig_den = gwdenoise.simpledenoise(sig, wave, threshfunc='hard', th=50000, Nit=2, sigma_method='MADall')
    #
    # n_zeros, n_coeffs = gwdenoise.sparsity(sig_w_den)
    #
    # # th = 10, sparsity = 94% 1163.0 coefficients
    # # th = 10, sparsity = 95% 950
    # # th = 50, sparsity = 96% 644
    #
    # day = 24 * 3600



    # fp = myplots.fplot(plotconf='time')
    # fp.xscale = 'linear'
    # fp.yscale = 'linear'
    # fp.draw_frame = True
    # fp.ylabel = r'Fractional frequency'
    # fp.legendloc = 'upper left'
    # fp.xlabel = 'Time [days]'
    # fp.ylabel = 'TDI A [fractional frequency]'
    # # fp.xlims = [24.7, 25.1]
    # fig1, ax1 = fp.plot([t_sig / day, t_sig / day],
    #                     [sig, sig_den],
    #                     ['k', 'r'], [2, 2], ['Original', 'Compressed ('+str(100*n_zeros/n_coeffs)+'%)'],
    #                     linestyles=['solid', 'dashed'], zorders=[2, 2])
    #
    # plt.show()



    # # Short study
    # wave_list = ['db4', 'sym4', 'dmey', 'coif4', 'bior4.4']
    # th_list = [1000, 5000, 10000, 50000, 100000, 150000, 200000, 250000, 300000]
    #
    # for wave in wave_list:
    #     err = np.zeros(len(th_list))
    #     spar = np.zeros(len(th_list))
    #
    #     for i in range(len(th_list)):
    #         sig_w_den, sig_den = gwdenoise.simpledenoise(sig, wave, threshfunc='hard', th=th_list[i], Nit=1,
    #                                                      sigma_method='MADall')
    #
    #         n_zeros, n_coeffs = gwdenoise.sparsity(sig_w_den)
    #
    #         spar[i] = n_zeros/n_coeffs
    #         err[i] = np.sum( np.abs(sig - sig_den)**2 )
    #     plt.plot(spar, err, label=wave)
    #
    #
    # plt.legend()
    # plt.show()

