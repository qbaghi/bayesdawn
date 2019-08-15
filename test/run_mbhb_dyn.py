from bayesdawn import psdmodel


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

    import numpy as np
    from dynesty import NestedSampler
    # from dynesty import utils as dyfunc
    # import ptemcee
    from matplotlib import pyplot as plt
    # import pywt
    # from gwavelets import gwplot, gwdenoise
    import datetime
    # import myplots
    import time
    import h5py
    # from scipy import signal
    from bayesdawn.waveforms import lisaresp
    from bayesdawn import gwmodel, dasampler, datamodel
    from bayesdawn.utils import loadings
    from bayesdawn import samplers
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
    nlive = model_cls.ndim_tot * (model_cls.ndim_tot + 1) // 2
    # log_likelihood = lambda x: model_cls.log_likelihood(x, spectrum_list, y_fft_list)
    sampler_cls = samplers.extended_nested_sampler(model_cls.log_likelihood, model_cls.ptform, model_cls.ndim_tot,
                                                   nlive=nlive, logl_args=(spectrum_list, y_fft_list))
    # ==================================================================================================================
    # Test of likelihood calculation
    # ==================================================================================================================
    # Intrinsic parameters
    # params_intr = np.array([params[i] for i in [0, 1, 2, 3, 4, 8, 9]])
    # model_cls.log_likelihood(params_intr, psd_ae, sig_fft)


    # ==================================================================================================================
    # Creation of sampling class instance
    das = dasampler.FullModel(model_cls, psd_cls, dat_cls, sampler_cls, outdir='./', prefix='samples', n_wind=500,
                              n_wind_psd=50000, imputation=False, psd_estimation=False, normalized=False)


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

    # ==================================================================================================================
    # Run sampler
    # ==================================================================================================================
    sampler_cls0 = NestedSampler(das.log_likelihood, model_cls.ptform, model_cls.ndim_tot,
                                 logl_args=(das.spectrum, das.y_fft))


    now = datetime.datetime.now()
    prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_")
    out_dir = '/Users/qbaghi/Codes/data/results_dynesty/local/'
    print("start sampling...")
    # das.run(n_it=100000, n_update=100, n_thin=1, n_save=100, save_path=out_dir + prefix + 'chains_temp.hdf5')
    results = sampler_cls0.run_nested(maxiter=100000)
    print("done.")
    # ==================================================================================================================
    # Save data
    # ==================================================================================================================
    fh5 = h5py.File(out_dir + prefix + 'chains.hdf5', 'w')
    # fh5.create_dataset("chains/chain", data=das.sampler_cls.results.samples)
    fh5.create_dataset("chains/chain", data=results.samples)
    fh5.close()