import numpy as np
import h5py
import pywt
from gwavelets import gwplot, gwdenoise
from matplotlib import pyplot as plt
import myplots
from bayesdawn.waveforms import lisaresp


def load_simulation(hdf5_name,
                    param_keylist=['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi'],
                    signal_keylist=['tdi_a', 'tdi_e', 'tdi_ts']):

    # Load data
    fh5 = h5py.File(hdf5_name, 'r')
    params = [fh5["parameters/" + par][()]for par in param_keylist]
    time_vect = fh5["signal/time"][()]
    signal_list = [fh5["signal/" + sig][()] for sig in signal_keylist]
    noise_list = [fh5["noise/" + sig][()] for sig in signal_keylist]
    fh5.close()

    return time_vect, signal_list, noise_list, params


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import time
    # FTT modules
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import fft, ifft

    # Input file name
    hdf5_name = '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5'
    # Load simulation data
    time_vect, signal_list, noise_list, params = load_simulation(hdf5_name)

    sig = signal_list[0] / 1e-21# [200000:220000]
    t_sig = time_vect[:]# [200000:220000]
    del_t = t_sig[1] - t_sig[0]
    n = len(sig)
    fs = 1 / del_t
    tobs = n / fs

    # Instantiate waveform class
    signal_cls = lisaresp.MBHBWaveform()
    # Compute waveform in frequency domain
    freq = np.fft.fftfreq(n) * fs
    f_cut = freq[(freq >= 1e-5) & (freq <= 1e-2)]

    t1 = time.time()
    af, ef, tf = signal_cls.compute_signal_freq(f_cut, params, del_t, tobs, channel='TDIAET', real_imag=False, ldc=False)
    t2 = time.time()
    print(t2-t1)

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

