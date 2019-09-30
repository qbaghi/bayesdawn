



if __name__ == '__main__':

    import numpy as np
    from scipy import linalg as LA
    from matplotlib import pyplot as plt
    import time
    # import tdi
    # import librosa

    from bayesdawn.waveforms import lisaresp
    from bayesdawn import gwmodel, dasampler
    from bayesdawn import datamodel
    from bayesdawn.utils import loadings

    # FTT modules
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import fft, ifft

    # ==================================================================================================================
    # Input file name
    hdf5_name = '/Users/qbaghi/Codes/data/simulations/mbhb/simulation_3.hdf5'
    # Load simulation data
    time_vect, signal_list, noise_list, params = loadings.load_simulation(hdf5_name)

    i1 = 0 # np.int(216160 - 2**17)
    i2 = signal_list[0].shape[0]# 216160
    # sig = signal_list[0]/1e-21# [200000:220000]
    t_sig = time_vect[:]# [200000:220000]
    del_t = t_sig[1] - t_sig[0]
    scale = 1.0# 1e-21


    # y_list = [(signal_list[j][i1:i2] + noise_list[j][i1:i2])/scale for j in range(len(signal_list))]
    # y_fft_list = [fft(y0) for y0 in y_list]
    # n = len(y_list[0])
    # fs = 1 / del_t
    # tobs = n * del_t

    # ==================================================================================================================
    # Instantiate waveform class
    signal_cls = lisaresp.MBHBWaveform()

    # ==================================================================================================================
    # Compute the least-squares matrix to fit for extrinsic parameters using exact instrisic parameters
    # Signal alone
    # signal_only = [signal[i1:i2] for signal in signal_list]
    signal_only = [datamodel.TimeSeries(signal.astype(np.float)[i1:i2], del_t=del_t) for signal in signal_list]
    noisy_signal = [datamodel.TimeSeries((signal_list[i]+noise_list[i]).astype(np.float)[i1:i2], del_t=del_t)
                    for i in range(len(signal_list))]
    # signal_only = [datamodel.TimeSeries(signal[i1:i2], del_t=del_t) for signal in signal_list]
    # Convert in frequency domain
    signal_only_fft = [signal.dft(wind='rect', normalized=False) for signal in signal_only]
    noisy_signal_fft = [signal.dft(wind='rect', normalized=False) for signal in noisy_signal]
    n = len(signal_only[0])
    fs = 1 / del_t
    tobs = n * del_t
    f = np.fft.fftfreq(n)*fs
    # Restric frequency band
    fmin = 1e-5
    fmax = 1e-2
    i_band = np.where((f >= fmin) & (f <= fmax))[0]
    f_band = f[i_band]
    # Transform parameters list in numpy array and select only intrinsic parameters
    params_arr = np.array(params)
    params_intr = params_arr[signal_cls.i_intr]
    # Compute design matrix
    t1 = time.time()
    a_list = signal_cls.design_matrix_freq(f_band, params_intr,  del_t, tobs, complex=True)

    # Estimate coefficients on channel A
    beta = LA.pinv(np.dot(a_list[0].T.conj(), a_list[0])).dot(np.dot(a_list[0].T.conj(), signal_only_fft[0][i_band]))
    # mydata = np.concatenate([signal_only_fft[0][i_band].real, signal_only_fft[0][i_band].imag])
    # beta = LA.pinv(np.dot(a_list[0].T.conj(), a_list[0])).dot(np.dot(a_list[0].T.conj(), mydata))
    # Derive signal
    data_est = np.dot(a_list[0], beta)
    t2 = time.time()
    print("Signal calculation from reduced model took " + str(t2-t1))
    # signal_est = data_est[0:i_band.shape[0]] + 1j * data_est[i_band.shape[0]:2*i_band.shape[0]]
    signal_est = data_est[:]

    # Test with exact generation
    t1 = time.time()
    signal_exact = signal_cls.compute_signal_freq(f_band, params_arr, del_t, tobs, channel='TDIAET', ldc=False)
    t2 = time.time()
    print("Signal calculation from full model took " + str(t2-t1))

    # ==================================================================================================================
    # Plots
    plt.figure(0)
    plt.loglog(f_band, np.abs(signal_only_fft[0][i_band]), 'k', linewidth=2, label='Analysed data')
    plt.loglog(f_band, np.abs(signal_exact[0]), 'b--', label='Regenerated waveform')
    plt.loglog(f_band, np.abs(signal_est), 'r:', label='Reduced model')
    plt.legend()

    plt.figure(1)
    plt.semilogx(f_band, np.angle(signal_only_fft[0][i_band]), 'k', linewidth=2, label='Analysed data')
    plt.semilogx(f_band, np.angle(signal_exact[0]), 'b--', label='Regenerated waveform')
    plt.semilogx(f_band, np.angle(signal_est), 'r:', label='Reduced model')
    plt.legend()

    # z = signal_only[0].qtransform()
    # librosa.display.specshow(librosa.amplitude_to_db(z, ref=np.max), sr=fs,
    #                          x_axis='time', y_axis='cqt_note')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrum')
    # plt.tight_layout()
    #
    plt.show()
