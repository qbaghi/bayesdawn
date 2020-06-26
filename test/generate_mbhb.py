# -*- coding: utf-8 -*-
# Author: Quentin Baghi 2017
#
# Code to generate noisy LISA Data with binary black hole merger source


import h5py



def save_simulation(time_vector, noise_list, signal_list, signal_keylist, params, param_keylist, save_file_path='./'):

    # Number of parameters
    fh5 = h5py.File(save_file_path, 'a')
    fh5.create_group("parameters")
    for i in range(len(param_keylist)):
        fh5.create_dataset("parameters/" + param_keylist[i], data=params[i])
    # TDI signals
    fh5.create_group("signal")
    fh5.create_group("noise")
    fh5.create_dataset("signal/time", data=time_vector)
    for i in range(len(signal_keylist)):
        fh5.create_dataset("signal/" + signal_keylist[i], data=signal_list[i])
        fh5.create_dataset("noise/" + signal_keylist[i], data=noise_list[i])

    fh5.close()


if __name__ == '__main__':

    from LISAhdf5 import ParsUnits
    from lisabeta.lisa import ldctools
    import time
    import clht
    import numpy as np
    from bayesdawn.waveforms import lisaresp
    from bayesdawn import datamodel
    from plottools import myplots
    from matplotlib import pyplot as plt
    import tdi

    # FTT modules
    import pyfftw

    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import fft, ifft

    # =========================================
    # Set simulation sampling and duration
    # =========================================
    # Set number of observation points
    N = 2 ** 18
    # Sampling frequency
    fs = 0.1
    # Sampling cadence
    del_t = 1 / fs
    # Set observation time
    tobs = N / fs
    day = 24 * 3600


    # =========================================
    # Set source parameter values in LDC format
    # =========================================
    # LDC format
    p = ParsUnits()
    p.addPar('ObservationDuration', tobs, 'Second')
    p.addPar('Cadence', del_t, 'Second')
    p.addPar("EclipticLatitude", 40 * np.pi / 180, 'Radian')
    p.addPar("EclipticLongitude", 20 * np.pi / 180, 'Radian')
    p.addPar('Distance', 100000, 'mpc')
    p.addPar('Mass1', 1e6, 'SolarMass')
    p.addPar('Mass2', 2e6, 'SolarMass')
    p.addPar('Redshift', clht.Dl2z(100000), 'dimensionless')
    p.addPar('CoalescenceTime', 25 * day, 'Second')
    p.addPar('PhaseAtCoalescence', 0, 'Radian')
    p.addPar('Inclination', 0.1, 'Radian')
    p.addPar("Polarisation", 0.2, 'Radian')
    p.addPar('PolarAngleOfSpin1', 0, 'Radian')
    p.addPar('PolarAngleOfSpin2', 0, 'Radian')
    p.addPar('Spin1', 0, 'MassSquared')
    p.addPar('Spin2', 0, 'MassSquared')

    params = ldctools.get_params_from_LDC(p)
    param_keylist = ['m1', 'm2', 'xi1', 'xi2', 'tc', 'dist', 'inc', 'phi0', 'lam', 'beta', 'psi']

    # # Luminosity distance - redshift conversion
    # # Compute redshift from luminosity distance
    # z = clht.Dl2z(dist)
    # # Compute lunimosity distance from redshift
    # dl = clht.z2Dl(z)

    # ==================================================================================================================
    # Generate waveform
    # ==================================================================================================================
    # Instantiate waveform class
    signal_cls = lisaresp.MBHBWaveform()
    # Compute waveform in frequency domain
    n_fft = N
    freq = np.fft.fftfreq(n_fft) * fs
    n_pos = np.int((n_fft-1)/2)
    # f_pos = np.abs(np.concatenate(([freq[1]], freq[1:n_pos + 2])))
    f_pos = np.abs(freq[0:n_pos + 2])
    t1 = time.time()
    af, ef, tf = signal_cls.compute_signal_freq(f_pos, params, del_t, tobs, channel='TDIAET', ldc=False)
    t2 = time.time()
    print("LISABeta time: " + str(t2-t1))
    t3 = time.time()
    af2, ef2, tf2 = signal_cls.compute_signal_freq(f_pos, params, del_t, tobs, channel='TDIAET', ldc=True)
    t4 = time.time()
    print("LDC time: " + str(t4 - t3))
    freq_signal_sym = [np.concatenate((sf[0:n_pos + 1], np.conjugate(sf[1:n_pos + 2][::-1]))) for sf in [af, ef, tf]]
    freq_signal_sym2 = [np.concatenate((sf[0:n_pos + 1], np.conjugate(sf[1:n_pos + 2][::-1]))) for sf in [af2, ef2, tf2]]
    # Generate waveform in time domain
    t_vect = np.arange(0, N) / fs
    time_signal = [ifft(sf_sym)[0:N] for sf_sym in freq_signal_sym]
    time_signal2 = [ifft(sf_sym)[0:N] for sf_sym in freq_signal_sym2]

    # Now let's say we want to generate the data from t_start to t_end
    t_end = 25.1 * day
    t_start = t_end - 2 ** 17 * del_t
    t_obs3 = t_end - t_start
    n3 = np.int(t_obs3/del_t)
    freq3 = np.fft.fftfreq(n3) * fs
    n_pos3 = np.int((n3 - 1) / 2)
    f_pos3 = np.abs(freq3[0:n_pos3 + 2])
    af3, ef3, tf3 = signal_cls.compute_signal_freq(f_pos3, params, del_t, t_end, channel='TDIAET', ldc=False,
                                                   tref=0)
    freq_signal_sym3 = [np.concatenate((sf[0:n_pos3 + 1], np.conjugate(sf[1:n_pos3 + 2][::-1])))
                        for sf in [af3, ef3, tf3]]
    time_signal3 = [ifft(sf_sym)[0:n3] for sf_sym in freq_signal_sym3]
    t_vect3 = np.arange(t_start, t_end, del_t)

    # ==================================================================================================================
    # Generate noise
    # ==================================================================================================================
    freq_psd = np.fft.fftfreq(2*N) * fs
    freq_psd[0] = freq_psd[1]
    psd_ae = tdi.noisepsd_AE(np.abs(freq_psd), model='SciRDv1')
    psd_t = tdi.noisepsd_T(np.abs(freq_psd), model='SciRDv1')
    a_noise = datamodel.generate_noise_from_psd(np.sqrt(psd_ae), fs, myseed=np.int(1234))[0:N]
    e_noise = datamodel.generate_noise_from_psd(np.sqrt(psd_ae), fs, myseed=np.int(5678))[0:N]
    t_noise = datamodel.generate_noise_from_psd(np.sqrt(psd_t), fs, myseed=np.int(9101))[0:N]


    # ==================================================================================================================
    # Form data
    # ==================================================================================================================
    a_meas = np.real(a_noise + time_signal[0])
    e_meas = np.real(e_noise + time_signal[1])
    t_meas = np.real(t_noise + time_signal[2])


    # ==================================================================================================================
    # Save data
    # ==================================================================================================================
    # save_simulation(t_vect, [a_noise, e_noise, t_noise], time_signal,
    #                 ['tdi_a', 'tdi_e', 'tdi_ts'], params, param_keylist,
    #                 save_file_path='/Users/qbaghi/Codes/data/simulations/mbhb/simulation_4.hdf5')
    #
    # ==================================================================================================================
    # Plots
    # ==================================================================================================================
    fp = myplots.fplot(plotconf='time')
    fp.xscale = 'linear'
    fp.yscale = 'linear'
    fp.draw_frame = True
    fp.ylabel = r'Fractional frequency'
    fp.legendloc = 'upper left'
    fp.xlabel = 'Time [days]'
    fp.ylabel = 'TDI a_mat [fractional frequency]'
    fp.xlims = [24.7, 25.1]
    fig1, ax1 = fp.plot([t_vect / day, t_vect3 / day],
                        [time_signal[0], time_signal3[0]],
                        ['k', 'r'], [2, 2], ['Signal alone', 'Cropped signal'],
                        linestyles=['solid', 'dashed'], zorders=[1, 2])
    fig1.savefig('./lisabeta_waveform3.pdf')
    # fig1, ax1 = fp.plot([t_vect / day, t_vect / day],
    #                     [a_meas, time_signal[0]],
    #                     ['k', 'r'], [1, 2], ['Measured', 'Signal alone'],
    #                     linestyles=['solid', 'solid'], zorders=[1, 2])
    # fig1, ax1 = fp.plot([t_vect / day, t_vect / day],
    #                     [time_signal[0], time_signal2[0]],
    #                     ['k', 'r'], [2, 2], ['Lisabeta', 'LDC'],
    #                     linestyles=['solid', 'dashed'], zorders=[2, 2])

    # fp2 = myplots.fplot(plotconf='frequency')
    # fp2.draw_frame = True
    # fp2.ylabel = r'Fractional frequency'
    # fp2.legendloc = 'upper left'
    # fp2.xlabel = 'Frequency [Hz]'
    # fp2.ylabel = 'TDI amplitude [fractional frequency]'
    # fp2.xscale = 'log'
    # fp2.yscale = 'log'
    # fig2, ax2 = fp2.plot([f_pos, f_pos3],
    #                      [np.abs(af), np.abs(af3)],
    #                      ['k', 'r'],
    #                      [2, 2],
    #                      ['TDI a_mat full', 'TDI a_mat cropped'],
    #                      linestyles=['solid', 'dashed'],
    #                      zorders=[1, 2])
    plt.figure("Frequency domain")
    # plt.loglog(f_pos, np.abs(af), 'k', linewidth=2, label='TDI a_mat full')
    plt.loglog(f_pos3, np.abs(af3), 'r:', linewidth=2, label='TDI a_mat cropped')
    plt.loglog(f_pos3, np.abs(fft(time_signal3[0])[0:n_pos3 + 2]), 'b--', linewidth=2, label='TDI a_mat FFT')
    plt.legend()
    plt.show()




