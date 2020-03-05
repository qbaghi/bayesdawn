# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07

Testing the reduced likelihood model for galactic binaries
such that
h = A beta

@author: qbaghi

"""

import numpy as np
from bayesdawn.waveforms import lisaresp, wavefuncs
from scipy import linalg
from matplotlib import pyplot as plt
import unittest


class TestLisarespMethods(unittest.TestCase):

    def test_design_matrix_freq(self, display=True):

        # Instantiate signal class
        signal_cls = lisaresp.UCBWaveform(wavefuncs.v_func_gb, phi_rot=0,
                                          armlength=2.5e9,
                                          nc=11)
        # Choose GW source parameters
        signal_params = np.array([6.40000000e-23, 6.63200000e-01,
                                  5.78000000e+00,
                                  3.97000000e+00,
                                  3.67279633e+00, 3.05949265e+00,
                                  6.22000000e-03, 3.47000000e-12])
        # Simulation duration, sampling, and analysis bandwidth
        n = 524288
        fs = 0.2
        tobs = n / fs
        f = np.fft.fftfreq(n) * fs
        inds = np.where((f > 6.1e-3) & (f < 6.3e-3))[0]
        f_band = f[inds]

        # Generate the waveform in Fourier domain in single-link measurements
        # Estimation of signal only
        h1, h2, h3 = signal_cls.compute_signal_freq(f[inds],
                                                    signal_params, 1/fs,
                                                    tobs,
                                                    channel='phasemeters')

        signal_fft = np.array([h1, h2, h3, h3, h1, h2]).T

        # Now compute the design matrices
        param_intr = signal_params[[4, 5, 6, 7]]
        mat_list = signal_cls.design_matrix_freq(f[inds], param_intr, 1/fs,
                                                 tobs,
                                                 channel='phasemeters')
        # Choose channel number
        i = 4
        # # Convert the matrix for the fit, that is, a 3 nf x k matrix
        # # Keep only the first 2 paramters
        # # mat_arr_fit = np.vstack([mat for mat in mat_list])
        # mat_arr_fit = mat_list[i][:, 0:2]
        # # signal_flatten = np.hstack(signal_fft)
        # signal_flatten = signal_fft[:, i]
        #
        # # Least squares estimation
        # cov = linalg.pinv(mat_arr_fit.conj().T.dot(mat_arr_fit))
        # beta = cov.dot(mat_arr_fit.conj().T.dot(signal_flatten))
        # print(beta)
        #
        # # Compare the results
        # signal_fft_est = np.dot(mat_arr_fit, beta)

        # Now do the full fit
        mat_full = np.vstack([mat[:, 0:2] for mat in mat_list])
        signal_full = np.concatenate([signal_fft[:, i]
                                      for i in range(signal_fft.shape[1])])
        # Least squares estimation
        cov = linalg.pinv(mat_full.conj().T.dot(mat_full))
        beta = cov.dot(mat_full.conj().T.dot(signal_full))

        # Compare the results
        signal_full_est = np.dot(mat_full, beta)
        n_inds = len(inds)
        signal_fft_est = np.array([signal_full_est[k * n_inds:(k+1)*n_inds]
                                   for k in range(signal_fft.shape[1])]).T

        # Compute residuals
        res = signal_fft - signal_fft_est
        mse = np.sqrt(np.mean(np.abs(res)**2))

        if display:
            # Plots
            plt.figure(0)
            plt.loglog(f_band, np.abs(signal_fft[:, i]), 'b--',
                       label='Generated waveform')
            plt.loglog(f_band, np.abs(signal_fft_est[:, i]), 'r:',
                       label='Reduced model')
            plt.legend()

            plt.figure(1)
            plt.semilogx(f_band, np.angle(signal_fft[:, i]), 'b--',
                         label='Generated waveform')
            plt.semilogx(f_band, np.angle(signal_fft_est[:, i]), 'r:',
                         label='Reduced model')
            plt.legend()

            plt.show()

        print("MSE is " + str(mse))
        print("MSE should be lower than " + str(1e-34))
        self.assertTrue(mse < 1e-34)


if __name__ == '__main__':

    unittest.main()
