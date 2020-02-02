#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:09:36 2019

@author: qbaghi
"""

# For MBHB only, use MLDC code
import GenerateFD_SignalTDIs
import LISAConstants as LC
# from lisabeta.lisa import lisa
import lisabeta.lisa.lisa as lisa
import lisabeta.tools.pyspline as pyspline
import numpy as np
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from . import coeffs

# Global variables
# Indices of parameters in the full parameter vector
# m1, m2, chi1, chi2, Deltat, dist, inc, phi, lambd, beta, psi
i_dist = 5
i_inc = 6
i_phi0 = 7
i_psi = 10
i_tc = 4
# Indices of extrinsic parameters
i_ext = [i_dist, i_inc, i_phi0, i_psi]
# Indices of intrinsic parameters
i_intr = [0, 1, 2, 3, 4, 8, 9]


def indices_low_freq(channel):
    """

    Returns the indices of the integrated gravitational strain signal for the
    low frequency approximation of the tdi response, such that

    dTDI = (1-D^2)(1+D)(H_i - H_j)

    Parameters
    ----------
    channel : string
        tdi channel among {'X1','Y1','Z1'}

    Returns
    -------
    i : scalar integer
        index of first arm
    j : scalar integer
        index of second arm


    """

    if channel == 'X1':
        i = 2
        j = 3

    elif channel == 'Y1':
        i = 3
        j = 1

    elif channel == 'Z1':
        i = 1
        j = 2

    return i, j


def optimal_order(theta, f_0):
    """
    Function calculating the optimal number of sidebands that must be
    calculated in the Bessel approximation to make a good approximation of the
    frequency modulated signal

    """

    R = LC.ua
    d0 = R * np.sin(theta) / LC.c
    # Modulation index
    mf = 2 * np.pi * f_0 * d0
    # Empirical model
    nc = np.int(1.2185 * mf + 5.625) + 1

    return nc


def convert_xyz_to_aet(x, y, z):
    a = (z - x) / np.sqrt(2.0)
    e = (x - 2.0 * y + z) / np.sqrt(6.0)
    t = (x + y + z) / np.sqrt(3.0)
    return a, e, t


def form_design_matrix(uc, k_p, k_c, complex=False):
    """
    Construct the design matrix A such that the GW signal can be written as
    s = A * beta

    Parameters
    ----------
    uc : 2d numpy array
        matrix Uc(f) of waveform basis functions, size n x 5
    k_p : bytearray
        Coefficient k+ of the waveform decomposition
    k_c : bytearray
        Coefficient k+ of the waveform decomposition

    Returns
    -------
    a_tmp : ndarray
        design matrix, size 2n x 4, where n is the number of frequencies

    """

    k_p_tot = np.concatenate((k_p, np.conj(k_p)))
    k_c_tot = np.concatenate((k_c, np.conj(k_c)))

    ac_plus = np.dot(uc, k_p_tot)  # C_func*Dxi_p
    ac_cros = np.dot(uc, k_c_tot)  # C_func*Dxi_c

    if not complex:
        a_tmp = np.empty((np.int(2 * ac_plus.shape[0]), 4), dtype=np.float64)
        a_tmp[:, 0] = np.concatenate((ac_plus.real, ac_plus.imag))
        a_tmp[:, 1] = np.concatenate((ac_cros.real, ac_cros.imag))
        a_tmp[:, 2] = np.concatenate((ac_plus.imag, -ac_plus.real))
        a_tmp[:, 3] = np.concatenate((ac_cros.imag, -ac_cros.real))
    else:
        a_tmp = np.empty((ac_plus.shape[0], 4), dtype=np.complex128)
        a_tmp[:, 0] = ac_plus  # C_func*Dxi_p
        a_tmp[:, 1] = ac_cros  # C_func*Dxi_c
        a_tmp[:, 2] = -1j * ac_plus  # S_func*Dxi_p
        a_tmp[:, 3] = -1j * ac_cros  # S_func*Dxi_c

    return a_tmp


class GWwaveform(object):
    """
    Class to compute the LISA response to an incoming gravitational wave.

    """

    def __init__(self, phi_rot=0, armlength=2.5e9):
        """


        Parameters
        ----------
        phi_rot : scalar float
            initial angle of LISA constellation
        armlength : scalar float
            Arm length (default is 2.5e9 m)

        """

        self.phi_rot = phi_rot
        self.armlength = armlength

        self.R = LC.ua
        self.f_T = 1. / LC.year


class UCBWaveform(GWwaveform):
    """
    Class to compute the LISA response to an incoming gravitational wave.

    """

    def __init__(self, v_func, phi_rot=0, armlength=2.5e9, nc=15):
        """


        Parameters
        ----------
        v_func : callable
            function of frequency giving the Fourier transform of
            exp(-j*Phi(t)), where Phi(t) is the phase of the gravitatonal wave
        fs : scalar float
            sampling frequency
        phi_rot : scalar float
            initial angle of LISA constellation
        armlength : scalar float
            Arm length (default is 2.5e9 m)
        nc : scalar integer
            order of the Bessel function decomposition


        """

        super().__init__(phi_rot=phi_rot, armlength=armlength)

        # Name of intrinsic parameters
        self.names = ['theta', 'phi', 'f_0', 'f_dot']
        # Phase function
        self.v_func = v_func
        # For Fourier series decomposition: m = 0, 1, .. 4
        self.nc = nc
        self.m_max = 4
        self.m_vect = np.arange(0, self.m_max + 1)
        self.jw2 = []

    def o2i(self, x):
        """
        convert Bessel order to vector index
        Example:
        Order n corresponds to the n + nc + 4 th entry in the vector

        n_arr = [- nc - 4, .. nc + 4]

        """

        return np.int(x + self.nc + 4)

    def bessel_decomp_pos(self, v_minus_list, jw, e_vect, n_vect, m):
        """

        This function calculates y_c(f) and y_s(f), basic quantities needed in
        the frequency response to gravitational wave,
        ONLY KEEPING THE TERMS THAT ARE SIGNIFICANT FOR POSITIVE FREQUENCIES.

        Parameters
        ----------
        v_minus : list of array_like
            stored values of v(f + k f_t)
        jw : list of complex scalar
            stored values of jv(n, 2*np.pi*f_0*d0)
        e_vect : list of complex scalars
            stored values of np.exp(1j*n*varphi)
        n_vect : array_like
            array of indices from -nc to nc (possibly with some skipped)
        m : scalar integer
            positive or negative integer indicating the harmonic of f_T
            (yearly period)

        Returns
        -------
        w : numpy array
            vector of values of w(f) (size N) calculated at given frequencies

        """

        # when v = FT(e^(-jPhi))
        v_minus = [np.conj(v_minus_list[self.o2i(-m-n)])
                   * e_vect[np.int(n + self.nc)] for n in n_vect]
        # when v = FT(e^(jPhi))
        # v_minus = [v_minus_list[self.o2i(n + m)] * e_vect[np.int(n + self.nc)]
        #            for n in n_vect]

        y_c = sum([jw[k]*1/2.*v_minus[k] for k in range(len(v_minus))])
        y_s = -1j*y_c

        return y_c, y_s

    def u_matrices(self, f, params, ts, t_start, t_end, derivative=2):
        """

        Compute the frequency matrix W involved in the calculation of the basis
        functions

        .. math::

            \tilde{\dot{u}}^{(i)}_{c,\alpha}(f) = Uc K^{ij}_{\alpha}

        Parameters
        ----------
        f : array_like or list of array_like
            frequencies where to evalutate the function w(f)
        params : array_like
            vector of extrinsinc parameters: theta, phi, f_0, f_dot
        v_func : callable
            function of frequency giving the Fourier transform of
            exp(-j*Phi(t)), where Phi(t) is the phase of the gravitatonal wave
        ts : scalar float
            Sampling time
        t_start : scalar float
            Starting observation time
        t_end: scalar float
            End observation time
        derivative : scalar integer
            order of the derivative, integer >=0

        Returns
        -------
        Uc : 2d numpy array
            matrix Uc(f) of size N x 5
        Us : 2d numpy array
            matrix Us(f) of size N x 5

        """

        theta = params[0]
        phi = params[1]
        f_0 = params[2]
        f_dot = params[3]

        d0 = self.R * np.sin(theta) / LC.c

        varphi = phi + np.pi/2

        # Precompute the Bessel coefficients f, f_0, f_dot, ts, T1, T2
        # when v = FT(e^(-jPhi))
        # v_minus_list = [self.v_func(-f + k * self.f_T, f_0, f_dot, ts, t_start,
        #                             t_end)
        #                 for k in range(-self.nc - self.m_max,
        #                                self.nc + self.m_max)]
        # (f, f_0, f_dot, T, ts)
        v_minus_list = [self.v_func(-f + k * self.f_T, f_0, f_dot, t_end, ts)
                        for k in range(-self.nc - self.m_max,
                                       self.nc + self.m_max + 1)]
        # when v = FT(e^(+jPhi))
        # v_minus_list = [self.v_func(f + k * self.f_T, f_0, f_dot, ts, t_start,
        #                             t_end)
        #                 for k in range(-self.nc - self.m_max,
        #                                self.nc + self.m_max + 4)]

        n_vect = np.arange(-self.nc, self.nc + 1)
        e_vect = np.exp(1j*n_vect*varphi)

        jw = [special.jv(n, 2*np.pi*f_0*d0) for n in n_vect]

        # Valid only for positive frequencies:
        u = [self.bessel_decomp_pos(v_minus_list, jw, e_vect, n_vect, m)
             for m in - self.m_vect]
        u.extend([u[0]])  # Avoid computing m=0 twice
        u.extend([self.bessel_decomp_pos(v_minus_list, jw, e_vect, n_vect, m)
                  for m in self.m_vect[1:]])

        jomega = (2*np.pi*1j*f)**derivative
        uc = np.array([jomega * z[0] for z in u]).T
        us = np.array([jomega * z[1] for z in u]).T

        return uc, us

    def design_matrix_freq(self, uc, us, k_p, k_c):
        """
        Compute design matrix such that the TDI variable (fist generation)
        can be written as

        TDI(f) = A(theta,phi,f0,f_dot,f) beta(A_p,A_c,phi_0,psi)

        beta = (gamma_p,sigma_p,gamma_c,sigma_c)

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        params : array_like
            vector of extrinsinc parameters: theta, phi, f_0, f_dot
        ts : scalar float
            Sampling time
        t_start : scalar float
            Starting observation time
        t_end: scalar float
            End observation time
        channel : string
            tdi channel among {'X1','Y1','Z1'}
            Can also be a list of channels, like ['X1','Y1','Z1']. In that case
            the output is also a list of matrices.


        Returns
        -------
        A : numpy array
            model matrix of size (N x K)


        """

        # # Compute model matrix in frequency domain (in fractional frequency)
        # uc = self.u_matrices(f, params, ts, t_start, t_end,
        #                      derivative=derivative)
        #
        # return [form_design_matrix(uc, k_p[i], k_c[i], complex=complex)
        #         for i in range(len(k_p))]

        # # Indices of arms to be considered for the TDI variable
        # i, j = pyFLR.indices_low_freq(tdi)
        # N = len(f)
        #
        # # Compute coefficients of the decomposition of basis functions
        # u_alpha k_p, k_c = k_coeffs(params,Phi_rot,i,j)

        k_p_tot = np.concatenate((k_p, np.conj(k_p)))
        k_c_tot = np.concatenate((k_c, np.conj(k_c)))

        # Compute response in the slowly varying response approximation
        a_tmp = np.empty((uc.shape[0], 4), dtype=np.complex128)
        a_tmp[:, 0] = np.dot(uc, k_p_tot)  # C_func*Dxi_p
        a_tmp[:, 1] = np.dot(uc, k_c_tot)  # C_func*Dxi_c
        a_tmp[:, 2] = np.dot(us, k_p_tot)  # S_func*Dxi_p
        a_tmp[:, 3] = np.dot(us, k_c_tot)  # S_func*Dxi_c
        # a_mat = (armlength/LC.c)**2*a_tmp

        return a_tmp

    def compute_signal_freq(self, f, params, del_t, tobs, channel='TDIAET'):
        """
        Compute LISA's response to the incoming galactic binary GW in the
        frequency domain

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        params : array_like
            vector of parameters: a0, incl, phi_0, psi, theta, phi, f_0, f_dot
        tobs : scalar float
            Observation Duration
        ts : scalar float
            sampling cadence
        channel : string
            tdi channel among {'X1','X2','X3'}
        tref : float
            starting time of the waveform

        Returns
        -------
        ch_interp : list of numpy arrays
            list containing the 3 TDI responses in channels A, E and T,
            expressed in fractional frequency amplidudes,
            such that ch_interp[i] corresponds to fft(ch[i]) without any
            normalization

        """

        # Extract physical parameters
        a0, incl, phi_0, psi, theta, phi, f_0, f_dot = params
        # Compute constant amplitude coefficients
        beta = coeffs.beta_gb(a0, incl, phi_0, psi)
        # Vector of intrinsic parameters
        param_intr = np.array([theta, phi, f_0, f_dot])

        # domain and in each channel
        if channel == 'phasemeters':
            print(channel)
            pre = (self.armlength / (4 * LC.c))
            derivative = 1
            # There is a mixing to convert it in the phasemeter measurements!
            # i_mix = [2, 0, 1]
            # 3 1 2
            # theta,phi,f_0,f_dot

            # Compute the required coefficients response depending on channel
            kp3, kc3 = coeffs.k_coeffs_single(param_intr, self.phi_rot, 3)
            kp1, kc1 = coeffs.k_coeffs_single(param_intr, self.phi_rot, 1)
            kp2, kc2 = coeffs.k_coeffs_single(param_intr, self.phi_rot, 2)

            k_p_list = [kp3, kp1, kp2]
            k_c_list = [kc3, kc1, kc2]

        elif (channel == 'TDIAET') | (channel == 'TDIXYZ'):
            print(channel)
            pre = (self.armlength / LC.c) ** 2
            derivative = 2

            # Compute the required coefficients response depending on channel
            i, j = indices_low_freq('X1')  # 23
            kp23, kc23 = coeffs.k_coeffs(params, self.phi_rot, i, j)
            i, j = indices_low_freq('Y1')  # 31
            kp31, kc31 = coeffs.k_coeffs(params, self.phi_rot, i, j)
            i, j = indices_low_freq('Z1')  # 12
            kp12, kc12 = coeffs.k_coeffs(params, self.phi_rot, i, j)

            k_p_list = [kp23, kp31, kp12]
            k_c_list = [kc23, kc31, kc12]

        # Compute model matrix in frequency domain (in fractional frequency)
        uc, us = self.u_matrices(f, param_intr, del_t, 0, tobs,
                                 derivative=derivative)

        mat_list = [self.design_matrix_freq(uc, us, k_p_list[i], k_c_list[i])
                    for i in range(len(k_p_list))]

        # Transform to complex number
        ch_list = [pre * mat.dot(beta) for mat in mat_list]

        if channel == 'TDIAET':
            a, e, t = convert_xyz_to_aet(ch_list[0], ch_list[1], ch_list[2])
            return a, e, t
        else:
            return ch_list[0], ch_list[1], ch_list[2]


class MBHBWaveform(GWwaveform):
    """
    Class to compute the LISA response to an incoming gravitational wave.

    """

    def __init__(self, phi_rot=0, armlength=2.5e9, reduced=False):
        """


        Parameters
        ----------
        v_func : callable
            function of frequency giving the Fourier transform of exp(-j*Phi(t)),
            where Phi(t) is the phase of the gravitatonal wave
        fs : scalar float
            sampling frequency
        phi_rot : scalar float
            initial angle of LISA constellation
        armlength : scalar float
            Arm length (default is 2.5e9 m)
        nc : scalar integer
            order of the Bessel function decomposition


        """

        super().__init__(phi_rot=phi_rot, armlength=armlength)

        # Name of intrinsic parameters
        self.names = ['m1', 'm2', 'a1', 'a2', 'beta', 'psi', 'tc']

        # For PhenomD waveform
        # --------------------
        # fRef=0 means fRef=fpeak in PhenomD or maxf if out of range
        self.fRef = 0.0
        self.t0 = 0.0

        # Indices of parameters in the full parameter vector
        self.i_dist = 5
        self.i_inc = 6
        self.i_phi0 = 7
        self.i_psi = 10
        # Indices of extrinsic parameters
        self.i_ext = [5, 6, 7, 10]
        # Indices of intrinsic parameters
        self.i_intr = [0, 1, 2, 3, 4, 8, 9]

    def interpolate_waveform(self, fr_in, fr_out, x, real_imag=False):

        real_func = spline(fr_in, np.real(x), ext='zeros')
        imag_func = spline(fr_in, np.imag(x), ext='zeros')
        xf_real = real_func(fr_out)
        xf_imag = imag_func(fr_out)

        if real_imag:
            return xf_real, xf_imag
        else:
            return xf_real + 1j * xf_imag

    def shift_time(self, f, xf, delay):

        return xf * np.exp(-2j * np.pi * f * delay)

    def compute_signal_freq(self, f, params, del_t, tobs, channel='TDIAET', ldc=False, tref=0):
        """
        Compute LISA's response to the incoming MBHB GW in the frequency domain

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        params : array_like
            vector of parameters: phi0, m1, m2, a1, a2, dist, inc, lam, beta, psi, tc
        Tobs : scalar float
            Observation Duration
        ts : scalar float
            sampling cadence
        channel : string
            tdi channel among {'X1','X2','X3'}
        ldc : boolean
            whether to use LDC or Lisabeta waveform
        tref : float
            starting time of the waveform

        Returns
        -------
        ch_interp : list of numpy arrays
            list containing the 3 TDI responses in channels A, E and T, expressed in fractional frequency amplidudes,
            such that ch_interp[i] corresponds to fft(ch[i]) without any normalization

        """

        # phi0, m1, m2, a1, a2, dist, inc, lam, beta, psi, tc = params
        m1 = params[0]
        m2 = params[1]
        xi1 = params[2]
        xi2 = params[3]
        tc = params[4]
        dist = params[5]
        inc = params[6]
        phi0 = params[7]
        lam = params[8]
        beta = params[9]
        psi = params[10]

        if ldc:
            # TDI response on native grid
            fr, x, y, z, wfTDI = GenerateFD_SignalTDIs.MBHB_LISAGenerateTDIfast(phi0, self.fRef, m1, m2, xi1, xi2,
                                                                                dist, inc,
                                                                                lam,
                                                                                beta,
                                                                                psi,
                                                                                tobs,
                                                                                tc,
                                                                                del_t, tShift=0, fmin=1.e-5,
                                                                                fmax=0.5 / del_t, frqs=None, resf=None)

            ch_interp = [self.interpolate_waveform(fr, f, ch) for ch in [x, y, z]]
            a, e, t = convert_xyz_to_aet(ch_interp[0], ch_interp[1], ch_interp[2])

            if (channel == 'TDIAET') | (channel == ['A', 'E', 'T']):
                # print(channel)
                # a, e, t = convert_xyz_to_aet(ch_interp[0], ch_interp[1], ch_interp[2])
                ch_interp = [a / del_t, e / del_t, t / del_t]
            elif channel == ['A']:
                ch_interp = a / del_t
            elif channel == ['E']:
                ch_interp = e / del_t
            elif channel == ['T']:
                ch_interp = t / del_t

        else:
            # TDI response
            wftdi = lisa.GenerateLISATDI(params, tobs=tobs, minf=1e-5, maxf=1., tref=tref, torb=0., TDItag='TDIAET',
                                         acc=1e-4, order_fresnel_stencil=0, approximant='IMRPhenomD',
                                         responseapprox='full', frozenLISA=False, TDIrescaled=False)
            signal_freq = lisa.GenerateLISASignal(wftdi, f)
            ch_interp = [self.shift_time(f, signal_freq['ch1'], tobs).conj() / del_t,
                         self.shift_time(f, signal_freq['ch2'], tobs).conj() / del_t,
                         self.shift_time(f, signal_freq['ch3'], tobs).conj() / del_t]
            # Devide by del_t to be consistent with the unnormalized DFT

        # Interpolate the response on required grid
        # return signal_freq['ch1'], signal_freq['ch2'], signal_freq['ch3']
        # return ch_interp[0] #, ch_interp[1], ch_interp[2]
        return ch_interp

    def single_design_matrix(self, tdi_resp_plus, tdi_resp_cros):

        a_mat = np.empty((2 * tdi_resp_plus.shape[0], 2), dtype=np.float64)
        a_mat[:, 0] = np.concatenate((tdi_resp_plus.real, tdi_resp_plus.imag))
        a_mat[:, 1] = np.concatenate((tdi_resp_cros.real, tdi_resp_cros.imag))
        # a_mat[:, 2] = np.concatenate((tdi_resp_plus.imag, -tdi_resp_plus.real))
        # a_mat[:, 3] = np.concatenate((tdi_resp_cros.imag, -tdi_resp_cros.real))

        return a_mat

    def design_matrix_freq(self, f, params_intr, del_t, tobs, channel='TDIAET',
                           complex=False, tref=0):
        """
        Compute design matrix such that the TDI variable (fist generation)
        can be written as

        TDI(f) = A(theta,phi,f0,f_dot,f) beta(A_p,A_c,phi_0,psi)

        beta = (gamma_p,sigma_p,gamma_c,sigma_c)

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        params_intr : array_like
            vector of intrinsinc parameters
        del_t : scalar float
            sampling cadence
        tdi : string
            tdi channel among {'X1','X2','X3'}
        tobs : float
            observation time
        channel: str
            TDI channels to consider
        complex : bool
            whether to used complex-valued or real-valued design matrices


        Returns
        -------
        A : numpy array
            model matrix of size (N x K)


        """

        # LISABETA

        # # Indices of parameters in the full parameter vector
        # self.i_dist = 5
        # self.i_inc = 6
        # self.i_phi0 = 7
        # self.i_psi = 10
        # # Indices of extrinsic parameters
        # self.i_ext = [5, 6, 7, 10]
        # # Indices of intrinsic parameters
        # self.i_intr = [0, 1, 2, 3, 4, 8, 9]
        # m1, m2, chi1, chi2, tc, dist, inc, phi, lambd, beta, psi = params
        params_1 = np.zeros(11)
        params_2 = np.zeros(11)

        # Same intrinsic parameters
        params_1[self.i_intr] = params_intr
        params_2[self.i_intr] = params_intr
        # Initial phase
        params_1[self.i_phi0] = 0
        params_2[self.i_phi0] = 0
        # Luminosity distance (Mpc)
        params_1[self.i_dist] = 1e3
        params_2[self.i_dist] = 1e3
        # Inclination
        params_1[self.i_inc] = 0  # 0.5 * np.pi
        params_2[self.i_inc] = 0  # 0.5 * np.pi
        # Polarization angle
        params_1[self.i_psi] = 0.0
        params_2[self.i_psi] = np.pi / 4

        # amp = 2* G * mu / (dl * c**2)

        # Amp, iota, psi_m, phi0
        # 1. 1.e-21, 0.5*np.pi, 0.0, 0.0
        # 2. 1.e-21, 0.5*np.pi, 0.25*np.pi, 0.0

        # Calculate the response on required grid
        tdi_response_plus = self.compute_signal_freq(f, params_1, del_t, tobs, channel=channel, tref=tref)
        tdi_response_cros = self.compute_signal_freq(f, params_2, del_t, tobs, channel=channel, tref=tref)

        if not complex:

            mat_list = [self.single_design_matrix(tdi_response_plus[i], tdi_response_cros[i])
                        for i in range(len(tdi_response_plus))]

        else:

            mat_list = [np.array([tdi_response_plus[i], tdi_response_cros[i]]).T for i in range(len(tdi_response_plus))]

        # # Compute model matrix in frequency domain (in fractional frequency)
        # # Uc,Us = self.u_matrices(f,params,ts,Tstart,Tend,nc = self.nc, derivative = 2)
        # Uc = self.u_matrices(f, params, ts, Tstart, Tend, nc=self.nc, derivative=2)
        #
        # k_p_tot = np.concatenate((k_p, np.conj(k_p)))
        # k_c_tot = np.concatenate((k_c, np.conj(k_c)))
        #
        #
        # # gamma_p,gamma_c,sigma_p,sigma_c
        # A_tmp = np.empty((2 * len(f), 4), dtype=np.float64)
        # Ac_plus = np.dot(Uc, k_p_tot)  # C_func*Dxi_p
        # Ac_cros = np.dot(Uc, k_c_tot)  # C_func*Dxi_c
        # A_tmp[:, 0] = np.concatenate((Ac_plus.real, Ac_plus.imag))
        # A_tmp[:, 1] = np.concatenate((Ac_cros.real, Ac_cros.imag))
        # A_tmp[:, 2] = np.concatenate((Ac_plus.imag, -Ac_plus.real))
        # A_tmp[:, 3] = np.concatenate((Ac_cros.imag, -Ac_cros.real))
        #
        # A = (self.armlength / LC.c) ** 2 * A_tmp

        return mat_list


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
    # print("Length of initial frequency grid " + str(len(fs)))

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
        tr_int = tr

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


def design_matrix(params_intr, freq, tobs, tref=0, t_offset=52.657,
                  channels=None):
    """
    Design matrix for reduced order likelihood
    Parameters
    ----------
    par_intr : ndarray
        instrinsic sampling parameters Mc, q, tc, chi1, chi2, np.sin(bet), lam
    freq :
    tobs
    tref
    t_offset
    channels

    Returns
    -------

    """

    # m1, m2, chi1, chi2, tc, dist, inc, phi, lambd, beta, psi = params
    params_1 = np.zeros(11)
    params_2 = np.zeros(11)
    # Save intrinsic parameters
    params_1[i_intr] = params_intr
    params_2[i_intr] = params_intr
    # Luminosity distance (Mpc)
    params_1[i_dist] = 1e3
    params_2[i_dist] = 1e3
    # # Coalescence time
    # params_1[i_tc] = 0  # 0.5 * np.pi
    # params_2[i_tc] = tobs/2  # 0.5 * np.pi
    # Polarization angle
    params_1[i_psi] = 0.0
    params_2[i_psi] = np.pi / 4
    if channels is None:
        channels = [1, 2, 3]

    # TDI response 1
    wftdi_1 = lisa.GenerateLISATDI(params_1, tobs=tobs, minf=1e-5, maxf=1., tref=tref, torb=0., TDItag='TDIAET',
                                   acc=1e-4, order_fresnel_stencil=0, approximant='IMRPhenomD',
                                   responseapprox='full', frozenLISA=False, TDIrescaled=False)
    # TDI response 2
    # wftdi_2 = lisa.GenerateLISATDI(params_2, tobs=tobs, minf=1e-5, maxf=1., tref=tref, torb=0., TDItag='TDIAET',
    #                                acc=1e-4, order_fresnel_stencil=0, approximant='IMRPhenomD',
    #                                responseapprox='full', frozenLISA=False, TDIrescaled=False)

    fs1 = wftdi_1['freq']
    # fs2 = wftdi_2['freq']
    # tr1 = [wftdi_1['transferL' + str(np.int(i))] for i in channels]
    # tr2 = [wftdi_2['transferL' + str(np.int(i))] for i in channels]

    if freq is not None:
        amp = pyspline.resample(freq, fs1, wftdi_1['amp'])
        phase = pyspline.resample(freq, fs1, wftdi_1['phase'])
        # amp2 = pyspline.resample(freq, fs2, wftdi_1['amp'])
        # phase2 = pyspline.resample(freq, fs2, wftdi_1['phase'])
        tr_int1 = [pyspline.resample(freq, fs1, wftdi_1['transferL' + str(np.int(i))]) for i in channels]
        # tr_int2 = [pyspline.resample(freq, fs2, wftdi_2['transferL' + str(np.int(i))]) for i in channels]
    else:
        freq = wftdi_1['freq']
        tr_int1 = [wftdi_1['transferL' + str(np.int(i))] for i in channels]
        # tr_int2 = [wftdi_2['transferL' + str(np.int(i))] for i in channels]

    h = amp * np.exp(1j * phase)
    # h2 = amp2 * np.exp(1j * phase2)
    # Phasor for the time shift
    z = np.exp(-2j * np.pi * freq * t_offset)

    # mat_list = [np.vstack((h * tr_int1[i - 1] * z, h2 * tr_int2[i - 1] * z)).conj().T for i in channels]
    mat_list = [np.array([h * tr_int1[i - 1] * z]).conj().T for i in channels]

    return mat_list  # , tr_int1, tr_int2

# def design_matrix(par_intr, freq_data, tobs, tref=0, t_offset=52.657, channels=None, minf=1e-5, maxf=1., acc=1e-4,
#                   dist=1e3, torb=0, LISAconst=pyresponse.LISAconstProposal, responseapprox='full',
#                   frozenLISA=False, TDIrescaled=False):
#     """
#     Design matrix for reduced order likelihood
#     Parameters
#     ----------
#     par_intr : ndarray
#         instrinsic sampling parameters Mc, q, tc, chi1, chi2, np.sin(bet), lam
#     freq :
#     tobs
#     tref
#     t_offset
#     channels
#
#     Returns
#     -------
#
#     """
#     # np.array([Mc, q, tc, chi1, chi2, np.sin(bet), lam])
#     # Convert likelihood parameters into waveform-compatible parameters
#     params_intr = like_to_waveform_intr(par_intr)
#     # The full set of parameters is m1, m2, chi1, chi2, tc, dist, incl, phi0, lam, bet, psi
#     # Indices of intrinsic parameters m1, m2, chi1, chi2, tc, lam, bet
#     # Initial phase: same!
#     phi0 = 0
#     # Inclination: same!
#     inc = 0
#     # Polarization angle: 45 degree angle difference
#     psi_1 = 0.0
#     psi_2 = np.pi / 4
#     # TDI response
#     m1, m2, chi1, chi2, tc, lambd, beta = params_intr
#     # Units
#     q = m1 / m2
#     m1_SI = m1 * pyconstants.MSUN_SI
#     m2_SI = m2 * pyconstants.MSUN_SI
#     dist_SI = dist * 1e6 * pyconstants.PC_SI
#     M = m1 + m2
#     Ms = M * pyconstants.MTSUN_SI
#     if tobs is not None:
#         minf = np.fmax(minf, pytools.funcNewtonianfoft(m1, m2, tobs * pyconstants.YRSID_SI))
#     Mfmin = Ms * minf
#     Mfmax = Ms * maxf
#     Mfmax_model = 0.2
#     Mfmax = np.fmin(Mfmax, Mfmax_model)
#     maxf = Mfmax / Ms
#
#     # Combined frequency grid for the waveform
#     gridfreqClass = pytools.FrequencyGrid(minf, maxf, M, q, acc=acc)
#     gridfreq = gridfreqClass.get_freq()
#
#     # Generate IMRPhenomD waveform and add time shift
#     # Calling with fRef=0., phiRef=0. defines the source frame
#     phiRef = 0.
#     fRef = 0.
#     wfClass = pyIMRPhenomD.IMRPhenomDh22AmpPhase(gridfreq, phiRef, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
#     freq, amp, phase = wfClass.get_waveform()
#     phase += 2 * np.pi * freq * tc
#
#     # Build spline for tf
#     phaseov2pisplineClass = pyspline.CubicSpline(freq, phase / (2*np.pi))
#     tfspline = phaseov2pisplineClass.get_spline_d()
#
#     # Global shift of phase to account for
#     phase += 2 * np.pi * freq * tref
#
#     # ==================================================================================================================
#     # Compute TDI response for the two polarizations
#     # ==================================================================================================================
#     l0 = 2
#     m0 = 2
#     tdi_class_1 = pyresponse.LISAFDresponseTDI3Chan(gridfreq, tfspline, torb, l0, m0, inc, phi0, lambd, beta, psi_1,
#                                                  TDI='TDIAET', LISAconst=LISAconst, responseapprox=responseapprox,
#                                                  frozenLISA=frozenLISA, TDIrescaled=TDIrescaled)
#     tdi_class_2 = pyresponse.LISAFDresponseTDI3Chan(gridfreq, tfspline, torb, l0, m0, inc, phi0, lambd, beta, psi_2,
#                                                  TDI='TDIAET', LISAconst=LISAconst, responseapprox=responseapprox,
#                                                  frozenLISA=frozenLISA, TDIrescaled=TDIrescaled)
#
#     # phaseRdelay, transferL1, transferL2, transferL3
#     transfer_list_1 = tdi_class_1.get_response()
#     transfer_list_2 = tdi_class_2.get_response()
#
#     # ==================================================================================================================
#     # Interpolation of the response
#     # ==================================================================================================================
#     if channels is None:
#         channels = [1, 2, 3]
#
#     if freq_data is not None:
#         amp = pyspline.resample(freq_data, freq, amp)
#         phase = pyspline.resample(freq_data, freq, phase)
#         tr_int_1 = [pyspline.resample(freq_data, freq, transfer_list_1[i]) for i in channels]
#         tr_int_2 = [pyspline.resample(freq_data, freq, transfer_list_2[i]) for i in channels]
#     else:
#         tr_int_1 = [transfer_list_1[i] for i in channels]
#         tr_int_2 = [transfer_list_2[i] for i in channels]
#
#     # GW strain
#     h = amp * np.exp(1j * phase)
#     # Phasor for the time shift
#     z = np.exp(-2j * np.pi * freq_data * t_offset)
#     # Compute response for all channels
#     ch_interp_1 = [np.conj(h * tr_int_1[i - 1] * z) for i in channels]
#     ch_interp_2 = [np.conj(h * tr_int_2[i - 1] * z) for i in channels]
#     # Build design matrix
#     mat_list = [np.array([ch_interp_1[i], ch_interp_2[i]]).T for i in range(len(ch_interp_1))]
#
#     return mat_list
