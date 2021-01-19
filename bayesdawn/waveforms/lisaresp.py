#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:09:36 2019

@author: qbaghi
"""

# For MBHB only, use MLDC code
try:
    import GenerateFD_SignalTDIs
    import LISAConstants as LC
except ImportError:
    print("Proceed without MLDC packages.")
# from lisabeta.lisa import lisa
try:
    import lisabeta.lisa.lisa as lisa
    import lisabeta.tools.pyspline as pyspline
except ImportError:
    print("Proceed without lisabeta package.")
import numpy as np
import time
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from . import coeffs
import pyfftw
from ..utils import physics
from pyfftw.interfaces.numpy_fft import fft, ifft, rfft
pyfftw.interfaces.cache.enable()

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

    dTDI = (1-d^2)(1+d)(H_i - H_j)

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
    mf = 2 * np.pi * f_0 * np.abs(d0)
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
    Construct the design matrix a_mat such that the GW signal can be written as
    s = a_mat * beta

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


def compute_full_series(v_list, c_mat):
    """Short summary.

    Parameters
    ----------
    v_list : ndarra
        list of nc values v(f - n / Tobs) where v is the exponential GW
        phase for frequencies f. Size 2nc x nf
    c_mat : ndarray
        Fourier series coefficients of a_slow, size n_channel x 2nc x 2

    Returns
    -------
    series: ndarray
        Array of size n_channel x nc x 2 giving the Fourier-series approximation
        of a_slow(f) in the frequency domain.
        a_slow(f) = sum_{n=-nc}^{+nc} c[n] v(f - n / Tobs)

    """
    
    # np.array([v_list]).T has size nf x 2nc
    # We used to multiply a 2nc x 2 matrix by a nf x 2nc matrix
    # to get a 2nc x nf x 2 array
    return np.sum(c_mat * np.array([v_list]).T, axis=1)



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
        self.f_t = 1. / LC.year


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
            vector of values of w(f) (size n_data) calculated at given frequencies

        """

        # when v = FT(e^(-jPhi))
        v_minus = [np.conj(v_minus_list[self.o2i(-m-n)])
                   * e_vect[np.int(n + self.nc)] for n in n_vect]
        # when v = FT(e^(jPhi))
        # v_minus = [v_minus_list[self.o2i(n + m)] * e_vect[np.int(n + self.nc)]
        #            for n in n_vect]

        y_c = sum([jw[k]*1/2.*v_minus[k] for k in range(len(v_minus))])
        # y_s = -1j*y_c

        # return y_c, y_s
        return y_c

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
            matrix Uc(f) of size n_data x 5
        Us : 2d numpy array
            matrix Us(f) of size n_data x 5

        """

        theta = params[0]
        phi = params[1]
        f_0 = params[2]
        f_dot = params[3]

        d0 = self.R * np.sin(theta) / LC.c
        varphi = phi + np.pi/2

        # Precompute the Bessel coefficients f, f_0, f_dot, ts, T1, T2
        v_minus_list = [self.v_func(-f + k * self.f_t, f_0, f_dot, t_end, ts)
                        for k in range(-self.nc - self.m_max,
                                       self.nc + self.m_max + 1)]

        n_vect = np.arange(-self.nc, self.nc + 1)
        e_vect = np.exp(1j*n_vect*varphi)

        jw = [special.jv(n, 2*np.pi*f_0*d0) for n in n_vect]

        # List of ucm for all m, valid only for positive frequencies:
        u = [self.bessel_decomp_pos(v_minus_list, jw, e_vect, n_vect, m)
             for m in - self.m_vect]
        u.extend([u[0]])  # Avoid computing m=0 twice+
        u.extend([self.bessel_decomp_pos(v_minus_list, jw, e_vect, n_vect, m)
                  for m in self.m_vect[1:]])

        # jomega = (2*np.pi*1j*f)**derivative
        # uc = np.array([jomega * z[0] for z in u]).T
        # us = np.array([jomega * z[1] for z in u]).T

        # uc = np.array([jomega * z for z in u]).T
        #  us = - 1j * uc
        # return uc  # , us
        return np.transpose(u * np.array([(2*np.pi*1j*f)**derivative]))

    def single_design_matrix_freq(self, uc, k_p, k_c, full=True):
        """
        Compute design matrix such that the measurements y
        can be written as

        y(f) = a_mat(theta,phi,f0,f_dot,f) beta(A_p,A_c,phi_0,psi)

        beta = (gamma_p,sigma_p,gamma_c,sigma_c)

        Parameters
        ----------
        uc : ndarray
            waveform matrix of shape nf x n_coeff
        k_p : ndarray
            vector of + coefficients (size n_coeff = 5)
        k_c : ndarray
            vector of x coefficients (size n_coeff)
        full : bool
            If True, outputs the full design matrix of size nf x 4
            If False, outputs only half of it (as the second and third columns
            are proportional to the first and the second, respectively)

        Returns
        -------
        a_mat : numpy array
            model matrix of size (n_data x K)


        """

        k_p_tot = np.concatenate((k_p, np.conj(k_p)))
        k_c_tot = np.concatenate((k_c, np.conj(k_c)))

        # # Compute response in the slowly varying response approximation
        # a_tmp = np.empty((uc.shape[0], 4), dtype=np.complex128)
        # a_tmp[:, 0] = np.dot(uc, k_p_tot)  # C_func*Dxi_p
        # a_tmp[:, 1] = np.dot(uc, k_c_tot)  # C_func*Dxi_c
        # a_tmp[:, 2] = np.dot(us, k_p_tot)  # S_func*Dxi_p
        # a_tmp[:, 3] = np.dot(us, k_c_tot)  # S_func*Dxi_c
        # # a_mat = (armlength/LC.c)**2*a_tmp

        if full:
            # Alternate way, using the fact that us = - 1j * uc:
            a_tmp = np.empty((uc.shape[0], 4), dtype=np.complex128)
            a_tmp[:, 0] = np.dot(uc, k_p_tot)  # C_func*Dxi_p
            a_tmp[:, 1] = np.dot(uc, k_c_tot)  # C_func*Dxi_c
            a_tmp[:, 2] = - 1j * a_tmp[:, 0]  # S_func*Dxi_p
            a_tmp[:, 3] = - 1j * a_tmp[:, 1]  # S_func*Dxi_c
        else:
            a_tmp = np.empty((uc.shape[0], 2), dtype=np.complex128)
            a_tmp[:, 0] = np.dot(uc, k_p_tot)  # C_func*Dxi_p
            a_tmp[:, 1] = np.dot(uc, k_c_tot)  # C_func*Dxi_c
            # a_tmp = np.hstack([uc, uc]).dot()
            # a_tmp = np.hstack([uc.dot(kt) for kt in [k_p_tot, k_c_tot]])

        return a_tmp

    def design_matrix_freq(self, f, param_intr, del_t, tobs,
                           channel='TDIAET', full=True):
        """
        Compute LISA's response to the incoming galactic binary GW in the
        frequency domain

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        param_intr : array_like
            vector of intrinsic parameters:
            colatitude theta [rad] = Ecliptic Latitude + pi/2
            longitude phi [rad] = Ecliptic longitude - pi
            frequency f_0 [Hz],
            frequency derivative f_dot [Hz^2]
        tobs : scalar float
            Observation Duration
        ts : scalar float
            sampling cadence
        channel : string
            tdi channel among {'X1','X2','X3'}
        full : bool
            If True, outputs the full design matrices of size nf x 4
            If False, outputs only half of them (as for each matrix, the second
            and third columns are proportional to the first and the second,
            respectively)


        Returns
        -------
        mat_list : list of numpy arrays
            list containing the design matrices


        """

        # Compute response coefficients for channels s1, s2, s3
        k_p_list, k_c_list, pre, derivative = self.compute_response_coeffs(
            param_intr, channel)
        # Compute model matrix in frequency domain (in fractional frequency)
        uc = self.u_matrices(f, param_intr, del_t, 0, tobs,
                             derivative=derivative)
        # Compute the matrices corresponding to each channel h1, h2, h3
        mat_list = [pre * self.single_design_matrix_freq(uc, k_p_list[i],
                                                         k_c_list[i],
                                                         full=full)
                    for i in range(len(k_p_list))]
        # If phasemeter model is required, there should be 6 matrices
        # (one for each link)
        if channel == 'phasemeters':
            # You have to include primed channels: h3, h1, h2
            mat_list.extend([mat_list[2], mat_list[0], mat_list[1]])

        return mat_list

    def compute_response_coeffs(self, param_intr, channel='TDIAET'):
        """Compute the coefficients needed for calculating the response

        Parameters
        ----------
        param_intr : ndarray
            Instrinsic parameters
        channel : str
            Type of data channel among {'phasemegers, 'TDIAET', 'TDIXYZ'}

        Returns
        -------
        k_p_list : list
            list of + coefficients
        k_c_list : list
            list of x coefficients
        pre : float
            prefactor to apply to the response
        derivative : integer
            order of the derivative to take to compute the response

        """
        # domain and in each channel
        if channel == 'phasemeters':
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
            pre = (self.armlength / LC.c) ** 2
            derivative = 2
            # Compute the required coefficients response depending on channel
            i, j = indices_low_freq('X1')  # 23
            kp23, kc23 = coeffs.k_coeffs(param_intr, self.phi_rot, i, j)
            i, j = indices_low_freq('Y1')  # 31
            kp31, kc31 = coeffs.k_coeffs(param_intr, self.phi_rot, i, j)
            i, j = indices_low_freq('Z1')  # 12
            kp12, kc12 = coeffs.k_coeffs(param_intr, self.phi_rot, i, j)

            k_p_list = [kp23, kp31, kp12]
            k_c_list = [kc23, kc31, kc12]

        return k_p_list, k_c_list, pre, derivative

    def compute_signal_freq(self, f, params, del_t, tobs, channel='TDIAET'):
        """
        Compute LISA's response to the incoming galactic binary GW in the
        frequency domain

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        params : array_like
            vector of parameters:
            amplitude a0 [GW strain],
            inclination incl [rad],
            initial phase phi_0 [rad],
            polarization angle psi [rad],
            colatitude theta [rad] = Ecliptic Latitude + pi/2
            longitude phi [rad] = Ecliptic longitude - pi
            frequency f_0 [Hz],
            frequency derivative f_dot [Hz^2]
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
            list containing the 3 TDI responses in channels a_mat, E and T,
            expressed in fractional frequency amplidudes,
            such that ch_interp[i] corresponds to fft(ch[i]) without any
            normalization

        Notes
        -----
        The Ecliptic latitude (or declination) should varies in the interval
        [-pi/2, pi /2], therefore theta is in [0, pi]
        Therefore, the colatitude theta is in the interval [-pi/2, pi /2]
        The ecliptic longitude varies from 0 to 2pi.
        Therefore the longitude phi varies in [-pi, pi]

        """

        # Extract physical parameters
        a0, incl, phi_0, psi, theta, phi, f_0, f_dot = params
        # Compute constant amplitude coefficients
        beta = coeffs.beta_gb(a0, incl, phi_0, psi)
        # Vector of intrinsic parameters
        param_intr = np.array([theta, phi, f_0, f_dot])
        # Compute response coefficients
        k_p_list, k_c_list, pre, derivative = self.compute_response_coeffs(
            param_intr, channel)
        # Compute model matrix in frequency domain (in fractional frequency)
        # uc, us = self.u_matrices(f, param_intr, del_t, 0, tobs,
        #                          derivative=derivative)
        uc = self.u_matrices(f, param_intr, del_t, 0, tobs,
                             derivative=derivative)

        # mat_list = [self.design_matrix_freq(uc, us, k_p_list[i], k_c_list[i])
        #             for i in range(len(k_p_list))]
        mat_list = [self.single_design_matrix_freq(uc,
                                                   k_p_list[i],
                                                   k_c_list[i],
                                                   full=True)
                    for i in range(len(k_p_list))]

        # Transform to complex number
        ch_list = [pre * mat.dot(beta) for mat in mat_list]

        if channel == 'TDIAET':
            a, e, t = convert_xyz_to_aet(ch_list[0], ch_list[1], ch_list[2])
            return a, e, t
        else:
            return ch_list[0], ch_list[1], ch_list[2]

    def transfer_function(self, f, params, del_t, tobs, channel='phasemeters'):
        """Compute an approximation of LISA's transfer function G(F) for a
        monochromatic wave, for phasemeter or TDI measurements.
        This is such that
        y(f) = G_{+}(f) h_source_{+}(f) + G_{x}(f) h_source_{x}(f)
        The total transfer function is
        G(f)^2 = |G_{+}(f)|^2 + |G_{x}(f)|^{2}

        We take f_source = f (the TF is computed at the source's main frequency)

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        params : array_like
            vector of parameters:
            amplitude a0 [GW strain],
            inclination incl [rad],
            initial phase phi_0 [rad],
            polarization angle psi [rad],
            colatitude theta [rad] = Ecliptic Latitude + pi/2
            longitude phi [rad] = Ecliptic longitude - pi
            frequency f_0 [Hz],
            frequency derivative f_dot [Hz^2]
        tobs : scalar float
            Observation Duration
        ts : scalar float
            sampling cadence
        channel : string
            tdi channel among {'X1','X2','X3'}

        Returns
        -------
        g_p_list : list of ndarrays
            transfer function for the + polarization calculated at f
        g_c_list : list of ndarrays
            transfer function for the x polarization calculated at f

        """
        # Extract physical parameters
        a0, incl, phi_0, psi, theta, phi, f_0, f_dot = params
        # Vector of intrinsic parameters
        param_intr = np.array([theta, phi, f_0, f_dot])
        d0 = self.R * np.sin(theta) / LC.c
        varphi = phi + np.pi/2
        m_vect = np.arange(0, 5)
        e_vect = np.exp(1j*m_vect*varphi)

        # domain and in each channel
        if channel == 'phasemeters':
            pre = (self.armlength / (4 * LC.c))
            derivative = 1
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

        # Include derivative
        jomega = (2*np.pi*1j*f)**derivative
        # Bessel coefficients stored in a nf x 5 matrix, with derivative coeff
        jw = np.array([special.jv(m, 2*np.pi*f*d0) * jomega for m in m_vect]).T
        # Multiply exp(jm * varphi) by k_p[m]
        ek_p = [e_vect * k_p for k_p in k_p_list]
        ek_c = [e_vect * k_c for k_c in k_c_list]
        minus_1_m = np.array((-1)**m_vect)
        # Transfer function for the + polarization, sum over m
        g_p_list = [pre * np.sum((co + minus_1_m * np.conj(co)) * jw, axis=1)
                    for co in ek_p]
        # Transfer function for the x polarization, sum over m
        g_c_list = [pre * np.sum((co + minus_1_m * np.conj(co)) * jw, axis=1)
                    for co in ek_c]

        return g_p_list, g_c_list


# -----------------------------------------------------------------------------
class UCBWaveformFull(GWwaveform):
    """
    Class to compute the LISA response to an incoming gravitational wave.

    """

    def __init__(self, v_func, del_t, tobs,
                 phi_rot=0, armlength=2.5e9, nc=2**9, arm_list=None,
                 e=0):
        """

        Parameters
        ----------
        v_func : callable
            function of frequency giving the Fourier transform of
            exp(+j*Phi(t)), where Phi(t) is the phase of the gravitatonal wave
        fs : scalar float
            sampling frequency
        phi_rot : scalar float
            initial angle of LISA constellation
        armlength : scalar float
            Arm length (default is 2.5e9 m)
        nc : scalar integer
            order of the Bessel function decomposition
        arm_list : list of callables
            list of armlength function wrt time, in the order:
            L1, L2, L3, L1', L2', L3'
        e : float
            LISA's orbit excentricity

        """

        super().__init__(phi_rot=phi_rot, armlength=armlength)

        # Name of intrinsic parameters
        self.names = ['theta', 'phi', 'f_0', 'f_dot']
        # Phase function
        self.v_func = v_func
        # For the number of time samples in the Fourier series
        self.nc = nc
        # Sampling time
        self.del_t = del_t
        # Observation duration
        self.tobs = tobs
        # Cosine and sine of LISA orbital phase
        self.dt = None
        self.t_samples = []
        self.cosphit_mat = []
        self.sinphit_mat = []
        self.cs_mat = []
        # Precompute them at time samples
        self.precompute_lisa_phase()
        # List of arm lengths functions vs time
        if arm_list is None:
            self.arm_list = [self.armlength for i in range(3)]
            self.arm_list_prime = [self.armlength for i in range(3)]
        else:
            self.arm_list = [arm(self.t_samples) for arm in arm_list[0:3]]
            self.arm_list_prime = [arm(self.t_samples)
                                   for arm in arm_list[3:6]]
        # Eccentricity
        self.e = e
        # List of arm indices measured on each optical bench i
        self.index_list = [3, 1, 2, 2, 3, 1]
        self.prime_list = [False, False, False, True, True, True]

    def update_arm_list(self, arm_list):
        """

        Parameters
        ----------
        arm_list : list of callables
            list of armlength function wrt time, in the order:
            L1, L2, L3, L1', L2', L3'


        """
        # List of arm lengths functions vs time
        self.arm_list = [arm(self.t_samples) for arm in arm_list[0:3]]
        self.arm_list_prime = [arm(self.t_samples)
                               for arm in arm_list[3:6]]

    def precompute_lisa_phase(self):
        """
        Compute terms in Phi_T(t) on a minimal grid in the time domain.
        """
        f_sample = 2 * self.nc
        self.t_samples, self.dt = np.linspace(0, self.tobs, f_sample + 2,
                                              endpoint=False, retstep=True)

        self.cosphit_mat = np.array([np.cos(2*np.pi*m*self.f_t*self.t_samples)
                                     for m in range(5)]).T
        self.sinphit_mat = np.array([np.sin(2*np.pi*m*self.f_t*self.t_samples)
                                     for m in range(5)]).T
        self.cs_mat = np.hstack((self.cosphit_mat, self.sinphit_mat))

    def compute_modulation(self, theta, phi, i, channel='phasemeters'):

        if channel == 'phasemeters':
            # Compute the required coefficients response depending on channel
            k_pi, k_ci = coeffs.xi_coeffs(theta, phi, self.phi_rot, i)
        # Compute modulation functions
        xi_pi = self.cs_mat.dot(k_pi)
        xi_ci = self.cs_mat.dot(k_ci)

        return xi_pi, xi_ci

    def compute_slow_part(self, theta, phi, f_0, f_dot, i,
                          channel='phasemeters', prime=False):
        """
        Compute the slow-varying part of the response in OB i
        in the time domain, such that

        s_i = Re{ A_slow_i(t) e^{j Phi(t)} }

        Parameters
        ----------
        theta : scalar float
            colatitude angle: theta = beta + pi/2 if beta is the ecliptic
            latitude [rad]
        phi : scalar float
            longitude angle such taht phi = lam - pi where lam is the ecliptic
            longitude [rad]
        f_0 : scalar float
            wave frequency [Hz]
        i : int
            index of the considered arm among {1, 2, 3}

        Returns
        -------
        a_slow : ndarray
            design matrix of size 2nc x 2

        """

        # Compute modulation function
        xpi, xci = self.compute_modulation(theta, phi, i, channel=channel)
        # Compute prefractor
        a, b = coeffs.kn_coeffs(theta, phi, self.phi_rot, i)
        kni = self.cosphit_mat[:, 0:3].dot(a) + self.sinphit_mat[:, 0:3].dot(b)
        # Compute k . r_0
        kr0 = - self.R * np.sin(theta) * (
                self.cosphit_mat[:, 1] * np.cos(phi)
                + self.sinphit_mat[:, 1] * np.sin(phi))

        # # Compute k.r_i
        # # In FD waveform
        # a, b = coeffs.ku_coeffs(theta, phi, self.phi_rot, i)
        # kui = self.cosphit_mat[:, 0:3].dot(a) + self.sinphit_mat[:, 0:3].dot(b)
        # # Modulation phase, i.e. k . r_0
        # d = - self.R * np.sin(theta) * (
        #         self.cosphit_mat[:, 1] * np.cos(phi)
        #         + self.sinphit_mat[:, 1] * np.sin(phi)) / physics.c_light
        # d_re = d + (2 * self.R * self.e * kui) / physics.c_light
        #
        # if not prime:
        #     # Compute coefficients for k.n_{i+2}
        #     a2, b2 = coeffs.ku_coeffs(theta, phi,
        #                               self.phi_rot, coeffs.cyclic_perm(i + 2))
        #     # k.n_{i+2}
        #     ku_em = self.cosphit_mat[:,
        #             0:3].dot(a2) + self.sinphit_mat[:, 0:3].dot(b2)
        #     d_em = d + (2 * self.R * self.e * ku_em
        #                 + self.arm_list[i - 1]) / physics.c_light
        #     prefact = 1 / (2 * (1 - kni))
        # else:
        #     # Compute coefficients for k.n_{i+1}
        #     a2, b2 = coeffs.ku_coeffs(theta, phi,
        #                               self.phi_rot, coeffs.cyclic_perm(i + 1))
        #     # k.n_{i+1}
        #     ku_em = self.cosphit_mat[:,
        #             0:3].dot(a2) + self.sinphit_mat[:, 0:3].dot(b2)
        #     d_em = d + (2 * self.R * self.e * ku_em
        #                 + self.arm_list_prime[i - 1]) / physics.c_light
        #     prefact = 1 / (2 * (1 + kni))

        if not prime:
            # Reception index
            i_re = coeffs.cyclic_perm(i + 1)
            # Emission index
            i_em = coeffs.cyclic_perm(i + 2)
            # Armlength
            arm_length = self.arm_list[i - 1]
            # Response denominator
            denom = 2 * (1 - kni)

        else:
            # Reception index
            i_re = coeffs.cyclic_perm(i + 2)
            # Emission index
            i_em = coeffs.cyclic_perm(i + 1)
            # Armlength
            arm_length = self.arm_list_prime[i - 1]
            # Response denominator
            denom = 2 * (1 + kni)

        # Reception time
        # --------------
        a_re, b_re = coeffs.ku_coeffs(theta, phi, self.phi_rot, i_re)
        ku_re = self.cosphit_mat[:,
                0:3].dot(a_re) + self.sinphit_mat[:, 0:3].dot(b_re)
        d_re = (kr0 + 2 * self.e * self.R * ku_re) / physics.c_light

        # Emission time
        # -------------
        a_em, b_em = coeffs.ku_coeffs(theta, phi, self.phi_rot, i_em)
        ku_em = self.cosphit_mat[:,
                0:3].dot(a_em) + self.sinphit_mat[:, 0:3].dot(b_em)
        d_em = (kr0 + 2 * self.R * self.e * ku_em + arm_length)/physics.c_light

        # Reception time
        dt_re = d_re * (1 - f_dot * (d_re + 2 * self.t_samples) / (2 * f_0))
        # Emission time
        dt_em = d_em * (1 - f_dot * (d_em + 2 * self.t_samples) / (2 * f_0))

        phasing = (np.exp(-2j * np.pi * f_0 * dt_re)
                   - np.exp(-2j * np.pi * f_0 * dt_em))

        # If v_func is monochromatic, we must include the f_dot dependence
        # of the GW phase in the slow part:
        phasing *= np.exp(1j * np.pi * f_dot * self.t_samples**2)

        # Sinc formulation
        # pref = np.pi * f_0 * self.arm_list[i-1] / LC.c
        # prefact = 1j * pref * np.sinc(pref * (1 - kni))
        # Copmute
        # # Compute the phase perturbation
        # delta_phi = np.pi * d * (2 * f_0 + f_dot * (2 * self.t_samples - d))
        # # Compute slow part design matrix (for a+, ax)
        # pref = np.array([prefact * np.exp(-1j * delta_phi)]).T

        # # Computation of the phasing term
        # ar, br = coeffs.kn_bar_coeffs(theta, phi, self.phi_rot, i)
        # kn_bar_i = self.cosphit_mat[:, 0:3].dot(ar)\
        #            + self.sinphit_mat[:, 0:3].dot(br)
        # kr_bar_i = kr0 + 2 * self.e * self.R * kn_bar_i
        # kr_bar_i = kr0[:]
        # d = (2 * kr_bar_i + self.arm_list[i-1]) / LC.c
        # phasing = np.exp(-1j * np.pi * f_0 * d)
        return np.array([phasing / denom]).T * np.array([xpi, xci]).T
        # return np.array([prefact * phasing]).T * np.array([xpi, xci]).T
        # return np.array([prefact * phasing * xpi, prefact * phasing * xci]).T

    def design_matrix_freq(self, f, param_intr, channel='phasemeters'):
        """
        Compute the list of frequency-domain design matrix mat[i] such that
        LISA's response
        to GW in OB i is

        s_i(f) = mat[i] * beta

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        param_intr : array_like
            vector of intrinsic parameters:
            colatitude theta [rad] = Ecliptic Latitude + pi/2
            longitude phi [rad] = Ecliptic longitude - pi
            frequency f_0 [Hz],
            frequency derivative f_dot [Hz^2]
        channel : string
            type of channel: {'TDIAET', 'phasemeters'}

        Returns
        -------
        mat_list : list of numpy arrays
            list containing the 3 design matrices of size nf x 2


        """

        # Extract instrinsic parameters
        theta, phi, f_0, f_dot = param_intr
        # Compute slow part of nc x 2 design matrices in time domain
        # index_list = [3, 1, 2]
        # a_slow_list = [self.compute_slow_part(theta, phi, f_0, f_dot, i,
        #                                       channel=channel)
        #                for i in index_list]
        # t1 = time.time()
        a_slow_list = [self.compute_slow_part(theta, phi, f_0, f_dot,
                                              self.index_list[j],
                                              channel=channel,
                                              prime=self.prime_list[j])
                       for j in range(len(self.index_list))]
        # Compute its FFC to get Fourier series coefficients
        # (list ot nc x 2 matrices)
        # t2 = time.time()
        # print("Slow part computation: " + str(t2 - t1))
        # t1 = time.time()
        c_list = fft(np.array(a_slow_list), axis=1) / self.t_samples.size
        # Compute corresponding frequency vector
        f_vect = np.fft.fftfreq(c_list[0].shape[0],
                                d=self.tobs / c_list[0].shape[0])
        # t2 = time.time()
        #print("FFT: " + str(t2 - t1))
        # print("Shape of c_list" + str(c_list.shape))
        # t1 = time.time()
        # Form the grid of frequency differences, size nf x nc
        f_grid = np.array([f]) - np.array([f_vect]).T
        # t2 = time.time()
        # print("Frequency grid: " + str(t2 - t1))
        # Assume that v_func includes f_dot dependence
        # v_list = self.v_func(f_grid, f_0, f_dot, self.tobs, self.del_t)
        # Assume that v_func is monochromatic
        # t1 = time.time()
        v_list = self.v_func(f_grid, f_0, self.tobs, self.del_t)
        # t2 = time.time()
        # print("Waveform grid: " + str(t2 - t1))
        # print("Shape of v_list" + str(v_list.shape))
        # v_list = [self.v_func(f - f_vect[k], f_0, f_dot, self.tobs, self.del_t)
        #           for k in range(c_list[0].shape[0])]
        # t1 = time.time()
        a_mat_list = [self.compute_series(v_list, c_list[i]) / 2
                      for i in range(len(c_list))]
        # t2 = time.time()
        # print("Series computation: " + str(t2 - t1))
        # # Extend with 3 primed channels: h3, h1, h2
        # if channel == 'phasemeters':
        #     # You have to include primed channels: h3, h1, h2
        #     a_mat_list.extend([a_mat_list[2], a_mat_list[0], a_mat_list[1]])

        return a_mat_list

    # @staticmethod
    def compute_series(self, v_list, c):
        """Short summary.

        Parameters
        ----------
        v_list : list of ndarrays
            list of nc values v(f - n / Tobs) where v is the exponential GW
            phase for frequencies f. Each list item is an array of
            size nf.
        c : ndarray
            Fourier series coefficients of a_slow, size nc x 2

        Returns
        -------
        series: ndarray
            Array of size nf x 2 giving the Fourier-series approximation
            of a_slow(f) in the frequency domain.

        """

        # For n = -2nc..2nc, get a 2nc x nf x 2 array
        return np.sum(c * np.array([v_list]).T, axis=1)

    def compute_signal_freq(self, f, params, channel='phasemeters'):
        """
        Compute LISA's response to the incoming galactic binary GW in the
        frequency domain

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        params : array_like
            vector of parameters:
            amplitude a0 [GW strain],
            inclination incl [rad],
            initial phase phi_0 [rad],
            polarization angle psi [rad],
            colatitude theta [rad] = Ecliptic Latitude + pi/2
            longitude phi [rad] = Ecliptic longitude - pi
            frequency f_0 [Hz],
            frequency derivative f_dot [Hz^2]
        channel : string
            tdi channel among {'X1','X2','X3'}

        Returns
        -------
        ch_interp : list of numpy arrays
            list containing the 3 TDI responses in channels a_mat, E and T,
            expressed in fractional frequency amplidudes,
            such that ch_interp[i] corresponds to fft(ch[i]) without any
            normalization

        Notes
        -----
        The Ecliptic latitude (or declination) should varies in the interval
        [-pi/2, pi /2], therefore theta is in [0, pi]
        Therefore, the colatitude theta is in the interval [-pi/2, pi /2]
        The ecliptic longitude varies from 0 to 2pi.
        Therefore the longitude phi varies in [-pi, pi]

        """

        # Extract physical parameters
        a0, incl, phi_0, psi, theta, phi, f_0, f_dot = params
        # Compute the constant extrinsinc amplitude coefficients
        h0_p = a0 * (1 + np.cos(incl) ** 2)
        h0_c = 2 * a0 * np.cos(incl)
        # Compute complex amplitudes
        a_p = h0_p * np.exp(1j * phi_0)
        a_c = 1j * h0_c * np.exp(1j * phi_0)
        # Compute complex vector of effective amplitudes
        beta = np.array([a_p * np.cos(2 * psi) + a_c * np.sin(2 * psi),
                         a_c * np.cos(2 * psi) - a_p * np.sin(2 * psi)])
        # Compute design matrices
        param_intr = np.array([theta, phi, f_0, f_dot])
        mat_list = np.array(self.design_matrix_freq(f, param_intr,
                                                    channel=channel))

        return mat_list.dot(beta)


# ------------------------------------------------------------------------------
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

    def compute_signal_freq(self, f, params, del_t, tobs, channel='TDIAET',
                            ldc=False, tref=0):
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
            list containing the 3 TDI responses in channels a_mat, E and T,
            expressed in fractional frequency amplidudes,
            such that ch_interp[i] corresponds to fft(ch[i]) without any
            normalization

        """

        # phi0, m1, m2, a1, a2, dist, inc, lam, beta, psi, tc = params
        # if ldc:
        #     # TDI response on native grid
        #     m1 = params[0]
        #     m2 = params[1]
        #     xi1 = params[2]
        #     xi2 = params[3]
        #     tc = params[4]
        #     dist = params[5]
        #     inc = params[6]
        #     phi0 = params[7]
        #     lam = params[8]
        #     beta = params[9]
        #     psi = params[10]
        #     fr, x, y, z, wfTDI =GenerateFD_SignalTDIs.MBHB_LISAGenerateTDIfast(
        #         phi0, self.fRef, m1, m2, xi1, xi2, dist, inc, lam, beta, psi,
        #         tobs, tc, del_t, tShift=0, fmin=1.e-5, fmax=0.5 / del_t,
        #         frqs=None, resf=None)
        #
        #     ch_interp = [self.interpolate_waveform(fr, f, ch) for ch in [x, y, z]]
        #     a, e, t = convert_xyz_to_aet(ch_interp[0], ch_interp[1], ch_interp[2])
        #
        #     if (channel == 'TDIAET') | (channel == ['a_mat', 'E', 'T']):
        #         # print(channel)
        #         # a, e, t = convert_xyz_to_aet(ch_interp[0], ch_interp[1], ch_interp[2])
        #         ch_interp = [a / del_t, e / del_t, t / del_t]
        #     elif channel == ['a_mat']:
        #         ch_interp = a / del_t
        #     elif channel == ['E']:
        #         ch_interp = e / del_t
        #     elif channel == ['T']:
        #         ch_interp = t / del_t
        #
        # else:
        # TDI response
        wftdi = lisa.GenerateLISATDI(params, tobs=tobs, minf=1e-5, maxf=1.,
                                     tref=tref, torb=0., TDItag='TDIAET',
                                     acc=1e-4, order_fresnel_stencil=0,
                                     approximant='IMRPhenomD',
                                     responseapprox='full', frozenLISA=False,
                                     TDIrescaled=False)
        signal_freq = lisa.GenerateLISASignal(wftdi, f)
        # Divide by del_t to be consistent with the unnormalized DFT
        ch_interp = [self.shift_time(f, signal_freq['ch1'], tobs).conj()/del_t,
                     self.shift_time(f, signal_freq['ch2'], tobs).conj()/del_t,
                     self.shift_time(f, signal_freq['ch3'], tobs).conj()/del_t]

        # Interpolate the response on required grid
        # return signal_freq['ch1'], signal_freq['ch2'], signal_freq['ch3']
        # return ch_interp[0] #, ch_interp[1], ch_interp[2]
        return ch_interp

    def single_design_matrix(self, tdi_resp_plus, tdi_resp_cros):

        a_mat = np.empty((2 * tdi_resp_plus.shape[0], 2), dtype=np.float64)
        a_mat[:, 0] = np.concatenate((tdi_resp_plus.real, tdi_resp_plus.imag))
        a_mat[:, 1] = np.concatenate((tdi_resp_cros.real, tdi_resp_cros.imag))

        return a_mat

    def design_matrix_freq(self, f, params_intr, del_t, tobs, channel='TDIAET',
                           complex=False, tref=0):
        """
        Compute design matrix such that the TDI variable (fist generation)
        can be written as

        TDI(f) = a_mat(theta,phi,f0,f_dot,f) beta(A_p,A_c,phi_0,psi)

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
        a_mat : numpy array
            model matrix of size (n_data x K)


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
        params_1[self.i_inc] = 0.5 * np.pi # 0 
        params_2[self.i_inc] = 0.5 * np.pi # 0 
        # Polarization angle
        params_1[self.i_psi] = 0.0
        params_2[self.i_psi] = np.pi / 4

        # amp = 2* G * mu / (dl * c**2)

        # Amp, iota, psi_m, phi0
        # 1. 1.e-21, 0.5*np.pi, 0.0, 0.0
        # 2. 1.e-21, 0.5*np.pi, 0.25*np.pi, 0.0

        # Calculate the response on required grid
        tdi_response_plus = self.compute_signal_freq(f, params_1, del_t, tobs,
                                                     channel=channel,
                                                     tref=tref)
        tdi_response_cros = self.compute_signal_freq(f, params_2, del_t, tobs,
                                                     channel=channel,
                                                     tref=tref)

        if not complex:
            mat_list = [self.single_design_matrix(tdi_response_plus[i],
                                                  tdi_response_cros[i])
                        for i in range(len(tdi_response_plus))]
        else:
            mat_list = [np.array([tdi_response_plus[i],
                                  tdi_response_cros[i]]).T
                        for i in range(len(tdi_response_plus))]

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


def lisabeta_template(params, freq, tobs, tref=0, t_offset=52.657,
                      channels=None):
    """

    Parameters
    ----------
    params : ndarray
        full vector of parameters with format:  
        m1     Redshifted mass of body 1 (solar masses)  
        m2     Redshifted mass of body 2 (solar masses)  
        chi1   Dimensionless spin of body 1 (in [-1, 1])  
        chi2   Dimensionless spin of body 2 (in [-1, 1])  
        Deltat Time shift (in seconds)  
        dist   Luminosity distance (Mpc)  
        inc    Inclination angle (rad)  
        phi    Observer's azimuthal phase (rad)  
        lambda Source longitude in SSB-frame (rad)  
        beta   Source latitude in SSB-frame (rad)  
        psi    Polarization angle (rad) 
    freq : ndarray
        array of frequency bins
    tobs : float
        observation time
    tref : float
        reference time (starting time of time series? TBD)
    toffset
        offset time to match LDC waveform.

    Returns
    -------
    ch_interp : list of ndarrays
        list of waveform values interpolated at frequency bins.
        What is the normalization?

    """

    if channels is None:
        channels = [1, 2, 3]

    # TDI response
    wftdi = lisa.GenerateLISATDI(params, tobs=tobs, minf=1e-5, maxf=1.,
                                 tref=tref, torb=0., TDItag='TDIAET',
                                 acc=1e-4, order_fresnel_stencil=0,
                                 approximant='IMRPhenomD',
                                 responseapprox='full', frozenLISA=False,
                                 TDIrescaled=False)

    signal_freq = generate_lisa_signal(wftdi, freq, channels)

    # Phasor for the time shift
    z = np.exp(-2j * np.pi * freq * t_offset)
    # Get coalescence time
    ch_interp = [(signal_freq['ch' + str(np.int(i))] * z).conj()
                 for i in channels]

    return ch_interp


# def design_matrix(params_intr, freq, tobs, tref=0, t_offset=52.657,
#                   channels=None, complex_amplitudes=1):
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

#     Returns
#     -------
#     mat_list : list
#         list of design matrices for each channel

#     """

#     # First column
#     # m1, m2, chi1, chi2, tc, dist, inc, phi, lambd, beta, psi = params
#     params_1 = np.zeros(11)
#     # Save intrinsic parameters
#     params_1[i_intr] = params_intr
#     # Luminosity distance (Mpc)
#     params_1[i_dist] = 1e3
#     # Inclination
#     params_1[i_inc] = 0.5 * np.pi
#     # Polarization angle
#     params_1[i_psi] = 0.0
    
#     if channels is None:
#         channels = [1, 2, 3]

#     # TDI response 1
#     wftdi_1 = lisa.GenerateLISATDI(params_1, tobs=tobs, minf=1e-5, maxf=1.,
#                                    tref=tref, torb=0., TDItag='TDIAET',
#                                    acc=1e-4, order_fresnel_stencil=0,
#                                    approximant='IMRPhenomD',
#                                    responseapprox='full', frozenLISA=False,
#                                    TDIrescaled=False)

#     if freq is not None:
#         fs1 = wftdi_1['freq']
#         amp = pyspline.resample(freq, fs1, wftdi_1['amp'])
#         phase = pyspline.resample(freq, fs1, wftdi_1['phase'])
#         tr_int1 = [pyspline.resample(freq, fs1, 
#                                      wftdi_1['transferL' + str(np.int(i))])
#                    for i in channels]
#     else:
#         freq = wftdi_1['freq']
#         tr_int1 = [wftdi_1['transferL' + str(np.int(i))] for i in channels]
    
#     # Complex strain
#     h = amp * np.exp(1j * phase)
#     # Phasor for the time shift
#     z = np.exp(-2j * np.pi * freq * t_offset)
#     # If only one column
#     if complex_amplitudes == 1:
#         mat_list = [np.array([h * tr_int1[i - 1] * z]).conj().T 
#                     for i in channels]
#     # Second column if required
#     elif complex_amplitudes == 2:
#         params_2 = np.zeros(11)
#         # Save intrinsic parameters
#         params_2[i_intr] = params_intr
#         # Luminosity distance (Mpc)
#         params_2[i_dist] = 1e3
#         # Inclination
#         params_2[i_inc] = 0.5 * np.pi
#         # Polarization angle
#         params_2[i_psi] = np.pi / 4
        
#         # TDI response 2
#         wftdi_2 = lisa.GenerateLISATDI(params_2, tobs=tobs, minf=1e-5, maxf=1.,
#                                        tref=tref, torb=0., TDItag='TDIAET',
#                                        acc=1e-4, order_fresnel_stencil=0,
#                                        approximant='IMRPhenomD',
#                                        responseapprox='full', frozenLISA=False,
#                                        TDIrescaled=False)
#         if freq is not None:
#             fs2 = wftdi_2['freq']
#             amp2 = pyspline.resample(freq, fs2, wftdi_1['amp'])
#             phase2 = pyspline.resample(freq, fs2, wftdi_1['phase'])
#             tr_int2 = [pyspline.resample(freq, fs2,
#                                          wftdi_2['transferL' + str(np.int(i))])
#                        for i in channels]
#         else:
#             freq = wftdi_2['freq']
#             tr_int2 = [wftdi_2['transferL' + str(np.int(i))] for i in channels]

#         h2 = amp2 * np.exp(1j * phase2)
#         mat_list = [np.vstack((h * tr_int1[i - 1] * z,
#                                h2 * tr_int2[i - 1] * z)).conj().T
#                     for i in channels]

#     return mat_list# , tr_int1, tr_int2


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
    params_0[i_dist] = 1e3
    # Inclination
    params_0[i_inc] = 0.5 * np.pi
    # 4 elements
    params_list = [params_0 for j in range(4)]
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
    elements = [lisabeta_template(params, freq, tobs, 
                                  tref=tref, 
                                  t_offset=t_offset, 
                                  channels=channels) for params in params_list]
    # Compute the design matrices for each channel
    mat_list = [np.vstack([el[i] for el in elements]).T 
                for i in range(len(elements[0]))]

    return mat_list
