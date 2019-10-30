#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:09:36 2019

@author: qbaghi
"""


import copy
import numpy as np
import LISAConstants as LC
import tdi
from scipy import special, linalg
from .coeffs import k_coeffs
# from lisabeta.lisa import lisa
import lisabeta.lisa.lisa as lisa
from scipy.interpolate import InterpolatedUnivariateSpline as spline

# For MBHB only, use MLDC code
import GenerateFD_SignalTDIs
import lisabeta.waveforms.bbh.pyIMRPhenomD as pyIMRPhenomD
import lisabeta.lisa.pyresponse as pyresponse
import lisabeta.pyconstants as pyconstants
import lisabeta.tools.pytools as pytools
import lisabeta.tools.pyspline as pyspline
from bayesdawn.utils import physics


def optimal_order(theta, f_0):
    """
    Function calculating the optimal number of sidebands that must be calculated
    in the Bessel approximation to make a good approximation of the frequency
    modulated signal

    """

    R = LC.ua
    d0 = R * np.sin(theta) / LC.c
    # Modulation index
    mf = 2*np.pi*f_0*d0
    # Empirical model
    nc = np.int(1.2185*mf+5.625)+1

    return nc


def convert_xyz_to_aet(X, Y, Z):

    A = (Z - X)/np.sqrt(2)
    E = (X - 2*Y + Z)/np.sqrt(6)
    T = (X + Y + Z)/np.sqrt(3)

    return A, E, T


class GWwaveform(object):
    """
    Class to compute the LISA response to an incoming gravitational wave.

    """

    def __init__(self, Phi_rot=0, armlength=2.5e9):
        """


        Parameters
        ----------
        Phi_rot : scalar float
            initial angle of LISA constellation
        armlength : scalar float
            Arm length (default is 2.5e9 m)

        """

        self.Phi_rot = Phi_rot
        self.armlength = armlength

        self.R = LC.ua
        self.f_T = 1. / LC.year


class UCBWaveform(GWwaveform):
    """
    Class to compute the LISA response to an incoming gravitational wave.
    
    """

    def __init__(self, v_func, Phi_rot=0, armlength=2.5e9, nc=15):
        """
        
        
        Parameters
        ----------
        v_func : callable
            function of frequency giving the Fourier transform of exp(-j*Phi(t)),
            where Phi(t) is the phase of the gravitatonal wave
        fs : scalar float
            sampling frequency
        Phi_rot : scalar float
            initial angle of LISA constellation
        armlength : scalar float
            Arm length (default is 2.5e9 m)
        nc : scalar integer
            order of the Bessel function decomposition        
        
        
        """

        super().__init__(Phi_rot=Phi_rot, armlength=armlength)


        # Name of intrinsic parameters
        self.names = ['theta', 'phi', 'f_0', 'f_dot']
        # Phase function
        self.v_func = v_func
        # For Fourier series decomposition:
        self.M = 4
        self.m_vect = np.arange(0, self.M+1)
        self.nc = nc
        self.jw2 = []

    def indices_low_freq(self, channel):
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

    def o2i(self, x, nc):
        """
        convert Bessel order to vector index
    
        """
    
        return np.int(x+self.nc+4)

    def bessel_decomp_pos(self, v_minus_list, Jw, e_vect, nc, m):
        """
    
        This function calculates y_c(f) and y_s(f), basic quantities needed in the
        frequency response to gravitational wave, ONLY KEEPING THE TERMS THAT
        ARE SIGNIFICANT FOR POSITIVE FREQUENCIES.
    
        Parameters
        ----------
        v_minus : list of array_like
            stored values of conj(v)(-f+kf_t)
        Jw : list of complex scalar
            stored values of jv(n, 2*np.pi*f_0*d0)*np.exp(-1j*n*varphi)
        e_vect : list of complex scalars
            stored values of np.exp(1j*n*varphi)
        nc : scalar integer
            order of the decomposition
        m : scalar integer
            positive or negative integer indicating the harmonic of f_T (yearly period)
    
        Returns
        -------
        w : numpy array
            vector of values of w(f) (size N) calculated at given frequencies
    
        """
    
        # NEW BESSEL DECOMPOSITION:
        n_vect = np.arange(-nc, nc+1)
    
        v_minus = [np.conj(v_minus_list[self.o2i(-m-n,nc)])*e_vect[np.int(n+nc)] for n in n_vect]
    
        y_c = sum([Jw[k]*1/2.*v_minus[k] for k in range(len(v_minus))])
        #y_s = sum( [ -Jw[k]*1j/2.*v_minus[k] for k in range(len(v_minus)) ] )
        #y_s = -1j*y_c
        return y_c#,y_s


    def u_matrices(self, f, params, ts, Tstart, Tend, nc=15, derivative=2):
        """
    
        Compute the frequency matrix W involved in the calculation of the basis
        functions
    
        .. math::
    
            \tilde{\dot{u}}^{(i)}_{c,\alpha}(f)  = Uc K^{ij}_{\alpha}
    
        Parameters
        ----------
        f : array_like or list of array_like
            frequencies where to evalutate the function w(f)
        params : array_like
            vector of extrinsinc parameters: theta,phi,f_0,f_dot
        v_func : callable
            function of frequency giving the Fourier transform of exp(-j*Phi(t)),
            where Phi(t) is the phase of the gravitatonal wave
        ts : scalar float
            Sampling time
        Tstart : scalar float
            Starting observation time
        Tend: scalar float
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
    
        # Precompute the Bessel coefficients
        v_minus_list = [self.v_func(-f+k*self.f_T,f_0,f_dot,ts,Tstart,Tend) for k in range(-nc-self.M,nc+self.M+4)]
    
        n_vect = np.arange(-nc,nc+1)
        e_vect = np.exp(1j*n_vect*varphi)
    
        Jw = [special.jv(n, 2*np.pi*f_0*d0) for n in n_vect]
    
#        U = [self.bessel_decomp_pos_c(v_minus_list,Jw,e_vect,nc,m) for m in -m_vect]
#        U.extend([U[0]]) # Avoid computing m=0 twice
#        U.extend([self.bessel_decomp_pos_c(v_minus_list,Jw,e_vect,nc,m) for m in m_vect[1:]])
#    
#        jw2 = (2*np.pi*1j*f)**derivative
#        Uc = np.array([ jw2 * z[0] for z in U]).T
#    
#        return Uc
    
        U = [self.bessel_decomp_pos(v_minus_list,Jw,e_vect,nc,m) for m in -self.m_vect]
        U.extend([U[0]]) # Avoid computing m=0 twice
        U.extend([self.bessel_decomp_pos(v_minus_list,Jw,e_vect,nc,m) for m in self.m_vect[1:]])
        #t2 = time.time()
        #print("---- U decomposition took " + str(t2-t1) + " sec.")
        
        jw2 = (2*np.pi*1j*f)**derivative
        Uc = np.array([jw2 * z for z in U]).T
        #Us = np.array([ jw2 * z[1] for z in U]).T
        #Us = -1j*Uc


        return Uc#,Us

    def design_matrix_freq(self, f, params, ts, Tstart, Tend, channel='X1'):
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
            vector of extrinsinc parameters: theta,phi,f_0,f_dot
        Tobs : scalar float
            Observation Duration
        ts : scalar float
            sampling cadence
        tdi : string
            tdi channel among {'X1','X2','X3'}
    
    
        Returns
        -------
        A : numpy array
            model matrix of size (N x K)
    
    
        """
    
        # Indices of arms to be considered for the TDI variable
        i, j = self.indices_low_freq(channel)
    
        # Compute coefficients of the decomposition of basis functions u_alpha
        k_p, k_c = k_coeffs(params, self.Phi_rot, i, j)
    
    
    
        # Compute model matrix in frequency domain (in fractional frequency)
        #Uc,Us = self.u_matrices(f,params,ts,Tstart,Tend,nc = self.nc, derivative = 2)
        Uc = self.u_matrices(f, params, ts, Tstart, Tend, nc = self.nc, derivative = 2)
    
        k_p_tot = np.concatenate((k_p, np.conj(k_p)))
        k_c_tot = np.concatenate((k_c, np.conj(k_c)))
    
#        # Compute response in the slowly varying response approximation
#        A_tmp = np.empty((len(f),4),dtype = np.complex128)
#        A_tmp[:,0] = np.dot(Uc,k_p_tot) #C_func*Dxi_p
#        A_tmp[:,1] = np.dot(Uc,k_c_tot) #C_func*Dxi_c
#        A_tmp[:,2] = -1j* A_tmp[:,0] #S_func*Dxi_p
#        A_tmp[:,3] = -1j*A_tmp[:,1] #S_func*Dxi_c

        #gamma_p,gamma_c,sigma_p,sigma_c
        A_tmp = np.empty((2*len(f), 4), dtype=np.float64)
        Ac_plus = np.dot(Uc, k_p_tot) #C_func*Dxi_p
        Ac_cros = np.dot(Uc, k_c_tot) #C_func*Dxi_c
        A_tmp[:, 0] = np.concatenate((Ac_plus.real, Ac_plus.imag))
        A_tmp[:, 1] = np.concatenate((Ac_cros.real, Ac_cros.imag))
        A_tmp[:, 2] = np.concatenate((Ac_plus.imag, -Ac_plus.real))
        A_tmp[:, 3] = np.concatenate((Ac_cros.imag, -Ac_cros.real))

        A = (self.armlength/LC.c)**2*A_tmp
    
        return A


class MBHBWaveform(GWwaveform):
    """
    Class to compute the LISA response to an incoming gravitational wave.

    """

    def __init__(self, Phi_rot=0, armlength=2.5e9, reduced=False):
        """


        Parameters
        ----------
        v_func : callable
            function of frequency giving the Fourier transform of exp(-j*Phi(t)),
            where Phi(t) is the phase of the gravitatonal wave
        fs : scalar float
            sampling frequency
        Phi_rot : scalar float
            initial angle of LISA constellation
        armlength : scalar float
            Arm length (default is 2.5e9 m)
        nc : scalar integer
            order of the Bessel function decomposition


        """

        super().__init__(Phi_rot=Phi_rot, armlength=armlength)

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

        return xf*np.exp(-2j*np.pi*f*delay)

    def compute_signal_freq(self, f, params, del_t, tobs, channel='TDIAET', ldc=False, tref=0):
        """
        Compute LISA's response to the incoming MBHB GW in the frequency domain

        Parameters
        ----------
        f : array_like
            frequencies where to compute the matrix
        params : array_like
            vector of extrinsinc parameters: theta,phi,f_0,f_dot
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
            a, e, t = tdi.AET(ch_interp[0], ch_interp[1], ch_interp[2])

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
            ch_interp = [self.shift_time(f, signal_freq['ch1'],  tobs).conj() / del_t,
                         self.shift_time(f, signal_freq['ch2'],  tobs).conj() / del_t,
                         self.shift_time(f, signal_freq['ch3'],  tobs).conj() / del_t]
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

    def design_matrix_freq(self, f, params_intr, del_t, tobs, channel='TDIAET', complex=False, tref=0):
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
        params_1[self.i_inc] = 0 # 0.5 * np.pi
        params_2[self.i_inc] = 0 # 0.5 * np.pi
        # Polarization angle
        params_1[self.i_psi] = 0.0
        params_2[self.i_psi] = np.pi/4


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


def like_to_waveform_intr(par_intr):
    """
    Convert reduced-likelihood intrinsic parameters into waveform-compatible parameters
    Parameters
    ----------
    par : array_like
        intrinsic parameter vector for sampling the posterior:
        Mc, q, tc, chi1, chi2, np.sin(bet), lam

    Returns
    -------
    params_intr : array_like
        instrinsic parameters m1, m2, chi1, chi2, tc, bet, psi

    """

    # Explicit the vector of paramters
    mc, q, tc, chi1, chi2, sb, lam = par_intr

    # Convert chirp mass into individual masses
    m1, m2 = physics.compute_masses(mc, q)

    # Convert Source latitude in SSB-frame
    bet = np.arcsin(sb)

    # Form the waveform-compatible intrinsic parameter vector
    params_intr = m1, m2, chi1, chi2, tc, lam, bet

    return params_intr


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


def design_matrix(par_intr, freq_data, tobs, tref=0, t_offset=52.657, channels=None, minf=1e-5, maxf=1., acc=1e-4,
                  dist=1e3, torb=0, LISAconst=pyresponse.LISAconstProposal, responseapprox='full',
                  frozenLISA=False, TDIrescaled=False):
    """
    Design matrix for reduced order likelihood
    Parameters
    ----------
    params_intr : ndarray
        instrinsic parameters
    freq :
    tobs
    tref
    t_offset
    channels

    Returns
    -------

    """

    # Convert likelihood parameters into waveform-compatible parameters
    params_intr = like_to_waveform_intr(par_intr)
    # The full set of parameters is m1, m2, chi1, chi2, tc, dist, incl, phi0, lam, bet, psi
    # Indices of intrinsic parameters m1, m2, chi1, chi2, tc, bet, psi
    # Initial phase: same!
    phi0 = 0
    # Inclination: same!
    inc = 0
    # Polarization angle: 45 degree angle difference
    psi_1 = 0.0
    psi_2 = np.pi / 4
    # TDI response
    m1, m2, chi1, chi2, tc, lambd, beta = params_intr
    # Units
    q = m1 / m2
    m1_SI = m1 * pyconstants.MSUN_SI
    m2_SI = m2 * pyconstants.MSUN_SI
    dist_SI = dist * 1e6 * pyconstants.PC_SI
    M = m1 + m2
    Ms = M * pyconstants.MTSUN_SI
    if tobs is not None:
        minf = np.fmax(minf, pytools.funcNewtonianfoft(m1, m2, tobs * pyconstants.YRSID_SI))
    Mfmin = Ms * minf
    Mfmax = Ms * maxf
    Mfmax_model = 0.2
    Mfmax = np.fmin(Mfmax, Mfmax_model)
    maxf = Mfmax / Ms

    # Combined frequency grid for the waveform
    gridfreqClass = pytools.FrequencyGrid(minf, maxf, M, q, acc=acc)
    gridfreq = gridfreqClass.get_freq()

    # Generate IMRPhenomD waveform and add time shift
    # Calling with fRef=0., phiRef=0. defines the source frame
    phiRef = 0.
    fRef = 0.
    wfClass = pyIMRPhenomD.IMRPhenomDh22AmpPhase(gridfreq, phiRef, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
    freq, amp, phase = wfClass.get_waveform()
    phase += 2 * np.pi * freq * tc

    # Build spline for tf
    phaseov2pisplineClass = pyspline.CubicSpline(freq, phase / (2*np.pi))
    tfspline = phaseov2pisplineClass.get_spline_d()

    # Global shift of phase to account for
    phase += 2 * np.pi * freq * tref

    # ==================================================================================================================
    # Compute TDI response for the two polarizations
    # ==================================================================================================================
    l0 = 2
    m0 = 2
    tdi_class_1 = pyresponse.LISAFDresponseTDI3Chan(gridfreq, tfspline, torb, l0, m0, inc, phi0, lambd, beta, psi_1,
                                                 TDI='TDIAET', LISAconst=LISAconst, responseapprox=responseapprox,
                                                 frozenLISA=frozenLISA, TDIrescaled=TDIrescaled)
    tdi_class_2 = pyresponse.LISAFDresponseTDI3Chan(gridfreq, tfspline, torb, l0, m0, inc, phi0, lambd, beta, psi_2,
                                                 TDI='TDIAET', LISAconst=LISAconst, responseapprox=responseapprox,
                                                 frozenLISA=frozenLISA, TDIrescaled=TDIrescaled)

    transfer_list_1 = tdi_class_1.get_response()
    transfer_list_2 = tdi_class_2.get_response()

    # ==================================================================================================================
    # Interpolation of the response
    # ==================================================================================================================
    if channels is None:
        channels = [1, 2, 3]

    if freq_data is not None:
        amp = pyspline.resample(freq_data, freq, amp)
        phase = pyspline.resample(freq_data, freq, phase)
        tr_int_1 = [pyspline.resample(freq_data, freq, transfer_list_1[i]) for i in channels]
        tr_int_2 = [pyspline.resample(freq_data, freq, transfer_list_2[i]) for i in channels]
    else:
        tr_int_1 = [transfer_list_1[i] for i in channels]
        tr_int_2 = [transfer_list_2[i] for i in channels]

    # GW strain
    h = amp * np.exp(1j * phase)
    # Phasor for the time shift
    z = np.exp(-2j * np.pi * freq_data * t_offset)
    # Compute response for all channels
    ch_interp_1 = [np.conj(h * tr_int_1[i - 1] * z) for i in channels]
    ch_interp_2 = [np.conj(h * tr_int_2[i - 1] * z) for i in channels]
    # Build design matrix
    mat_list = [np.array([ch_interp_1[i], ch_interp_2[i]]).T for i in range(len(ch_interp_1))]

    return mat_list
