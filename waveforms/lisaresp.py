#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:09:36 2019

@author: qbaghi
"""


import copy
import numpy as np
import LISAConstants as LC
from scipy import special
from .coeffs import k_coeffs
import lisabeta
from scipy.interpolate import InterpolatedUnivariateSpline as spline

# For MBHB only, use MLDC code
import GenerateFD_SignalTDIs


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

        # Phase function
        self.v_func = v_func
        # For Fourier series decomposition:
        self.M = 4
        self.m_vect = np.arange(0,self.M+1)
        self.nc = nc
        self.jw2 = []

    def indices_low_freq(self, tdi):
        """
    
        Returns the indices of the integrated gravitational strain signal for the
        low frequency approximation of the tdi response, such that
    
        dTDI = (1-D^2)(1+D)(H_i - H_j)
    
        Parameters
        ----------
        tdi : string
            tdi channel among {'X1','Y1','Z1'}
    
        Returns
        -------
        i : scalar integer
            index of first arm
        j : scalar integer
            index of second arm
    
    
        """
    
        if tdi == 'X1':
            i = 2
            j = 3
    
        elif tdi == 'Y1':
            i = 3
            j = 1
    
        elif tdi == 'Z1':
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


    def design_matrix_freq(self, f, params, ts, Tstart, Tend, tdi='X1'):
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
        i, j = self.indices_low_freq(tdi)
    
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

    def __init__(self, Phi_rot=0, armlength=2.5e9):
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

        # For PhenomD waveform
        # --------------------
        # fRef=0 means fRef=fpeak in PhenomD or maxf if out of range
        self.fRef = 0.0
        self.t0 = 0.0

    def compute_signal_freq(self, f, params, del_t, tobs, tdi='TDIXYZ', real_imag=False):
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
        tdi : string
            tdi channel among {'X1','X2','X3'}
        real_imag : boolean
            if True, returns real and imaginary parts of the signal in distinct arrays. Otherwise, one single complex
            array is returned.

        """

        phi0, m1, m2, a1, a2, dist, inc, lam, beta, psi, tc = params

        # phi0, fRef, m1, m2, a1, a2, dist, inc, lam, beta, psi, Tobs, tc, del_t = params
        # TDI response on native grid
        fr, x, y, z, wfTDI = GenerateFD_SignalTDIs.MBHB_LISAGenerateTDIfast(phi0, self.fRef, m1, m2, a1, a2,
                                                                            dist, inc,
                                                                            lam,
                                                                            beta,
                                                                            psi,
                                                                            tobs,
                                                                            tc,
                                                                            del_t, tShift=0.0, fmin=1.e-5,
                                                                            fmax=0.5 / del_t, frqs=None, resf=None)

        # Interpolate the response on required grid
        real_func = spline(fr, np.real(x), ext='zeros')
        imag_func = spline(fr, np.imag(x), ext='zeros')
        xf_real = real_func(f)
        xf_imag = imag_func(f)

        if real_imag:
            return xf_real, xf_imag
        else:
            return xf_real + 1j * xf_imag

    def design_matrix_freq(self, f, params, del_t, t1, t2, tdi='TDIXYZ', complex=False):
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

        # phi0, m1, m2, a1, a2, dist, inc, lam, beta, psi, tc = params
        # m1, m2, chi1, chi2, del_t, dist, inc, phi, lambd, beta, psi = params
        tobs = t2 - t1

        params_1 = copy.deepcopy(params)
        params_2 = copy.deepcopy(params)

        # Initial phase
        params_1[0] = 0
        params_2[0] = 0
        # Luminosity distance (Mpc)
        params_1[5] = 1e3
        params_2[5] = 1e3
        # Inclination
        params_1[6] = 0 # 0.5 * np.pi
        params_2[6] = 0 # 0.5 * np.pi
        # Polarization angle
        params_1[9] = 0.0
        params_2[9] = np.pi/4


        # amp = 2* G * mu / (dl * c**2)

        # Amp, iota, psi_m, phi0
        # 1. 1.e-21, 0.5*np.pi, 0.0, 0.0
        # 2. 1.e-21, 0.5*np.pi, 0.25*np.pi, 0.0

        # wftdi = lisabeta.lisa.GenerateLISATDI(params, tobs=1., minf=1e-5, maxf=1., torb=0., TDItag='TDIAET',
        #                                       acc=1e-4, order_fresnel_stencil=0, approximant='IMRPhenomD',
        #                                       LISAconst=lisabeta.lisa.pyresponse.LISAconstProposal,
        #                                       responseapprox='full',
        #                                       frozenLISA=False, TDIrescaled=False)


        # Calculate the response on required grid
        ac_plus_real, ac_plus_imag = self.compute_signal_freq(f, params_1, del_t, tobs, tdi=tdi, real_imag=True)
        ac_cros_real, ac_cros_imag = self.compute_signal_freq(f, params_2, del_t, tobs, tdi=tdi, real_imag=True)


        if not complex:
            a_mat = np.empty((2 * len(f), 4), dtype=np.float64)
            a_mat[:, 0] = np.concatenate((ac_plus_real, ac_plus_imag))
            a_mat[:, 1] = np.concatenate((ac_cros_real, ac_cros_imag))
            a_mat[:, 2] = np.concatenate((ac_plus_imag, -ac_plus_real))
            a_mat[:, 3] = np.concatenate((ac_cros_imag, -ac_cros_real))
        else:
            a_mat = np.empty((len(f), 4), dtype=np.complex128)
            a_mat[:, 0] = ac_plus_real + 1j * ac_plus_imag
            a_mat[:, 1] = ac_cros_real + 1j * ac_cros_imag

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

        return a_mat

