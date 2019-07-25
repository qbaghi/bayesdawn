#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:11:56 2019

@author: qbaghi
"""
import numpy as np

from scipy import linalg as LA
from . import mhmcmc
from bayesdawn.bayesdawn.waveforms import wavefuncs, lisaresp
from .gaps import gapgenerator

# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft

        
        
class GWModel(object):
    """

    This class provide all the functions necessary to calcualte the likelihood
    of the gravitational wave data model



    """


    def __init__(self,names=['theta','phi','f_0','f_dot'],
        bounds=[[0,np.pi],[0.2*np.pi],[1e-4,1e-3],[1e-15,1e-10]],
        distribs = ['uniform','uniform','symbeta','uniform'],
        matmodeltype = 'chirping_GB_4p',
        timevect=[],
        tdi = 'X1',
        Phi_rot = 0,
        S_min = 1e-45,
        nc = 20,
        fmin = 1e-4,
        fmax = 1e-2,
        nsources = 1):

        # ======================================================================
        # Initialization of fixed data / parameters
        # ======================================================================
        # Number of GW sources
        self.nsources = nsources
        # Names of parameters
        self.names = names
        # Boundaries of parameters
        self.bounds = bounds
        self.lo,self.hi = self.get_lohi()
        # Number of parameters per source
        if nsources > 0:
            self.ndim = np.int(len(bounds)/nsources)
        else:
            self.ndim = 0
        # Type of prior distribution for each parameter
        self.distribs = distribs
        # Type of TDI channel
        self.tdi = tdi
        # Initial angle of LISA constellation
        self.Phi_rot = Phi_rot
        # Length of data
        self.N = len(timevect)


        # ======================================================================
        # Parameters for waveform computation
        # ======================================================================
        self.ts = timevect[1]-timevect[0]
        self.L = 2.5e9
        self.fs = 1/self.ts
        self.Tobs = self.N*self.ts
        
        # Order of the Bessel decomposition for frequency model
        self.nc = lisaresp.optimal_order(np.pi / 2, self.hi[2])
#        # Waveform model
#        if M == None:
#            # if there is no windowing use the standard frequency model
#            self.smodel = lisaresp.UCBWaveform(wavefuncs.v_func_gb,
#                                             Phi_rot=Phi_rot,
#                                             armlength=self.L,
#                                             nc=self.nc)        
#            self.T1 = 0
#            self.T2 = self.Tobs
#        else:
#            self.M = M
#            # if the windowing is a piece-wise tuckey window:
#            f_segs,self.T1,self.T2 = gapgenerator.compute_freq_times(self.M,self.ts)
            
        # Starting and end times of waveform model:
        self.T1 = 0
        self.T2 = self.Tobs        
        # Waveform model
            
        # Type of matrix model
        self.smodel = lisaresp.UCBWaveform(wavefuncs.v_func_gb,
                                           Phi_rot=Phi_rot,
                                           armlength=self.L,
                                           nc=self.nc)

        # Considered bandwidth for the fit
        self.fmin = fmin
        self.fmax = fmax
        # Frequency vector
        self.f = np.fft.fftfreq(self.N)*self.fs
        # Find corresponding indices in the frequency vector
        self.inds_pos = np.where((self.f>=fmin)&(self.f<=fmax))[0]
        self.inds_neg = self.N-self.inds_pos
        self.inds = np.concatenate((self.inds_pos,self.inds_neg))
        print("The estimation domain has size " + str(len(self.inds_pos)))
        # Number of positive frequencies considered
        self.npos = len(self.inds_pos)
        

    def matrix_model(self, f, params):
        """
    
        function of frequency f, source model parameters params describing the features of the analyzed data
    
        Parameters
        ----------
        model_type : string
            type of linear matrix model A(f) in the frequency domain such that the
            response writes s(f) = A(f)*b
    
        Returns
        -------
        A_matrix_func : callable

    
        """
    
        return self.smodel.design_matrix_freq(f, params, self.ts, self.T1, self.T2, tdi=self.tdi)


    def get_lohi(self):
        """
        Extract lower and upper bounds in separate vectors
        """
        
        lo = np.array([bound[0] for bound in self.bounds])
        hi = np.array([bound[1] for bound in self.bounds])
        
        return lo, hi
    
    
    def logp(self,x,lo,hi):
        return np.where(((x >= lo) & (x <= hi)).all(-1), 0.0, -np.inf)
    
    def logpo(self,x, lo, hi, i1, i2):
        
        return np.where(((x >= lo) & (x <= hi)).all(-1) & (x[i1] <= x[i2]) , 0.0, -np.inf)


    def log_prior(self,params):
        """
        Logarithm of the prior probabilitiy of parameters f_0 and f_dot

        Parameters
        ----------
        params : array_like
            vector of parameters in the orders of
            names=['f_0','f_dot']

        Returns
        -------
        logP : scalar float
            logarithm of the prior probability


        """
        #prior probability for f_0 and f_dot
        logs = [mhmcmc.logprob(params[i],self.distribs[i],self.bounds[i]) for i in range(len(params))]

        return np.sum(np.array(logs))


    def log_likelihood(self, params, S, y_fft):
        """
        Logarithm of the likelihood, optimized for FREQUENCY domain computations,
        and reduced to 2 parameters only.

        Parameters
        ----------
        params : array_like
            vector of parameters in the orders of
            names=['theta,'phi','f_0','f_dot']
        S : array_like
            noise spectrum, equal to fs * PSD / 2
        y_fft : array_like
            discrete fourier transform of the data


        Returns
        -------
        logL : scalar float
            logarithm of the likelihood

        """

        # # Update design matrix and derived quantities
        # A_freq,A_freq_w,ZI = self.compute_matrices(params, S)
        # # Data with real and imaginary part separated
        # yr = np.concatenate((y_fft[self.inds_pos].real, y_fft[self.inds_pos].imag))
        # N_bar = A_freq_w.conj().T.dot(yr)
        #
        # return np.real(N_bar.conj().T.dot(ZI.dot(N_bar)))


        # Update the frequency domain residuals
        z_fft_inds = y_fft[self.inds_pos] - self.draw_frequency_signal_onBW(params,S,y_fft)

        # Compute periodogram for relevant frequencies
        I_inds = np.abs(z_fft_inds)**2/self.N

        # Update reduced likelihood
        # return np.real(-0.5*np.sum(np.log(S[self.inds_pos]) + I_inds/S[self.inds_pos]))
        return np.real(-0.5 * np.sum(I_inds / S[self.inds_pos]))


    def draw_single_freq_signal_onBW(self, params, S, y_fft):
        """
        Compute deterministic signal model on the restricted bandwidth,
        for one single source only
        """
        # Update design matrix and derived quantities
        A_freq, A_freq_w, ZI = self.compute_matrices(params,S)

        #ZI = alfastfreq.compute_inverse_normal(A_freq,ApS)
        beta = self.draw_beta(ZI,A_freq_w,y_fft) 

        # Return the frequency domain GW signal
        #return np.dot(A_freq,beta)   
        y_gw_fft_2 = np.dot(A_freq,beta)

        return y_gw_fft_2[0:self.npos] + 1j*y_gw_fft_2[self.npos:]

    def draw_frequency_signal_onBW(self, params, S, y_fft):
        """
        Compute deterministic signal model on the restricted bandwidth only
        """

        if self.nsources > 0:
            y_gw_fft = self.draw_single_freq_signal_onBW(params, S, y_fft)
        else:
            y_gw_fft = np.zeros(self.npos, dtype=np.complex128)
                
        # Return the frequency domain GW signal
        return y_gw_fft     
    
    
    def draw_frequency_signal(self, params, S, y_fft):
        """
        Compute deterministic signal model on the full Fourier grid
        """
        
        y_gw_fft = np.zeros(self.N, dtype=np.complex128)

        y_gw_fft[self.inds_pos] = self.draw_frequency_signal_onBW(params,S,y_fft)
        y_gw_fft[self.inds_neg] = np.conj( y_gw_fft[self.inds_pos] )

        return y_gw_fft



    def compute_matrices(self,params,S):
        """

        Compute the design matrix, its weighted version, and the inverse normal
        matrix

        """
        if self.nsources == 1:
            # Update frequency model of the waveform
            #A_computed = self.matmodel(self.f[self.inds_pos],params,self)
            #A_freq = np.vstack((A_computed,np.conj(A_computed)))
            A_freq = self.matmodel(self.f[self.inds_pos],params)

        elif self.nsources > 1 :
            
            A_freq = np.hstack([self.matmodel(self.f[self.inds_pos],
                                              params[i*self.ndim:(i+1)*self.ndim]) for i in range(self.nsources)])
        
        # Weight matrix columns by the inverse of the PSD
        #A_freq_w = np.array([A_freq[:,j]/S[self.inds] for j in range(A_freq.shape[1])]).T
        S2 = np.concatenate((S[self.inds_pos],S[self.inds_pos]))
        A_freq_w = np.array([A_freq[:,j]/S2 for j in range(A_freq.shape[1])]).T
        
        # Inverse normal matrix
        
        ZI = LA.pinv( np.dot(np.transpose(A_freq).conj(),A_freq_w) )            

        return A_freq,A_freq_w,ZI


    def ls_estimate_beta(self,ZI,ApS,data_fft):
        """
        Generalized least-square estimate of the extrinsic parameters
        """
        
        data_real = np.concatenate((data_fft[self.inds_pos].real,
                                    data_fft[self.inds_pos].imag))
        #return np.real(np.dot( ZI, np.dot(np.transpose(ApS).conj(),data_fft[self.inds])))
        return np.dot( ZI, np.dot(np.transpose(ApS).conj(),data_real))
        #return np.dot( ZI, np.dot(np.transpose(ApS).conj(),data_fft[self.inds_pos]))


    def draw_beta(self,ZI,ApS,data_fft):
        """
        Draw the GW amplitude parameter vector beta from its conditional distribution

        Parameters
        ----------
        ZI : 2d numpy array
            inverse weighted normal matrix (A^* C^-1 A )^-1
        ApS : 2d numpy array
            weighted design matrix
        data_fft : array_like
            discrete fourier transform of data

        Returns
        -------
        beta : 1d numpy array
            random drawn of the conditional distribution of beta

        """
        return np.random.multivariate_normal(self.ls_estimate_beta(ZI,ApS,data_fft), ZI)
        #return np.real(np.dot( ZI, np.dot(np.transpose(ApS).conj(),data_fft[self.inds])))



    def compute_time_signal(self,y_gw_fft):
        """

        Compute the GW signal in the time domain given the
        Fourier-domain model in the relevant bandwidth.


        Parameters
        ----------
        y_gw_fft : array_like
            estimated GW signal in the frequency domain, in the Fourier grid

        Returns
        -------
        y_model : array_like
            estimated GW signal in the time domain


        """
        #y_gw_fft = np.zeros(self.N, dtype = np.complex128)
        #y_gw_fft[self.inds] = y_gw_fft_inds

        return np.real(ifft(y_gw_fft))

    def log_prob(self,params,params_aux):
        """
        Logarithm of the posterior probability function, optimized for
        FREQUENCY domain computations, reduced by Bessel decomposition to
        only 2 parameters : frequency and frequency derivative

        Parameters
        ----------
        params : array_like
            vector of parameters in the orders of
            names=['f_0','f_dot']

        Returns
        -------
        logp : scalar float
            logarithm of the posterior distribution

        """

        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params,params_aux)








class GWModelSeg(GWModel):
    """
    
    This class provide all the functions necessary to calcualte the likelihood
    of the gravitational wave data model, in the case of multiple segments 
    of data assumed to be independant. 
    
    """
    
    
    def __init__(self,M,names=['theta','phi','f_0','f_dot'],
                 bounds=[[0,np.pi],[0.2*np.pi],[1e-4,1e-3],[1e-15,1e-10]],
                 distribs = ['uniform','uniform','symbeta','uniform'],
                 matmodeltype = 'chirping_GB_4p',
                 data = [],
                 timevect=[],
                 tdi = 'X1',
                 Phi_rot = 0,
                 nc = 20,
                 fmin = 1e-4,
                 fmax = 1e-2,
                 nsources = 1):
        
        
        GWModel.__init__(self,names=names,
                         bounds=bounds,
                         distribs = distribs,
                         matmodeltype = matmodeltype,
                         data = data,
                         timevect=timevect,
                         tdi = tdi,
                         Phi_rot = Phi_rot,
                         nc = nc,
                         fmin = fmin,
                         fmax = fmax,
                         nsources = nsources)
        
        
        self.M = M
        self.N = len(M)
        
        f_segs,Tstarts,Tends = gapgenerator.compute_freq_times(self.M,self.ts)
        
        freq_list,Tstarts,Tends, self.inds_freq, self.inds_seg = \
        gapgenerator.select_freq(f_segs,Tstarts,Tends,self.fmin,self.fmax)
        
        self.Nstarts = [np.int(Ts*self.fs) for Ts in Tstarts] 
        self.Nends = [np.int(Te*self.fs) for Te in Tends] 
        self.Nseg = len(self.Nstarts)
        
        # All frequencies
        self.freqs = np.concatenate(freq_list)
        # Stack corresponding start end end times
        self.Tstarts = np.concatenate([Tstarts[j]*np.ones(len(self.inds_freq[j])) for j in range(self.Nseg)])
        self.Tends = np.concatenate([Tends[j]*np.ones(len(self.inds_freq[j])) for j in range(self.Nseg)])
        # Fourier grids for each segment
        self.fN = [f_segs[i] for i in self.inds_seg] 
        

        self.data_seg = [data[self.Nstarts[j]:self.Nends[j]] for j in range(self.Nseg)]
        
        # Prepare the data: Fourier transform each segment that can be used
        self.y_fft = [fft(dat) for dat in self.data_seg]
        
        y_fft_r = np.concatenate([self.y_fft[j][self.inds_freq[j]].real for j in range(len(self.y_fft))])
        y_fft_i = np.concatenate([self.y_fft[j][self.inds_freq[j]].imag for j in range(len(self.y_fft))])
        
        self.y_fft_f = np.concatenate((y_fft_r,y_fft_i))





    def matrix_model_seg(self,model_type):
        """
    
        Create a matrix model function depending on the type of source. 
        Assume several frequency data sets.
    
        Parameters
        ----------
        model_type : string
            type of linear matrix model A(f) in the frequency domain such that the
            response writes s(f) = A(f)*b
    
        Returns
        -------
        A_matrix_func : callable
            function of frequency f, source model parameters params, and model class
            cls describing the features of the analyzed data
    
        """
    
    
    
        if model_type == 'chirping_GB_4p':
    
            # chirping galactic binary, 4-parameter model matrix
            # params = [theta,phi,f_0,f_dot]
            def A_matrix_func(f,T1,T2,params):
                
                return self.smodel.design_matrix_freq(f,params,self.ts,T1,
                                                      T2,tdi = self.tdi)
    
    
        elif model_type == 'mono_GB_3p':
            # 3-parameter matrix model for monochromatic galactic binary source
            # params = [theta,phi,f_0]
            def A_matrix_func(f,T1,T2,params):
                params_mono = np.concatenate((params,[0]))
                # Update model matrix
                return self.smodel.design_matrix_freq(f,params_mono,self.ts,T1,
                                                      T2,tdi = self.tdi)

        return A_matrix_func


        
        
    def log_likelihood_seg(self,params,S):
        """
        Logarithm of the likelihood, optimized for FREQUENCY domain computations,
        in the case where the data are analysed segment by segment.
        
        Parameters
        ----------
        params : array_like
            vector of parameters in the orders of
            names=['theta,'phi','f_0','f_dot']
        S : array_like
            noise spectrum, equal to fs * PSD / 2

        
        
        Returns
        -------
        logL : scalar float
            logarithm of the likelihood
        
        """

        # Update design matrix and derived quantities
        A_freq,A_freq_w,ZI = self.compute_matrices_seg(params,S)
        # Data with real and imaginary part separated
        #yr = np.concatenate((y_fft[self.inds_pos].real,y_fft[self.ikeep].imag))

        
        N_bar = A_freq_w.conj().T.dot(self.y_fft_f)

        return np.real( N_bar.conj().T.dot(ZI.dot(N_bar)) )
        
         
            
        
    def compute_matrices_seg(self,params,Sf):
        """

        Compute the design matrix, its weighted version, and the inverse normal
        matrix
        
        Parameters
        ----------
        params : array_like
            vector of parameters in the orders of
            names=['theta,'phi','f_0','f_dot']      
            
        Sf : array_like
            noise spectrum, equal to fs * PSD /2, CALCULATED AT THE FREQUENCIES
            OF INTEREST !!
        

        """
        if self.nsources == 1:
            # Update frequency model of the waveform
            #A_computed = self.matmodel(self.f[self.inds_pos],params,self)
            #A_freq = np.vstack((A_computed,np.conj(A_computed)))
            A_freq = self.matmodel(self.freqs,self.Tstarts,
                                   self.Tends,params)

        elif self.nsources > 1 :
            
            A_freq = np.hstack([self.matmodel(self.freqs,
                                              self.Tstarts,
                                              self.Tends,
                                              params[i*self.ndim:(i+1)*self.ndim]) for i in range(self.nsources)])
        
        # Weight matrix columns by the inverse of the PSD
        #A_freq_w = np.array([A_freq[:,j]/S[self.inds] for j in range(A_freq.shape[1])]).T
        # For real and imaginary parts
        S2 = np.concatenate((Sf,Sf))
        A_freq_w = np.array([A_freq[:,j]/S2 for j in range(A_freq.shape[1])]).T
        
        # Inverse normal matrix
        ZI = LA.pinv( np.dot(np.transpose(A_freq).conj(),A_freq_w) )            

        return A_freq,A_freq_w,ZI

    def ls_estimate_beta_seg(self,ZI,ApS):
        """
        Generalized least-square estimate of the extrinsic parameters
        """
        
        #return np.real(np.dot( ZI, np.dot(np.transpose(ApS).conj(),data_fft[self.inds])))
        return np.dot( ZI, np.dot(np.transpose(ApS).conj(), self.y_fft_f))
    
    def draw_frequency_signal_onBW_seg(self, params, S):
        """
        Compute deterministic signal model on the restricted bandwidth,
        for one single source only
        """
        # Update design matrix and derived quantities
        A_freq,A_freq_w,ZI = self.compute_matrices_seg(params,S)

        #ZI = alfastfreq.compute_inverse_normal(A_freq,ApS)
        beta = self.draw_beta_seg(ZI,A_freq_w,self.y_fft_f) 

        # Return the frequency domain GW signal
        #return np.dot(A_freq,beta)   
        y_gw_fft_2 = np.dot(A_freq,beta)  
        
        return y_gw_fft_2[0:len(y_gw_fft_2)] + 1j*y_gw_fft_2[len(y_gw_fft_2):]
    
#        def draw_frequency_signal(self,params,S):
#            """
#            Compute deterministic signal model on the full Fourier grid
#            """
#            
#            y_gw_fft = [np.zeros(len(fk), dtype = np.complex128) for fk in self.fN]
#            
#            y_gw_fft[self.inds_freq] = self.draw_frequency_signal_onBW(params,S)
#            y_gw_fft[self.inds_neg] = np.conj( y_gw_fft[self.inds_pos] )
#    
#            return y_gw_fft