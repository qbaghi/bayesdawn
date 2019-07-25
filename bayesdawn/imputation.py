#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:24:27 2019

@author: qbaghi

This module provide classes to perform missing data imputation steps based on 
Gaussian conditional model
"""

from .gaps import gapgenerator
import numpy as np
import time
from scipy import linalg as LA
from mecm import noise


class nearestNeighboor(object):
    """

    Implement the (naive) nearest-neighboor method for missing data imputation.

    Parameters
    ----------
    M : array_like
        binary mask
    PSD : PSD class instance
        class allowing to compute the PSD and the autocovariance of the noise
    Na : scalar integer
        number of points to consider before each gap (for the conditional
        distribution of gap data)
    Na : scalar integer
        number of points to consider after each gap

    """
    def __init__(self, M, Na=150, Nb=150):

        # The binary mask
        self.M = M[:]
        # Starting and ending points of gaps
        self.N_starts, self.N_ends = gapgenerator.findEnds(M)
        gap_lengths = self.N_ends - self.N_starts
        self.N_max = np.int( Na + Nb + np.max(gap_lengths) )
        if self.N_max > 2000:
            raise ValueError("The maximum size of gap + conditional set is too high.")
        self.Na = Na
        self.Nb = Nb

        # Total length of the data
        self.N = len(M)
        # Number of gaps
        N_gaps = len(self.N_starts)

        # Indices of embedding segments around each gap
        # The edges of each segment is set such that there are Na + Nb observed
        # data around, unless another gap is present.

        if N_gaps == 1:
            # 2 segments
            self.indices = [np.arange(np.int(np.max([self.N_starts[0] - Na, 0])),
                                      np.int(np.min([self.N_ends[0] + Nb, self.N])))]

        if N_gaps > 1:
            # first segment
            self.indices = [np.arange(np.int(np.max([self.N_starts[0] - Na,0])),
            np.int(np.min([self.N_ends[0]+Nb,self.N_starts[1]])))]
            # most of the segments
            self.indices = self.indices + [np.arange(np.int(np.max([self.N_starts[j] - Na,self.N_ends[j-1]])),
            np.int(np.min([self.N_ends[j]+Nb,self.N_starts[j+1]]))) for j in range(1, N_gaps-1)]
            # last segment
            self.indices = self.indices + [np.arange(np.int(np.max([self.N_starts[N_gaps-1]- Na,self.N_ends[N_gaps-2]])),
            np.int(np.min([self.N_ends[N_gaps-1]+Nb,self.N])))]
        # self.indices = [np.arange(np.int(np.max([self.N_starts[j]- Na,0])),
        # np.int(np.min([self.N_ends[j]+Nb,N]))) for j in range(len(self.N_starts))]
        # Indices of missing data
        self.ind_mis = np.where(M == 0)[0]
        # Indices of observed data
        self.ind_obs = np.where(M == 1)[0]

    def conditionalDraw(self, z_o, S_2N, CooI, C_mo, ind_obs, ind_mis, M, C):
        """
        Function performing random draws of the complete data noise vector
        conditionnaly on the observed data.

        Parameters
        ----------
        z_o : numpy array
            vector of observed residuals (size No)
        S_2N : numpy array (size P >= 2N)
            PSD vector
        CooI : 2d numpy array
            Inverse of covariance matrix of observed data
        C_mo : 2d numpy array
            Matrix of covariance between missing data with observed data
        ind_obs : array_like (size No)
            vector of chronological indices of the observed data points in the
            complete data vector
        ind_mis : array_like (size No)
            vector of chronological indices of the missing data points in the
            complete data vector
        M : numpy array (size N)
            mask vector (with entries equal to 0 or 1)

        Returns
        -------
        eps : numpy array (size Nm)
            realization of the vector of missing noise given the observed data


        References
        ----------
        J. Stroud et al, Bayesian and Maximum Likelihood Estimation for Gaussian
        Processes on an Incomplete Lattice, 2014


        """

        # the size of the vector that is randomly drawn is
        # equal to the size of the mask.
        # e = np.real(noise.generateNoiseFromDSP(np.sqrt(S_2N*2.), 1.)[0:M.shape[0]])
        e = np.random.multivariate_normal(np.zeros(M.shape[0]), C[0:M.shape[0], 0:M.shape[0]])

        # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
        eps = e[ind_mis] + C_mo.dot(CooI.dot(z_o - e[ind_obs]))

        return eps

    def draw_missing_data(self, y, y_model, PSD):
        """

        Draw the missing data from their conditional distributions on the
        observed data

        Parameters
        ----------
        y : array_like
            vector of observed data (size N)
        y_model : array_like
            vector of modelled signal (size N)
        PSD : PSD_spline instance
            class to compute the noise PSD


        Returns
        -------
        y_rec : array_like
            realization of the full data vector conditionnally to the observed
            data

        """
        # Compute the autocovariance from the full PSD and restrict it to N_max
        # points
        t1 = time.time()
        R = PSD.calculate_autocorr(self.N)[0:self.N_max]
        t2 = time.time()
        print("Computation of autocovariance took " + str(t2-t1))
        # Compute the spectrum on 2*N_max points
        S2 = PSD.calculate(2*self.N_max)
        t1 = time.time()

        # Impute the missing data: estimation of missing residuals
        y_mis_res = self.imputation(y-y_model, R, S2)
        # Construct the full imputed data vector
        # at observed value this is the same
        y_rec = np.zeros(self.N, dtype=np.float64)
        y_rec[self.ind_obs] = y[self.ind_obs]
        y_rec[self.ind_mis] = y_mis_res + y_model[self.ind_mis]
        t2 = time.time()
        print("Missing data imputation took " + str(t2-t1))

        return y_rec

    def imputation(self, y, R, S2):
        """

        Nearest neighboor imputation

        Parameters
        ----------
        y : array_like
            observed residuals
        R : array_like
            autocovariance function until lag N_max
        S2 : array_like
            values of the noise spectrum calculated on a Fourier grid of size
            2 N_max

        Returns
        -------
        y_mis : 1d numpy array
            imputed missing value

        """

        C = LA.toeplitz(R)

        # ======================================================================
        # For precomputations
        # ======================================================================
        #indj_obs = np.where(M[indices[0]]==1)[0]
        #indj_mis = np.where(M[indices[0]]==0)[0]
        #y_mis = np.array([]) #np.zeros(len(ind_mis),dtype = np.float64)
        #C_oo = C[np.ix_(indj_obs,indj_obs)]
        #CooI = LA.inv(C_oo)

        # ======================================================================
        # Gap per gap imputation
        # ======================================================================
        results = [self.single_imputation(y[indj], self.M[indj], C, S2) for indj in self.indices]
        y_mis = np.concatenate(results)
        # y_rec = np.zeros(N, dtype = np.float64)
        # y_rec[self.ind_obs] = y[self.ind_obs]
        # y_rec[self.ind_mis] = y_mis

        #return y_rec
        return y_mis

    def single_imputation(self, yj, Mj, C, S_2N):
        """

        Sample the missing data distribution conditionally on the observed data

        """
        # Local indices of missing and observed data
        ind_obsj = np.where(Mj == 1)[0]
        ind_misj = np.where(Mj == 0)[0]

        C_mo = C[np.ix_(ind_misj, ind_obsj)]
        #C_mm = C[np.ix_(ind_misj,ind_misj)]
        CooI = LA.inv(C[np.ix_(ind_obsj, ind_obsj)])

        out = self.conditionalDraw(yj[ind_obsj], S_2N, CooI, C_mo, ind_obsj, ind_misj, Mj, C)
        #out = conditionalDraw2(yj[ind_obsj],C_mm,C_mo,CooI)

        return out

