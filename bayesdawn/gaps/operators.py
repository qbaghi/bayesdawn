# -*- coding: utf-8 -*-
"""
Created on Friday March 06

Time-domain and frequency domain operators useful for gapped data analysis

@author: qbaghi

"""
import numpy as np
import pyfftw
from bayesdawn.utils import fastoeplitz
from pyfftw.interfaces.numpy_fft import fft, ifft
from scipy import sparse, linalg
pyfftw.interfaces.cache.enable()


class MappingOperator(object):

    def __init__(self, inds, n_data):
        """Implement W_o and W_m operators mapping the observed/missing data
        to the full data vector and conversely.

        W_o matrix has size N_o x n_data

        Parameters
        ----------
        inds : array_like
            Indices of the identified data points in the full vector y
        n_data : int
            Size of the full data vector y


        """

        self.inds = inds
        self.n_data = n_data

    def build_matrix(self, sp=True):
        """
        Construct the sparse N_o x n_data matrix W_o

        First row:
        has 1 at the index for the first observed data point, ortherwise zero

        ith row:
        has 1 at the index corresponding to the ith observed data point

        Parameters
        ----------
        sp : bool
            if True, use a sparse representation of the matrix


        Returns
        -------
        w_o : scipy.sparse matrix
            W_o matrix

        Example
        -------

        If the full time series has size 5 and the observed points are 0, 1,
        and 4, then the matrix reads:

        [1 0 0 0 0]
        [0 1 0 0 0]
        [0 0 0 0 1]

        """
        if sp:
            data = np.ones(len(self.inds))
            row_ind = np.arange(0, len(self.inds))
            w_o = sparse.csc_matrix((data, (row_ind, self.inds)), 
                                    shape=(len(self.inds), self.n_data))
            return w_o
        else:
            return np.array([self.binary_vector(id) for id in self.inds])

    def binary_vector(self, id):

        v = np.zeros(self.n_data)
        v[id] = 1

        return v

    def apply(self, y):
        """Apply transformation

        y_o = W_o y

        Parameters
        ----------
        y : ndarray
            Data vector to be mapped, must be of size n_data

        Returns
        -------
        y_o : ndarray
            mapped data vector, size n_o = len(inds)

        """

        return y[self.inds]

    def apply_transpose(self, y_o):
        """Apply reverse transformation

        y = W_o^{T} y_o

        Parameters
        ----------
        y_o : ndarray
            mapped data vector, size n_o = len(inds)

        Returns
        -------
        y : ndarray
            vector of full size n_data whose entries at indices inds are equal
            to y_o

        """

        y = np.zeros(self.n_data, dtype=y_o.dtype)
        y[self.inds] = y_o

        return y_o

    def apply_fourier(self, y_tilde):
        """Apply transformation

        y_o = W_o F^* y_tilde

        Parameters
        ----------
        y_tilde : ndarray
            DFT Data vector or matrix to be mapped, must be of size n_data x k

        Returns
        -------
        y_o : ndarray
            mapped data vector, size n_o = len(inds)

        """

        if len(y_tilde.shape) == 1:
            return self.apply(ifft(y_tilde) / np.sqrt(self.n_data))
        elif len(y_tilde.shape) == 2:
            return np.array([
                self.apply(ifft(y_tilde[:, i]) / np.sqrt(self.n_data))
                for i in range(y_tilde.shape[1])]).T

    def apply_transpose_fourier(self, y_o):
        """Apply transformation

        y_tilde = F W_o^{T} y_o

        Parameters
        ----------
        y_o : ndarray
            mapped data vector, size n_o = len(inds)

        Returns
        -------
        y_tilde : ndarray
            DFT Data vector, size n_data x k

        """

        if len(y_o.shape) == 1:
            return fft(self.apply_transpose(y_o)) / np.sqrt(self.n_data)
        elif len(y_o.shape) == 2:
            return np.array(
                [fft(self.apply_transpose(y_o[:, i])) / np.sqrt(self.n_data)
                 for i in range(y_o.shape[1])]).T


class CovarianceOperator(object):
    
    def __init__(self, autocorr, inds_obs, inds_mis):
        """
        Toeplitz Noise covariance operator defined by a an autocovariance 
        function calculated at sampling times.

        Parameters
        ----------
        autocorr : dnarray
            autocovariance, size n
        inds_obs : array_like
            observed data indices, size n_o
        inds_mis : array_like
            missing data indices, size n_m
        """
        
        self.autocorr = autocorr
        self.inds_obs = inds_obs
        self.inds_mis = inds_mis
        self.no = len(inds_obs)
        self.nm = len(inds_mis)
        self.n = len(autocorr)
        
        self.limit = 1e4
        # Covariance of missing data
        self.c_mm = np.array([])
        # Square covariance at missing data points
        self.c2_mm = np.array([])
        # Cubic covariance at missing data points
        self.c3_mm = np.array([])
        
    def get_cmm(self):
        """
        Returns missing data covariance.
        """
        
        if self.c_mm == np.array([]):
            self.compute_cmm()
        
        return self.c_mm
    
    def compute_cmm(self, power=1):
        """
        Compute the missing data covariance matrix and store it.
        """
        
        id_mx, id_my = np.meshgrid(self.inds_mis, self.inds_mis)
        first_row = self.autocorr**power
        
        if power == 1:
            self.c_mm = first_row[np.abs(id_mx - id_my)]
        elif power == 2 :
            self.c2_mm = first_row[np.abs(id_mx - id_my)]
        elif power == 3 :
            self.c3_mm = first_row[np.abs(id_mx - id_my)]
        
    def dot(self, v):
        """
        Perform the dot product of the covariance matrix with a vector v

        Parameters
        ----------
        v : ndarray
            vector of size n
        """
        
        return fastoplitz.toeplitz_multiplication(v, self.autocorr, 
                                                  self.autocorr)
        
    def compute_k_mat(self):
        """
        Compute the matrix 
        K = I + Z^T Sigma Y involved in the Woodbury formula
        """
        i_mm = np.diag(np.ones(len(self.inds_mis)))
        self.k_mat = np.block([[i_mm + self.c_mm - sigma2_mm,        
                                self.c_mm],
                               [self.c_mm - (1 + self.c_mm)*sigma2_mm + sigma3_mm,
                            i_mm + sigma_mm_2 - sigma2_mm]])
    
    def get_coo(self):
        """
        Returns observed data covariance.
        """
        
        id_ox, id_oy = np.meshgrid(self.inds_obs, self.inds_obs)
        
        return self.autocorr[np.abs(id_ox - id_oy)]
    
    def get_full_cov(self, n_max=None):
        """
        Return full covariance in the time domain.
        """
        if n_max is None:
            n_max = self.n
        
        return linalg.toeplitz(self.autocorr[0:n_max])
        


def mask_matrix(mat, threshold=1e-3):

    inds = np.where(np.abs(mat) < np.max(np.abs(mat)) * threshold)
    matmask = np.zeros(mat.shape)
    matmask[inds] = 1
    mat_plot_masked = np.ma.array(mat, mask=matmask)

    return mat_plot_masked
