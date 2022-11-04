# -*- coding: utf-8 -*-
"""
@author: qbaghi

This module provide codes to perform efficient matrix-matrix and matrix-vector
computations.
"""
import numpy as np
from numpy import linalg as LA
from scipy import sparse
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
from . import fastoeplitz
pyfftw.interfaces.cache.enable()


def mat_vect_prod(y_in, ind_in, ind_out, mask, s_2n):
    """
    Linear operator that calculate Com y_in assuming that we can write:

    Com = M_o F* Lambda F M_m^T

    Parameters
    ----------
    y_in : numpy array
        input data vector
    ind_in : array_like
        array or list containing the chronological indices of the values
        contained in the input vector in the complete data vector
    ind_out : array_like
        array or list containing the chronological indices of the values
        contained in the output vector in the complete data vector
    M : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    N : scalar integer
        Size of the complete data vector
    s_2n : numpy array (size P >= 2N)
        Vector of DFT covariances. Should be be S(f) * fs / 2, where
        S(f) is the one-sided PSD.


    Returns
    -------
    y_out : numpy array
        y_out = Com * y_in transformed output vector of size N_out


    """

    # calculation of the matrix product Coo y, where y is a vector
    y = np.zeros(len(mask))  # + 1j*np.zeros(N)
    y[ind_in] = y_in

    n_fft = len(s_2n)

    return np.real(ifft(s_2n * fft(y, n_fft))[ind_out])


def matmat_prod(a_in, ind_in, ind_out, mask, s_2n):
    """
    Linear operator that calculates Coi * a_in assuming that we can write:

    Com = M_o F* Lambda F M_m^T

    Parameters
    ----------
    y_in : 2D numpy array
        input matrix of size (N_in x K)
    ind_in : array_like
        array or list containing the chronological indices of the values
        contained in the input vector in the complete data vector (size N_in)
    ind_out : array_like
        array or list containing the chronological indices of the values
        contained in the output vector in the complete data vector (size N_out)
    mask : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    N : scalar integer
        Size of the complete data vector
    s_2n : numpy array (size P >= 2N)
        Vector of DFT covariances. Should be be S(f) * fs / 2, where
        S(f) is the one-sided PSD.


    Returns
    -------
    A_out : numpy array
        Matrix (size N_out x K) equal to A_out = Com * a_in

    """
    N_in = len(ind_in)
    N_out = len(ind_out)

    (N_in_A,K) = np.shape(a_in)

    if N_in_A != N_in :
        raise TypeError("Matrix dimensions do not match")

    A_out = np.empty((N_out,K),dtype = np.float64)

    for j in range(K):
        A_out[:,j] = mat_vect_prod(a_in[:,j],ind_in,ind_out,mask,s_2n)

    return A_out


def precond_bicgstab(x0, b, a_func, n_it, stp, P, z0_hat=None, verbose=True):
    """
    Function solving the linear system
    x = A^-1 b
    with preconditioned bi-conjuage gradient algorithm


    Parameters
    ----------
    x0 : numpy array of size No
        initial guess for the solution (can be zeros(No) array)
    b : numpy array of size No
        observed vector (right hand side of the system)
    a_func: linear operator
        linear function of a vector x calculating A*x
    n_it : scalar integer
        number of maximal iterations
    stp : scalar float
        stp: stopping criteria
    P : scipy.sparse. operator
        preconditionner operator, calculating Px for all vectors x
    z0_hat : array_like (size N)
        first guess for solution, optional (default is None)



    Returns
    -------
    x : the reconstructed vector (numpy array of size N)
    """

    # Default first guess
    #z0_hat = None
    # Initialization of residual vector
    sr = np.zeros(n_it+1)
    #sz = np.zeros(n_it+1)
    k=0

    # Intialization of the solution
    b_norm = LA.norm(b)
    x = np.zeros(len(x0))
    x[:] = x0
    r = b - a_func(x0)
    z = P(r)

    p = np.zeros(len(r))
    p[:] = z

    z_hat = np.zeros(len(z))

    if z0_hat is None:
        z_hat[:] = z
    else:
        z_hat[:] = z0_hat

    sr[0] = LA.norm(r)

    # Iteration
    while (k < n_it) & (sr[k] > stp*b_norm):

        # Ap_k-1
        Ap = a_func(p)
        # Mq_k-1=Ap_k-1
        q = P(Ap)

        a = np.sum(np.conj(z)*z_hat) / np.sum(np.conj(q)*z_hat)

        x_12 = x + a*p
        r_12 = r - a*Ap
        z_12 = z - a*q

        Az_12 = a_func(z_12)
        s_12 = P(Az_12)

        w = np.sum(np.conj(z_12)*s_12) / np.sum(np.conj(s_12)*s_12)

        x = x_12 + w * z_12
        r = r_12 - w * Az_12
        z_new = z_12 - w * s_12

        b = a/w * np.sum(np.conj(z_new)*z_hat) / np.sum(np.conj(z)*z_hat)

        p = z_new + b * (p - w*q)

        z[:] = z_new
        sr[k + 1] = LA.norm(r)
        # zr[k+1] = LA.norm(z_new)

        # increment
        k = k + 1

        if verbose:
            if k % 20 == 0:
                print('PCG Iteration ' + str(k) + ' completed')
                print('Residuals = ' + str(sr[k])
                      + ' compared to criterion = '+str(stp * b_norm))

    print("Preconditioned BiCGSTAB algorithm ended with:")
    print(str(k) + "iterations." )
    info = 0

    if sr[k-1] > stp * b_norm:
        print("Attention: Preconditioned BiCGSTAB algorithm ended \
        without reaching the specified convergence criterium. Check quality of \
        reconstruction.")
        print("Current criterium: " +str(sr[k-1] / b_norm) + " > " + str(stp))
        info = 1

    # pcg_cls = PCG(a_func, n_it, stp, P)
    # x, sr = pcg_cls.solve(x0, b, verbose=verbose)
    #
    # info = 0
    # if sr[pcg_cls.k - 1] > stp * pcg_cls.b_norm:
    #     print("Attention: Preconditioned BiCGSTAB algorithm ended \
    #     without reaching the specified convergence criterium. Check quality \
    #     of reconstruction.")
    #     print("Current criterium: " + str(sr[pcg_cls.k - 1] / pcg_cls.b_norm)
    #           + " > " + str(stp))
    #     info = 1

    return x, sr, info  # ,sz


def print_pcg_status(info):
    """
    Function that takes the status result of the scipy.sparse.linalg.bicgstab
    algorithm and print it in an understandable way.
    """
    if info == 0:
        print("successful exit!")
    elif info > 0:
        print("convergence to tolerance not achieved")
        print("number of iterations: " + str(info))
    elif info < 0:
        print("illegal input or breakdown.")
        

def precond_linear_op(solver, N_out, N_in):

    P_func = lambda x: solver(x)
    PH_func = lambda x: solver(x)

    def Pmat_func(X):
        # Z = np.empty((N_out,X.shape[1]),dtype = np.float64)
        # for j in range(X.shape[1]):
        #     Z[:,j] = solver(X[:,j])
        return np.array([solver(X[:, j]) for j in range(X.shape[1])]).T

    p_op = sparse.linalg.LinearOperator(shape=(N_out, N_in), matvec=P_func,
                                        rmatvec=PH_func, matmat=Pmat_func,
                                        dtype=np.float64)

    return p_op


def cov_linear_op(ind_in, ind_out, mask, s_2n):
    """
    Construct a linear operator object that computes the operation C * v
    for any vector v, where C is a covariance matrix.


    Linear operator that calculate Com y_in assuming that we can write:

    Com = M_o F* Lambda F M_m^T

    Parameters
    ----------
    y_in : numpy array
        input data vector
    ind_in : array_like
        array or list containing the chronological indices of the values
        contained in the input vector in the complete data vector
    ind_out : array_like
        array or list containing the chronological indices of the values
        contained in the output vector in the complete data vector
    mask : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    s_2n : numpy array (size P >= 2N)
        Vector of DFT covariances. Should be be S(f) * fs / 2, where
        S(f) is the one-sided PSD.


    Returns
    -------
    Coi_op : scipy.sparse.linalg.LinearOperator instance
        linear opreator that computes the vector y_out = Com * y_in for any
        vector of size N_in

    """

    C_func = lambda x: mat_vect_prod(x, ind_in, ind_out, mask, s_2n)
    CH_func = lambda x: mat_vect_prod(x, ind_out, ind_in, mask, s_2n)
    Cmat_func = lambda X: matmat_prod(X, ind_in, ind_out, mask, s_2n)

    N_in = len(ind_in)
    N_out = len(ind_out)
    Coi_op = sparse.linalg.LinearOperator(shape=(N_out, N_in), matvec=C_func,
                                          rmatvec=CH_func, matmat=Cmat_func,
                                          dtype=np.float64)

    return Coi_op


def pcg_solve(ind_obs, mask, s_2n, b, x0, tol, maxiter, p_solver, pcg_algo):
    """
    Function that solves the problem Ax = b by calling iterative algorithms,
    using user-specified methods.
    Where A can be written as A = W_o F* D F W_o^T

    Parameters
    ----------
    ind_obs : array_like
        array of size n_o or list containing the chronological indices of the
        values contained in the observed data vector in the complete data
        vector
    mask : numpy array (size N)
        mask vector (with entries equal to 0 or 1)
    s_2n : numpy array (size P >= 2N)
        Vector of DFT covariances. Should be be S(f) * fs / 2, where
        S(f) is the one-sided PSD.
    b : numpy array
        vector of size n_o containing the right-hand side of linear system to
        solve
    x0 : numpy array
        vector of size n_o: first guess for the linear system to be solved
    tol : scalar float
        stopping criterium for the preconditioned conjugate gradient algorithm
    p_solver : sparse.linalg.factorized instance
        preconditionner matrix: linear operator which calculates an
        approximation of the solution: u_approx = C_OO^{-1} b for any vector b
    pcg_algo : string {'mine','scipy','scipy.bicgstab','scipy.bicg','scipy.cg',
        'scipy.cgs'}
        Type of preconditioned conjugate gradient (PCG) algorithm to use.


    Returns
    -------
    u : numpy array
        approximate solution of the linear system

    """

    n_o = len(ind_obs)

    if pcg_algo == 'mine':

        def coo_func(x):
            return mat_vect_prod(x, ind_obs, ind_obs, mask, s_2n)

        u, sr, info = precond_bicgstab(x0, b, coo_func, maxiter, tol, p_solver)

    elif 'scipy' in pcg_algo:
        coo_op = cov_linear_op(ind_obs, ind_obs, mask, s_2n)
        p_op = precond_linear_op(p_solver, n_o, n_o)
        tol_eff = np.min([tol, tol * LA.norm(b)])
        if (pcg_algo == 'scipy') | (pcg_algo == 'scipy.bicgstab'):
            u, info = sparse.linalg.bicgstab(coo_op, b, x0=x0, tol=tol_eff,
                                             maxiter=maxiter, M=p_op,
                                             callback=None)
            print_pcg_status(info)
        elif (pcg_algo == 'scipy.bicg'):
            u, info = sparse.linalg.bicg(coo_op, b, x0=x0, tol=tol_eff,
                                         maxiter=maxiter, M=p_op,
                                         callback=None)
            print_pcg_status(info)
        elif (pcg_algo == 'scipy.cg'):
            u, info = sparse.linalg.cg(coo_op, b, x0=x0, tol=tol_eff,
                                       maxiter=maxiter, M=p_op, callback=None)
            print_pcg_status(info)
        elif (pcg_algo == 'scipy.cgs'):
            u, info = sparse.linalg.cgs(coo_op, b, x0=x0, tol=tol_eff,
                                        maxiter=maxiter, M=p_op, callback=None)
            print_pcg_status(info)
        else:
            raise ValueError("Unknown PCG algorithm name")
        print("Value of || A * x - b ||/||b|| at exit:")
        print(str(LA.norm(coo_op.dot(u)-b)/LA.norm(b)))

    else:
        raise ValueError("Unknown PCG algorithm name")

    return u, info


def compute_precond(autocorr, mask, p=10, ptype='sparse', taper='Wendland2'):
    """
    For a given mask and a given PSD function, this function approximately 
    computes the linear operator x = C_OO^{-1} b for any vector b, 
    where C_OO is the covariance matrix of the observed data 
    (at points where mask==1).
    It uses an operator that is close to C_OO^{-1} but not exaclty it, 
    and is fast to compute.


    Parameters
    ----------
    autocorr : numpy array
        input autocovariance functions at each lag (size n_data)
    mask : numpy array
        mask vector
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function. This is needed to pre-conditionate the
        conjugate gradients.
    ptype : string {'sparse','circulant'}
        specifies the type of preconditioner matrix (sparse approximation of
        the covariance or circulant approximation of the covariance)
    taper : string {'Wendland1','Wendland2','Spherical'}
        Name of the taper function. This argument is only used if
        ptype='sparse'
    square : bool
        whether to build a square matrix. if False, then

    Returns
    -------
    solve : sparse.linalg.factorized instance
        preconditionner operator, calculating n_fft x for all vectors x

    """

    n_data = len(mask)

    # ======================================================================
    ind_obs = np.where(mask != 0)[0]  # .astype(int)
    # Calculate the covariance matrix of the complete data

    if ptype == 'sparse':
        # Preconditionning : use sparse matrix
        C = fastoeplitz.build_sparse_cov2(autocorr, p, n_data, 
                                          form="csc", taper=taper)
        # Calculate the covariance matrix of the observed data
        C_temp = C[:, ind_obs]
        # Preconditionner
        solve = sparse.linalg.factorized(C_temp[ind_obs, :])

    elif ptype == 'circulant':

        s_2n = np.real(fft(autocorr))
        n_fft = len(s_2n)

        def solve(v):
            return np.real(ifft(fft(v, n_fft)/s_2n, len(v)))

    return solve


def gls(dat, mat, sn):
    """
    Generalized least-square estimator.

    Parameters
    ----------
    dat : ndarray
        data vector, size n
    mat : ndarray
        design matrix, size n x p
    sn : ndarray
        variance vector, size n
        
    Returns
    -------
    amps : ndarray
        estimated amplitudes, size p
    """
    
    mat_weighted = mat / np.array([sn]).T
    amps = LA.pinv(np.dot(mat_weighted.conj().T, mat)).dot(
        np.dot(mat_weighted.conj().T, dat))
    
    return amps