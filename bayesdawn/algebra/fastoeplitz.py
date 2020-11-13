# -*- coding: utf-8 -*-
"""

@author: qbaghi

This module provide routines to perform fast toeplitz matrix computations
"""
from . import matrixalgebra
import numpy as np
from scipy import sparse
# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft


def toeplitz_multiplication(v, first_row, first_column):
    """

    Performs the matrix-vector product T * v where
    T is a Toeplitz matrix, using FFT

    Parameters
    ----------
    v : array_like
        input vector of size n_data
    first_row : array_like
        first row of the Toepltiz matrix (size n_data)
    first_column : array_like
        first column of the Toepltiz matrix (size n_data)

    Returns
    -------
    y : numpy array
        vector such that y = T * v

    """

    n = first_row.shape[0]
    a_2n_fft = fft(np.concatenate((first_row,[0],first_column[1:][::-1])))

    return np.real(ifft(a_2n_fft*fft(v, 2*n))[0:n])


def multiple_toepltiz_inverse(c_mat, lambda_n, a):
    """

    Efficiently compute several Toepltiz systems with the same Toepltiz
    matrix

    T xj = cj

    which is in matrix form

    T x_mat = c_mat

    Where T is a n_data x n_data Toeplitz matrix and c_mat is a n_data x n_knots matrix

    """

    n = c_mat.shape[0]
    #zero_vect = np.zeros(n_data)


    # PRECOMPUTATIONS
    # Cf. Step 2 of Ref. [1]
    #ae_2n = np.concatenate(([1],a,zero_vect))
    #ae_2n_fft = fft(ae_2n)
    ae_2n_fft = fft(np.concatenate(([1],a)),2*n)
    # using hermitian and real property of covariance matrices:
    # be_2n_fft = fft(be_2n)
    be_2n_fft = ae_2n_fft.conj() #np.real(ae_2n_fft) - 1j*np.imag(ae_2n_fft)

    signs = (-1)**(np.arange(2*n)+1)

    x_mat = np.empty(np.shape(c_mat))

    print("shape of c_mat is " + str(c_mat.shape))

    for j in range(c_mat.shape[1]):
        #ce_2n = np.zeros(2*n_data)
        #ce_2n[0:n_data] = c_mat[:,j]
        #ce_2n = np.concatenate((c_mat[:,j],zero_vect))
        #ce_2n_fft = fft(ce_2n)
        ce_2n_fft = fft(c_mat[:,j],2*n)
        u_2n = ifft( ae_2n_fft*ce_2n_fft )
        v_2n = ifft( be_2n_fft*ce_2n_fft )

        #pe_2n_fft = fft( np.concatenate((v_2n[0:n_data],zero_vect)) )
        #qe_2n_fft = fft( np.concatenate((u_2n[n_data:],zero_vect))  )
        pe_2n_fft = fft( v_2n[0:n] , 2*n )
        qe_2n_fft = fft( u_2n[n:] , 2*n  )

        we_2n = ifft( ae_2n_fft*pe_2n_fft + signs*be_2n_fft*qe_2n_fft )

        x_mat[:,j] = np.real(we_2n[0:n]/lambda_n)

    return x_mat


def toepltiz_inverse_jain(c, lambda_n, a):
    """

    Solve for the system Tx = c
    where T is a symmetric Toeplitz matrix
    from precomputed solution

    T z = e_1

    where

    z = (1/lambda_n) * [ 1  a ]^T

    where a is a n_data-1 vector.

    Here we follow
    [1] Jain, Fast Inversion of Banded Toeplitz Matrices by Circular
    Decompositions, 1978

    Parameters
    ----------

    c : array_like
        right-hand side vector of the Toeplitz system T x = c
    lambda_n : scalar float
        constant such that z = (1/lambda_n) * [ 1  a ]^T is solution of T z = e1
        where e1 = [1 0 .. 0].
    a : array_like
        vector of size n_data-1 such that a = lambda_n * [z1 .. zN-1] where z is the
        solution of the system T z = e1

    Returns
    -------
    x : numpy array
        vector of size n_data, solution of the problem T x = c


    """

    n = len(c)

    # Cf. Step 2 of Ref. [1]
    # 1)
    #ae_2n = np.concatenate(([1],a,np.zeros(n_data)))

    # be_2n = np.concatenate(([1],np.zeros(n_data-1),[0],a[::-1]))
    # ce_2n = np.concatenate((c,np.zeros(n_data)))

    # 2)
    #ce_2n_fft = fft(ce_2n)
    #ae_2n_fft = fft(ae_2n)
    ce_2n_fft = fft(c,2*n)
    ae_2n_fft = fft(np.concatenate(([1],a)),2*n)
    # using hermitian and real property of covariance matrices:
    # be_2n_fft = fft(be_2n)
    be_2n_fft = ae_2n_fft.conj()
    #be_2n_fft = np.real(ae_2n_fft) - 1j*np.imag(ae_2n_fft)
    u_2n = ifft( ae_2n_fft*ce_2n_fft )
    v_2n = ifft( be_2n_fft*ce_2n_fft )

    # 3)
    # pe_2n = np.zeros(2*n_data)
    # qe_2n = np.zeros(2*n_data)
    # pe_2n[0:n_data] = v_2n[0:n_data]
    # qe_2n[0:n_data] = u_2n[n_data:]

    # or better:
    #pe_2n_fft = fft( np.concatenate((v_2n[0:n_data],np.zeros(n_data))) )
    #qe_2n_fft = fft( np.concatenate((u_2n[n_data:],np.zeros(n_data)))  )
    # or even better:
    pe_2n_fft = fft( v_2n[0:n], 2*n )
    qe_2n_fft = fft( u_2n[n:] , 2*n )

    # 4)
    signs = (-1)**(np.arange(2*n)+1)
    we_2n = ifft( ae_2n_fft*pe_2n_fft + signs*be_2n_fft*qe_2n_fft )

    return np.real(we_2n[0:n]/lambda_n)


def teopltiz_precompute(r, p=10, nit=1000, tol=1e-4, precond='taper'):
    """
    Solve the system T y = e1 where T is symmetric Toepltiz.
    where e1 = [1 0 0 0 0 0].T to compute the vector a and lambda_n for
    further fast Toepltiz inversions.

    Compute the prefactor lambda_n and vector a_{n-1} such that the inverse of
    T writes
    T^{-1} = (1/lambda_n) * [ 1       a*_{n-1}
                        a_{n-1}   S_{n-1} ]

    Parameters
    ----------
    r : array_like
        autocovariance function (first row of the Toepltiz matrix)
    p : scalar integer
        maximum number of lags for the preconditionning
    Nit : scalar integer
        maximum number of iterations for PCG
    tol : scalar float
        relative error convergence criterium for the PCG algorithm
    precond : str in {'taper', 'circulant'}
        preconditionner to use

    Returns
    -------
    lambda_n : scalar float
        prefactor of the inverse of T
    a : numpy array
        vector of size n_data-1 involved in the computation of the inverse of T



    References
    ----------
    [1] Jain, Fast Inversion of Banded Toeplitz Matrices by Circular
    Decompositions, 1978

    """
    ndim = len(r)
    # First basis vector (of orthonormal cartesian basis)
    e1 = np.concatenate(([1],np.zeros(ndim-1)))
    # Compute spectrum
    s_2n = fft(np.concatenate((r, [0] , r[1:][::-1])))
    # Linear operator correponding to the Toeplitz matrix
    t_op = toepltiz_linear_op(ndim, s_2n)
    # Preconditionner to approximate T^{-1}
    if precond == 'taper':
        psolver = compute_toepltiz_precond(r, p=p)
    elif precond == 'circulant':
        psolver = toepltiz_linear_op(ndim, fft(r))
    # Build the associated linear operator
    p_op = matrixalgebra.precond_linear_op(psolver, ndim, ndim)
    # Initial guess
    z, info = sparse.linalg.bicgstab(t_op, e1, 
                                     x0=np.zeros(ndim),
                                     tol=tol,
                                     maxiter=nit,
                                     M=p_op,
                                     callback=None)
    matrixalgebra.print_pcg_status(info)

    lambda_n = 1/z[0]
    a = lambda_n * z[1:]

    return lambda_n, a


# ==============================================================================
def compute_toepltiz_precond(r, p=10, taper='Wendland2'):
    """
    Compute a sparse preconditionner for solving T x = b where T is Toeplitz
    and symmetric, defined by autocovariance R


    Parameters
    ----------
    R : numpy array
        input autocovariance functions at each lag (size n_data)
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function. This is needed to pre-conditionate the conjugate
        gradients.
    taper : string {'Wendland1','Wendland2','Spherical'}
        name of the taper function.

    Returns
    -------
    solve : sparse.linalg.factorized instance
        preconditionner operator, calculating P x for all vectors x

    """

    # Preconditionning : use sparse matrix
    T_approx = build_sparse_cov2(r, p, len(r), form="csc", taper = taper)
    # Preconditionner
    solve = sparse.linalg.factorized(T_approx)

    return solve




# ==============================================================================
def toepltiz_mat_vect_prod(y, s_2n):
    """
    Linear operator that calculate T y_in assuming that we can write:

    Com = F* Lambda F

    where Lambda is a P x P diagonal matrix and F is the P x n_data Discrete Fourier
    Transform matrix.

    Parameters
    ----------
    y : numpy array
        input data vector of size n_data
    S_2N : numpy array (size P >= 2N)
        PSD vector


    Returns
    -------
    y_out : numpy array
        y_out = T * y_in transformed output vector of size N_out


    """

    return np.real(ifft(s_2n * fft(y, len(s_2n)))[0:len(y)])


def toepltiz_linear_op(ndim, s_2n):
    """
    Construct a linear operator object that computes the operation C * v
    for any vector v, where C is a covariance matrix.


    Linear operator that calculate Com y_in assuming that we can write:

    T =  F* Lambda F

    Parameters
    ----------
    ndim : scalar integer
        size of the corresponding Toepltiz matrix
    s_2n : numpy array (size P >= 2N)
        PSD vector


    Returns
    -------
    t_op : scipy.sparse.linalg.LinearOperator instance
        linear opreator that computes the vector y_out = T * y_in for any
        vector of size n_data

    """

    t_func = lambda x: toepltiz_mat_vect_prod(x, s_2n)
    th_func = lambda x: toepltiz_mat_vect_prod(x, s_2n)
    tmat_func = lambda X: np.array([toepltiz_mat_vect_prod(X[:,j],s_2n) 
                                    for j in X.shape[1]]).T

    t_op = sparse.linalg.LinearOperator(shape=(ndim,ndim),
                                        matvec=t_func,
                                        rmatvec=th_func,
                                        matmat=tmat_func,
                                        dtype=np.float64)

    return t_op


def taper_covariance(h, theta, taper='Wendland1', tau=10):
    """
    Function calculating a taper covariance that smoothly goes to zero when h
    goes to theta. This taper is to be mutiplied by the estimated covariance
    autocorr of the process, to discard correlations larger than theta.

    Ref : Reinhard FURRER, Marc G. GENTON, and Douglas NYCHKA,
    Covariance Tapering for Interpolation of Large Spatial Datasets,
    Journal of Computational and Graphical Statistics, Volume 15, Number 3,
    Pages 502–523,2006

    Parameters
    ----------
    h : numpy array of size n_data
        lag
    theta : scalar float
        taper parameter
    taper : string {'Wendland1','Wendland2','Spherical'}
        name of the taper function


    Returns
    -------
    C_tap : numpy array
        the taper covariance function values (vector of size n_data)
    """

    ii = np.where(h <= theta)[0]

    if taper == 'Wendland1':

        c = np.zeros(len(h))

        c[ii] = (1.-h[ii]/np.float(theta))**4 * (1 + 4.*h[ii]/np.float(theta))

    elif taper == 'Wendland2':

        c = np.zeros(len(h))

        c[ii] = (1-h[ii]/np.float(theta))**6 * (1 + 6.*h[ii]/theta + \
        35*h[ii]**2/(3.*theta**2))

    elif taper == 'Spherical':

        c = np.zeros(len(h))

        c[ii] = (1-h[ii]/np.float(theta))**2 * (1 + h[ii]/(2*np.float(theta)) )

    elif taper == 'Hanning':
        c = np.zeros(len(h))
        c[ii] = (np.hanning(2*theta)[theta:2*theta])**2

    elif taper == 'Gaussian':

        c = np.zeros(len(h))
        sigma = theta/5.
        c[ii] = np.sqrt( 1./(2*np.pi*sigma**2) ) * np.exp( - h[ii]**2/(2*sigma**2) )

    elif taper == 'rectSmooth':

        c = np.zeros(len(h))
        c[h <= theta-tau] = 1.
        jj = np.where( (h >= theta-tau) & (h <= theta) )[0]
        c[jj] = 0.5*(1 + np.cos( np.pi*(h[jj]-theta+tau)/np.float(tau) ) )

    elif taper == 'modifiedWendland2':

        c = np.zeros(len(h))
        c[h<=theta-tau] = 1.
        jj = np.where( (h>=theta-tau) & (h<=theta) )[0]

        c[jj] = (1-(h[jj]-theta+tau)/np.float(tau))**6 * (1 + \
        6.*(h[jj]-theta+tau)/tau + 35*(h[jj]-theta+tau)**2/(3.*tau**2))

    elif taper == 'rect':

        c = np.zeros(len(h))
        c[h <= theta] = 1.

    return c


def build_sparse_cov2(autocorr, p, n_data, form=None, taper='Wendland2'):
    """
    This function constructs a sparse matrix which is a tappered, approximate
    version of the covariance matrix of a stationary process of autocovariance
    function autocorr and size n_data x n_data.

    Parameters
    ----------
    autocorr : numpy array
        input autocovariance functions at each lag (size n_data)
    p : scalar integer
        number of lags to calculate the tapered approximation of the
        autocoariance function.
    n_data : scalar integer
        Size of the complete data vector
    form : character string (or None), optional
        storage format of the sparse matrix (default is None)
    taper : string
        type of taper function to smoothly decrease the tapered covariance
        function down to zero.


    Returns
    -------
    C_tap : scipy.sparse matrix
        Tappered covariance matrix (size n_data x n_data) with p non-zero diagonals.


    """
    k = np.array([])
    values = list()
    tap = taper_covariance(np.arange(0, n_data), p, taper=taper)
    r_tap = autocorr[0:n_data] * tap

    # calculate the rows and columns indices of the non-zero values
    # Do it diagonal per diagonal
    for i in range(p+1):
        rf = np.ones(n_data - i) * r_tap[i]
        values.append(rf)
        k = np.hstack((k, i))
        # Symmetric values
        if i != 0:
            values.append(rf)
            k = np.hstack((k, -i))
    # [build_diagonal(values, r_tap, k, i) for i in range(p + 1)]

    return sparse.diags(values, k.astype(int), format=form,
                        dtype=autocorr.dtype)