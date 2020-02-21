import numpy as np
from scipy import sparse
# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft

from mecm import matrixalgebra
from mecm import mecm

def toeplitz_multiplication(v,first_row,first_column):
    """

    Performs the matrix-vector product T * v where
    T is a Toeplitz matrix, using FFT

    Parameters
    ----------
    v : array_like
        input vector of size N
    first_row : array_like
        first row of the Toepltiz matrix (size N)
    first_column : array_like
        first column of the Toepltiz matrix (size N)

    Returns
    -------
    y : numpy array
        vector such that y = T * v

    """

    N = len(v)
    a_2N_fft = fft(np.concatenate((first_row,[0],first_column[1:][::-1])))

    return np.real(ifft(a_2N_fft*fft(v,2*N))[0:N])


def multiple_toepltiz_inverse(c_mat,lambda_n,a):
    """

    Efficiently compute several Toepltiz systems with the same Toepltiz
    matrix

    T xj = cj

    which is in matrix form

    T x_mat = c_mat

    Where T is a N x N Toeplitz matrix and c_mat is a N x J matrix

    """

    N = c_mat.shape[0]
    #zero_vect = np.zeros(N)


    # PRECOMPUTATIONS
    # Cf. Step 2 of Ref. [1]
    #ae_2n = np.concatenate(([1],a,zero_vect))
    #ae_2n_fft = fft(ae_2n)
    ae_2n_fft = fft(np.concatenate(([1],a)),2*N)
    # using hermitian and real property of covariance matrices:
    # be_2n_fft = fft(be_2n)
    be_2n_fft = ae_2n_fft.conj() #np.real(ae_2n_fft) - 1j*np.imag(ae_2n_fft)

    signs = (-1)**(np.arange(2*N)+1)

    x_mat = np.empty(np.shape(c_mat))

    print("shape of c_mat is " + str(c_mat.shape))

    for j in range(c_mat.shape[1]):
        #ce_2n = np.zeros(2*N)
        #ce_2n[0:N] = c_mat[:,j]
        #ce_2n = np.concatenate((c_mat[:,j],zero_vect))
        #ce_2n_fft = fft(ce_2n)
        ce_2n_fft = fft(c_mat[:,j],2*N)
        u_2n = ifft( ae_2n_fft*ce_2n_fft )
        v_2n = ifft( be_2n_fft*ce_2n_fft )

        #pe_2n_fft = fft( np.concatenate((v_2n[0:N],zero_vect)) )
        #qe_2n_fft = fft( np.concatenate((u_2n[N:],zero_vect))  )
        pe_2n_fft = fft( v_2n[0:N] , 2*N )
        qe_2n_fft = fft( u_2n[N:] , 2*N  )

        we_2n = ifft( ae_2n_fft*pe_2n_fft + signs*be_2n_fft*qe_2n_fft )

        x_mat[:,j] = np.real(we_2n[0:N]/lambda_n)

    return x_mat


def toepltiz_inverse_jain(c,lambda_n,a):
    """

    Solve for the system Tx = c
    where T is a symmetric Toeplitz matrix
    from precomputed solution

    T z = e_1

    where

    z = (1/lambda_n) * [ 1  a ]^T

    where a is a N-1 vector.

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
        vector of size N-1 such that a = lambda_n * [z1 .. zN-1] where z is the
        solution of the system T z = e1

    Returns
    -------
    x : numpy array
        vector of size N, solution of the problem T x = c


    """

    N = len(c)

    # Cf. Step 2 of Ref. [1]
    # 1)
    #ae_2n = np.concatenate(([1],a,np.zeros(N)))

    # be_2n = np.concatenate(([1],np.zeros(N-1),[0],a[::-1]))
    # ce_2n = np.concatenate((c,np.zeros(N)))

    # 2)
    #ce_2n_fft = fft(ce_2n)
    #ae_2n_fft = fft(ae_2n)
    ce_2n_fft = fft(c,2*N)
    ae_2n_fft = fft(np.concatenate(([1],a)),2*N)
    # using hermitian and real property of covariance matrices:
    # be_2n_fft = fft(be_2n)
    be_2n_fft = ae_2n_fft.conj()
    #be_2n_fft = np.real(ae_2n_fft) - 1j*np.imag(ae_2n_fft)
    u_2n = ifft( ae_2n_fft*ce_2n_fft )
    v_2n = ifft( be_2n_fft*ce_2n_fft )

    # 3)
    # pe_2n = np.zeros(2*N)
    # qe_2n = np.zeros(2*N)
    # pe_2n[0:N] = v_2n[0:N]
    # qe_2n[0:N] = u_2n[N:]

    # or better:
    #pe_2n_fft = fft( np.concatenate((v_2n[0:N],np.zeros(N))) )
    #qe_2n_fft = fft( np.concatenate((u_2n[N:],np.zeros(N)))  )
    # or even better:
    pe_2n_fft = fft( v_2n[0:N], 2*N )
    qe_2n_fft = fft( u_2n[N:] , 2*N )

    # 4)
    signs = (-1)**(np.arange(2*N)+1)
    we_2n = ifft( ae_2n_fft*pe_2n_fft + signs*be_2n_fft*qe_2n_fft )

    return np.real(we_2n[0:N]/lambda_n)



def teopltiz_precompute(R,p=10,Nit = 1000,tol = 1e-4):
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
    R : array_like
        autocovariance function (first row of the Toepltiz matrix)


    Returns
    -------
    lambda_n : scalar float
        prefactor of the inverse of T
    a : numpy array
        vector of size N-1 involved in the computation of the inverse of T
    p : scalar integer
        maximum number of lags for the preconditionning
    Nit : scalar integer
        maximum number of iterations for PCG
    tol : scalar float
        relative error convergence criterium for the PCG algorithm


    References
    ----------
    [1] Jain, Fast Inversion of Banded Toeplitz Matrices by Circular
    Decompositions, 1978

    """
    N = len(R)
    # First basis vector (of orthonormal cartesian basis)
    e1 = np.concatenate(([1],np.zeros(N-1)))
    # Compute spectrum
    S_2N = fft(np.concatenate((R,[0],R[1:][::-1])))
    # Linear operator correponding to the Toeplitz matrix
    T_op = toepltizLinearOp(N,S_2N)
    # Preconditionner to approximate T^{-1}
    Psolver = computeToepltizPrecond(R,p=p)
    # Build the associated linear operator
    P_op = matrixalgebra.precondLinearOp(Psolver,N,N)
    # Initial guess
    z,info = sparse.linalg.bicgstab(T_op, e1, x0=np.zeros(N),tol=tol,
    maxiter=Nit,M=P_op,callback=None)
    matrixalgebra.printPCGstatus(info)

    lambda_n = 1/z[0]
    a = lambda_n * z[1:]

    return lambda_n,a



# ==============================================================================
def computeToepltizPrecond(R,p=10,taper = 'Wendland2'):
    """
    Compute a sparse preconditionner for solving T x = b where T is Toeplitz
    and symmetric, defined by autocovariance R


    Parameters
    ----------
    R : numpy array
        input autocovariance functions at each lag (size N)
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
    T_approx = mecm.build_sparse_cov2(R, p, len(R), form="csc", taper = taper)
    # Preconditionner
    solve =  sparse.linalg.factorized(T_approx)

    return solve




# ==============================================================================
def toepltizMatVectProd(y,S_2N):
    """
    Linear operator that calculate T y_in assuming that we can write:

    Com = F* Lambda F

    where Lambda is a P x P diagonal matrix and F is the P x N Discrete Fourier
    Transform matrix.

    Parameters
    ----------
    y : numpy array
        input data vector of size N
    S_2N : numpy array (size P >= 2N)
        PSD vector


    Returns
    -------
    y_out : numpy array
        y_out = T * y_in transformed output vector of size N_out


    """

    return np.real( ifft( S_2N * fft(y,len(S_2N)) )[0:len(y)] )



def toepltizLinearOp(N,S_2N):
    """
    Construct a linear operator object that computes the operation C * v
    for any vector v, where C is a covariance matrix.


    Linear operator that calculate Com y_in assuming that we can write:

    T =  F* Lambda F

    Parameters
    ----------
    N : scalar integer
        size of the corresponding Toepltiz matrix
    S_2N : numpy array (size P >= 2N)
        PSD vector


    Returns
    -------
    T_op : scipy.sparse.linalg.LinearOperator instance
        linear opreator that computes the vector y_out = T * y_in for any
        vector of size N

    """

    T_func = lambda x: toepltizMatVectProd(x,S_2N)
    TH_func = lambda x: toepltizMatVectProd(x,S_2N)
    Tmat_func = lambda X: np.array([toepltizMatVectProd(X[:,j],S_2N) for j in X.shape[1]]).T

    T_op = sparse.linalg.LinearOperator(shape = (N,N),matvec=T_func,
    rmatvec = TH_func,matmat = Tmat_func,dtype=np.float64)

    return T_op
