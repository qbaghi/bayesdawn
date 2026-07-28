# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:24:27 2019

@author: qbaghi

This module provide classes to perform missing data imputation steps based on
Gaussian conditional model
"""
import copy
import warnings
from numpy import ndarray
import numpy as np
from scipy import signal, linalg
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, ifft
from .algebra import matrixalgebra, fastoeplitz
from .gaps import gapgenerator, operators
from .noisegenerator import generate_noise_from_psd

# Enable the cache to save FFTW plan to perform faster fft for the subsequent calls of pyfftw
pyfftw.interfaces.cache.enable()


def toeplitz(r, inds):
    """
    Build a Toeplitz matrix from a vector r and a set of indices inds.
    
    Parameters
    ----------
    r : array_like
        vector of size n
    inds : array_like
        indices of the Toeplitz matrix

    Returns
    -------
    T : numpy array
        Toeplitz matrix of size len(inds) x len(inds)
    """
    ix, iy = np.meshgrid(inds, inds)

    indx = np.abs(ix - iy)

    return np.vstack([r[indx[i, :]] for i in range(indx.shape[0])])


class GaussianStationaryProcess(object):
    """

    Implement the (naive) nearest-neighboor method for missing data imputation.


    """

    def __init__(
        self,
        y_mean,
        mask,
        psd_cls,
        method="nearest",
        precond="taper",
        na=150,
        nb=150,
        p=60,
        tol=1e-6,
        n_it_max=1000,
        n_wood_max=5000,
    ):
        """

        Parameters
        ----------
        y_mean : array_like
            mean vector of the Gaussian process, size n
        mask : array_like
            binary mask
        psd_cls : psdmodel.PSD instance or callable
            power spectral density class. Should have a method called
            calculate() that takes a frequency vector as input.
            Alternatively, it can be a function that takes a frequency vector
            as input.
        method : str
            method to use to perform imputation.
            'nearest': nearest neighboors, approximate method.
            'PCG': preconjugate gradient, iterative exact method.
            'woodbury': low-rank formulation, non-iterative, exact method.
        precond : str
            Preconditionning methods among {'taper', 'circulant'}.
        na : scalar integer
            number of points to consider before each gap (for the conditional
            distribution of gap data)
        na : scalar integer
            number of points to consider after each gap
        p : int
            number of points to keep before truncation for the preconditionner
            (only if 'PCG' method is chosen)
        """

        # Masked data
        self.y_mean = copy.deepcopy(y_mean)
        # The binary mask
        self.mask = copy.deepcopy(mask)
        # The PSD
        self.psd_cls = copy.deepcopy(psd_cls)
        # Total length of the data
        self.n = len(mask)
        # Imputation method
        self.method = method
        # Preconditionning method
        self.precond = precond
        # Tappering number for sparse approximation of the covariance
        self.p = p
        # Error tolerance to reach to end PCG algorithm iterations
        self.tol = tol
        # Maximum number of iterations for the PCG algorithm
        self.n_it_max = n_it_max
        # Maximum missing data length accepted by Woodbury method
        self.n_wood_max = n_wood_max
        # Check whether there are gaps
        if np.any(self.mask == 0):
            # Starting and ending points of gaps
            self.n_starts, self.n_ends = gapgenerator.find_ends(mask)
            gap_lengths = self.n_ends - self.n_starts
            self.n_max = int(na + nb + np.max(gap_lengths))
            self.na = na
            self.nb = nb
            # Number of gaps
            self.n_gaps = len(self.n_starts)
            # Indices of missing data
            self.ind_mis = np.where(mask == 0)[0]
            # Indices of observed data
            self.ind_obs = np.where(mask == 1)[0]
        else:
            self.n_starts, self.n_ends = 0, self.n
            self.n_max = 0
            self.n_gaps = 0
            self.ind_mis = []
            self.ind_obs = np.arange(0, self.n).astype(int)
        # If the method is exact (not nearest neighboors) you need the full autocovariance
        if self.method != "nearest":
            self.n_max = len(mask)
        else:
            if self.n_max > 2000:
                warnings.warn(
                    "The maximum size of gap + conditional is high.", UserWarning
                )

        # Indices of embedding segments around each gap
        # The edges of each segment is set such that there are Na + Nb observed
        # data around, unless another gap is present.
        if self.n_gaps == 0:
            self.indices = None
            print("Time series does not contain gaps.")

        elif self.n_gaps == 1:
            # 2 segments
            self.indices = [
                np.arange(
                    int(np.max([self.n_starts[0] - na, 0])),
                    int(np.min([self.n_ends[0] + nb, self.n])),
                )
            ]

        elif self.n_gaps > 1:
            # first segment
            self.indices = [
                np.arange(
                    int(np.max([self.n_starts[0] - na, 0])),
                    int(np.min([self.n_ends[0] + nb, self.n_starts[1]])),
                )
            ]
            # most of the segments
            self.indices = self.indices + [
                np.arange(
                    int(np.max([self.n_starts[j] - na, self.n_ends[j - 1]])),
                    int(np.min([self.n_ends[j] + nb, self.n_starts[j + 1]])),
                )
                for j in range(1, self.n_gaps - 1)
            ]
            # last segment
            self.indices = self.indices + [
                np.arange(
                    int(
                        np.max(
                            [
                                self.n_starts[self.n_gaps - 1] - na,
                                self.n_ends[self.n_gaps - 2],
                            ]
                        )
                    ),
                    int(np.min([self.n_ends[self.n_gaps - 1] + nb, self.n])),
                )
            ]

        # Cache local gap-segment metadata to avoid repeated index extraction.
        if self.indices is None:
            self.segment_meta = None
        else:
            self.segment_meta = []
            for indj in self.indices:
                maskj = self.mask[indj]
                ind_obsj = np.where(maskj == 1)[0]
                ind_misj = np.where(maskj == 0)[0]
                self.segment_meta.append(
                    {
                        "indices": indj,
                        "mask": maskj,
                        "ind_obs": ind_obsj,
                        "ind_mis": ind_misj,
                        "segment_size": int(self.na + self.nb + len(ind_misj)),
                        "n_mis": len(ind_misj),
                    }
                )

        # ==
        # Store quantities that can be computed offline
        # ==

        # Autocovariance
        self.autocorr = None
        # Power spectral density computed on a frequency grid of size 2n
        self.s2 = None
        # Preconditionner for PCG or tapered methods
        self.solve = None
        # Inverted matrix for woodbury method
        self.sig_inv_mm_inv = None
        self.w_m_cls = None
        self.a = None
        self.lambda_n = None

    def update_psd(self, psd_cls):
        """
        Update the PSD class of the Gaussian stationary process

        Parameters
        ----------
        psd_cls : psdmodel.PSD instance
            New PSD class
        """

        self.psd_cls = copy.deepcopy(psd_cls)

    def update_mean(self, y_mean):
        """
        Update the mean vector of the Gaussian stationary process

        Parameters
        ----------
        y_mean : ndarray or list
            Mean vector (deterministic part).
        """

        self.y_mean = y_mean[:]

    def compute_offline(self):
        """
        Performs all necessary offline computations that depend on PSD and
        mean vector.
        """

        # Compute the autocovariance from the full PSD and restrict it to N_max
        # points
        if not isinstance(self.psd_cls, list):
            self.autocorr = self.psd_cls.calculate_autocorr(self.n)[0 : self.n_max]
            # Compute the spectrum on 2*N_max points
            self.s2 = self.psd_cls.calculate(2 * self.n_max)
        else:
            self.autocorr = [
                psd.calculate_autocorr(self.n)[0 : self.n_max] for psd in self.psd_cls
            ]
            self.s2 = [psd.calculate(2 * self.n_max) for psd in self.psd_cls]

        if self.method == "woodbury":
            if len(self.ind_mis) <= self.n_wood_max:
                print("Start Toeplitz system precomputations...")
                self.w_m_cls = operators.MappingOperator(self.ind_mis, self.n)
                w_m = self.w_m_cls.build_matrix(sp=False)
                # s_n = self.psd_cls.calculate(self.n_max)
                # sigma_inv_wmt = ifft(fft(w_m.T, axis=0) / np.array([s_n]).T, axis=0)
                if not isinstance(self.psd_cls, list):
                    autocorr = self.autocorr[:]
                else:
                    # Assume same autocovariance for every channel
                    autocorr = self.autocorr[0]
                # Precompute quantities for calculating the inverse of Sigma
                self.lambda_n, self.a = fastoeplitz.teopltiz_precompute(
                    autocorr,
                    p=self.p,
                    nit=self.n_it_max,
                    tol=self.tol,
                    method="levinson",
                    precond=self.precond,
                )
                sigma_inv_wmt = fastoeplitz.multiple_toepltiz_inverse(
                    w_m.T, self.lambda_n, self.a
                )
                self.sig_inv_mm_inv = linalg.pinv(w_m.dot(sigma_inv_wmt))

            else:
                msg = "Number of missing data is too large for woodbury method."
                raise ValueError(msg)

    def compute_preconditioner(self):
        """
        Precompute the pre-conditioner operator that looks like Coo

        """

        # Precompute solver if necessary
        if (self.method == "PCG") | (self.method == "tapered"):
            print("Build preconditionner...")
            if not isinstance(self.autocorr, list):
                self.solve = matrixalgebra.compute_precond(
                    self.autocorr, self.mask, p=self.p, taper="Wendland2"
                )
            else:
                self.solve = [
                    matrixalgebra.compute_precond(
                        autocorr, self.mask, p=self.p, taper="Wendland2"
                    )
                    for autocorr in self.autocorr
                ]
            # # For now, use the same preconditionner for all channels
            # self.solve = matrixalgebra.compute_precond(self.autocorr,
            #                                             self.mask,
            #                                             p=self.p,
            #                                             taper='Wendland2')
            print("Preconditionner built.")

    def impute(self, y, draw=True):
        """

        Draw the missing data from their conditional distributions on the
        observed data. The difference with the draw_missing_data method is that
        it checks whether there are gaps or not. If not, this function is
        identity.

        Parameters
        ----------
        y : ndarray or list
            masked data vector, size n
        draw : bool
            if True (default), the data vector is drawn from the conditional
            distribution given the observed data. If False, the expectation of
            the conditional distribution is returned (in that case the output
            is deterministic, as it does not involved any random number
            generation.)

        Returns
        -------
        y_rec : array_like
            realization of the full data vector conditionnally to the observed
            data, or its mean.

        """

        # If there is only one single channel
        if self.n_gaps > 0:
            return self.draw_missing_data(y, draw=draw)
        else:
            return y

    def draw_missing_data(self, y, draw=True):
        """

        Draw the missing data from their conditional distributions on the
        observed data

        Parameters
        ----------
        y : ndarray or list of ndarrays
            masked data y = mask * x. If a list is given, draw as many
            vectors as there are arrays in the list.
        draw : bool
            if True (default), the data vector is drawn from the conditional
            distribution given the observed data. If False, the expectation of
            the conditional distribution is returned (in that case the output
            is deterministic, as it does not involved any random number
            generation.)

        Returns
        -------
        y_rec : array_like
            realization of the full data vector conditionnally to the observed
            data

        """

        if self.autocorr is None:
            # print('recomputing offline elements')
            self.compute_offline()
        if ((self.method == "PCG") | (self.method == "tapered")) & (self.solve is None):
            self.compute_preconditioner()
            # print('recomputing preconditioner')
        # If there is only one array
        if isinstance(y, np.ndarray):
            # Impute the missing data: estimation of missing residuals
            y_mis_res = self.imputation(
                y - self.y_mean, self.autocorr, self.s2, solve=self.solve, draw=draw
            )
            # Construct the full imputed data vector
            # at observed value this is the same
            y_rec = copy.deepcopy(y)
            y_rec[self.ind_mis] = y_mis_res + self.y_mean[self.ind_mis]

        elif isinstance(y, list):
            y_mis_res = [
                self.imputation(
                    y[i] - self.y_mean[i],
                    self.autocorr[i],
                    self.s2[i],
                    solve=self.solve[i],
                    draw=draw,
                )
                for i in range(len(y))
            ]
            y_rec = copy.deepcopy(y)

            for i in range(len(y)):
                y_rec[i][self.ind_mis] = y_mis_res[i] + self.y_mean[i][self.ind_mis]

        else:
            raise ValueError("Unknown input type for y")

        return y_rec

    def apply_coo_inv(self, z_o, s2, solve=None):
        """

        Operator performing the product Coo^{-1} z on any vector z

        Parameters
        ----------
        z_o : array_like
            vector of size n_obs
        s2 : array_like
            One-sided PSD values calculated on a Fourier grid of size 2 N_max
            WARNING: used to be S(f) * fs / 2. Now the normalization is done
            inside the function.
        solve : linear operator
            preconditionner

        Returns
        -------
        x : 1d numpy array
            vector of size n_obs, such that x = Coo^{-1} z

        """

        # Compute the DFT covariances from the one-sided PSD
        # The actual covariance is npoints x S(f) * fs / 2 but the factor
        # of npoints is already accounted for in the IFFT normalization
        cov_2n = s2 * self.psd_cls.fs / 2.0

        if self.method == "tapered":
            # Approximately solve the linear system C_oo x = eps
            x = solve(z_o)
        elif self.method == "PCG":
            # Precompute solver if necessary
            if solve is None:
                # self.compute_preconditioner(r)
                raise ValueError("Please provide preconditionning operator")
            # First guess
            x0 = np.zeros(len(self.ind_obs))
            # Solve the linear system C_oo x = eps
            x, _ = matrixalgebra.pcg_solve(
                self.ind_obs,
                self.mask,
                cov_2n,
                z_o,
                x0,
                self.tol,
                self.n_it_max,
                solve,
                "scipy",
            )
        elif self.method == "woodbury":
            epsilon_masked = np.zeros(self.n)
            epsilon_masked[self.ind_obs] = z_o
            # Apply inverse sigma
            v_ = fastoeplitz.toepltiz_inverse_jain(
                epsilon_masked, self.lambda_n, self.a
            )
            y_ = np.zeros(self.n)
            y_[self.ind_mis] = self.sig_inv_mm_inv.dot(v_[self.ind_mis])
            e_ = v_ - fastoeplitz.toepltiz_inverse_jain(y_, self.lambda_n, self.a)
            x = e_[self.ind_obs]

        else:
            raise ValueError("Unknown imputation method.")

        return x

    def imputation(self, y, r, s2, solve=None, draw=True):
        """

        Impute the missing data using a conditional draw.

        Parameters
        ----------
        y : array_like
            masked residuals (size n_data)
        r : array_like
            autocovariance function until lag N_max
        s2 : array_like
            values of the noise one-sided PSD calculated on a Fourier grid of size
            2 N_max. WARNING: it used to be the noise spectrum S fs / 2
        solve : linear operator
            preconditionner
        draw : bool, optional
            if True (default), the missing data are drawn from their
            conditional distribution. If False, their conditional expectation
            is returned.


        Returns
        -------
        y_mis : 1d numpy array
            imputed missing value

        """

        if self.method == "nearest":
            # =================================================================
            # Gap per gap imputation
            # =================================================================
            if self.n_max <= 2000:
                c = linalg.toeplitz(r)
            else:
                c = None

            y_mis = np.empty(len(self.ind_mis), dtype=y.dtype)
            offset = 0
            for seg in self.segment_meta:
                yj = y[seg["indices"]]
                if draw:
                    out = self.single_imputation(
                        yj,
                        seg["mask"],
                        c,
                        r,
                        s2,
                        ind_obsj=seg["ind_obs"],
                        ind_misj=seg["ind_mis"],
                        segment_size=seg["segment_size"],
                    )
                else:
                    out = self.single_conditional_mean(
                        yj,
                        seg["mask"],
                        c,
                        r,
                        s2,
                        ind_obsj=seg["ind_obs"],
                        ind_misj=seg["ind_mis"],
                        segment_size=seg["segment_size"],
                    )

                n_mis = seg["n_mis"]
                y_mis[offset : offset + n_mis] = out
                offset += n_mis

        else:
            if draw:
                # For missing data draw:
                e = generate_noise_from_psd(s2, self.psd_cls.fs)[0 : self.n]
                u = self.apply_coo_inv(
                    y[self.ind_obs] - e[self.ind_obs], s2, solve=solve
                )
                # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
                y_mis = e[self.ind_mis] + matrixalgebra.mat_vect_prod(
                    u, self.ind_obs, self.ind_mis, self.mask, s2
                )
            else:
                # For conditional mean computation:
                # Compute u = C_oo^{-1} z_o
                u = self.apply_coo_inv(y[self.ind_obs], s2, solve=solve)
                # Compute the missing data conditional mean via z|o = Cmo u
                y_mis = matrixalgebra.mat_vect_prod(
                    u, self.ind_obs, self.ind_mis, self.mask, s2
                )

        return y_mis

    def single_imputation(
        self,
        yj,
        maskj,
        c,
        r,
        psd_2n,
        threshold=2000,
        ind_obsj=None,
        ind_misj=None,
        segment_size=None,
    ):
        """
        Sample the missing data distribution conditionally on the observed
        data, using direct brute-force computation.

        Parameters
        ----------
        yj : ndarray
            segment of masked data residuals
        maskj : ndarray
            local mask
        c : ndarray
            covariance matrix of sized nj x nj
        r : ndarray
            autocovariance computed until lag n_max
        psd_2n : ndarray
            One-sided PSD computed on a Fourier grid of size 2nj
        threshold : int, optional
            Threshold for the size of the neighbooring segments, above which
            the methods switches from matrix-based to FFT-based.

        Returns
        -------
        eps : ndarray
            imputed missing data, of size len(np.where(maskj == 0)[0])

        """
        # Compute the DFT covariances from the one-sided PSD
        # The actual covariance is npoints x S(f) * fs / 2 but the factor
        # of npoints is already accounted for in the IFFT normalization
        cov_2n = psd_2n * self.psd_cls.fs / 2.0

        # Local indices of missing and observed data
        if ind_obsj is None:
            ind_obsj = np.where(maskj == 1)[0]
        if ind_misj is None:
            ind_misj = np.where(maskj == 0)[0]

        # Compute the size of the neighbooring observed points + gap size
        if segment_size is None:
            segment_size = int(self.na + self.nb + len(ind_misj))

        # If the size is below some threshold, apply full-matrix method:
        if segment_size <= threshold:
            c_mo = c[np.ix_(ind_misj, ind_obsj)]
            c_oo = c[np.ix_(ind_obsj, ind_obsj)]
            # out = self.conditional_draw(yj[ind_obsj], psd_2n, c_oo_inv, c_mo,
            #                             ind_obsj, ind_misj, maskj, c)
            e = np.random.multivariate_normal(
                np.zeros(maskj.shape[0]), c[0 : maskj.shape[0], 0 : maskj.shape[0]]
            )

            # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
            rhs = yj[ind_obsj] - e[ind_obsj]
            eps = e[ind_misj] + c_mo.dot(linalg.solve(c_oo, rhs, assume_a="pos"))

        # Otherwise, use FFT-based method:
        else:
            # Covariance of observed data and its inverse
            c_oo = toeplitz(r, ind_obsj)

            # Covariance missing / observed data : matrix operator
            def c_mo(v):
                return matrixalgebra.mat_vect_prod(v, ind_obsj, ind_misj, maskj, cov_2n)

            e = generate_noise_from_psd(psd_2n, self.psd_cls.fs)[0 : maskj.shape[0]]

            # Z u | o = Z_tilde_u + Cmo Coo^-1 ( Z_o - Z_tilde_o )
            rhs = yj[ind_obsj] - e[ind_obsj]
            eps = e[ind_misj] + c_mo(linalg.solve(c_oo, rhs, assume_a="pos"))

        return eps

    def single_conditional_mean(
        self,
        yj,
        maskj,
        c,
        r,
        psd_2n,
        threshold=2000,
        ind_obsj=None,
        ind_misj=None,
        segment_size=None,
    ):
        """
        Compute the conditional expectation of missing data given the observed
        data, using direct brute-force computation
        (to be used on short segments with the nearest-neighboor method.)

        Parameters
        ----------
        yj : ndarray
            segment of masked data
        maskj : ndarray
            local mask
        c : ndarray
            covariance matrix of sized nj x nj
        r : ndarray
            autocovariance computed until lag n_max
        psd_2n : ndarray
            One-sided PSD computed on a Fourier grid of size 2nj
        threshold : int, optional
            Threshold for the size of the neighbooring segments, above which
            the methods switches from matrix-based to FFT-based.

        Returns
        -------
        mu_mis_j : ndarray
            conditional expectation of missing data,
            of size len(np.where(maskj == 0)[0])

        """

        # Compute the DFT covariances from the one-sided PSD
        # The actual covariance is npoints x S(f) * fs / 2 but the factor
        # of npoints is already accounted for in the IFFT normalization
        cov_2n = psd_2n * self.psd_cls.fs / 2.0

        # Local indices of missing and observed data
        if ind_obsj is None:
            ind_obsj = np.where(maskj == 1)[0]
        if ind_misj is None:
            ind_misj = np.where(maskj == 0)[0]

        # Compute the size of the neighbooring observed points + gap size
        if segment_size is None:
            segment_size = int(self.na + self.nb + len(ind_misj))

        # If the size is below some threshold, apply full-matrix method:
        if segment_size <= threshold:
            c_mo = c[np.ix_(ind_misj, ind_obsj)]
            c_oo = c[np.ix_(ind_obsj, ind_obsj)]
            mu_mis_j = c_mo.dot(linalg.solve(c_oo, yj[ind_obsj], assume_a="pos"))

        # Otherwise, use FFT-based method:
        else:
            # Covariance of observed data and its inverse
            c_oo = toeplitz(r, ind_obsj)

            # Covariance missing / observed data : matrix operator
            def c_mo(v):
                return matrixalgebra.mat_vect_prod(v, ind_obsj, ind_misj, maskj, cov_2n)

            mu_mis_j = c_mo(linalg.solve(c_oo, yj[ind_obsj], assume_a="pos"))

        return mu_mis_j


class MultivariateGaussianStationaryProcess(object):
    """
    Skeleton class for multivariate, correlated Gaussian time-series
    imputation from a PSD-CSD spectral matrix.

    Notes
    -----
    This class mirrors the public API of GaussianStationaryProcess while
    operating on arrays with shape (n_data, n_channels).
    The numerical kernels for block-covariance application and conditional
    draws are intentionally left as TODOs.
    """

    def __init__(
        self,
        y_mean,
        mask,
        psd_cls,
        method="nearest",
        precond="taper",
        na=150,
        nb=150,
        p=60,
        tol=1e-6,
        n_it_max=1000,
        n_wood_max=5000,
        shared_mask=True,
    ):
        """
        Parameters
        ----------
        y_mean : array_like
            Mean time series with shape (n_data, n_channels).
        mask : array_like
            If shared_mask is True, expected shape is (n_data,).
            Otherwise expected shape is (n_data, n_channels).
        psd_cls : object or callable
            PSD/CSD provider.
        method, precond, na, nb, p, tol, n_it_max, n_wood_max :
            Same meaning as in GaussianStationaryProcess.
        shared_mask : bool
            If True, one time mask is shared by all channels.
        """

        self.y_mean = np.asarray(copy.deepcopy(y_mean))
        if self.y_mean.ndim != 2:
            raise ValueError("y_mean must be a 2D array of shape (n_data, n_channels)")

        self.n, self.n_chan = self.y_mean.shape
        self.shared_mask = shared_mask

        raw_mask = np.asarray(copy.deepcopy(mask))
        if self.shared_mask:
            if raw_mask.ndim != 1 or raw_mask.shape[0] != self.n:
                raise ValueError(
                    "With shared_mask=True, mask must be 1D with length n_data"
                )
            self.mask = raw_mask
            self.mask_time = raw_mask
        else:
            if raw_mask.ndim != 2 or raw_mask.shape != self.y_mean.shape:
                raise ValueError(
                    "With shared_mask=False, mask must have shape (n_data, n_channels)"
                )
            self.mask = raw_mask
            # Conservative definition: a time index is observed only if all
            # channels are observed.
            self.mask_time = np.prod(raw_mask, axis=1)

        self.psd_cls = copy.deepcopy(psd_cls)
        self.method = method
        self.precond = precond
        self.p = p
        self.tol = tol
        self.n_it_max = n_it_max
        self.n_wood_max = n_wood_max
        self.na = na
        self.nb = nb

        # Gap bookkeeping on the time axis
        if np.any(self.mask_time == 0):
            self.n_starts, self.n_ends = gapgenerator.find_ends(self.mask_time)
            gap_lengths = self.n_ends - self.n_starts
            self.n_max = int(na + nb + np.max(gap_lengths))
            self.n_gaps = len(self.n_starts)
            self.ind_mis_t = np.where(self.mask_time == 0)[0]
            self.ind_obs_t = np.where(self.mask_time == 1)[0]
        else:
            self.n_starts, self.n_ends = 0, self.n
            self.n_max = 0
            self.n_gaps = 0
            self.ind_mis_t = np.array([], dtype=int)
            self.ind_obs_t = np.arange(0, self.n).astype(int)

        if self.method != "nearest":
            self.n_max = self.n
        elif self.n_max > 2000:
            warnings.warn("The maximum size of gap + conditional is high.", UserWarning)

        self.indices = self._build_gap_segments(na, nb)

        # Segment-level cache (time-domain view)
        if self.indices is None:
            self.segment_meta = None
        else:
            self.segment_meta = []
            for indj in self.indices:
                maskj = self.mask_time[indj]
                ind_obsj = np.where(maskj == 1)[0]
                ind_misj = np.where(maskj == 0)[0]
                self.segment_meta.append(
                    {
                        "indices": indj,
                        "mask": maskj,
                        "ind_obs": ind_obsj,
                        "ind_mis": ind_misj,
                        "segment_size": int(self.na + self.nb + len(ind_misj)),
                        "n_mis": len(ind_misj),
                    }
                )

        # Offline caches for multivariate covariance model
        self.crosscorr = None
        self.s2_matrix = None
        self.solve = None
        self._cov_block_cache = {}

    def _build_gap_segments(self, na, nb):
        """Build time segments around each gap (same logic as univariate)."""

        if self.n_gaps == 0:
            return None

        if self.n_gaps == 1:
            return [
                np.arange(
                    int(np.max([self.n_starts[0] - na, 0])),
                    int(np.min([self.n_ends[0] + nb, self.n])),
                )
            ]

        indices = [
            np.arange(
                int(np.max([self.n_starts[0] - na, 0])),
                int(np.min([self.n_ends[0] + nb, self.n_starts[1]])),
            )
        ]
        indices = indices + [
            np.arange(
                int(np.max([self.n_starts[j] - na, self.n_ends[j - 1]])),
                int(np.min([self.n_ends[j] + nb, self.n_starts[j + 1]])),
            )
            for j in range(1, self.n_gaps - 1)
        ]
        indices = indices + [
            np.arange(
                int(
                    np.max(
                        [
                            self.n_starts[self.n_gaps - 1] - na,
                            self.n_ends[self.n_gaps - 2],
                        ]
                    )
                ),
                int(np.min([self.n_ends[self.n_gaps - 1] + nb, self.n])),
            )
        ]
        return indices

    def update_psd(self, psd_cls):
        """Update PSD/CSD provider."""

        self.psd_cls = copy.deepcopy(psd_cls)
        self.crosscorr = None
        self.s2_matrix = None
        self.solve = None
        self._cov_block_cache = {}

    def update_mean(self, y_mean):
        """Update multichannel mean model."""

        y_mean = np.asarray(y_mean)
        if y_mean.shape != self.y_mean.shape:
            raise ValueError("y_mean must keep shape (n_data, n_channels)")
        self.y_mean = copy.deepcopy(y_mean)

    def compute_offline(self):
        """
        Prepare multivariate covariance quantities from PSD/CSD model.
        """

        if self.method != "nearest":
            raise NotImplementedError(
                "Only method='nearest' is currently implemented for "
                "MultivariateGaussianStationaryProcess."
            )

        if not self.shared_mask:
            raise NotImplementedError(
                "Only shared_mask=True is currently implemented for "
                "MultivariateGaussianStationaryProcess."
            )

        if hasattr(self.psd_cls, "calculate"):
            self.s2_matrix = np.asarray(self.psd_cls.calculate(2 * self.n_max))
        elif callable(self.psd_cls):
            f = np.fft.fftfreq(2 * self.n_max) * self.psd_cls.fs
            self.s2_matrix = np.asarray(self.psd_cls(f))
        else:
            raise ValueError(
                "psd_cls must provide a calculate method or be a callable "
                "returning a PSD-CSD matrix."
            )

        if self.s2_matrix.ndim != 3:
            raise ValueError(
                "Multivariate PSD must have shape (n_freq, n_chan, n_chan)"
            )
        if (
            self.s2_matrix.shape[1] != self.n_chan
            or self.s2_matrix.shape[2] != self.n_chan
        ):
            raise ValueError(
                "PSD-CSD matrix channel dimensions must match y_mean second dimension"
            )

        if hasattr(self.psd_cls, "calculate_autocorr"):
            crosscorr = np.asarray(self.psd_cls.calculate_autocorr(self.n))
            if crosscorr.ndim != 3:
                raise ValueError(
                    "calculate_autocorr must return a 3D array for multivariate mode"
                )
            if crosscorr.shape[0] == self.n_chan and crosscorr.shape[1] == self.n_chan:
                self.crosscorr = crosscorr[:, :, 0 : self.n_max]
            elif (
                crosscorr.shape[1] == self.n_chan and crosscorr.shape[2] == self.n_chan
            ):
                self.crosscorr = np.transpose(crosscorr[0 : self.n_max], (1, 2, 0))
            else:
                raise ValueError(
                    "Unexpected autocorrelation shape for multivariate mode"
                )
        else:
            # Fallback: infer lag-domain cross-correlations from spectral matrices.
            corr = ifft(self.s2_matrix, axis=0)
            self.crosscorr = np.real(np.transpose(corr[0 : self.n_max], (1, 2, 0)))

        # Invalidate cached covariance blocks whenever PSD-derived quantities change.
        self._cov_block_cache = {}

    def compute_preconditioner(self):
        """
        Build block preconditioner for multivariate C_oo systems.
        """

        raise NotImplementedError(
            "Multivariate preconditioner construction is not implemented yet."
        )

    def impute(self, y, draw=True):
        """Impute missing entries in a multichannel time series."""

        y = np.asarray(y)
        if y.shape != self.y_mean.shape:
            raise ValueError("y must have shape (n_data, n_channels)")
        if self.n_gaps > 0:
            return self.draw_missing_data(y, draw=draw)
        return y

    def draw_missing_data(self, y, draw=True):
        """Draw missing multichannel data from conditional distribution."""

        if self.crosscorr is None or self.s2_matrix is None:
            self.compute_offline()

        if ((self.method == "PCG") | (self.method == "tapered")) and (
            self.solve is None
        ):
            self.compute_preconditioner()

        y_res = y - self.y_mean
        y_mis_res = self.imputation(
            y_res,
            self.crosscorr,
            self.s2_matrix,
            solve=self.solve,
            draw=draw,
        )

        y_rec = copy.deepcopy(y)
        if self.shared_mask:
            y_rec[self.ind_mis_t, :] = y_mis_res + self.y_mean[self.ind_mis_t, :]
        else:
            raise NotImplementedError(
                "Channel-specific mask scattering is not implemented yet."
            )

        return y_rec

    def apply_coo_inv(self, z_o, s2, solve=None):
        """
        Apply inverse observed-observed block covariance to a vector/matrix.
        """

        raise NotImplementedError(
            "Multivariate C_oo^{-1} application is not implemented yet."
        )

    def imputation(self, y, r, s2, solve=None, draw=True):
        """
        Multivariate conditional imputation core.
        """

        if self.method != "nearest":
            raise NotImplementedError(
                "Only method='nearest' is currently implemented for multivariate mode."
            )

        y_mis = np.empty((len(self.ind_mis_t), self.n_chan), dtype=y.dtype)
        offset = 0
        for seg in self.segment_meta:
            yj = y[seg["indices"], :]
            c_seg = self._build_segment_block_toeplitz(yj.shape[0])
            if draw:
                out = self.single_imputation(
                    yj,
                    seg["mask"],
                    c=c_seg,
                    r=r,
                    psd_2n=s2,
                    ind_obsj=seg["ind_obs"],
                    ind_misj=seg["ind_mis"],
                    segment_size=seg["segment_size"],
                )
            else:
                out = self.single_conditional_mean(
                    yj,
                    seg["mask"],
                    c=c_seg,
                    r=r,
                    psd_2n=s2,
                    ind_obsj=seg["ind_obs"],
                    ind_misj=seg["ind_mis"],
                    segment_size=seg["segment_size"],
                )
            n_mis = seg["n_mis"]
            y_mis[offset : offset + n_mis, :] = out
            offset += n_mis

        return y_mis

    def _flatten_channel_major(self, yj_sub):
        """Flatten a (n_time, n_chan) array as channel-wise concatenated series."""

        return np.concatenate([yj_sub[:, k] for k in range(self.n_chan)])

    def _reshape_channel_major_missing(self, vec, n_mis):
        """Map channel-major missing vector back to (n_mis, n_chan)."""

        return vec.reshape(self.n_chan, n_mis).T

    def _channel_major_time_indices(self, ind_time, segment_length):
        """Map local time indices to channel-major flattened vector indices."""

        return np.concatenate(
            [ind_time + k * segment_length for k in range(self.n_chan)]
        )

    def _build_segment_block_toeplitz(self, segment_length):
        """
        Build channel-major covariance block Toeplitz for one segment length.
        """

        if segment_length in self._cov_block_cache:
            return self._cov_block_cache[segment_length]

        if self.crosscorr is None:
            raise ValueError("crosscorr is not available, call compute_offline first")

        if segment_length > self.crosscorr.shape[2]:
            raise ValueError(
                "segment length exceeds available autocorrelation lag range"
            )

        blocks = [
            [
                linalg.toeplitz(self.crosscorr[i, j, 0:segment_length])
                for j in range(self.n_chan)
            ]
            for i in range(self.n_chan)
        ]
        cov = np.block(blocks)
        self._cov_block_cache[segment_length] = cov
        return cov

    def _build_local_covariance_blocks(self, ind_misj, ind_obsj, segment_length=None):
        """
        Build C_mo and C_oo for one segment using index selection from the
        channel-major block Toeplitz covariance.
        """

        if segment_length is None:
            segment_length = int(
                np.max(np.concatenate((ind_misj, ind_obsj))).astype(int) + 1
            )

        c = self._build_segment_block_toeplitz(segment_length)
        ind_mis_flat = self._channel_major_time_indices(ind_misj, segment_length)
        ind_obs_flat = self._channel_major_time_indices(ind_obsj, segment_length)

        c_mo = c[np.ix_(ind_mis_flat, ind_obs_flat)]
        c_oo = c[np.ix_(ind_obs_flat, ind_obs_flat)]

        return c_mo, c_oo

    def _build_local_covariance(self, ind_row, ind_col, segment_length=None):
        """Select block covariance between two local time index sets."""

        if segment_length is None:
            segment_length = int(
                np.max(np.concatenate((ind_row, ind_col))).astype(int) + 1
            )

        c = self._build_segment_block_toeplitz(segment_length)
        ind_row_flat = self._channel_major_time_indices(ind_row, segment_length)
        ind_col_flat = self._channel_major_time_indices(ind_col, segment_length)
        return c[np.ix_(ind_row_flat, ind_col_flat)]

    def single_imputation(
        self,
        yj,
        maskj,
        c,
        r,
        psd_2n,
        threshold=2000,
        ind_obsj=None,
        ind_misj=None,
        segment_size=None,
    ):
        """
        Segment-level multivariate conditional draw.
        """

        if ind_obsj is None:
            ind_obsj = np.where(maskj == 1)[0]
        if ind_misj is None:
            ind_misj = np.where(maskj == 0)[0]

        n_mis = len(ind_misj)
        if n_mis == 0:
            return np.empty((0, self.n_chan), dtype=yj.dtype)

        seg_len = yj.shape[0]
        if c is None:
            c = self._build_segment_block_toeplitz(seg_len)
        ind_obs_flat = self._channel_major_time_indices(ind_obsj, seg_len)
        ind_mis_flat = self._channel_major_time_indices(ind_misj, seg_len)

        c_mo = c[np.ix_(ind_mis_flat, ind_obs_flat)]
        c_oo = c[np.ix_(ind_obs_flat, ind_obs_flat)]
        c_mm = c[np.ix_(ind_mis_flat, ind_mis_flat)]

        y_obs = self._flatten_channel_major(yj[ind_obsj, :])
        u = linalg.solve(c_oo, y_obs, assume_a="pos")
        mu_mis = c_mo.dot(u)

        c_om = c_mo.T
        cond_cov = c_mm - c_mo.dot(linalg.solve(c_oo, c_om, assume_a="pos"))
        # Symmetrize and lightly regularize to mitigate round-off issues.
        cond_cov = 0.5 * (cond_cov + cond_cov.T)
        cond_cov = cond_cov + 1e-12 * np.eye(cond_cov.shape[0])

        draw_mis = np.random.multivariate_normal(mu_mis, cond_cov)

        return self._reshape_channel_major_missing(draw_mis, n_mis)

    def single_conditional_mean(
        self,
        yj,
        maskj,
        c,
        r,
        psd_2n,
        threshold=2000,
        ind_obsj=None,
        ind_misj=None,
        segment_size=None,
    ):
        """
        Segment-level multivariate conditional mean.
        """

        if ind_obsj is None:
            ind_obsj = np.where(maskj == 1)[0]
        if ind_misj is None:
            ind_misj = np.where(maskj == 0)[0]

        if len(ind_misj) == 0:
            return np.empty((0, self.n_chan), dtype=yj.dtype)

        seg_len = yj.shape[0]
        if c is None:
            c = self._build_segment_block_toeplitz(seg_len)
        ind_obs_flat = self._channel_major_time_indices(ind_obsj, seg_len)
        ind_mis_flat = self._channel_major_time_indices(ind_misj, seg_len)

        c_mo = c[np.ix_(ind_mis_flat, ind_obs_flat)]
        c_oo = c[np.ix_(ind_obs_flat, ind_obs_flat)]
        y_obs = self._flatten_channel_major(yj[ind_obsj, :])

        rhs = linalg.solve(c_oo, y_obs, assume_a="pos")
        mu_mis_vec = c_mo.dot(rhs)

        return self._reshape_channel_major_missing(mu_mis_vec, len(ind_misj))
