
import numpy as np
import ptemcee
import h5py
from . import gaps

# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft


class FullModel(object):

    def __init__(self, y, mask, p0_aux, signal_cls, psd_cls=None, imp_cls=None, outdir='./', rescale=None,
                 n_wind=500, n_wind_psd=500000, prefix='samples'):
        # Data vector in time domain
        self.y = y
        self.N = len(y)

        # Spectrum and data DFT
        self.S, self.y_fft = p0_aux

        # Signal model class
        self.signal_cls = signal_cls
        # noise model class
        self.psd_cls = psd_cls
        # Missing data imputation model class
        self.imp_cls = imp_cls
        # Output directory path
        self.outdir = outdir

        self.psd_file_name = prefix + '_psd.hdf5'

        # Signal parameter boundaries
        self.lo, self.hi = signal_cls.get_lohi()
        self.ndim = len(self.lo)

        # Windowing smoothing parameter (optimized for signal estimation)
        self.n_wind = n_wind
        self.w = gaps.gapgenerator.modified_hann(self.signal_cls.N, n_wind=self.n_wind)
        # Normalization constant for signal amplitude
        self.K1 = np.sum(self.w)

        # Windowing smoothing parameter (optimized for noise psd estimation)
        self.n_wind_psd = n_wind_psd

        if self.imp_cls is not None:
            # If the missing data imputation is activated, then the window for PSD does not have gaps
            self.w_psd = gapgenerator.modified_hann(self.signal_cls.N, n_wind=self.n_wind_psd)

        else:
            # If the missing data imputation is disabled, we have to take the gaps (if any) into account in the
            # windowing
            if any(mask == 0):
                nd, nf = gapgenerator.findEnds(mask)
                self.w_psd = gapgenerator.windowing(nd, nf, self.N, window='modified_hann', n_wind=self.n_wind_psd)
            else:
                self.w_psd = gapgenerator.modified_hann(self.signal_cls.N, n_wind=self.n_wind_psd)

        # Normalization constant for noise amplitude
        self.K1_psd = np.sum(self.w_psd)
        # Normalization constant for noise power spectrum
        self.K2_psd = np.sum(self.w_psd ** 2)
        # Windowed signal DFT optimized for PSD estimation
        self.y_fft_psd = fft(self.y * self.w_psd)

        # Rescaling of parameters
        if rescale is None:
            self.rescale = np.ones(self.ndim)
        else:
            self.rescale = rescale
            self.lo *= 1. / rescale
            self.hi *= 1. / rescale

        self.psd_samples = []
        self.psd_logpvals = []
        self.psd_save = 1


    def logl(self, params, S, y_fft):
        """
        log-likelihood taking into account parameter rescaling.
        """

        return self.signal_cls.log_likelihood(params * self.rescale, S, y_fft)

    def update_miss(self, pos0):
        """

        Update missing data imputation

        Parameters
        ----------
        pos : array_like
            vector of current parameter values

        Returns
        -------

        """

        # IMPUTATION STEP
        if self.imp_cls is not None:

            if self.psd_cls is None:
                # Draw the signal
                y_gw_fft = self.signal_cls.draw_frequency_signal(pos0 * self.rescale, self.S)
                # Inverse Fourier transform back in time domain
            y_gw = np.real(ifft(y_gw_fft))
            # Draw the missing data
            y_rec = self.imp_cls.draw_missing_data(self.y, y_gw, self.psd_cls)
            # Calculate the DFT of the reconstructed data
            self.y_fft = fft(y_rec * self.w) * self.signal_cls.N / self.K1
            # Update the data DFT for the psd estimation
            self.y_fft_psd = fft(y_rec * self.w_psd)

    def update_psd(self, pos0, npsd):
        """
        Update noise PSD parameters

        Parameters
        ----------
        pos0 : array_like
            vector of current parameter values
        npsd : integer
            number of draws during one MCMC psd update

        Returns
        -------

        """

        # PSD POSTERIOR STEP
        if self.psd_cls is not None:
            # Draw the signal (windowed DFT)
            y_gw_fft = self.signal_cls.draw_frequency_signal(pos0 * self.rescale, self.S, self.y_fft)
            # Calculation of the model residuals
            z_fft = self.y_fft_psd - self.K1_psd / self.N * y_gw_fft
            # Update periodogram
            self.psd_cls.set_periodogram(z_fft, K2=self.K2_psd)
            # Draw the PSD
            psd_samples, logpvalues = self.psd_cls.sample_psd(npsd)
            # Last PSD value
            self.psd_cls.logSc = psd_samples[npsd - 1, :]
            # Update PSD function and Fourier spectrum
            self.psd_cls.update_psd_func(self.psd_cls.logSc)
            # Update new value of the spectrum for the posterior step
            self.S = self.psd_cls.calculate(self.N)
            # Store psd parameter samples
            self.psd_samples = np.vstack((self.psd_samples, psd_samples))
            # Store corresponding log-posterior values
            self.psd_logpvals = np.concatenate((self.psd_logpvals, logpvalues))
            # fh5 = h5py.File(self.outdir + self.psd_file_name, 'a')
            # dset_samples = fh5["psd/samples"]
            # dset_samples.resize(dset_samples.shape[0] + npsd, axis=0)
            # dset_samples[-npsd:, :] = psd_samples
            # dset_logpvals = fh5["psd/logpvals"]
            # dset_logpvals.resize(dset_logpvals.shape[0] + npsd, axis=0)
            # dset_logpvals[-npsd:] = logpvalues
            # fh5.close()

    def update_aux(self, pos0, npsd):
        """

        Update all auxiliary parameters at once

        Parameters
        ----------
        pos0 : array_like
            vector of current parameter values
        npsd : integer
            number of draws during one MCMC psd update

        Returns
        -------

        """

        # Missing data imputation step
        self.update_miss(pos0)
        # PSD parameter posterior step
        self.update_psd(pos0, npsd)

    def initialize_aux(self, pos0, npsd):
        """
        Initialization of auxiliary parameters

        Parameters
        ----------
        pos0 : array_like
            first guess for signal parameters



        """

        # # Initialization of parameter values
        # pos = np.copy(p0)

        # Initialization of missing data
        self.update_miss(pos0)

        # Initialization of PSD parameters and missing data
        if self.psd_cls is not None:
            # Draw the signal (windowed DFT)
            y_gw_fft = self.signal_cls.draw_frequency_signal(pos0 * self.rescale, self.S, self.y_fft)
            # Calculation of the model residuals
            z_fft = self.y_fft_psd - self.K1_psd / self.N * y_gw_fft
            # Initialization of periodogram
            self.psd_cls.set_periodogram(z_fft, K2=self.K2_psd)
            # Initialization of the PSD
            self.psd_cls.estimate_from_I(self.psd_cls.I)
            # Update new value of the spectrum
            self.S = self.psd_cls.calculate(self.N)
            # Store first value of PSD
            logSc0 = np.zeros((1, self.psd_cls.logSc.shape[0]))
            logSc0[0, :] = self.psd_cls.logSc
            self.psd_samples = logSc0
            # Calculate first value of loglikelihood
            self.psd_logpvals = np.array([self.psd_cls.psd_posterior(logSc0)])
            # # Initialization of PSD parameter file
            # fh5 = h5py.File(self.outdir + self.psd_file_name, 'a')
            # # Clear all data sets
            # if "psd/samples" in fh5:
            #     del fh5["psd/samples"]
            # if "psd/logpvals" in fh5:
            #     del fh5["psd/logpvals"]
            # dset_samples = fh5.create_dataset("psd/samples", np.shape(self.psd_samples), maxshape=(None, None),
            #                                   dtype=np.float64, chunks=(npsd, self.psd_cls.logSc.shape[0]))
            # dset_logpvals = fh5.create_dataset("psd/logpvals", np.shape(self.psd_logpvals), maxshape=(None,),
            #                                    dtype=np.float64, chunks=(npsd,))
            # dset_samples[:] = self.psd_samples
            # dset_logpvals[:] = self.psd_logpvals
            # fh5.close()

    def reset_psd_samples(self):

        # Store first value of PSD
        logSc0 = np.zeros((1, self.psd_cls.logSc.shape[0]))
        logSc0[0, :] = self.psd_cls.logSc
        self.psd_samples = logSc0
        # Calculate first value of loglikelihood
        self.psd_logpvals = np.array([self.psd_cls.psd_posterior(logSc0)])

    def save_psd_samples(self):
        """

        Returns
        -------

        """

        # Initialization of PSD parameter file
        fh5 = h5py.File(self.outdir + self.psd_file_name, 'a')
        # # Clear all data sets
        # if "psd/samples" + str(self.psd_save) in fh5:
        #     del fh5["psd/samples" + str(self.psd_save)]
        # if "psd/logpvals" + str(self.psd_save) in fh5:
        #     del fh5["psd/logpvals" + str(self.psd_save)]
        dset_samples = fh5.create_dataset("psd/samples" + str(self.psd_save), np.shape(self.psd_samples))
        dset_logpvals = fh5.create_dataset("psd/logpvals" + str(self.psd_save), np.shape(self.psd_logpvals))
        dset_samples[:] = self.psd_samples
        dset_logpvals[:] = self.psd_logpvals
        fh5.close()
        self.psd_save += 1


class fullSampler(FullModel):

    def __init__(self, y, mask, p0_aux, signal_cls, psd_cls=None, imp_cls=None, outdir='./', rescale=None,
                 prefix='samples', order=None, nwalkers=12, ntemps=10, n_wind=500, n_wind_psd=500000):
        """
        
        Class to sample GW signal parameters, noise parameters and missing data
        
        Parameters
        ---------- 
        y : array_like
            data in the time domain, possiby masked by a binary mask (missing
            values are indicated by zeros)
        mask : array_like
            binary mask vector with entries 0 if data is missing and 1 if data
            is observed.
        p0_aux : list of array_like
            initial guess for the auxiliary data (S,y_fft). We assume that 
            y_fft is properly normalized so that it takes into account any 
            possible windowing or masking applied to the time series
        signal_cls : instance of GibbsModel
            the class providing the log likelihood function and auxiliary parameters
            update functions.
        psd_cls : instance of PSDSpline class
            class defining the noise model and methods necessary to update its parameters
        imp_cls : instance of the approximp class
            class defining the way to perform missing data imputation
        outdir : string
            where to store the psd results and other things
        rescale : array_like
            vector of same size of the GW signal parameter vector, which apply
            a rescaling of the parameters (can be usefull for numerical reasons,
            so that all parameters have roughly the same magnitude)
        nwalkers : scalar integer
            number of chains to use
        ntemps : scalar integer
            number of temperature in the parallel-tempering scheme
        n_wind : scalar integer
            smoothing parameter of the Tukey window applied to the signal to 
            prevent noise leakage for the GW parameter estimation
        n_wind_psd : scalar integer
            smoothing parameter of the Tukey window applied to the residuals to
            prevent leakage for noise PSD estimation
            


            
        Returns
        -------
        noise_params_new : array_like
            updated PSD parameters
            
            
            
            
        """

        FullModel.__init__(self, y, mask, p0_aux, signal_cls, psd_cls=psd_cls, imp_cls=imp_cls, outdir=outdir,
                           rescale=rescale, n_wind=n_wind, n_wind_psd=n_wind_psd, prefix=prefix)

        if order is None:
            self.logp = self.signal_cls.logp
            self.logpargs = [self.lo, self.hi]
        else:
            self.logp = self.signal_cls.logpo
            self.logpargs = [self.lo, self.hi, order[0], order[1]]

        # ---------------------------------------------------------------------
        # Instantiate the GW parameter sampler
        # ---------------------------------------------------------------------
        # Parallel-tempered MCMC with several chains and affine invariant
        # swaps
        self.sampler = ptemcee.Sampler(nwalkers, self.ndim, self.logl,
                                       self.logp, ntemps=ntemps,
                                       loglargs=p0_aux, logpargs=self.logpargs)
        # Define the sampling function
        self.sample = self.sampler.sample

    def set_loglargs(self, loglargs):
        """
        Update auxiliary parameters of the MCMC sampler
        
        """
        self.sampler._likeprior.loglargs = loglargs

    def get_loglargs(self):
        """
        Get auxiliary parameters of the MCMC sampler
        
        """

        return self.sampler._likeprior.loglargs

    def run(self, p0, nit=100000, nupdate=1000, npsd=10, thin=10):
        """Metropolis-Hastings within Gibbs sampler using `PTMCMCSampler`
    
        The parameters are bounded in the finite interval described by ``lo`` and
        ``hi`` (including ``-np.inf`` and ``np.inf`` for half-infinite or infinite
        domains).
    
        If run in an interactive terminal, live progress is shown including the
        current sample number, the total required number of samples, time elapsed
        and estimated time remaining, acceptance fraction, and autocorrelation
        length.
    
        Sampling terminates when all chains have accumulated the requested number
        of independent samples.
    
        Parameters
        ----------
        nit : int, optional
            Minimum number of independent samples.
        ntemps : int, optional
            Number of temperatures.
        nupdate : int, optional
            Cadence at which updating the auxiliary parameters
        npsd : int,optional
            number of Metropolis-Hastings steps to update the PSD

            
            
        Returns
        -------
        chain : `numpy.ndarray`
            The thinned and flattened posterior sample chain,
            with at least ``nindep`` * ``nwalkers`` rows
            and exactly ``ndim`` columns.
    
        Other parameters
        ----------------
        kwargs :
            Extra keyword arguments for `ptemcee.Sampler`.
            *Tip:* Consider setting the `pool` or `vectorized` keyword arguments in
            order to speed up likelihood evaluations.

        """

        # Initialization of auxiliary parameters
        self.initialize_aux(p0[0, 0, :], npsd)

        # Initialization of parameter values
        pos = np.copy(p0)

        # Initialization of iteration counter
        i = 0
                
        for pos, lnlike0, lnprob0 in self.sample(pos, nit, thin=thin, storechain=True):
            
            if i % nupdate == 0:
                
                print("Update of auxiliary parameters at iteration " + str(i))
                self.update_aux(pos[0, 0, :], npsd)
                self.set_loglargs((self.S, self.y_fft))

            i = i+1           

        fullchain = self.sampler.chain[:]

        return fullchain

