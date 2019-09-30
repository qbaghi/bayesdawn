import numpy as np
import h5py
from . import gaps
import ptemcee

# FTT modules
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import fft, ifft


class FullModel(object):

    def __init__(self, signal_cls, psd_cls, dat_cls, sampler_cls, outdir='./', n_wind=500, n_wind_psd=500000,
                 prefix='samples', n_psd=10, imputation=False, psd_estimation=False, normalized=False, rescaled=True):
        """

        Parameters
        ----------
        signal_cls : instance of GWModel
            the class providing the log likelihood function and auxiliary parameters
            update functions.
        psd_cls : instance of PSDSpline class
            class defining the noise model and methods necessary to update its parameters
        dat_cls : instance of the DataModel class
            class defining the way to perform missing data imputation
        sampler_cls : instance of sampler PTEMCEE, NestedSampler, etc.
            class charaterizing the monte-carlo sampler to sample the posterior distribution
        outdir : string
            where to store the psd results and other things
        n_wind : scalar integer
            smoothing parameter of the Tukey window applied to the signal to
            prevent noise leakage for the GW parameter estimation
        n_wind_psd : scalar integer
            smoothing parameter of the Tukey window applied to the residuals to
            prevent leakage for noise PSD estimation
        prefix : string
            prefix in name of files to be saved
        n_psd : int,optional
            number of Metropolis-Hastings steps to update the PSD
        imputation : boolean
            flag telling whether to perform imputation of missiong data
        pse_estimation: boolean
            flag telling whether to perform PSD estimation
        normalized : bool
            if True, the log-likelihood is properly normalized (likelihood integral equals 1)
        rescaled : bool
            if True, the parameter ranges are all linearly rescaled to unit intervals [0, 1]


        """

        # Signal model class
        self.signal_cls = signal_cls
        # noise model class
        self.psd_cls = psd_cls
        # Missing data imputation model class
        self.dat_cls = dat_cls
        # Sampling class
        self.sampler_cls = sampler_cls
        # Output directory path
        self.outdir = outdir
        self.psd_file_name = prefix + '_psd.hdf5'
        # Windowing smoothing parameter (optimized for signal estimation)
        self.n_wind = n_wind
        self.w = gaps.gapgenerator.modified_hann(self.dat_cls.N, n_wind=self.n_wind)
        # Normalization constant for signal amplitude
        self.K1 = np.sum(self.w)
        # Windowing smoothing parameter (optimized for noise psd estimation)
        self.n_wind_psd = n_wind_psd
        # If there are missing data, then the window for PSD does not have gaps
        if any(dat_cls.mask == 0):
            nd, nf = gaps.gapgenerator.findEnds(dat_cls.mask)
            self.w_psd = gaps.gapgenerator.windowing(nd, nf, self.dat_cls.N, window='modified_hann',
                                                     n_wind=self.n_wind_psd)
        else:
            self.w_psd = gaps.gapgenerator.modified_hann(self.dat_cls.N, n_wind=self.n_wind_psd)
        # Normalization constant for noise amplitude
        self.K1_psd = np.sum(self.w_psd)
        # Normalization constant for noise power spectrum
        self.K2_psd = np.sum(self.w_psd ** 2)
        # Windowed signal DFT optimized for PSD estimation
        self.y_fft_psd = self.dat_cls.dft(self.dat_cls.y, self.w_psd)
        # Windowed signal DFT optimized for signal estimation
        self.y_fft = self.dat_cls.dft(self.dat_cls.y, self.w)
        # Spectrum value (can be a numpy array or a list)
        self.spectrum = self.psd_cls.calculate(self.dat_cls.N)
        self.psd_samples = []
        self.psd_logpvals = []
        self.psd_save = 1
        # Number of psd draws for each update
        self.n_psd = n_psd
        # Imputation flag
        self.imputation = imputation
        # PSD estimation flag
        self.psd_estimation = psd_estimation
        # Log-likelihood normalization flag
        self.normalized = normalized
        # Normalizing constant of the likelihood
        self.log_norm = self.compute_log_norm()
        # Rescaling of parameters flag
        self.rescaled = rescaled

        if signal_cls.order is None:
            self.logp = self.signal_cls.logp
            self.logpargs = [self.signal_cls.lo, self.signal_cls.hi]
        else:
            self.logp = self.signal_cls.logpo
            self.logpargs = [self.signal_cls.lo, self.signal_cls.hi, signal_cls.order[0], signal_cls.order[1]]

        if not rescaled:
            self.uniform2params = lambda u: u
            self.params2uniform = lambda x: x
        else:
            self.uniform2params = lambda u: self.signal_cls.formtp(u)
            self.params2uniform = lambda x: self.signal_cls.ptform(x)

        # Reset log-likelihood with proper normalization and auxiliary variables
        self.sampler_cls.update_log_likelihood(self.log_likelihood, (self.spectrum, self.y_fft))
        # Reset log-prior with proper normalization
        self.sampler_cls.update_log_prior(self.signal_cls.logp, (self.params2uniform(self.signal_cls.lo),
                                                                 self.params2uniform(self.signal_cls.hi)))

    def log_likelihood(self, x, *args):
        """
        Log-likelihood including proper PSD normalization and conversion from unitless to physical parameters

        Parameters
        ----------
        x
        args

        Returns
        -------

        """
        # Adds the normalization to the likelihood
        #self.log_likelihood = lambda x, *args: \
            #self.signal_cls.log_likelihood(self.uniform2params(x), *args) + self.log_norm
        return self.signal_cls.log_likelihood(self.uniform2params(x), *args) + self.log_norm

    def compute_log_norm(self):

        if self.normalized:
            if type(self.spectrum) == np.array:
                # Normalization constant (calculated once and for all)
                log_norm = np.real(-0.5 * (np.sum(np.log(self.spectrum)) + self.signal_cls.N * np.log(2 * np.pi)))
            elif type(self.spectrum) == list:
                # If the noise spectrum is a list of spectra corresponding to each TDI channel, concatenate the spectra
                # in a single array
                # Restricted spectrum
                # spectrum_arr = np.concatenate([spect[self.signal_cls.inds_pos] for spect in self.spectrum])
                spectrum_arr = np.concatenate([spect for spect in self.spectrum])
                log_norm = np.real(-0.5 * (np.sum(np.log(spectrum_arr)))
                                   + len(self.y_fft) * self.signal_cls.N * np.log(2 * np.pi))
        else:
            log_norm = 0

        return log_norm

    def compute_frequency_residuals(self, y_gw_fft):
        """
        Compute the residuals used for the PSD estimation
        Parameters
        ----------
        y_gw_fft : numpy.ndarray or list of arrays
            GW signal in the Fourier domain

        Returns
        -------
        z_fft : numpy.ndarray or list
            residuals in Fourier domain

        """

        if type(y_gw_fft) == np.ndarray:
            z_fft = self.y_fft_psd - self.K1_psd / self.dat_cls.N * y_gw_fft
        elif type(y_gw_fft) == list:
            z_fft = [self.y_fft_psd[i] - self.K1_psd / self.dat_cls.N * y_gw_fft[i] for i in range(len(y_gw_fft))]

        return z_fft

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
        # Convert values to physical parameter values if necessary
        params = self.uniform2params(pos0)
        # Inverse Fourier transform back in time domain (can be a numpy array or a list of arrays)
        y_gw = self.signal_cls.compute_time_signal(params)
        # Draw the missing data (can be a numpy array or a list of arrays)
        y_rec = self.dat_cls.imputation(y_gw, self.psd_cls.psd_list)
        # Calculate the DFT of the reconstructed data
        self.y_fft = fft(y_rec * self.w) * self.dat_cls.N / self.K1
        # Update the data DFT for the psd estimation
        self.y_fft_psd = fft(y_rec * self.w_psd)

    def update_psd(self, pos0):
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
        # Convert values to physical parameter values if necessary
        params = self.uniform2params(pos0)
        # Draw the signal (windowed DFT)
        y_gw_fft = self.signal_cls.compute_frequency_signal(params)
        # Calculation of the model residuals
        z_fft = self.compute_frequency_residuals(y_gw_fft)
        # Update periodogram
        self.psd_cls.set_periodogram(z_fft, K2=self.K2_psd)
        # Draw the PSD
        psd_samples, logp_values = self.psd_cls.sample_psd(self.npsd)
        # Last PSD value
        self.psd_cls.logSc = psd_samples[self.npsd - 1, :]
        # Update PSD function and Fourier spectrum
        self.psd_cls.update_psd_func(self.psd_cls.logSc)
        # Update new value of the spectrum for the posterior step
        self.signal_cls.spectrum = self.psd_cls.calculate(self.N)
        # Store psd parameter samples
        self.psd_samples = np.vstack((self.psd_samples, psd_samples))
        # Store corresponding log-posterior values
        self.psd_logpvals = np.concatenate((self.psd_logpvals, logp_values))
        # fh5 = h5py.File(self.outdir + self.psd_file_name, 'a')
        # dset_samples = fh5["psd/samples"]
        # dset_samples.resize(dset_samples.shape[0] + npsd, axis=0)
        # dset_samples[-npsd:, :] = psd_samples
        # dset_logpvals = fh5["psd/logpvals"]
        # dset_logpvals.resize(dset_logpvals.shape[0] + npsd, axis=0)
        # dset_logpvals[-npsd:] = logpvalues
        # fh5.close()
        # Update the spectrum and the normalizing constant
        self.spectrum = self.psd_cls.calculate(self.dat_cls.N)
        self.log_norm = self.compute_log_norm()

    def initialize_aux(self, pos0):
        """
        Initialization of auxiliary parameters

        Parameters
        ----------
        pos0 : array_like
            first guess for signal parameters



        """

        # Initialization of missing data
        self.update_miss(pos0)
        # Initialization of PSD parameters and missing data
        # Draw the signal (windowed DFT)
        y_gw_fft = self.signal_cls.draw_frequency_signal(pos0, self.spectrum, self.y_fft)
        # Calculation of the model residuals
        z_fft = self.y_fft_psd - self.K1_psd / self.N * y_gw_fft
        # Initialization of periodogram
        self.psd_cls.set_periodogram(z_fft, K2=self.K2_psd)
        # Initialization of the PSD
        self.psd_cls.estimate_from_I(self.psd_cls.I)
        # Update new value of the spectrum
        self.spectrum = self.psd_cls.calculate(self.N)
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

    def update_aux(self, pos0):
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
        if self.imputation:
            self.update_miss(pos0)
        # PSD parameter posterior step
        if self.psd_estimation:
            self.update_psd(pos0)
        self.sampler_cls.update_log_likelihood(self.log_likelihood, (self.spectrum, self.y_fft))

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

    def run(self, n_it=100000, n_update=1000, n_thin=5, n_save=1000, save_path='./chains.hdf5'):
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
        n_it : int, optional
            Minimum number of independent samples.
        n_update : int, optional
            Cadence at which updating the auxiliary parameters


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

        # # Initialization of auxiliary parameters
        # self.initialize_aux(p0, n_psd)
        if self.imputation | self.psd_estimation:
            callback = self.update_aux
        else:
            callback = None

        self.sampler_cls.run(n_it, n_update, n_thin, n_save, callback=callback, pos0=None, save_path=save_path)


class fullSampler(FullModel):

    def __init__(self, signal_cls, psd_cls, dat_cls, outdir='./', prefix='samples', order=None,
                 nwalkers=12, ntemps=10, n_wind=500, n_wind_psd=500000, imputation=False, psd_estimation=False):
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
        dat_cls : instance of the approximp class
            class defining the way to perform missing data imputation
        outdir : string
            where to store the psd results and other things
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
        imputation : boolean
            flag telling whether to perform imputation of missiong data
        pse_estimation: boolean
            flag telling whether to perform PSD estimation
            


            
        Returns
        -------
        noise_params_new : array_like
            updated PSD parameters
            
            
            
            
        """

        FullModel.__init__(self, signal_cls, psd_cls, dat_cls, outdir=outdir, n_wind=n_wind, n_wind_psd=n_wind_psd,
                           prefix=prefix)

        self.imputation = imputation
        self.psd_estimation = psd_estimation

        if any(dat_cls.mask == 0):
            nd, nf = gaps.gapgenerator.findEnds(dat_cls.mask)
            self.w_psd = gaps.gapgenerator.windowing(nd, nf, self.dat_cls.N, window='modified_hann',
                                                     n_wind=self.n_wind_psd)

        if order is None:
            self.logp = self.signal_cls.logp
            self.logpargs = [self.signal_cls.lo, self.signal_cls.hi]
        else:
            self.logp = self.signal_cls.logpo
            self.logpargs = [self.signal_cls.lo, self.signal_cls.hi, order[0], order[1]]

        # ---------------------------------------------------------------------
        # Instantiate the GW parameter sampler
        # ---------------------------------------------------------------------
        # Parallel-tempered MCMC with several chains and affine invariant
        # swaps
        self.sampler = ptemcee.Sampler(nwalkers, self.signal_cls.ndim_tot, signal_cls.log_likelihood,
                                       self.logp, ntemps=ntemps,
                                       loglargs=(self.spectrum, self.y_fft), logpargs=self.logpargs)
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
        if self.imputation:
            self.update_miss(pos0)
        # PSD parameter posterior step
        if self.psd_estimation:
            self.update_psd(pos0, npsd)

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
                self.set_loglargs((self.spectrum, self.y_fft))

            i = i+1           

        fullchain = self.sampler.chain[:]

        return fullchain






