from bayesdawn import samplers
import numpy as np


class PosteriorModel(object):

    def __init__(self, signal_cls, bounds, rescaled=False):
        """

        Parameters
        ----------
        bounds : list of lists
            parameter boundaries
        rescaled : bool
            if True, the parameter ranges are all linearly rescaled to unit intervals [0, 1]

        """

        # Signal and likelihood model instance
        self.signal_cls = signal_cls

        # Boundaries of parameters
        self.bounds = bounds
        self.lo = np.array([bound[0] for bound in self.bounds])
        self.hi = np.array([bound[1] for bound in self.bounds])
        self.rescaled = rescaled

        if rescaled:
            self.lo_rescaled = np.zeros(len(bounds))
            self.hi_rescaled = np.ones(len(bounds))
        else:
            self.lo_rescaled = self.lo
            self.hi_rescaled = self.hi

        # Effective size of the model
        self.n_eff = self.signal_cls.N * len(self.signal_cls.channels)

        # Normalizing constant of the likelihood
        self.log_norm = 0

    def compute_log_norm(self, spectrum):

        if type(spectrum) == np.array:
            # Normalization constant (calculated once and for all)
            self.log_norm = np.real(-0.5 * (np.sum(np.log(self.spectrum)) + self.signal_cls.N * np.log(2 * np.pi)))
        elif type(spectrum) == list:
            # If the noise spectrum is a list of spectra corresponding to each TDI channel, concatenate the spectra
            # in a single array
            # Restricted spectrum
            # spectrum_arr = np.concatenate([spect[self.posterior_cls.inds_pos] for spect in self.spectrum])
            spectrum_arr = np.concatenate([spect for spect in spectrum])
            self.log_norm = np.real(-0.5 * (np.sum(np.log(spectrum_arr))) + len(spectrum) * self.signal_cls.N * np.log(2 * np.pi))

    def log_likelihood(self, u, *args):
        """
        log_likelihood wrapper taking into account the rescaling of parameters

        Parameters
        ----------
        u : bytearray
            parameter values in [0, 1] interval
        args : tuple
            other likelihood arguments (not rescaled)

        Returns
        -------

        """

        if self.rescaled:
            return self.signal_cls.log_likelihood(self.uniform2param(u), *args) + self.log_norm
        else:
            return self.signal_cls.log_likelihood(u, *args) + self.log_norm

    def logp(self, x, lo, hi):

        return np.where(((x >= lo) & (x <= hi)).all(-1), 0.0, -np.inf)

    def logpo(self, x, i1, i2):

        return np.where(((x >= self.lo_rescaled) & (x <= self.hi_rescaled)).all(-1) & (x[i1] <= x[i2]), 0.0, -np.inf)

    # def log_prior(self, params):
    #     """
    #     Logarithm of the prior probabilitiy of parameters f_0 and f_dot
    #
    #     Parameters
    #     ----------
    #     params : array_like
    #         vector of parameters in the orders of
    #         names=['f_0','f_dot']
    #
    #     Returns
    #     -------
    #     logP : scalar float
    #         logarithm of the prior probability
    #
    #
    #     """
    #
    #     #prior probability for f_0 and f_dot
    #     logs = [samplers.logprob(params[i], self.distribs[i], self.bounds[i]) for i in range(len(params))]
    #
    #     return np.sum(np.array(logs))

    def log_prob(self, params, params_aux):
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
        return lp + self.log_likelihood(params, params_aux)

    def uniform2param(self, u):
        """

        Convert numbers drawn from uniform distribution to physical parameter values

        Parameters
        ----------
        u : numpy array
            arrays of floats in interval [0, 1]

        Returns
        -------
        x : numpy array
            arrays of floats in interval [lo, hi]

        """

        return (self.hi - self.lo) * u + self.lo

    def param2uniform(self, x):
        """

        Convert physical parameter values to numbers in the interval [0, 1]

        Parameters
        ----------
        x : numpy array
            arrays of floats in interval [lo, hi]


        Returns
        -------
        u : numpy array
            arrays of floats in interval [0, 1]

        """

        return (x - self.lo) / (self.hi - self.lo)

    def compute_time_signal(self, u):

        return self.signal_cls.compute_time_signal(self.uniform2param(u))

    def compute_frequency_signal(self, u):

        return self.signal_cls.compute_frequency_signal(self.uniform2param(u))


