from matplotlib import pyplot as plt
import numpy as np
from bayesdawn.postproc import resanalysis
from bayesdawn.utils import physics
from bayesdawn.waveforms import lisaresp
from dynesty import plotting as dyplot


def postprocess(chain, lnprob, config, params, n_burn=500, n_thin=1, n_bins=40, k=4):
    """

    Parameters
    ----------
    chain : ndarray
        numpy array containing the posterior samples
    config : configparser.ConfigParser instance
        configuration object containing the parameters of the MCMC run
    params : ndarray
        full vector of true parameter values
    n_burn : int
        number of samples to discard at the beginning of the chains
    k : int
        multiple of standard deviation up to which the cornerplots are restricted

    Returns
    -------
    fig : matplotlib.pyplot.figure  instance
        figure where the cornerplot is drawn
    axes :  matplotlib.pyplot.axes instance
        axes of fig

    """

    chain_eff = chain[:, :, chain[0, 0, :, 0] != 0, :]
    print("Shape of non-zero sample array: " + str(chain_eff.shape))
    chain_eff = chain_eff[0, :, n_burn::n_thin, :]

    # Load simulation parameters
    # Get all parameter name keys
    names_full = [key for key in config['ParametersLowerBounds']]
    # Get prior bound values
    bounds_full = [[float(config['ParametersLowerBounds'][name]), float(config['ParametersUpperBounds'][name])]
                   for name in names_full]
    # Get all parameter name keys
    names_full = [key for key in config['ParametersLowerBounds']]

    # truths_vect = params[:]
    # offset = np.array([bound[0] for bound in bounds])

    # Get true parameter values, only for intrinsic parameters
    # if config["Model"].getboolean("reduced"):

    # Convert waveform parameters to actually sampled parameters
    par = physics.waveform_to_like(params)
    # mc, q, tc, chi1, chi2, np.log10(dl), np.cos(incl), np.sin(bet), lam, psi, phi0
    inds_intr = [0, 1, 2, 3, 4, 7, 8]

    if not config["Model"].getboolean("reduced"):
        par0 = np.array(par)[inds_intr]
        chain_eff = chain_eff[:, :, inds_intr]
        names = np.array(names_full)[inds_intr]
        lo = np.array([bound[0] for bound in bounds_full])[inds_intr]
        hi = np.array([bound[1] for bound in bounds_full])[inds_intr]

    else:
        names = np.array(names_full)
        par0 = np.array(par)
        lo = np.array([bound[0] for bound in bounds_full])
        hi = np.array([bound[1] for bound in bounds_full])

    # offset = lo
    # scales = hi - lo
    offset = 0
    scales = 1
    # chain_eff = chain_eff * (hi - lo) + lo
    # if not config["Model"].getboolean("rescaled"):
    #     # Convert them in the interval [0, 1]
    #     params0_u = (params0 - lo) / (hi - lo)
    #     # Rescale between [0, 1]
    #     chain_rescaled = (chain_eff - lo) / (hi - lo)
    # else:
    #     params0_u = params0[:]
    #     chain_rescaled = chain_eff[:]

    # limits = [[0, 1] for i in range(chain_rescaled.shape[-1])]
    print("Shape of effective chain: " + str(chain_eff.shape))
    chain_flatten = chain_eff.reshape((-1, chain_eff.shape[2]))
    medians = np.median(chain_flatten, axis=0)
    stds = np.std(chain_flatten, axis=0)
    limits = [[medians[i] - k * stds[i], medians[i] + k * stds[i]] for i in range(chain_flatten.shape[1])]
    print("Shape of flattened chain: " + str(chain_flatten.shape))
    print("Length of parameter vector: " + str(len(par0)))

    fig, axes = resanalysis.cornerplot([chain_flatten],
                                       par0, offset, scales, names,
                                       colors=['k', 'gray', 'blue'],
                                       limits=limits, fontsize=16,
                                       bins=n_bins, truth_color='red', figsize=(9, 8.5), linewidth=1)

    # results = {'samples': chain_flatten,
    #            'logvol': lnprob[0, :, n_burn::n_thin].reshape((-1, lnprob.shape[2])),
    #            'weights': np.ones(chain_flatten.shape[0])}

    # fig, axes = dyplot.cornerpoints(results, dims=None, thin=1, span=None, cmap='plasma', color=None,
    #                                 kde=False, nkde=1000, plot_kwargs=None, labels=names,
    #                                 label_kwargs=None, truths=par0, truth_color='red',
    #                                 truth_kwargs=None, max_n_ticks=5, use_math_text=False,
    #                                 fig=None)

    plt.show()

    return fig, axes