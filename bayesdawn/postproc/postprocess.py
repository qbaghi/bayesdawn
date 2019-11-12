from matplotlib import pyplot as plt
import numpy as np
from bayesdawn.postproc import resanalysis
from scipy import stats


def postprocess(chain, lnprob, names, par0, n_burn=500, n_thin=1, n_bins=40, k=4):
    """

    Parameters
    ----------
    chain : ndarray
        numpy array containing the posterior samples
    names : list of str
        list of parameter names
    par0 : ndarray
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
    offset = 0
    scales = 1

    print("Shape of effective chain: " + str(chain_eff.shape))
    chain_flatten = chain_eff.reshape((-1, chain_eff.shape[2]))
    medians = np.median(chain_flatten, axis=0)
    # stds = np.std(chain_flatten, axis=0)
    stds = stats.median_absolute_deviation(chain_flatten, axis=0)
    limits = [[medians[i] - k * stds[i], medians[i] + k * stds[i]] for i in range(chain_flatten.shape[1])]
    print("Shape of flattened chain: " + str(chain_flatten.shape))
    print("Length of parameter vector: " + str(len(par0)))

    fig, axes = resanalysis.cornerplot([chain_flatten],
                                       par0, offset, scales, names,
                                       colors=['k', 'gray', 'blue'],
                                       limits=limits, fontsize=16,
                                       bins=n_bins, truth_color='red', figsize=(9, 8.5), linewidth=1,
                                       plot_datapoints=False, smooth=1.0, smooth1d=2.0)

    plt.show()

    return fig, axes