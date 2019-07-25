import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib as mpl


def lognormvar(mu, sig2):
    """

    # The variance of the exponential of a gaussian distribution of mean mu and variance sigma2 is
    # [exp(sigma2)-1]*exp(2*mu + sigma2)

    Parameters
    ----------
    mu
    sig2

    Returns
    -------
    sig2_exp

    """

    return (np.exp(sig2) - 1) * np.exp(2 * mu + sig2)




mpl.rcdefaults()
mpl.rcParams['font.family'] = 'serif'
#    mpl.rcParams['xtick.major.size'] = 6
#    mpl.rcParams['ytick.major.size'] = 6
#    mpl.rcParams['xtick.minor.size'] = 3
#    mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['text.usetex'] = True
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['lines.markeredgewidth'] = 1
# mpl.rcParams['legend.handletextpad'] = 0.3
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.figsize'] = 6, 5
mpl.rcParams['figure.autolayout'] = True

# ==========================================================================
# Barplots for Bayes factors
# ==========================================================================

# delta_f = np.array([1e-7, 1e-8, 1e-9, 1e-10])
# ind = np.arange(len(delta_f))
# X = [ind, ind, ind]
# # For Complete data
# logB_complete = np.array([104.00453313, 38.62348088, 24.73178091, 22.09517497])
# logB_complete_err = np.array([0.29505523, 0.36471844, 0.299414, 0.28620051])
#
# # For gapped data and windowing method
# logB_window = np.array([-0.49605142, -0.43551874, 2.64928399, 0.49093671])
# logB_window_err = np.array([0.0329986, 0.0270233, 0.02233843, 0.021304])
# # For gapped data and DA method
# logB_DA = np.array([106.71811731, 40.0273956, 28.16463517, 28.52405347])
# logB_DA_err = np.array([0.30590103, 0.36643779, 0.30234179, 0.29136273])
delta_f = np.array([1e-7, 1e-8, 1e-9])
ind = np.arange(len(delta_f))
X = [ind, ind, ind]

# For A1 = 2 and A2 = 1.5
#########################
# # For Complete data
# logB_complete = np.array([104.00453313, 38.62348088, 24.73178091])
# logB_complete_err = np.array([0.29505523, 0.36471844, 0.299414])
# # For gapped data and windowing method
# logB_window = np.array([-0.49605142, -0.43551874, 2.64928399])
# logB_window_err = np.array([0.0329986, 0.0270233, 0.02233843])
# # For gapped data and DA method
# logB_DA = np.array([106.71811731, 40.0273956, 28.16463517])
# logB_DA_err = np.array([0.30590103, 0.36643779, 0.30234179])

# # # For same amplitude and random gaps
# # ####################################
# # For Complete data
# logB_complete = np.array([163.34696504, 67.80116829, 25.7267448, 17.91274735])
# logB_complete_err = np.array([0.42592654, 0.42749419, 0.34967362, 0.34339197])
# # For gapped data and windowing method
# logB_window = np.array([-1.13950843, -4.07411988, 3.42889368, 4.32991547])
# logB_window_err = np.array([0.03478457, 0.03308532, 0.02548193, 0.02501538])
# # For gapped data and DA method
# logB_DA = np.array([183.12607074, 80.69904393, 29.69120347, 20.95783957])
# logB_DA_err = np.array([0.44352533, 0.43656447, 0.35776373, 0.34505469])

# # For same amplitude and antenna gaps
# ####################################
# # For Complete data
# logB_complete = np.array([163.34696504, 67.80116829, 25.7267448, 17.91274735])
# logB_complete_err = np.array([0.42592654, 0.42749419, 0.34967362, 0.34339197])
# # For gapped data and windowing method
# logB_window = np.array([103.40334817, 39.53571004, 14.11789999, 7.55843489])
# logB_window_err = np.array([0.28714883, 0.28373448, 0.22992236, 0.22418521])
# # For gapped data and DA method
# logB_DA = np.array([177.92905604, 73.42671259, 28.69277735, 20.8526008])
# logB_DA_err = np.array([0.43819421, 0.43299554, 0.35643442, 0.34653156])


# For same amplitude and random gaps and EMCEE calculation
###########################################################
# For Complete data
logB_complete = np.array([177.86674067, 84.55074765, 44.46961875, 36.84705732])
logB_complete_err = np.array([20.48698424, 20.81466185, 18.93563272, 17.79046023])
# For gapped data and windowing method
logB_window = np.array([5.04291534,  1.56493368, 7.19801673, 7.48835142])
logB_window_err = np.array([13.38003147, 12.55370689, 9.37689494, 8.50952161])
# For gapped data and DA method
logB_DA = np.array([197.58831862, 97.83338462, 48.31879245, 39.74701148])
logB_DA_err = np.array([20.79938212, 21.04350754, 19.81964137, 18.33507994])


# # # For same amplitude and periodic gaps and EMCEE calculation
# # ###########################################################
# # For Complete data
# logB_complete = np.array([177.86674067, 84.55074765, 44.46961875, 36.84705732])
# logB_complete_err = np.array([20.48698424, 20.81466185, 18.93563272, 17.79046023])
# # For gapped data and windowing method
# logB_window = np.array([118.113445, 56.22315571, 31.37840321, 24.62008196])
# logB_window_err = np.array([18.21771166, 18.65324688, 17.08435837, 16.564876])
# # For gapped data and DA method
# logB_DA = np.array([192.28590794, 90.33114342, 47.50040924, 39.72815791])
# logB_DA_err = np.array([20.31379255, 20.51712949, 18.83398667, 17.8824588])


logY = [logB_complete, logB_window, logB_DA]
logY_err = [logB_complete_err, logB_window_err, logB_DA_err]
linewidths = [0.05, 0.05, 0.05]
linestyles = ['solid', 'solid', 'solid']
colors = ['black', 'gray', 'blue']
labels = ['Complete data', 'Gapped data, windowing', 'Gapped data, DA method']

# width of the bars
barWidth = 0.15

# # Choose the height of the error bars (bars1)
# yer1 = [0.5, 0.4, 0.5]
#
# # Choose the height of the error bars (bars2)
# yer2 = [1, 0.7, 1]

# The x position of bars
r1 = np.arange(len(logY[0]))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r = [r1, r2, r3]

# Choose name of categories
index = [r'$10^{-7}$', r'$10^{-8}$', r'$10^{-9}$', r'$10^{-10}$']

# Confidence interval on exp(Y)
Y_err = []
Y = []
for i in range(len(logY)):

    Y.append(np.exp(logY[i]))
    ci = np.zeros((2,len(logY[i])))
    # lower bound
    ci[0, :] = np.exp(logY[i]) - np.exp(logY[i] - logY_err[i])
    # upper bound
    ci[1, :] = np.exp(logY[i] + logY_err[i]) - np.exp(logY[i])
    Y_err.append(ci)

    # Y.append(logY[i])
    # ci = np.zeros((2, len(logY[i])))
    # # lower bound
    # ci[0, :] = logY[i] - logY_err[i]
    # # upper bound
    # ci[1, :] = logY[i] + logY[i]
    # Y_err.append(ci)

    # Y_err.append(np.sqrt(lognormvar(logY[i], logY_err[i]**2)))

# Create blue bars
for i in range(len(Y)):
    plt.bar(r[i], Y[i], width=barWidth, color=colors[i], edgecolor='black', capsize=7, label=labels[i],
            yerr=Y_err[i], alpha=0.75)

# general layout

plt.xticks([r + barWidth for r in range(len(Y[0]))], index)
# plt.ylabel(r'$\log B_{21}$')
plt.ylabel(r'$B_{21}$')
# plt.ylabel(r'$\log B_{21}$')
plt.xlabel(r'$\Delta f$ [Hz]')
plt.hlines(20, 0 - 3*barWidth, len(Y[0])+barWidth, colors='red', linewidth=1, linestyles='dashed')#, label=r"Positive threshold $B_{12} = 3$-$20$")
# plt.hlines(3, 0 - 3*barWidth, len(Y[0])+barWidth, colors='red', linewidth=1, linestyles='dashed')
plt.legend()
plt.xlim([0-3*barWidth,len(Y[0])-barWidth])
plt.ylim([1e-6, 1e96])
#plt.text(len(Y[0])- 3.5*barWidth, 40, r"$3$-$20$", color='red')
plt.text(len(Y[0])- 4*barWidth, 40, r"$B_{12}$=20", color='red')
plt.yscale('log')
#plt.text(len(Y)- 4.1*barWidth, 0.1, r"$B_{12} = 3$", color='red')
# plt.minorticks_on()
plt.savefig('/Users/qbaghi/Documents/articles/papers/papers/gaps/figures/2sources_bayes_factors/bayes_2sources_2e-4Hz_randgaps.pdf')
# plt.savefig('/Users/qbaghi/Documents/articles/papers/papers/gaps/figures/2sources_bayes_factors/bayes_2sources_2e-4Hz_antgaps.pdf')

# Show graphic
plt.show()
