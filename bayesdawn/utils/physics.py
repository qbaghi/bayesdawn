import numpy as np
import pyFDresponse as fd_resp


def compute_masses(mc, q):
    """

    Parameters
    ----------
    mc : float
        chirp mass (any unit)
    q : float
        mass ratio

    Returns
    -------
    m1 : float
        mass of object 1
    m2 : float
        mass of object 2

    """

    ma = (q / ((q + 1.) ** 2)) ** (-3.0 / 5.0)
    mb = ((q / ((q + 1.) ** 2)) ** (-6.0 / 5.0) - 4.0 * (q / ((q + 1.) ** 2)) ** (-1.0 / 5.0)) ** 0.5

    m1 = 0.5 * mc * (ma + mb)
    m2 = 0.5 * mc * (ma - mb)

    return m1, m2


def like_to_waveform(par):
    """
    Convert likelihood parameters into waveform-compatible parameters
    Parameters
    ----------
    par : array_like
        parameter vector for sampling the posterior:
        Mc, q, tc, chi1, chi2, np.log10(DL), np.cos(incl), np.sin(bet), lam, psi, phi0

    Returns
    -------
    params : array_like
       parameter vector for LISABeta waveform: m1, m2, chi1, chi2, del_t, dist, incl, phi0, lam, bet, psi

    """

    # Explicit the vector of paramters
    mc, q, tc, chi1, chi2, logdl, ci, sb, lam, psi, phi0 = par

    # Convert chirp mass into individual masses
    m1, m2 = compute_masses(mc, q)

    # Convert base-10 logarithmic luminosity distance into lum. dist. in Mega parsecs
    dl = 10.0 ** logdl

    # Convert inclination
    incl = np.arccos(ci)
    # Convert Source latitude in SSB-frame
    bet = np.arcsin(sb)

    params = np.array([m1, m2, chi1, chi2, tc, dl, incl, phi0, lam, bet, psi])

    return params


def waveform_to_like(params):
    """
    Convert lisabeta waveform parameters to sampled parameters
    Parameters
    ----------
    params

    Returns
    -------

    """

    m1, m2, chi1, chi2, tc, dl, incl, phi0, lam, bet, psi = params

    mc = fd_resp.funcMchirpofm1m2(m1, m2)
    q = m1 / m2

    # transforming into sampling parameters
    par = np.array([mc, q, tc, chi1, chi2, np.log10(dl), np.cos(incl), np.sin(bet), lam, psi, phi0])

    return par


def like_to_waveform_intr(par_intr):
    """
    Convert likelihood parameters into waveform-compatible parameters
    Parameters
    ----------
    par : array_like
        parameter vector for sampling the posterior:
        Mc, q, tc, chi1, chi2, np.log10(DL), np.cos(incl), np.sin(bet), lam, psi, phi0

    Returns
    -------
    params : array_like
       parameter vector for LISABeta waveform: m1, m2, chi1, chi2, del_t, dist, incl, phi0, lam, bet, psi

    """

    # Explicit the vector of paramters
    mc, q, tc, chi1, chi2, sb, lam = par_intr

    # Convert chirp mass into individual masses
    m1, m2 = compute_masses(mc, q)

    # Convert Source latitude in SSB-frame
    bet = np.arcsin(sb)

    params = np.array([m1, m2, chi1, chi2, tc, lam, bet])

    return params