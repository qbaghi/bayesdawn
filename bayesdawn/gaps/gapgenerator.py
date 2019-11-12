import numpy as np
import copy
import warnings


def modified_hann(n_data, n_wind=60):
    """

    Modified Hann window as in Carre and Porter, 2010

    """

    w = np.ones(n_data)
    n = np.arange(n_data)

    if 2 * n_wind <= n_data:
        n_w = copy.copy(n_wind)
    else:
        n_w = np.int(n_wind / 2.)
        warnings.warn("Size of window decay is larger than half the window size",
                      UserWarning)

    # w[0:n_w] = 0.5 * ( 1 - np.cos( 2*np.pi * n[0:n_w] / (2*n_w) ) )
    w[0:n_w] = 0.5 * (1 - np.cos(np.pi * n[0:n_w] / n_w))

    # w[N-n_w:] = 0.5 * ( 1 - np.cos( 2*np.pi * (n[N-n_w:]-N+1+2*n_w) / (2*n_w) ) )
    w[n_data - n_w:] = 0.5 * (1 - np.cos(np.pi * (n[n_data - n_w:] - n_data + 1) / n_w))

    return w


def windowing(nd, nf, n, window='rect', n_wind=160):
    """
    Function that produces a mask vector M from the index locations of gaps edges.
    The non-zero values can be chosen to be just ones (rectangular window, default)
    or apodization windows smoothly going to zero at the gap edges.


    @param nd : vector containing the indices of the begening of each hole
    @type nd : nh x 1 vector where nh is the number of holes
    @param nf : vector containing the indices of the end of each hole
    @type nf : nh x 1 vector where nh is the number of holes
    @param n : size of the data
    @type n : integer scalar
    n_wind

    @param window : option to choose the type of window to apply
    @type window :
    @return: ...
        The vector mw containg the softened hole vector M with the selected
        type of window

    Parameters
    ----------

    """

    # import numpy as np

    # Number of holes
    nh = len(nd)

    # Windowed mask initialization
    mw = np.zeros((n))

    if window == 'rect':
        windowfunc = lambda x: np.ones(x)
        # First window
        mw[0:nd[0]] = windowfunc(nd[0])

        for k in range(0, nh - 1):
            mw[nf[k]:nd[k + 1]] = windowfunc(nd[k + 1] - nf[k])

        # last window :
        mw[nf[nh - 1]:n] = windowfunc(n - nf[nh - 1])

    else:

        if window == 'hann':
            windowfunc = lambda x: np.hanning(x)
        elif window == 'blackman':
            windowfunc = lambda x: np.blackman(x)
        elif window == 'modified_hann':
            windowfunc = lambda x: modified_hann(x, n_wind)

        # First window
        # we want that the window has value zero at entry n = Nd[0]
        mw[0:nd[0] + 1] = windowfunc(nd[0] + 2)[1:]
        #
        #        for k in range(0,nh-1) :
        #
        #            # We have mw[Nf[k]] = 1 (end of gap), so we don't want the window
        #            # to be zero here.
        #            # We also have mw[Nd[k+1]] = 0. The window must end here.
        #            mw[Nf[k]+1:Nd[k+1]+1] = windowfunc(Nd[k+1] - Nf[k])
        #
        # last window :
        mw[nf[nh - 1] - 1:n] = windowfunc(n - nf[nh - 1] + 2)[:-1]

        # Extend to take into account beginning and end of time series
        #        Nde = np.concatenate(([-1],Nd,[N]))
        #        Nfe = np.concatenate(([0],Nf,[N+1]))
        for k in range(0, nh - 1):
            # We have mw[Nf[k]] = 1 (end of gap), so we don't want the window
            # to be zero here.
            # We also have mw[Nd[k+1]] = 0. The window must end here.
            mw[nf[k] - 1:nd[k + 1] + 1] = windowfunc(nd[k + 1] - nf[k] + 2)

    return mw


def generategaps(n_data, fs, n_gaps, t_gaps, gap_type='random', f_gaps=1e-2, wind_type='rect', std_loc=0, std_dur=0):
    """
    Function that generates the indices of begening and end of data holes, and
    then uses the windowing function to creates the corresponding mask vector.


    Parameters
    ----------
    n_data : scalar integer
        size of the data
    fs : scalar float
        sampling frequency (Hz)
    n_gaps : scalar integer
        total number of gaps in the time series
    t_gaps : scalar float or array_like
        duration of each gaps (sec)
    gap_type : string {'random','periodic'}
        option to choose the type of window to apply
    f_gaps : scalar float
        frequency of the gaps in the periodic case, optional
    wind_type :  string
        type of window function, optional (default is rectangular window)
    std_loc : scalar float
        standard deviation of the gap locations (seconds, apply for periodic
        gap only)
    std_dur : scalar float
        standard deviation of the gap duration (seconds)


    @return: ...
        The vector MW containg the hole vector M with the selected
        type of holes
    """

    np.random.seed()

    if 'random' in gap_type:  # N_gaps of T_gaps seconds

        # Taille du trou en nombre de points
        if isinstance(t_gaps, (int, float, np.int, np.float64)):
            d_n = np.int(t_gaps * fs) * np.ones(n_gaps)
            d_n = d_n.astype(np.int)
        elif isinstance(t_gaps, (list, tuple, np.ndarray)):
            d_n = np.array(fs * t_gaps).astype(np.int)
        # Small deviations in the gap duration
        d_n = d_n + std_dur * fs * np.random.normal(loc=0.0, scale=1.0, size=len(d_n))
        # Uniform random location of gaps
        if 'poisson' in gap_type:
            # Calculate the average inter-gap inverval
            inter_gap_mean = (n_data - np.sum(d_n)) / (n_gaps + 1)
            # Draw the inter-gap durations from a Poisson distribution
            inter_gap = np.random.exponential(scale=inter_gap_mean, size=n_gaps)
            # Set the gaps locations
            nd = np.empty(n_gaps, dtype=np.int)
            ref_point = 0
            for g in range(n_gaps):
                nd[g] = ref_point + inter_gap[g]
                ref_point = nd[g] + d_n[g]

        else:
            nd = np.sort((np.random.rand(n_gaps) * n_data).astype(np.int))
        # End of gaps
        nf = (nd + d_n).astype(np.int)
        # Remove overlapping
        for k in range(n_gaps - 1):
            if nd[k + 1] - nd[k] <= d_n[k]:
                nd[k + 1] = nd[k] + d_n[k]
                nf[k + 1] = nf[k] + d_n[k]

        # Keep only hole locations that do not exceed data span
        print("Shape of nf: " + str(len(nf)))
        print("Shape of nd: " + str(len(nd)))
        nf = nf[nf < n_data - 1]
        nd = nd[nf < n_data - 1]

    elif gap_type == 'periodic':

        # Number of holes :
        n_gaps = np.int(f_gaps * n_data / fs)
        print("Warning: number of gaps derived from f_gaps: " + str(n_gaps))
        # Random location of holes
        nd = np.zeros(n_gaps)
        # Calculate CDF for all n
        nd = np.arange(fs / f_gaps, n_data, fs / f_gaps).astype(np.int)
        # Introduce some randomness on the gap locations
        nd = (nd + std_loc * fs * np.random.normal(loc=0.0, scale=1.0, size=len(nd))).astype(np.int)
        # Length of gaps in term of data points, including possible deviations
        d_n = t_gaps * fs + std_dur * fs * np.random.normal(loc=0.0, scale=1.0, size=len(nd))
        d_n = d_n.astype(np.int)
        # d_n = (T_gaps*fs*np.ones(len(nd))).astype(np.int)

        # Fin des trous
        nf = nd + d_n
        # Sort the holes
        # Remove overlapping
        for k in range(len(nd) - 1):

            if (nd[k + 1] - nd[k] <= d_n[k]):
                nd[k + 1] = nd[k] + d_n[k]
                nf[k + 1] = nf[k] + d_n[k]
        # Keep only hole locations that do not exceed data span
        nf = nf[nf < n_data - 1]
        nd = nd[nf < n_data - 1]

    else:
        raise ValueError('Unknown gap type')

    return nd, nf


def find_ends(m):
    """
    Function finding the ends of the holes defined by the mask M.


    @param m : mask vector
    @type m : N x 1 vector where N is the number of data

    @return: ...
        @param Nd : vector containing the indices of the begening of each hole
        @type Nd : Nh x 1 vector where Nh is the number of holes
        @param Nf : vector containing the indices of the end of each hole
        @type Nf : Nh x 1 vector where Nh is the number of holes
    """

    i_mis = np.where(m == 0)[0]
    N_mis = len(i_mis)
    # Recalculate the holes ends
    Nd_eff = np.array([])
    Nf_eff = np.array([])

    Nd_eff = np.append(Nd_eff, i_mis[0])
    for n in range(N_mis - 1):

        # If the next hole is not just after this one
        if i_mis[n + 1] != i_mis[n] + 1:
            # End of holes is just after
            Nf_eff = np.append(Nf_eff, i_mis[n] + 1)
            # beginning of next hole is at i_mis[n+1]
            Nd_eff = np.append(Nd_eff, i_mis[n + 1])

    # Last missing data
    Nf_eff = np.append(Nf_eff, i_mis[N_mis - 1] + 1)

    return Nd_eff.astype(np.int), Nf_eff.astype(np.int)


def segmentedges(M):
    """
    Find the segment edges 
    """
    Nd, Nf = find_ends(M)
    seg_starts = np.concatenate(([0], Nf))
    seg_ends = np.concatenate((Nd, [len(M)]))

    return seg_starts, seg_ends


def segmentlengths(M):
    """
    Compute segments lengths from mask
    """
    seg_starts, seg_ends = segmentedges(M)

    return (seg_ends - seg_starts).astype(np.int)


def segmentwise(y, M):
    """
    Separate the masked data in a list of segments

    """

    Nd, Nf = find_ends(M)
    seg_starts = np.concatenate(([0], Nf))
    seg_ends = np.concatenate((Nd, [len(M)]))
    # segment_ffts = [fft(y[seg_starts[j]:seg_ends[j]],n=n) for j in range(len(seg_starts))]

    return [y[seg_starts[j]:seg_ends[j]] for j in range(len(seg_starts))]


# y_segs = gg.segmentwise(data,self.M)
# Nstart,Nend = gg.segmentedges(self.M)
# y_segs_fft = [fft(seg) for seg in y_segs]
# f_segs = [np.fft.fftfreq(len(seg))*fs for seg in y_segs]


def compute_freq_times(M, ts):
    """
    Function that computes the begining and end times of each segment of 
    
    Parameters
    ----------
    M : array_like
        binary mask
    ts : array_like
        sampling time
    """
    # Edges of segments
    Nstarts, Nends = segmentedges(M)
    # Lengths of segments
    slen = (Nends - Nstarts).astype(np.int)
    # y_segs_fft = [fft(seg) for seg in y_segs]
    f_segs = [np.fft.fftfreq(Ns) / ts for Ns in slen]

    return f_segs, Nstarts * ts, Nends * ts


def findinds(f, fmin, fmax):
    return np.where((f >= fmin) & (f <= fmax))[0]


def select_freq(f_segs, Tstarts, Tends, f1, f2):
    """
    
    for a list of frequency vectors, select frequencies in a given interval.
    Discard vectors for which no frequency lies in the interval.
    
    """

    # For each frequency vector, indices of frequencies within the required 
    # interval
    inds = [findinds(f_seg, f1, f2) for f_seg in f_segs]
    # Then discard the empty ones
    ipos = [i for i in range(len(f_segs)) if len(inds[i]) > 0]
    inds_freq = [inds[i] for i in ipos]

    #    T1 = np.concatenate([Tstarts[j]*np.ones(len(f_segs_ind[j])) for j \
    #                         in range(len(f_segs_ind)) if len(f_segs_ind[j])>1])
    #    T2 = np.concatenate([Tends[j]*np.ones(len(f_segs_ind[j])) for j \
    #                         in range(len(f_segs_ind)) if len(f_segs_ind[j])>1])

    #    T1 = [Tstarts[j]*np.ones(len(inds[j])) for j in ipos]
    #    T2 = [Tends[j]*np.ones(len(inds[j])) for j in ipos]
    T1 = [Tstarts[j] for j in ipos]
    T2 = [Tends[j] for j in ipos]

    freqs = [f_segs[j][inds[j]] for j in ipos]

    return freqs, T1, T2, inds_freq, ipos


if __name__ == '__main__':

    import h5py
    import time
    import datetime
    from matplotlib import pyplot as plt

    N = 2 ** 22
    fs = 0.1
    ts = 10

    # wind_type = 'rect'
    wind_type = 'modified_hann'

    # gap_config = "antenna"
    # gap_config = "random"
    gap_config = "periodic"
    # gap_config = "micrometeorites"

    if gap_config == "antenna":

        # Gap frequency
        # f_gaps = 1/(10*3600.*24)
        f_gaps = 2 / (10 * 3600. * 24)
        # Gap location deviation
        std_loc = 3600 * 24
        # Gap duration
        # L_gaps = 2*3600.
        L_gaps = 3600.
        # Gap duration deviation
        std_dur = 10 * 60

        M = generategaps(N, fs, np.int(N * ts * f_gaps), L_gaps,
                         gap_type='periodic', f_gaps=f_gaps, wind_type=wind_type,
                         std_loc=std_loc, std_dur=std_dur)

    elif gap_config == "micrometeorites":

        # Gap frequency
        f_gaps = 1 / (3600. * 24)
        # Gap location deviation
        std_loc = 3600.
        # Gap duration
        L_gaps = 10 * 60.
        # Gap duration deviation
        std_dur = 60.

        M = generategaps(N, fs, np.int(N * ts * f_gaps), L_gaps,
                         gap_type='random_poisson', f_gaps=f_gaps, wind_type=wind_type,
                         std_loc=std_loc, std_dur=std_dur)

    elif gap_config == "random":

        # Gap frequency
        f_gaps = 1 / (3600. * 24)
        # Gap location deviation
        std_loc = 3600.
        # Gap duration
        L_gaps = 10 * 60.
        # Gap duration deviation
        std_dur = 60.

        M = generategaps(N, fs, np.int(N * ts * f_gaps), L_gaps,
                         gap_type='random', f_gaps=f_gaps, wind_type=wind_type,
                         std_loc=std_loc, std_dur=std_dur)

    elif gap_config == "periodic":

        f_gaps = 1 / (14 * 3600 * 24)
        # f_gaps = 1/86400.
        L_gaps = 7 * 3600.
        M = generategaps(N, fs, np.int(N * ts * f_gaps), L_gaps,
                         gap_type='periodic', f_gaps=f_gaps, wind_type=wind_type)

    # file_path = "/Users/qbaghi/Codes/data/masks/"
    # file_name = gap_config
    # # Prepare data files to save
    # now = datetime.datetime.now()
    # prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_") + "mask_"
    # file_name = file_path + prefix + gap_config + '.hdf5'
    # fh5 = h5py.File(file_name,'w')
    # fh5.create_dataset("mask", data = M)
    # fh5.close()

    file_path = "/Users/qbaghi/Codes/data/masks/"

    file_name1 = file_path + "2018-10-29_15h22-12_mask_antenna.hdf5"
    fh5 = h5py.File(file_name1, 'r')
    M1 = fh5['mask'].value
    fh5.close()

    file_name2 = file_path + "2018-11-14_15h48-51_mask_micrometeorites.hdf5"
    fh5 = h5py.File(file_name2, 'r')
    M2 = fh5['mask'].value
    fh5.close()

    Nd1, Nf1 = find_ends(M1)
    Nd2, Nf2 = find_ends(M2)
    MW1 = windowing(Nd1, Nf1, N, window='modified_hann', n_wind=260)
    MW2 = windowing(Nd2, Nf2, N, window='modified_hann', n_wind=260)

    #
    # # Prepare data files to save
    # now = datetime.datetime.now()
    # prefix = now.strftime("%Y-%m-%d_%Hh%M-%S_") + "mask_"
    # fh5 = h5py.File(file_path + prefix + 'blend'+ '.hdf5','w')
    # fh5.create_dataset("mask", data = M1*M2)
    # fh5.close()

    # Prepare data files to save
    now = datetime.datetime.now()
    fh5 = h5py.File(file_name1[:-5] + '_hann260.hdf5', 'w')
    fh5.create_dataset("mask", data=MW1)
    fh5.close()
    fh5 = h5py.File(file_name2[:-5] + '_hann260.hdf5', 'w')
    fh5.create_dataset("mask", data=MW2)
    fh5.close()
