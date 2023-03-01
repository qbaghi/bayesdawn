# Gap filling investigation support for LISA inference with lisabeta (maybe more generally)
# Time-domain functions implementation by Eleonora Castelli NASA-GSFC 2022
# based on previous version by John Baker NASA-GSFC 2021 

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal
from scipy.stats import norm
from cycler import cycler
import sys

from ldc.lisa.noise import get_noise_model
import lisaorbits
from bayesdawn import datamodel, psdmodel

# Function to print all attributes of hdf5 file recursively
def print_attrs(name, obj):
    shift = name.count('/') * '    '
    print(shift + name)
    for key, val in obj.attrs.items():
        print(shift + '    ' + f"{key}: {val}")
        
# Function to import LDC 2 data and convert them in more convenient format 
def load_tdi_timeseries(fname, 
                        import_datasets = ['obs','clean','sky','noisefree', 'gal'], 
                        generate_additional_datasets = True,
                        additional_datasets = ['clean_gapped', 'noise_gapped', 'sky_gapped', 'noiseglitch_gapped']):
    """
    Loads LDC-2 TDI time-series from HDF5 file and packs them in a dictionary.
    Dictionary keys are the HDF5 dataset groups.
    Each key items are the corresponding TDI time-series in a numpy rec-array with fields ['t', 'X', 'Y', 'Z'].
    
    Parameters:
    -----------
        fname: string
            LDC-2 HDF5 filename with .h5 extension
        import_datasets: list
            list of HDF5 groups containing /tdi datasets
        generate_additional_datasets: bool
            flag to generate additional datasets
        additional_datasets: list
            list of desired additional datasets
    
    Returns:
    --------
        tdi: dict
            dict with keys corresponding to HDF5 groups (import_datasets + additional_datasets)
            each key item is a numpy rec-array with fields ['t', 'X', 'Y', 'Z']
    """
    
    fid = h5py.File(fname)
    tdi={}
    for ds in import_datasets:
        tdi[ds] = fid[ds+'/tdi'][()].squeeze()
    # generate noise-only dataset
    tdi['noise'] = np.copy(tdi['clean'])
    for comb in tdi['clean'].dtype.names[1:]: # loop on the TDI combination names
        tdi['noise'][comb] = tdi['clean'][comb] - tdi['sky'][comb] - tdi['gal'][comb]
    # generate glitch and noiseglitch datasets
    tdi['glitch'] = np.copy(tdi['noisefree'])
    tdi['noiseglitch'] = np.copy(tdi['noise'])
    for comb in tdi['noisefree'].dtype.names[1:]: # loop on the TDI combination names
        # build glitch = noisefree - signal
        tdi['glitch'][comb] = tdi['noisefree'][comb] - tdi['sky'][comb] - tdi['gal'][comb]
        tdi['glitch'][comb][np.isnan(tdi['obs']['X'])] = 0
        # build noiseglitch = noise + glitch
        tdi['noiseglitch'][comb] = tdi['noise'][comb] + tdi['glitch'][comb]
    # generate gapped datsets by zeroing out the gaps    
    if generate_additional_datasets:
        dt = tdi['obs']['t'][1]-tdi['obs']['t'][0]
        gaps = tdi['obs']['t'][np.isnan(tdi['obs']['X'])]
        # find start times of gaps by taking first point and the point for which 
        # the difference between two subsequent samples is more than dt
        gapsstart = np.hstack([np.array([gaps[0]]), (gaps[1:])[gaps[1:]-gaps[:-1]>dt]])
        # same thing with endpoints, but reversed
        gapsend = np.hstack([(gaps[:-1])[gaps[1:]-gaps[:-1]>dt], np.array([gaps[-1]])])
        # get the indexes for the gaps
        gaps_indices = np.vstack([gapsstart, gapsend])
        # set gaps to zero
        # generate gapped dataset
        for ds, dsgap in zip(['clean','noise','sky','noiseglitch'], additional_datasets):
            tdi[dsgap] = np.copy(tdi[ds])
            for comb in tdi['obs'].dtype.names[1:]:
                tdi[dsgap][comb][np.isnan(tdi['obs']['X'])] = 0
    for comb in tdi['obs'].dtype.names[1:]:
        tdi['obs'][comb][np.isnan(tdi['obs']['X'])] = 0
    return tdi

# Convert LDC2 TDI data to orthogonal TDI combinations
def build_orthogonal_tdi(tdi_xyz, skip = 100):
    """
    Builds orthogonal TDi combinations and packs them in a recarray with fields ['t', A', 'E', 'T'].
    
    Parameters:
    -----------
        tdi_xyz: dict or numpy rec-array
            LDC imported data. Format can be either dict containing multiple numpy recarrays 
            or a single numpy rec-array with fields ['t', 'X', 'Y', 'Z']
        skip: integer
            number of samples to skip to remove margin effects
    
    Returns:
    --------
        data: dict or numpy rec-array
            either dict containing multiple numpy recarrays 
            or a single numpy rec-array with fields ['t', A', 'E', 'T'].
    """
    if type(tdi_xyz) is dict:
        data = {}
        for k in tdi_xyz.keys():
            # load tdi A, E, T
            A = (tdi_xyz[k]['Z'][skip:] -   tdi_xyz[k]['X'][skip:])/np.sqrt(2)
            E = (tdi_xyz[k]['X'][skip:] - 2*tdi_xyz[k]['Y'][skip:] + tdi_xyz[k]['Z'][skip:])/np.sqrt(6)
            T = (tdi_xyz[k]['X'][skip:] +   tdi_xyz[k]['Y'][skip:] + tdi_xyz[k]['Z'][skip:])*float(1./np.sqrt(3))

            data[k] = np.rec.fromarrays([tdi_xyz[k]['t'][skip:], A, E, T], names = ['t', 'A', 'E', 'T'])
            for comb in data[k].dtype.names[1:]:
                data[k][comb][np.isnan(data[k][comb])] = 0
    
    elif type(tdi_xyz) is np.ndarray:
        # load tdi A, E, T
        A = (tdi_xyz['Z'][skip:] -   tdi_xyz['X'][skip:])/np.sqrt(2)
        E = (tdi_xyz['X'][skip:] - 2*tdi_xyz['Y'][skip:] + tdi_xyz['Z'][skip:])/np.sqrt(6)
        T = (tdi_xyz['X'][skip:] +   tdi_xyz['Y'][skip:] + tdi_xyz['Z'][skip:])*float(1./np.sqrt(3))

        data = np.rec.fromarrays([tdi_xyz['t'][skip:], A, E, T], names = ['t', 'A', 'E', 'T'])
        for comb in data.dtype.names[1:]:
            data[comb][np.isnan(data[comb])] = 0
    
    return data

#Transform LDC data to Fourier domain
def makeFDdata(data):
    t = data['t']
    del_t = ( t[-1] - t[0] ) / ( len(t) - 1 )
    # print('del_t',del_t)
    if isinstance(data,dict): 
        chans = list(data.keys())
    elif isinstance(data,(np.ndarray,np.recarray)):
        chans = list(data.dtype.fields.keys())
    chans.remove('t')
    # print('channel names are',chans)
    chandata = [ data[ch] for ch in chans ]
    newchans = ['f'] + chans
    # print('newchans',newchans)
    chFT = [np.fft.rfft(data[ch]*del_t).conj() for ch in chans]
    Nf=len(chFT[0])
    df=0.5/del_t/(Nf-1)
    # print(df,1/del_t/len(t))
    fr=np.arange(Nf)*df
    fdata = np.rec.fromarrays([fr]+chFT, names = newchans)
    if t[0]!=0 and False:
        for ch in chans:
            # print(fdata['f'].shape,fdata[ch].shape)
            fdata[ch]*=np.exp(-1j*2*np.pi*fdata['f']*t[0])
    # print(fdata.shape)
    # print(fdata.dtype)
    return fdata      

#Corresponding inverse transform
def makeTDdata(data,t0=0,flow=None):
    print('f0',data['f'][0])
    del_t = 0.5/data['f'][-1]
    if isinstance(data,dict): 
        chans = list(data.keys())
    elif isinstance(data,(np.ndarray,np.recarray)):
        chans = list(data.dtype.fields.keys())
    chans.remove('f')
    #print('channel names are',chans)
    if flow is not None:
        f=data['f']
        window=np.ones_like(f)
        window[f<flow]=(np.cos((f[f<flow]/flow-1)*np.pi)+1)/2
    else: window = 1
    chandata = [ data[ch]*window for ch in chans ]
    newchans = ['t'] + chans
    #print('newchans',newchans)
    tdata = np.rec.fromarrays(ldctools.ComputeTDfromFD(*chandata, del_t), names = newchans)
    tdata['t']+=t0
    #print(fdata.shape)
    #print(fdata.dtype)
    return tdata 

# FFT and PSD evaluation function
def fft_olap_psd(data_array, chan=None, fs=None, navs = 1, detrend = True, win = 'taper', scale_by_freq = True, plot = False):
    '''
    Evaluates one-sided FFT and PSD of time-domain data.
    
    Parameters:
    -----------
        data_array : numpy rec-array 
            time-domain data whose fields are 't' for time-base 
            and 'A', 'E', 'T' for the case of orthogonal TDI combinations.
        channel : str
            single channel name 
        navs : int
            number of averaging segments used to evaluate fft
            *** TO DO: implement navs > 1
        detrend : bool
            apply detrending of data
        scale_by_freq : bool
            scale data by frequency
        plot : bool
            plot comparison between PSD and scipy.signal.welch evaluation
        
    Returns:
    --------
        fft_freq : numpy array
            array of frequencies
        PSD_own : numpy array
            array of PSD
        fft_WelchEstimate_oneSided : numpy array
            array of fft values
        scalefac : float
            scale factor to convert fft^2 values into PSD
    '''
    # Number of segments
    ndata = data_array.shape[0]
    if fs is None:
        if data_array.dtype.names: 
            dt = data_array['t'][1]-data_array['t'][0]
            fs = 1/dt
        else: # assume dt is 5 seconds
            raise ValueError("Specify the sampling frequency of data using keyword fs, e.g. fs = 0.01 ")
    if chan is None:
        if not data_array.dtype.names: 
            data = data_array
        else:
            raise ValueError("Specify a channel name using keyword chan, e.g. chan = 'A' ")
    elif type(chan) is str:
        data = data_array[chan]
            
    overlap_fac = 0.5
    navs = navs
    segment_size = np.int32(ndata/navs) # Segment size = 50 % of data length
    if win == 'hanning':
        window = scipy.signal.hann(segment_size) # Hann window
    elif win == 'blackmanharris':
        window = scipy.signal.blackmanharris(segment_size)
    elif win == 'taper':
        window = scipy.signal.windows.tukey(segment_size, alpha=0.3)
    # signal.welch
    f, Pxx_spec = scipy.signal.welch(data, fs=fs, window=window, 
                        detrend='constant',average='mean',nperseg=segment_size)

    ## Own implementation
    # Number of segments
    overlap_fac = 0.5
    baseSegment_number = np.int32(ndata/segment_size) # Number of initial segments
    total_segments =  np.int32(baseSegment_number + ((1-overlap_fac)**(-1) - 1 ) * (baseSegment_number - 1)) # No. segments including overlap
    overlap_size = overlap_fac*segment_size
    fft_size = segment_size
    detrend = True # If true, removes signal mean
    scale_by_freq = True
    # PSD size = N/2 + 1 
    PSD_size = np.int32(fft_size/2)+1

    if scale_by_freq:
        # Scale the spectrum by the norm of the window to compensate for
        # windowing loss; see Bendat & Piersol Sec 11.5.2.
        S2 = np.sum((window)**2) 
    else:
        # In this case, preserve power in the segment, not amplitude
        S2 = (np.sum(window))**2

    fft_segment = np.empty((total_segments,fft_size),dtype=np.complex64)
    for i in range(total_segments):
        offset_segment = np.int32(i* (1-overlap_fac)*segment_size)
        current_segment = data[offset_segment:offset_segment+segment_size]
        # Detrend (Remove mean value)   
        if detrend :
            current_segment = current_segment - np.mean(current_segment)
        windowed_segment = np.multiply(current_segment,window)
        fft_segment[i] = np.fft.fft(windowed_segment,fft_size) # fft automatically pads if n<nfft
        
    # Add FFTs of different segments
    fft_sum = np.zeros(fft_size,dtype=np.complex128)
    for segment in fft_segment:
         fft_sum += segment

    # Signal manipulation factors      

    # Normalization including window effect on power
    powerDensity_normalization = 1/S2
    # Averaging decreases FFT variance
    powerDensity_averaging = 1/total_segments
    # Transformation from Hz.s to Hz spectrum
    if scale_by_freq:
        powerDensity_transformation = 1/fs
    else:
        powerDensity_transformation = 1
    # assess scale factor    
    scalefac = 2 * powerDensity_averaging * powerDensity_normalization * powerDensity_transformation
    # Make oneSided estimate 1st -> N+1st element
    fft_WelchEstimate_oneSided = fft_sum[0:PSD_size]
    # Convert FFT values to power density in U**2/Hz
    PSD_own = np.square(np.abs(fft_WelchEstimate_oneSided)) * scalefac
    # Generate frequencies
    fft_freq = np.fft.rfftfreq(fft_size, 1/fs)

    if plot:
        fig, ax = plt.subplots(1,1, dpi = 120)
        ax.loglog(fft_freq, (PSD_own), label = 'my own',ls='-')
        ax.loglog(f, (Pxx_spec), label = 'welch',ls='--')
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('Linear spectrum [V RMS]')
        ax.set_title('Power spectrum (sciy.signal.welch)')
        ax.legend()
        ax.set_xlim([f[1], dt/2])
        ax.grid()
    return fft_freq[1:], PSD_own[1:], fft_WelchEstimate_oneSided[1:], scalefac


def generate_freq_data(data, split = False, win='blackmanharris'):
    '''
    Applies `fft_olap_psd` to each group inside data and groups results in a single `numpy.recarray` with the same structure of time-domain data.
    
    Parameters:
    -----------
        data : numpy rec-array 
            time-domain data whose fields are ['t', 'A', 'E', 'T']
        split : bool
            False will output another recarray with fields ['f', 'A', 'E', 'T']
            True will output an ordered ndarray to input to the MBHB search code 
            containing ['f', 'A_real', 'A_imaginary', 'E_real', 'E_imaginary', 'T_real', 'T_imaginary']
        
    Returns:
    --------
        fdata : numpy rec-array  
            frequency domain fft data whose fields are 'f', 'A', 'E', 'T'
        psddata : numpy rec-array 
            frequency domain psd data whose fields are 'f', 'A', 'E', 'T'
        fftscalefac : float
    '''
    if split:
        # tdi label names
        fdata = np.zeros(shape = (7,np.int32(data.shape[0]/2)))
        psddata = np.zeros(shape = (4,np.int32(data.shape[0]/2)))
        tdi = ['A','A','A','E','E','T','T']
        for n in range(7):
            f, psd, fft, fftscalefac = fft_olap_psd(data, chan = tdi[n], win=win)
            if n == 0:
                fdata[n] = f
                psddata[n] = f
            else:
                if n%2!=0:
                    fdata[n] = fft.real
                else:
                    fdata[n] = fft.imag
                    psddata[int(n/2)] = psd
        
        return fdata.T, psddata.T, fftscalefac

    else:
        # tdi label names
        names = data.dtype.names[1:]

        fdata = np.recarray(shape = (np.int32(data.shape[0]/2),), 
                           dtype={'names':('f',)+names, 'formats':[np.float64]+3*[np.complex128]})
        psddata = np.recarray(shape = (np.int32(data.shape[0]/2),), 
                           dtype={'names':('f',)+names, 'formats':4*[np.float64]})
        for tdi in names:
            f, psd, fft, fftscalefac = fft_olap_psd(data, chan = tdi, win=win)
            fdata[tdi] = fft
            psddata[tdi] = psd
        fdata['f']=f
        psddata['f']=f
    
        return fdata, psddata, fftscalefac

def pack_FD_data(array,channels=[ 'A', 'E', 'T' ]):
    '''
    Pack a numpy array to recarray as used in other functions here.

    Arguments:
         array: nd.array
                Unlabeled with columns ordered as [ f, ch1R, ch1I, ch2R, ... ] 
      channels: list of str, default ['A','E'] 
                Names for the result channels
    Returns:
      np.recarray with columns labeled as, eg, [ 'f', 'A', 'E', ...]

    Pack a numpy array with data with columns ordered as 
    [ f, ch1R, ch1I, ch2R, ... ] into a recarray formatted as other functions 
    in here use, with labeled columns and complex-valued channels, and column 
    names, eg [ 'f', 'A', 'E', ...]. If there the number of array columns and
    channel names are not consistent, the names are applied, sequentially until
    they, or the data columns are exhausted.

    '''
    
    if array.shape[1]%2 == 0:
        raise ValueError('Expected an odd number of columns')

    #Add check of data type?

    nchan = ( array.shape[1] - 1 ) // 2
    if nchan > len(channels): nchan = len(channels)
    names=channels[:nchan]
    
    data = np.recarray(shape = (len(array),), 
                       dtype={'names':['f']+names, 'formats':[np.float64]+nchan*[np.complex128]})
    data['f']=array[:,0]
    for ich in range(nchan):
        data[names[ich]]=array[:,1+ich*2]+1j*array[:,2+ich*2]

    return data
    
###### define compare spectra function for time-series
def plot_compare_spectra_timeseries(data, noise_model='spritz', freq_bands = None, fmax = 2e-2, 
                                    tdi_vars = 'orthogonal', labels = ['Signal + Noise','Noise','Signal'], 
                                    save = False, fname = None):
    '''
    A utility for plotting spectral comparisons starting from time-domain data instead
    of frequency-domain data.
    
    Parameters:
    -----------
        data : numpy rec-array 
            time-domain data whose fields are 't' for time-base 
            and 'A', 'E', 'T' for the case of orthogonal TDI combinations.
        noise_model : string
            name of noise model to input to the LDC get_noise_model function.   
        freq_bands : list
            list of frequencies splitting the data into multiple frequency bands to evaluate
            the histogram of fft real and imag part of the whitened data
        fmax : float scalar
            maximum frequency at which we want to cut the analysis
        tdi_vars : string
            key identifying the name of TDI variables under analysis. Default is 'orthogonal', 
            resulting in 'A', 'E', 'T'. Other option is 'from_file', which acquires TDI names
            from time-series file
        labels : list of strings
            labels for data variables
        save : bool
            flag for saving output plot. Defaults to False.
        fname : string
            path and filename to save plot
        
    Returns:
    --------
        grid plot of noise spectra compared to histogram of real and imaginary parts of the fft.
    '''
    # set max value of frequency
    fmax = fmax
    # tdi label names
    if tdi_vars == 'orthogonal': 
        names = ['A', 'E']#, 'T']
    elif tdi_vars == 'from_file':
        names = data.dtype.names[1:]
    # number of channels
    nchan = len(names)
    if freq_bands:
        nfbands = 3
    else:
        nfbands = 1
    # set up labels    
    data_labels = labels
    S={}
    
    # create figure
    fig = plt.figure(figsize=[2*nchan*nfbands,3*nchan],constrained_layout=True, dpi=120)
    # create 3x1 subfigures for each channel
    subfigs = fig.subfigures(nrows = 1, ncols = nchan)
    
    for row, subrow in enumerate(subfigs):
        subrow.suptitle(f'Channel {names[row]}', size = 'xx-large')
        axs = subrow.subfigures(nrows = 3, ncols = 1)
        
        ax = axs[0].subplots(1,1)
        ax.set_xlabel(r"Frequency [Hz]")
        ax.set_ylabel("sqrt(PSD) [1/Hz]") 
        for idx, d in enumerate(data):
            dt = d['t'][1]-d['t'][0]
            f, psd, fft, _ = fft_olap_psd(d, chan = names[row])
            ax.loglog(f[f<fmax], np.sqrt(psd[f<fmax]), label=data_labels[idx])
        if noise_model == 'spritz':
            orbits = lisaorbits.KeplerianOrbits(dt=86400.0, 
                                                L=2500000000.0, 
                                                a=149597870700.0, 
                                                lambda1=0, 
                                                m_init1=0, 
                                                kepler_order=2)
        Nmodel = get_noise_model(noise_model, f, wd=0, orbits=orbits, t_obs=len(d)*dt)
        S = Nmodel.psd(tdi2=True, option=names[row], freq=f, equal_arms=False)
        ax.loglog(f[f<fmax], np.sqrt(S[f<fmax]), label=names[row]+" PSD model")  
        ax.grid()
        ax.legend()
        
        ax = axs[1].subplots(nrows=1, ncols = 3, sharey=True)
        axs[1].suptitle('Real part deviation - whitened data')
        # assess number of bins from noise data
        nbins = int(np.sqrt(len(fft[f<fmax])))
        # create linspace for gaussian noise
        x = np.linspace(-6,6,nbins)
        ax[0].set_ylabel('Count density')
        if freq_bands:
            flims = [f[0]] + freq_bands + [fmax]
            n = 2  # group size
            m = 1  # overlap size
            flim=[flims[i:i+n] for i in range(0, len(flims)-m, n-m)]
        else:
            flim=[f[0],f[-1]]
        for i, a in enumerate(ax):
            fband = np.logical_and(f>=flim[i][0],f<flim[i][1])
            for idx, d in enumerate(data):
                _, _, fft, fft_scalefac = fft_olap_psd(d, chan = names[row])
                # set up scale factor for fft
                scalefac = np.sqrt(2*fft_scalefac)
                a.hist(fft[fband].real*scalefac/np.sqrt(S[fband]),
                     bins = nbins,
                     density = True,
                     label = data_labels[idx])
            a.plot(x, scipy.stats.norm.pdf(x), label='Normal distribution')
            a.grid()
            a.set_xlim([-6, 6])
            a.set_title('{:0.1f}-{:0.1f} mHz'.format((flim[i][0]*1e3),(flim[i][1]*1e3)))

        ax = axs[2].subplots(nrows=1, ncols = 3, sharey=True)
        axs[2].suptitle('Imaginary part deviation - whitened data')
        # assess number of bins from noise data
        nbins = int(np.sqrt(len(fft[f<fmax])))
        # create linspace for gaussian noise
        x = np.linspace(-6,6,nbins)
        ax[0].set_ylabel('Count density')
        if freq_bands:
            flims = [f[0]] + freq_bands + [fmax]
            n = 2  # group size
            m = 1  # overlap size
            flim=[flims[i:i+n] for i in range(0, len(flims)-m, n-m)]
        else:
            flim=[f[0],f[-1]]
        for i, a in enumerate(ax):
            fband = np.logical_and(f>=flim[i][0],f<flim[i][1])
            for idx, d in enumerate(data):
                _, _, fft, fft_scalefac = fft_olap_psd(d, chan = names[row])
                # set up scale factor for fft
                scalefac = np.sqrt(2*fft_scalefac)
                a.hist(fft[fband].imag*scalefac/np.sqrt(S[fband]),
                     bins = nbins,
                     density = True,
                     label = data_labels[idx])
            a.plot(x, scipy.stats.norm.pdf(x), label='Normal distribution')
            a.grid()
            a.set_xlim([-6, 6])
            a.set_title('{:0.1f}-{:0.1f} mHz'.format((flim[i][0]*1e3),(flim[i][1]*1e3)))
        
    if save:
        if fname is None:
            raise ValueError('Missing fname for figure!')
        fig.savefig(fname + '_compare_spectra.png', dpi = 120, bbox_inches='tight', facecolor='white')

        

def get_ldc_gap_mask(data, mode):
    """
    Extracts gap times or indexes from LDC data. 
    Adapted from LDC Spritz analysis notebook https://gitlab.in2p3.fr/LISA/LDC/-/blob/develop/notebooks/LDC2b-Spritz.ipynb
    
    Parameters:
    -----------
        data: dict or numpy rec-array
            LDC imported data. Format can be either dict containing multiple numpy recarrays 
            or a single numpy rec-array with fields ['t', 'X', 'Y', 'Z'] or ['t', 'A', 'E', 'T'] 
        mode: str
            'times': returns gap start and stop times
            'index': returns gap start and stop indexes
    
    Returns:
    --------
        gaps: numpy ndarray
            vstack numpy ndarray containing start times on first row and stop times on second row
    """
    if type(data) is dict:
        data = data[list(data.keys())[0]]
        
    dt = data['t'][1]-data['t'][0]
    fs = 1/dt
    # find all data points included in the gaps
    gaps_data = data['t'][(data[data.dtype.names[1]])==0]
    # find start times of gaps by taking first point and the point for which 
    # the difference between two subsequent samples is more than dt
    gapstarts = np.hstack([np.array([gaps_data[0]]), (gaps_data[1:])[gaps_data[1:]-gaps_data[:-1]>dt]])
    # same thing with endpoints, but reversed
    gapends = np.hstack([(gaps_data[:-1])[gaps_data[1:]-gaps_data[:-1]>dt], np.array([gaps_data[-1]])])
    if mode == 'times':
        # get the times for the gaps
        gaps = np.vstack([gapstarts, gapends])
    elif mode == 'index':
        idxstarts = ((gapstarts - data['t'][0])*fs).astype(int)
        idxends = ((gapends - data['t'][0])*fs).astype(int)
        gaps = np.vstack([idxstarts, idxends])
    return gaps


def construct_gap_mask(n_data,n_gaps=30,gap_length=10,verbose=False,seed=None):
    '''
    Construct a set of gaps which can be applied to gap-less data.

    Returns a dictionary with a mask for the gaps and other info
    '''
    if seed is not None: np.random.seed(seed=seed)
    mask = np.ones(n_data)
    gapstarts = (n_data * np.random.random(n_gaps)).astype(int)
    gapends = (gapstarts+gap_length).astype(int)
    for k in range(n_gaps): mask[gapstarts[k]:gapends[k]]= 0
    if verbose:
        print("Defined gaps:")
        for k in range(n_gaps):
            print("  gap"+str(k),"("+str(gapstarts[k])+":"+str(gapends[k])+")")
    return {'mask':mask,'starts':gapstarts,'ends':gapends}


def view_gaps(ts, ys, yg, 
              maskinfo=None, gapstarts=None, gapends=None, nwing=150, 
              channels=['A', 'E', 'T'], labels=None, 
              histogram = True, noise_model = 'spritz',
              save = False, fname = None):
    '''
    A development utility for making plots of the gap relevant data
    
    Parameters:
        ts : ndarray
            time-base
        ys : list of ndarrays
            original data 
        yg : list of ndarrays
        maskinfo : gap mask
        gapstarts : list
        gapends : list
        nwing : int
        channels : list of strings
        labels : list of strings
        save : bool
        fname : string
        
    Returns:
        ratio (optional)
    '''
    if maskinfo:
        gapstarts = maskinfo['starts']
        gapends = maskinfo['ends']
    ngap=len(gapstarts)
    nchan=len(channels)
    ratio = np.zeros((2,3))
    if histogram:
        rows = 4
    else:
        rows = 2
    # create figure
    fig = plt.figure(figsize=[5*ngap,3*nchan*rows],constrained_layout=True)
    # create 3x1 subfigures for each channel
    subfigs = fig.subfigures(nrows = nchan, ncols = 1)
    for chan, subfig in enumerate(subfigs):
        subfig.suptitle(f'Channel {channels[chan]}', size = 'xx-large')
        # create rows x ngap subplots per subfig
        axs = subfig.subplots(nrows = rows, ncols = ngap, sharey='row')
        for i in range(ngap):
            # assess start and end bounds for the data stretch to plot
            i0 = gapstarts[i]-nwing
            iend = gapends[i]+nwing
            # assign ax for first row with original data
            ax = axs[0][i]
            ax.set_prop_cycle(cycler(color=['tab:blue', 'tab:orange'],linestyle=['-',':']))
            l = 0
            for yi in ys:
                ax.plot(ts[i0:iend],yi[chan][i0:iend], label = labels[l])
                l+=1
            ax.plot(ts[i0:iend],yg[chan][i0:iend], label='gapped', color='tab:cyan', ls='-')
            ax.legend(loc = 'upper right')
            ax.grid()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude')
            # assign ax for second row with gap data
            ax = axs[1][i]
            ax.set_prop_cycle(cycler(color=['tab:purple', 'tab:olive'], linestyle=['-', ':']))
            std = []
            for yi in ys:
                std += [np.std(yi[chan][i0:iend] - yg[chan][i0:iend])]
                ax.plot(ts[i0:iend],yi[chan][i0:iend] - yg[chan][i0:iend])
            if len(ys)>1:
                ax.set_title('std ratio = {:.2f}'.format(std[1]/std[0]))
                ratio[chan][i] = std[1]/std[0]
            if labels is not None: ax.legend(labels=[l+' - gapped ' for l in labels])
            ax.grid()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude')
            
            if histogram:
                for j in range(2,4):
                    # evaluate adherence of PSD to noise model data real part
                    # assign ax for second row with gap data
                    ax = axs[j,i]
                    ax.set_ylabel('Count density')
                    idx = 0
                    fs = 1/(ts[1]-ts[0])
                    for l, yi in enumerate(ys):
                        f, _, fft, fft_scalefac = fft_olap_psd(yi[chan][i0:iend], fs=fs)
                        # assess number of bins from noise data
                        nbins = int(np.sqrt(len(fft)))
                        # create linspace for gaussian noise
                        x = np.linspace(-6,6,nbins)
                        # set up scale factor for fft
                        scalefac = np.sqrt(2*fft_scalefac)
                        # Comparison with LISA Orbits
                        if noise_model == 'spritz':
                            orbits = lisaorbits.KeplerianOrbits(dt=86400.0, 
                                                                L=2500000000.0, 
                                                                a=149597870700.0, 
                                                                lambda1=0, 
                                                                m_init1=0, 
                                                                kepler_order=2)
                        Nmodel = get_noise_model(noise_model, f, wd=0, orbits=orbits, t_obs=len(yi[chan])/fs)
                        S = Nmodel.psd(tdi2=True, option=channels[chan], freq=f, equal_arms=False)
                        if j%2:
<<<<<<< HEAD
                            plotvals=fft.imag*scalefac/np.sqrt(S)
                            ax.set_xlabel('Imag part deviation - whitened data')
                            ax.hist(plotvals,
                                 bins = nbins,
                                 density = True,
                                    label=labels[l])#+' '+str(int(min(plotvals)))+'<x<'+str(int(max(plotvals))))
=======
                            ax.set_xlabel('Imag part deviation - whitened data')
                            ax.hist(fft.imag*scalefac/np.sqrt(S),
                                 bins = nbins,
                                 density = True,
                                 label=labels[l])
>>>>>>> 0f0b7a2 (Edit plotting functions)
                        else:
                            ax.set_xlabel('Real part deviation - whitened data')
                            ax.hist(fft.real*scalefac/np.sqrt(S),
                                 bins = nbins,
                                 density = True,
                                 label=labels[l])
                    ax.plot(x,scipy.stats.norm.pdf(x), label='noise model', color='tab:green')
                    ax.set_xlim([-6, 6])
                    ax.grid()
                    ax.legend()               
    if save:
        if fname is not None:
            fig.savefig(fname + '_gaps.png', dpi = 120, bbox_inches='tight', facecolor='white')
            return ratio
        
def std_ratio_eval(ts, ys, yg, 
              maskinfo=None, gapstarts=None, gapends=None, nwing=100, 
              channels=['A', 'E', 'T']):
    '''
    A development utility for making plots of the gap relevant data
    
    Parameters
        ts
        ys
        yg
        maskinfo
        gapstarts
        gapends
        nwing
        channels
        
    Returns:
        ratio
    
    '''
    if maskinfo:
        gapstarts = maskinfo['starts']
        gapends = maskinfo['ends']
    n=len(gapstarts)
    nchan=len(channels)
    ratio = np.zeros((2,3))
    
    for chan in range(nchan):
        for i in range(n):
            i0 = gapstarts[i]-nwing
            iend = gapends[i]+nwing
            std = []
            for yi in ys:
                std += [np.std(yi[chan][i0:iend] - yg[chan][i0:iend])]
            if len(ys)>1:
                ratio[chan][i] = std[1]/std[0]
    return ratio
            
# Embed the PSD function in a class
# psdmodel is imported from bayesdawn
class LDCModelPSD(psdmodel.PSD):
    '''
    Specialization of the bayesdawn psd model class which connects LDC noise models to lisabeta PSD models.
    
    Parameters
    ----------
    n_data : array_like
        vector of size N_DSP continaing the noise DSP calculated at frequencies
        between -fe/N_DSP and fe/N_DSP where fe is the sampling frequency and N
        is the size of the time series (it will be the size of the returned
        temporal noise vector b)
    fs : scalar integer
        Size of the output time series
    noise_model : scalar float
        sampling frequency
    channel : string
        seed of the random number generator

    Returns
    -------
        bf : numpy array
        frequency sample of the colored noise (size N)
    '''

    def __init__(self, ndata, fs, noise_model, channel, fmin=None, fmax=None):
        # instantiates the PDS estimator from function psdmodel.PSD
        self.noise_model = noise_model
        self.channel = channel
        self.ndata = ndata
        self.fs = fs
        psdmodel.PSD.__init__(self, ndata, fs, fmin=None, fmax=None)
        if fmax is not None:
            self.f = self.f[self.f<fmax]

    def psd_fn(self, x):
        # returns the psd function defined earlier
        tobs = self.ndata / self.fs
        orbits = lisaorbits.KeplerianOrbits(dt=86400.0, 
                                    L=2500000000.0, 
                                    a=149597870700.0, 
                                    lambda1=0, 
                                    m_init1=0, 
                                    kepler_order=2) 
        
        Nmodel = get_noise_model(self.noise_model, x, wd=0, orbits=orbits, t_obs=tobs)
        return Nmodel.psd(tdi2=True, option=self.channel, freq=x, equal_arms=False)           
            

def create_imputation(data, channel, mask, noise_model = 'spritz'):
    '''
    Initialize imputation
    
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 23d04ac (Add functions to create and update imputation)
    Parameters
    ----------
    data : numpy rec-array 
        time-domain data array of gapped data whose fields are 't' for time-base 
        and 'A', 'E', 'T' for the case of orthogonal TDI combinations.
    channel : string
        TDI channel name
    noise_model : string
        noise model for LDC release. Defaults to 'spritz'.
    
    Returns
    -------
        psd_cls : psd cls object
            PSD class used for imputation initialization
        imp_cls : imp cls object
            imputation bayesdawn class
        y_res : numpy array
            residual data after subtraciton of signal from reconstructed data
    '''
    y_masked = data[channel]
    ndata = len(y_masked)
    dt = data['t'][1]-data['t'][0]
    fs = 1.0/dt
    s = np.zeros(len(mask))  # for residual 'signal' is zero
    # Initialize the (reconstructed) data residuals
    y_res = np.squeeze(np.array(mask*(y_masked - s)).T) # (ymasked - s)
    # initialize PSD-0 to the LDC unequal arm noise model
    psd_cls = LDCModelPSD(ndata, fs, noise_model = 'spritz', channel = channel)
    # instantiate imputation class
    imp_cls = datamodel.GaussianStationaryProcess(s, mask, psd_cls, method='nearest', na=100, nb=100)
    # Compute PSD dependent terms
    imp_cls.compute_offline() 
    # Impute missing data for iteration 0
    y_rec = imp_cls.impute(y_res, draw=True)
    # Update the data residuals
    y_res = y_rec - s
    
    return psd_cls, imp_cls, y_res
        
    
def update_imputation(data_rec, imp_cls, channel, fit_type = 'log_spline', fit_dof=15, fmin=7e-6):
    '''
    Update imputation
    
    Parameters
    ----------
    data_rec : numpy rec-array 
        time-domain data array of reconstructed data whose fields are 't' for time-base 
        and 'A', 'E', 'T' for the case of orthogonal TDI combinations.
    imp_cls : imp cls object
            imputation bayesdawn class previously initialized    
    channel : string
        TDI channel name
    fit_type : string
        type of PSD modeling fit to be applied to the reconstructed data
        options are None, 'fit_poly', 'fit_spline', 'fit_logpoly', 'fit_logspline'
    
    Returns
    -------
        psdmod : psd class object
            PSD class obtained as output for the fit
        imp_cls : imp cls object
            imputation bayesdawn class
        y_res : numpy array
            residual data after subtraciton of signal from reconstructed data
    '''
    # Do update PSD
    # FT data residuals
    fd = makeFDdata(data_rec)
    y_res = data_rec[channel]
    s = np.zeros(len(y_res))  # for residual 'signal' is zero
    # Instantiate PSD estimator
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    psdmod = psdmodel.ModelFDDataPSD(data=fd, 
                      channel=channel, 
                      fit_type=fit_type,
                      fit_dof=fit_dof,
                      smooth_df=4e-4,
                      fmin=fmin,
                      offset_log_fit=True)
    sys.stdout = save_stdout
    # Update PSD
    imp_cls.update_psd(psdmod)
    # Re-compute of PSD-dependent terms
    imp_cls.compute_offline()
    # Imputation of missing data by randomly drawing from their conditional distribution
    y_rec = imp_cls.impute(y_res, draw=True)
    # Update the data residuals
    y_res = y_rec - s
    
<<<<<<< HEAD
    return psdmod, imp_cls, y_res
=======
    if figname:
        return data_rec, figname
    else: 
        return data_rec
>>>>>>> 0f0b7a2 (Edit plotting functions)
=======
    return psdmod, imp_cls, y_res
>>>>>>> 23d04ac (Add functions to create and update imputation)
