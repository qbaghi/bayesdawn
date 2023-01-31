# Gap filling investigation support for LISA inference with lisabeta (maybe more generally)
# Time-domain functions implementation by Eleonora Castelli NASA-GSFC 2022
# based on previous version by John Baker NASA-GSFC 2021 

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal
from scipy.stats import norm
from cycler import cycler

from ldc.lisa.noise import get_noise_model
from bayesdawn import datamodel, psdmodel

# Function to print all attributes of hdf5 file recursively
def print_attrs(name, obj):
    shift = name.count('/') * '    '
    print(shift + name)
    for key, val in obj.attrs.items():
        print(shift + '    ' + f"{key}: {val}")
        
# Function to import LDC 2 data and convert them in more convenient format 
def load_tdi_timeseries(fname, 
                        import_datasets = ['obs','clean','sky','noisefree'], 
                        generate_additional_datasets = True,
                        additional_datasets = ['clean_gapped', 'noise_gapped', 'sky_gapped']):
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
    for comb in tdi['clean'].dtype.names[1:]:
        tdi['noise'][comb] = tdi['clean'][comb] - tdi['sky'][comb]
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
        for ds, dsgap in zip(['clean','noise','sky'], additional_datasets):
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
                data[k][comb][np.isnan(data[k]['A'])] = 0
    
    elif type(tdi_xyz) is np.ndarray:
        # load tdi A, E, T
        A = (tdi_xyz['Z'][skip:] -   tdi_xyz['X'][skip:])/np.sqrt(2)
        E = (tdi_xyz['X'][skip:] - 2*tdi_xyz['Y'][skip:] + tdi_xyz['Z'][skip:])/np.sqrt(6)
        T = (tdi_xyz['X'][skip:] +   tdi_xyz['Y'][skip:] + tdi_xyz['Z'][skip:])*float(1./np.sqrt(3))

        data = np.rec.fromarrays([tdi_xyz['t'][skip:], A, E, T], names = ['t', 'A', 'E', 'T'])
        for comb in data.dtype.names[1:]:
            data[comb][np.isnan(data['A'])] = 0
    
    return data

#Transform LDC data to Fourier domain
def makeFDdata(data):
    t = data['t']
    del_t = ( t[-1] - t[0] ) / ( len(t) - 1 )
    #print('del_t',del_t)
    if isinstance(data,dict): 
        chans = list(data.keys())
    elif isinstance(data,(np.ndarray,np.recarray)):
        chans = list(data.dtype.fields.keys())
    chans.remove('t')
    #print('channel names are',chans)
    chandata = [ data[ch] for ch in chans ]
    newchans = ['f'] + chans
    #print('newchans',newchans)
    chFT = [np.fft.rfft(data[ch]*del_t).conj() for ch in chans]
    Nf=len(chFT[0])
    df=0.5/del_t/(Nf-1)
    #print(df,1/del_t/len(t))
    fr=np.arange(Nf)*df
    fdata = np.rec.fromarrays([fr]+chFT, names = newchans)
    if t[0]!=0 and False:
        for ch in chans:
            #print(fdata['f'].shape,fdata[ch].shape)
            fdata[ch]*=np.exp(-1j*2*np.pi*fdata['f']*t[0])
    #print(fdata.shape)
    #print(fdata.dtype)
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
        else: # assume dt is 5 seconds
            raise ValueError("Specify the sampling frequency of data using keyword fs, e.g. fs = 0.01 ")
        fs = 1/dt
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
def plot_compare_spectra_timeseries(data, noise_models, fmax = 2e-2, tdi_vars = 'orthogonal', labels = ['Signal + Noise','Noise','Signal'], save = False, fname = None):
    '''
    A utility for plotting spectral comparisons starting from time-domain data instead
    of frequency-domain data.
    
    Parameters:
    -----------
        data : numpy rec-array 
            time-domain data whose fields are 't' for time-base 
            and 'A', 'E', 'T' for the case of orthogonal TDI combinations.
        noise_models : list of arrays
            PSD of noise model pertaining to the data under analysis. 
            Evaluated via the LDC get_noise_model function.     
        fmax : float scalar
            maximum frequency at which we want to cut the analysis. 
        tdi_vars : string
            Key identifying the name of TDI variables under analysis. Default is 'orthogonal', 
            resulting in 'A', 'E', 'T'. Other option is 'from_file', which acquires TDI names
            from time-series file. **** TO DO: uniform this
        
    Returns:
    --------
        grid plot of noise spectra compared with histogram of real and imaginary parts of the fft.
    '''
    # set max value of frequency
    fmax = fmax
    # tdi label names
    if tdi_vars == 'orthogonal': 
        names = ['A', 'E', 'T']
    elif tdi_vars == 'from_file':
        names = data.dtype.names[1:]
    # set up labels    
    data_labels = labels

    fig, axs = plt.subplots(3,len(noise_models),figsize=[19.2,12],dpi=120)
    for n in names[:len(noise_models)]: 
        ax = axs[0,names.index(n)]
        ax.set_title(r"Channel "+(n))
        ax.set_xlabel(r"Frequency [Hz]")
        ax.set_ylabel("sqrt(PSD) [1/Hz]") 
        idx = 0
        for d in data:
            f, psd, fft, _ = fft_olap_psd(d, chan = n)
            ax.loglog(f[f<fmax], np.sqrt(psd[f<fmax]), label=data_labels[idx])
            idx += 1
        ax.loglog(f[f<fmax], np.sqrt(noise_models[names.index(n)][f<fmax]), label=n+" PSD model")  
        ax.grid()
        ax.legend()

        # evaluate adherence of PDS to noise model data real part
        ax = axs[1,names.index(n)]
        
        # assess number of bins from noise data
        nbins = int(np.sqrt(len(fft[f<fmax])))
        # create linspace for gaussian noise
        x = np.linspace(-6,6,nbins)

    
        ax.set_xlabel('Real part deviation')
        ax.set_ylabel('Count density')
        idx = 0
        for d in data:
            _, _, fft, fft_scalefac = fft_olap_psd(d, chan = n)
            # set up scale factor for fft
            scalefac = np.sqrt(2*fft_scalefac)
            ax.hist(fft[f<fmax].real*scalefac/np.sqrt(noise_models[names.index(n)][f<fmax]),
                 bins = nbins,
                 density = True,
                 label = data_labels[idx])
            idx += 1
        ax.plot(x,norm.pdf(x), label='Normal distribution')
        ax.grid()
        ax.set_xlim([-6, 6])
        ax.legend()
        
        # evaluate adherence of PDS to noise model data real part
        ax = axs[2,names.index(n)]
        ax.set_xlabel('Imag part deviation')
        ax.set_ylabel('Count density')
        idx = 0
        for d in data:
            _, _, fft, fft_scalefac = fft_olap_psd(d, chan = n)
            # set up scale factor for fft
            scalefac = np.sqrt(2*fft_scalefac)
            ax.hist(fft[f<fmax].imag*scalefac/np.sqrt(noise_models[names.index(n)][f<fmax]),
                 bins = nbins,
                 density = True,
                 label = data_labels[idx])
            idx += 1
        ax.plot(x,norm.pdf(x), label='Normal distribution')
        ax.set_xlim([-6, 6])
        ax.grid()
        ax.legend()
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


def detect_glitch_outliers(data, plot = True, threshold = 10):
    """
    Detects glitches as outliers in the data. 
    Adapted from LDC Spritz analysis notebook https://gitlab.in2p3.fr/LISA/LDC/-/blob/develop/notebooks/LDC2b-Spritz.ipynb
    
    Parameters:
        data: numpy rec-array
            LDC imported data. Format can be either dict containing multiple numpy recarrays 
            or a single numpy rec-array with fields ['t', 'X', 'Y', 'Z'] or ['t', 'A', 'E', 'T'] 
            Lowpassed data work better for this purpose.     
        plot: bool
            True or False in order to get a plot of the detected outliers.
    
    Returns:
        peaks: numpy ndarray
            indexes of all detected outliers from the width of the distribution.
    """
    n = data.dtype.names[1]
    gaps = get_ldc_gap_mask(dataobs, mode='index')
    
    mad = scipy.stats.median_abs_deviation(data[n])
    median = np.median(data[n][0:gaps[0][0]])
    maxval = np.max(np.abs(data[n]))
    peaks, properties = scipy.signal.find_peaks(np.abs(data[n]), height=threshold*mad, threshold=None, distance=1)

    if plot:
        fig, ax = plt.subplots(1,1,figsize=(8,4),dpi=100)
        ax.plot(data['t'], data[n], label='Filtered data')
        ax.vlines(data['t'][peaks], ymin=-1.1*maxval, ymax=1.1*maxval, color='red', linestyle='dashed', label='Detected outliers')
        ax.set_ylabel('TDI'+n)
        ax.set_xlabel('Time [s]')      
        ax.legend()
        ax.grid()

    return peaks


def mask_glitches(data, peaks, glitchnum):
    """
    Extracts gap times or indexes from LDC data. 
    Adapted from LDC Spritz analysis notebook https://gitlab.in2p3.fr/LISA/LDC/-/blob/develop/notebooks/LDC2b-Spritz.ipynb
    
    Parameters:
        data: dict or numpy rec-array
            LDC imported data. Format can be either dict containing multiple numpy recarrays 
            or a single numpy rec-array with fields ['t', 'X', 'Y', 'Z'] or ['t', 'A', 'E', 'T'] 
        peaks: numpy ndarray
            indexes of all detected outliers from the width of the distribution
        glitchnum: int
            expected number of glitches (in order to cut out outliers coming from the GW signal)
    
    Returns:
        gaps: numpy ndarray
            vstack numpy ndarray containing start times on first row and stop times on second row
    """
    data_mask = np.copy(data)
    glitchlen = int(data['t'][peaks[1]] - data['t'][peaks[0]])

    if type(data) is dict:
        for k in data_mask.keys():
            for tdi in data_mask[k].dtype.names[1:]:
                for pk in peaks[:2*glitchnum]:
                    data_mask[k][tdi][pk-glitchlen:pk+glitchlen] = 0.0
    else:
        print(data_mask.dtype)
        for tdi in data_mask.dtype.names[1:]:
            print(tdi)
            for pk in peaks[:2*glitchnum]:
                data_mask[tdi][pk-glitchlen:pk+glitchlen] = 0.0

    return data_mask

def view_gaps(ts, ys, yg, 
              maskinfo=None, gapstarts=None, gapends=None, nwing=100, 
              channels=['A', 'E', 'T'], labels=None, save = False, fname = None):
    '''
    A development utility for making plots of the gap relevant data
    
    Parameters:
        ts
        ys
        yg
        maskinfo
        gapstarts
        gapends
        nwing
        channels
        labels
        save
        fname
        
    Returns:
        ratio (optional)
    '''
    if maskinfo:
        gapstarts = maskinfo['starts']
        gapends = maskinfo['ends']
    n=len(gapstarts)
    nchan=len(channels)
    ratio = np.zeros((2,3))
    # create figure
    fig = plt.figure(figsize=[5*n,4*nchan*2],constrained_layout=True)
    #     fig, axs = plt.subplots(nchan*2,n,figsize=[6.4*n,4.8*nchan*2],squeeze=False)
    # create 3x1 subfigures for each channel
    subfigs = fig.subfigures(nrows = nchan, ncols = 1)
    for chan, subfig in enumerate(subfigs):
        subfig.suptitle(f'Channel {channels[chan]}', size = 'xx-large')
        # create 2xn subplots per subfig
        axs = subfig.subplots(nrows = 2, ncols = n, sharey=True)
        for j in range(nchan):
            for i in range(n):
                i0 = gapstarts[i]-nwing
                iend = gapends[i]+nwing
                ax = axs[j][i]
                if j==0:
                    ax.set_prop_cycle(cycler(color=['tab:blue', 'tab:green'],linestyle=['-',':']))
                    l = 0
                    for yi in ys:
                        ax.plot(ts[i0:iend],yi[chan][i0:iend], label = labels[l])
                        l+=1
                    ax.plot(ts[i0:iend],yg[chan][i0:iend], label='gapped data', color='tab:orange', ls='-')
                    ax.legend(loc = 'upper right')
                else:
                    ax.set_prop_cycle(cycler(color=['tab:purple', 'tab:cyan'], linestyle=['-', ':']))
                    std = []
                    for yi in ys:
                        std += [np.std(yi[chan][i0:iend] - yg[chan][i0:iend])]
                        ax.plot(ts[i0:iend],yi[chan][i0:iend] - yg[chan][i0:iend])
                    if len(ys)>1:
                        ax.set_title('std ratio = {:.2f}'.format(std[1]/std[0]))
                        ratio[chan][i] = std[1]/std[0]
                    if labels is not None: ax.legend(labels=['gap = '+l+' - gapped data ' for l in labels])
                ax.grid()
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Ampitude []')
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
        psdmodel.PSD.__init__(self, ndata, fs, fmin=None, fmax=None)
        if fmax is not None:
            self.f = self.f[self.f<fmax]

    def psd_fn(self, x):
        # returns the psd function defined earlier
        tobs = ndata / fs
        orbits = lisaorbits.KeplerianOrbits(dt=cfg['dt_orbits'], 
                                    L=cfg['nominal_arm_length'], 
                                    a=149597870700.0, 
                                    lambda1=0, 
                                    m_init1=0, 
                                    kepler_order=cfg['kepler_order']) 
        
        Nmodel = get_noise_model(self.noise_model, x, wd=0, orbits=orbits, t_obs=tobs)
        return Nmodel.psd(tdi2=True, option=self.channel, freq=x, equal_arms=False)           
            

def LDC_imputation(data_masked, maskinfo, psd_correction, names = ['A', 'E', 'T'], figname = None):
    # create empty arrays for the imputation
    imp_cls = []
    psd_cls = []
    y_res = []
    # set up flags and variables
    mask = maskinfo['mask']
    psd_correction = False
    data_rec = data_masked.copy()

    # instantiate the PSD noise class
    if psd_correction:
        if figname: figname = figname + 'corrected' 
        for tdi in names:
            psd_cls.append(LDCCorrectedModelPSD(ndata, fs, noise_model = 'spritz', channel = tdi, polyfit = poly))
    else:
        if figname: figname = figname + 'original' 
        for tdi in names:
            psd_cls.append(LDCModelPSD(ndata, fs, noise_model = 'spritz', channel = tdi))#, fmax = 15e-3))    

    # Perform data imputation
    ### NB this can be streamlined a little bit more and/or transformed into a function 
    for tdi in range(len(names)):
        y_masked = data_masked[names[tdi]]
        s = np.zeros(len(mask))  #for residual 'signal' is zero
        # instantiate imputation class
        imp_cls += [datamodel.GaussianStationaryProcess(s, mask, psd_cls[tdi], method='PCG', na=50*fs, nb=50*fs)]
        # Initialize the (reconstructed) data residuals
        y_res = np.squeeze(np.array(y_masked).T) # (ymasked - s)   
        t1 = time.time()
        # Re-compute of PSD-dependent terms
        imp_cls[tdi].compute_offline()
        # Imputation of missing data by randomly drawing from their conditional distribution
        y_res = imp_cls[tdi].impute(y_masked, draw=True)
        # Update the data residuals
        t2 = time.time()
        print("The imputation / PSD estimation for combination " + names[tdi] + " in iteration "+ str(i) +" took " + str(t2-t1))
        data_rec[names[tdi]] = y_res
    
    if figname:
        return data_rec, figname
    else: 
        return data_rec


#The PSD model class
class ModelFDDataPSD(psdmodel.PSD):
    '''
    Specialization of the bayesdawn psd model class which provides a bayesdawn PSD model 
    build on a possible combination of analytic (LDC) noise models possibly with smoothing 
    and/or additional data fitting.
    
    Parameters
    ----------
    data : numpy rec-array (or dict)
        Multichannel Fourier data set [named columns 'f','A','E',...
    channel : str
        Tag specifying the channel to use for this instance, eg 'A', 'E'
    fit_type :  str = 'poly' 
        Tag for the type of model fitting to apply, or None to use the LDC model.
        Supported ['poly', 'log_poly', None]
    fit_dof : int [default 4]
        Number of degrees of freedom in the fit.
    fit_logx : bool [default True]
        If True use log(f) as the coordinate for fitting functions.
    noise_model : str [default 'spritz']
        Tag for the LDC base model to use for ratio, or None for no ratio.
    smooth_df: float [default None]
        Freq width scale to set number of points for smoothing
    fmin: float [default 1e-5]
        Minimum frequency for result
    fmax: float [default None]
        Maximum frequency for result
        
        
    Returns
    -------
    bayesdawn.psfmodel.PSD object
    
    For the specified channel analytic PSD model will be generated. If noise_model is not None,
    then the model will be generated by a ratio against the specified analytic noise model.
    
    If smooth_df is given, then the analytic noise model will be smoothed, by convolution with a
    Hanning window. 
    
    The type of fitting is specifed by fit_type, with 'poly' providing a fit using np.polyfit with 
    polynomial degree fit_dof-1. A variation on this is 'log_poly' which fits the ratio of the log 
    of the two quantities, scaled by the sum of their max values. Other methods may be added. 
    
    If fit_type is None, then the analytic model (possibly smoothed) is applied directly and only 'f' 
    will be used from the data.
    
    Note 
    ----
    
    Long-term, it probably makes more sense to fold the functionality here into the underlying
    bayesdawn.psdmodel.PSD code, which already includes a MCMC sampled spline fit for the PSD. Missing
    there is the crutch we use here for starting from, and scaling against an approximate analytic model.
    '''

    def __init__(self, data, channel, fit_type='poly',fit_dof=4, fit_logx=True, noise_model='spritz', smooth_df=None, fmin=1e-5, fmax=None):
    
        self.channel = channel
        self.fit_type = fit_type
        self.chdata=data[channel]

        f=data['f']
        df=(f[-1]-f[0])/(len(f)-1) 
        self.df=df
        
        fs=f[-1]*2
        ndata=(len(f)-1)*2
        self.ndata=ndata
        
        if fmax is not None:
            self.chdata=self.chdata[f<=fmax]
            f = f[f<=fmax]
        if fmin is not None:
            self.chdata=self.chdata[f>=fmin]
            f = f[f>=fmin]
            
        self.fin=f.copy()
        
        psdmodel.PSD.__init__(self, ndata, fs, fmin=fmin, fmax=fmax)
        
        self.scalefac=2*self.df
        
        if noise_model is not None:
            Nmodel=get_noise_model(noise_model, f)
            Sinit = Nmodel.psd(tdi2=True, option=self.channel, freq=f)
            if smooth_df is not None:
                nsmooth=int(smooth_df/df)
                w=np.hanning(nsmooth)
                smooth=lambda s:np.convolve(w/w.sum(),s,mode='same')
                Sinit=smooth(Sinit)
        else: 
            Sinit=None
            if fit_type is None: raise ValueError("Must specify at least fit_type or noise_model")
        self.Sinit=None
        if Sinit is not None:
            self.logSinit=scipy.interpolate.interp1d(np.log(f),np.log(Sinit),fill_value="extrapolate")
            self.Sinit=lambda x:np.exp(self.logSinit(np.log(x)))
        
        self.fit=None
        if fit_type is None: return
        self.fit_dof=fit_dof
        
        self.fit_logx=fit_logx
        if fit_logx: x=np.log(f)
        else: x=f
    
        if not fit_type.startswith('log_'):
            print('Fitting on linear-scale')
            if fit_type=='poly' or fit_type=='spline':
                y=np.abs(self.chdata)**2*self.scalefac
                
                if self.Sinit is not None:
                    #print('Sinit min/max',min(Sinit),max(Sinit))
                    y=y/self.Sinit
        else:
            print('Fitting on log-scale')
            if fit_type=='log_poly' or fit_type=='log_spline':
                y=np.abs(self.chdata)**2*self.scalefac
                if self.Sinit is not None:
                    zero=2*(max(y)+max(Sinit))
                    #print('Sinit min/max',min(Sinit),max(Sinit))
                    y=np.log(y/zero)/np.log(self.Sinit(self.fin)/zero)
                else:
                    zero=2*(max(y))
                    y=np.log(y/zero)
                self.fit_zero=zero
                
        if fit_type=='poly' or fit_type=='log_poly':              
            pf = np.polyfit(x,y,fit_dof+1,w=1/f)
            #print('poly fit:',pf)
            self.fit = np.poly1d(pf)
        elif fit_type=='spline' or fit_type=='log_spline':
            # I tried using the functionality in psdmodel.py but couldn't get it working.
            # to stay close to the existing implementation in psdmodel.py                    
            print(f[0],'< f < ',f[-1],'fmin/fmax=',fmin,fmax)
            n_knots=fit_dof-2 #dof=n_knots+2 (maybe, sort of guessing)                    
            knots=self.choose_knots(f) #This function needs f, not x
            xknots=knots
            if fit_logx: xknots=np.log(knots) #transform to x-space if needed
            fitspline=scipy.interpolate.LSQUnivariateSpline(x, y, xknots[1:-1], w=1/f, k=3, ext=3, check_finite=False)                    
            print('knots',knots)
            self.knots=knots
            self.fit=fitspline
        else:    
            raise ValueError('fit_type '+str(fit_type)+' not recognized.')
            

            
    def choose_knots(self,x):
        '''
        Arguments:
                  x : Numpy array
                      Data coordinates for which the spline will be applied. These are used for setting the 
                      spline range and for checking the suitability of the knots.
                 
        Returns:
             Numpy array with the ordered knots
             
        For spline data we want to set the location of the knots. There are diverse ways to do this
        of course.  Here we follow the scheme in bayesdawn.psdmodel (fixed).  For B-splines to work
        the knots have to meet a constraint, dependent on the data.  From the scipy documentation
        for LSQUnivariateSpline:
           "Knots t must satisfy the Schoenberg-Whitney conditions, i.e., there must be a subset of 
            data points x[j] such that t[j] < x[j] < t[j+k+1], for j=0, 1,...,n-k-2."
        If the conditions aren't met, knots will be dropped until the condition is met, this may result
        in an actual number of knots less than desired to realize the value of fit_dof. 
        [We could add a interative loop to try to improve that.]
        '''
        print('Finding knots in ',x[0],'< x <',x[-1])
        nknots=self.fit_dof #Approx guess of the relation of knots to dof
        t=psdmodel.choose_frequency_knots(nknots+2, freq_min=x[0], freq_max=x[-1])
        t=t[1:-1] #drop the boundaries (scipy will put them in)
        print(len(t),'knots before checking')
        
        #Now we check the Schoenberg-Whitney condition
        j=0
        k=3
        iscan=0
        while j+k+1<len(t):
            #First scan to the relevant part of the data
            while x[iscan]<=t[j] and iscan<len(x):iscan+=1
            #Now look for a data point in the window
            i=iscan
            #check whether the next point exists and is in the target range
            if iscan>=len(x) or x[iscan]>=t[j+k+1]: 
                #fails the condition. To fix, we remove an intermediate knot and try again
                jkill=j+k//2+1
                print('Dropping knot at ',t[jkill],'because no data in',t[j],'< t <',t[j+k+1])
                t=np.delete(t,jkill)
            else:
                j+=1
        print(len(t),'knots after ensuring condition.')
        return t
    
    def plot(self,show_fit=False, tag=None):
        
        if tag is None: 
            tag=self.fit_type
        if tag is None:
            tag='no fit'
            
        plt.loglog(self.fin,np.abs(self.chdata*np.sqrt(self.scalefac)),label='data ('+tag+')',alpha=0.3)
        plt.loglog(self.fin,np.sqrt(self.psd_fn(self.fin)),label='model ('+tag+')')
        plt.legend()                
            
        if show_fit and self.fit is not None and self.Sinit is not None:
            if self.fit_logx: x=np.log(self.fin)
            else: x=self.fin
            
            y=np.abs(self.chdata)**2*self.scalefac
            if not self.fit_type.startswith('log_'):
                if self.Sinit is not None:
                    #print('Sinit min/max',min(Sinit),max(Sinit))
                    y=y/self.Sinit(self.fin)
            else:
                zero=self.fit_zero
                y=np.log(y/zero)
                if self.Sinit is not None:
                    y=y/(self.logSinit(np.log(self.fin))-np.log(zero))
                    
            print('x',x[:5])
            print('y',y[:5])
            print('fit',y[:5])
                
            spline=self.fit_type.endswith('spline')
                
            plt.show()
            plt.semilogy(self.fin,(y),label='data ratio ('+tag+')')
            plt.semilogy(self.fin,self.fit(x),label='fit ('+tag+')')
            if spline:
                xknots=self.knots
                if self.fit_logx: xknots=np.log(xknots)
                
                plt.semilogy(self.knots,self.fit(xknots),'*',label='knots ('+tag+')')
            plt.legend()
            plt.show() 
            if False:
                plt.loglog(self.fin,(y),label='data ratio ('+tag+')')
                plt.loglog(self.fin,self.fit(x),label='fit ('+tag+')')
                plt.legend()
                plt.show() 

    def psd_fn(self, x):
        # returns the psd function defined earlier           

        if self.fit is not None:
            xx=x.copy()
            if self.fit_logx:
                xx=np.log(x)
            if not self.fit_type.startswith('log_'):
                dm = np.abs(self.fit(xx))            
                if self.Sinit is not None:
                    dm = dm*self.Sinit(x)
            else:
                logdm = self.fit(xx)
                zero=self.fit_zero
                if self.Sinit is not None:
                    logdm*=(self.logSinit(np.log(x))-np.log(zero))
                dm=zero*np.exp(logdm)
        else:
            dm = self.Sinit(x)
        return dm
