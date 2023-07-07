# Gap filling investigation support for LISA inference with lisabeta (maybe more generally)
# Time-domain functions implementation by Eleonora Castelli NASA-GSFC 2022
# based on previous version by John Baker NASA-GSFC 2021 

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from bayesdawn.connect import fillgaps

def BH_lowpass(data, t_win=100,t_sam=5,fs=10):
    """
    BH_lowpass, lowpass data convolving with a BH92 windowing function.
    data:  tuple of synchronously sampled time series
    t_win: window length, controls cut frequency
    t_sam: output sampling time
    fs:    input sampling frequency
    L Sala, December 2021
    modified by E Castelli, Sept 2022
    """
    names = data.dtype.names[1:]
    dt = 1/fs
    step_win = np.intc(t_win*fs)
    step_sam = np.intc(t_sam*fs)
    assert t_sam>=dt, 'Watch out, do not upsample your data.'
    assert np.isclose(t_sam*fs,int(t_sam*fs),rtol=1e-5), 'Downsampling time must be multiple of sampling time.'
    assert np.isclose(t_win*fs,int(t_win*fs),rtol=1e-5), 'Windowing time must be multiple of sampling time.'
    assert np.isclose(step_win/step_sam,int(step_win/step_sam),rtol=1e-5), 'Watch out, t_win must be multiple of t_sam.'
    
    dtarr = np.diff(data['t'])
    assert np.isclose(dtarr[0],dt,rtol=1e-5), 'Aaargh, sampling frequency is not consistent with data.' #just check fs
    assert np.allclose(dtarr,dt,rtol=1e-5), 'Aaargh, your data are not equally sampled in time.' #just check sampling time

    BHfilt = BH92(step_win) #build filter
    BHarea = np.sum(BHfilt)
    BHfilt = BHfilt/BHarea
    onearray = np.ones(step_win)/step_win

    #apply filter convolving
    outts = [np.convolve(data['t'],onearray,mode='valid')] #just a simple way to get times, computationally more expensive than linspace, but safer
    for tdi in names:
        outts += [np.convolve(data[tdi], BHfilt,  mode='valid')]
    #downsample it
    for i in range(len(outts)):
        outts[i] = outts[i][::step_sam]
    datalp = np.rec.fromarrays(outts, names = ['t', 'A', 'E', 'T'])
    return datalp

def BH92(M:int):
    z = np.arange(0,M)*2*np.pi/M
    return 0.35875 - 0.48829 * np.cos(z) + 0.14128 * np.cos(2*z) - 0.01168 * np.cos(3*z)

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
    gaps = fillgaps.get_ldc_gap_mask(data, mode='index')
    
    mad = scipy.stats.median_abs_deviation(data[n])
    to=len(data)
    if len(gaps>0):to=gaps[0][0]
    median = np.median(data[n][0:to])
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


def mask_glitches(data, peaks, glitchnum, n=1, longglitch=False):
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
        n: int
            multiplicative factor for the glitch data length
    
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
                    if longglitch:
                        data_mask[k][tdi][pk-n//50*glitchlen:pk+n*glitchlen] = 0.0
                    else:
                        data_mask[k][tdi][pk-n*glitchlen:pk+n*glitchlen] = 0.0
    else:
        print(data_mask.dtype)
        for tdi in data_mask.dtype.names[1:]:
            print(tdi)
            for pk in peaks[:2*glitchnum]:
                if longglitch:
                    data_mask[tdi][pk-n//50*glitchlen:pk+n*glitchlen] = 0.0
                else:
                    data_mask[tdi][pk-n*glitchlen:pk+n*glitchlen] = 0.0

    return data_mask
