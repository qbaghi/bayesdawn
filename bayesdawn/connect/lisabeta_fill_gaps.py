#Gap filling investigation support for LISA inference with lisabeta (maybe more generally)
#John Baker NASA-GSFC 2021 
import numpy as np
import matplotlib.pyplot as plt
from bayesdawn import datamodel, psdmodel
import lisabeta.lisa.pyLISAnoise as pyLISAnoise
from scipy.stats import norm

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

def reconstruct_time_domain(data,nchan):
    '''
    Transform data from Fourier domain (as implemented) to time domain.
    '''
    #Assume that fs are cell-centered labels of data elements
    #and evenly spaced in freq, over some band
    #assume data cols are freq, ch1.real, ch1.imag, ch2.real,... 
    fs=data[:,0]
    nd=len(data)
    df=(fs[-1]-fs[0])/(nd-1)
    nf=nd+int(fs[0]/df-.5)
    #print(fs[0],'< f <',fs[-1],' nd=',nd,' nf=',nf,'df=',df)
    #print('nd,nf,nf-nd',nd,nf,nf-nd)
    #print('f0,df,f0/df',fs[0],df,fs[0]/df)
    fftfs=df*np.arange(nf)
    nt=2*(nf-1)
    dt = 1/(fs[-1]*2)
    #print('nt,dt:',nt,dt)
    #tscalefac=np.sqrt(.5)/nt/df
    tscalefac=1/dt
    #print('recon scale:',tscalefac,nt,df)
    #Construct time-domain data
    #note y is the inverse-fft which is time-domain channel*dt in our usage
    y=[]
    for i in range(nchan):
        bufdata=1j*np.zeros(nf)
        bufdata[nf-nd:]=data[:,1+i*2]+1j*data[:,2+i*2]
        ##set ends to vanish
        #bufdata[0]=0;bufdata[-1]=0
        y += [np.fft.irfft(bufdata)*tscalefac]
        #print('y['+str(i)+'],shape=',y[i].shape)
        
    return y

def construct_specialized_data(yt,nchan,df,f0):
    '''
    Transform from time domain to Fourier domain (as used here)

    Should be effectively inverse of reconstruct_time_domain
    '''
    #Assume that fs are cell-centered labels of data elements
    #and evenly spaced in freq, over some band
    #assume data cols are freq, ch1.real, ch1.imag, ch2.real,... 
    nt=len(yt[0])
    nf=nt//2+1
    ioff=int(f0/df-.5)
    nd=nf-ioff
    fs=np.arange(nd)*df+f0
    #tscalefac=np.sqrt(.5)/nt/df
    #tscalefac=1/dt='fs'
    tscalefac=2*fs[-1]
    #print('con scale:',tscalefac,nt,df)
    yf=np.zeros((nd,1+nchan*2))
    yf[:,0]=fs
    for i in range(nchan):
        #print('len(yt)',len(yt[i]))
        yfft=np.fft.rfft(yt[i]/tscalefac)
        #print('nd,ioff,len(yfft[ioff:])',nd,ioff,len(yfft[ioff:]))
        yf[:,1+2*i] += yfft[ioff:].real
        yf[:,2+2*i] += yfft[ioff:].imag
        
    return yf

def view_gaps(ts,ys,maskinfo,nwing=20,labels=None):
    '''
    A development utility for making plots of the gap relevant data
    '''
    gapstarts=maskinfo['starts']
    gapends=maskinfo['ends']
    n=len(gapstarts)
    nchan=len(ys[0])
    fig, axs = plt.subplots(nchan*2,n,figsize=[6.4*n,4.8*nchan*2],squeeze=False)
    for i in range(n):
        i0=gapstarts[i]-nwing
        iend=gapends[i]+nwing
        for j in range(nchan):
            ax=axs[j*2,i]         
            for yi in ys:
                ax.plot(ts[i0:iend],yi[j][i0:iend].real)
            ax.plot(ts[i0:iend],np.abs(ys[0][j][i0:iend]))
            if labels is not None: ax.legend(labels=labels)
            ax=axs[j*2+1,i]
            for yi in ys[1:]:
                ax.plot(ts[i0:iend],yi[j][i0:iend].real-ys[0][j][i0:iend].real)
            if labels is not None: ax.legend(labels=[l+' - '+labels[0] for l in labels[1:]])
    plt.show()

def plot_compare_spectra(datasets,LISAnoise=None,nchan=3,labels=None,fs=None,PSDset=None):
    '''
    A utility for plotting spectral comparisons
    '''
    if fs is None:
        #print('plot_compare_spectra: Assuming list of np data.')
        fs=datasets[0][:,0]
        pdata=[]
        for dataset in datasets:
            pdatai=[]
            for ich in range(nchan):
                pdatai+=[dataset[:,ich*2+1]+1j*dataset[:,ich*2+2]]
            pdata+=[pdatai]
    else:
        #print('plot_compare_spectra: Assuming list of list-packed data on common freq grid.')
        pdata=datasets
    if labels is None:
        labels=[]
        for i in range(len(pdata)):labels.append('Set '+str(i+1))
    resc=False
    noises=[]
    for ich in range(nchan):
        if LISAnoise is not None:
            noises += [pyLISAnoise.LISANoisePSDFunction(LISAnoise,ich+1,TDIrescaled=resc).apply(fs)]
        elif PSDset is not None:
            noises += [PSDset[ich].calculate(fs)]
        else:
            raise ValueError['Must provide either LISAnoise or PSDset']
    fig, axs = plt.subplots(3,3,figsize=[19.2,14.4])
    nbins=int(np.sqrt(len(fs)))
    x=np.linspace(-6,6,nbins)
    for ich in range(nchan): 
        ax=axs[0,ich]
        ax.set_title(r"Channel "+str(ich+1))
        ax.set_xlabel(r"Frequency [Hz]")
        ax.set_ylabel("sqrt(PSD [1/Hz])")
        scalefac=np.sqrt(4*(fs[1]-fs[0]))
        iset=0
        for pdataset in pdata:
            iset=iset+1
            ax.loglog(fs, np.abs(pdataset[ich])*scalefac , label=labels[iset-1])
        ax.loglog(fs, np.sqrt(noises[ich]), label="PSD model")
        ax.legend()
        nn=len(fs)
        nbins=int(np.sqrt(nn))
        stds=np.zeros(len(pdata))
        for i in range(len(pdata)):
            stds[i]=np.std(pdata[i][ich][1:-1].real)
            stds[i]+=np.std(pdata[i][ich][1:-1].imag)
        #print('chan '+str(ich)+' data set stds:',stds)
        largesets=[pdata[i] for i in range(len(pdata)) if stds[i]/stds.mean()>0.001]                                             
        ax=axs[1,ich]
        for dataset in largesets:
            xx=dataset[ich][1:-1].real*scalefac/np.sqrt(noises[ich][1:-1])
            imax=np.argmax(xx)
            #print('i max:',imax,xx[imax])
            ax.hist(dataset[ich][1:-1].real*scalefac/np.sqrt(noises[ich][1:-1]),bins=nbins,density=True)
        ax.set_xlabel("real part deviation")
        ax.set_ylabel("count density")
        ax.legend(labels=[labels[i] for i in range(len(pdata)) if stds[i]/stds.mean()>0.001])
        ax.plot(x,norm.pdf(x))
        ax=axs[2,ich]
        for dataset in largesets:ax.hist(dataset[ich][1:-1].imag*scalefac/np.sqrt(noises[ich][1:-1]),bins=nbins,density=True)
        ax.set_xlabel("imag part deviation")
        ax.set_ylabel("count density")
        ax.plot(x,norm.pdf(x))

class bdPSDmodel(psdmodel.PSD):
    '''
    Specialization of the bayesdawn psd model class which connects to lisabeta PSD models
    '''

    def __init__(self, n_data, df, chan, LISAnoise, TDIrescaled=False,f0=0,dered_f0=0,dered_pow=2):
        psdmodel.PSD.__init__(self, n_data, df, fmin=None, fmax=None)
        self.LISAnoise=LISAnoise
        self.chan=chan+1
        self.TDIrescaled=TDIrescaled
        self.f0=f0-df
        self.df=df
        self.i0=int(f0/df+0.5)
        self.dered_f0=dered_f0
        self.dered_pow=dered_pow
        
    def psd_fn(self, x):
        #print('Calling pyLISAnoise with:\n  LISAnoise=',self.LISAnoise,"\n  chan=",self.chan,"\n  TDIrescaled=",resc,"\n  x=",x)
        
        xx=x
        psd = pyLISAnoise.LISANoisePSDFunction(self.LISAnoise,self.chan,TDIrescaled=self.TDIrescaled).apply(xx)
        #print('x.shape,psd.shape:',x.shape,psd.shape)        
        #psd[:self.i0]=psd[self.i0]
        if self.dered_f0>0:
            cut=self.dered_f0
            psd[xx<cut]*=(xx[xx<cut]/cut)**self.dered_pow #hack the low-f fall-off test
        return psd

def create_imputation(gapinfo,psd,nchan,method=None):
    '''
    Function to define a set of imputation models for multi-channel data

    gapinfo: a dict including gapinfo['mask']=gap_mask
    '''
    imp=[]
    mask=gapinfo['mask']
    for i in range(nchan):
        s=np.zeros(len(mask))  #for residual 'signal' is zero
        if method is not None and method=='woodbury':
            imp += [datamodel.GaussianStationaryProcess(s, mask, psd[i], method='woodbury')]
        elif method is None:
            imp += [datamodel.GaussianStationaryProcess(s, mask, psd[i], na=60, nb=60)]
        else: raise ValueError("Not recognized: method="+method)
        # perform offline computations
        imp[i].compute_offline()
        # If you want to update the deterministic signal (the mean of the Gaussian process)
        #imp_cls[i].update_mean(s)
        # If you want to update the PSD model
        #imp_cls[i].update_psd(psd_cls)
    return imp

def update_imputation(gapinfo,imp,nchan,resid_data,nfuzz=0,psd=None,verbose=False):
    '''
    Function to apply a set of imputation models to  multi-channel Fourier-domain residual data.
    '''
    #Expecting residual data in the format (eg for nchan=3)
    #[[f0,Ar0,Ai0,Er0,Ei0,Tr0,Ti0],[f1,Ar1,Ai1,...],...]
    #print('resid_data=',resid_data)
    mask=gapinfo['mask']
    resid=resid_data.copy()
    fs=resid[:,0]
    nd=len(resid)
    df=(fs[-1]-fs[0])/(nd-1) 
    #Optionally 'fuzz' the edges of the FD data before transforming
    #Based on the PSD
    #n_fft=sqrtS*n_fft*[np.sqrt(n_data*fs/4)] []-->[nt/2*sqrt(df)]
    if nfuzz>0 and psd is not None:
        nf=nd+int(fs[0]/df-.5)
        #print('Applying fuzz of size',nfuzz,'below f=',fs[nfuzz],'and above f=',fs[nf-nfuzz])
        nt=2*nf-1
        scale=nt*np.sqrt(df)/2/(fs[-1]*2)
        lowpsd=np.zeros((nfuzz,2*nchan))
        for i in range(nchan):
            lowpsd[:,2*i]=psd[i].psd_fn(np.arange(nfuzz)*df)
            lowpsd[:,2*i+1]=lowpsd[:,2*i]
        lowfuzz=np.random.normal(size = (nfuzz,2*nchan))
        lowfuzz*=np.sqrt(lowpsd)*scale
        dlowfuzz=resid[:nfuzz,1:]-lowfuzz
        resid[:nfuzz,1:]=lowfuzz
        hipsd=np.zeros((nfuzz,2*nchan))
        for i in range(nchan):
            hipsd[:,2*i]=psd[i].psd_fn(np.arange(nf-nfuzz,nf)*df)
            hipsd[:,2*i+1]=hipsd[:,2*i]        
        hifuzz=np.random.normal(size = (nfuzz,2*nchan))
        hifuzz*=np.sqrt(hipsd)*scale
        dhifuzz=resid[nf-nfuzz:,1:]-hifuzz
        resid[nf-nfuzz:,1:]=hifuzz
    y=reconstruct_time_domain(resid,nchan)
    y_rec=[]
    for i in range(nchan):
        # Impute missing data
        #y_masked=mask*y[i]
        y_masked=y[i]
        #y_rec += [imp[i].draw_missing_data(y_masked)]
        y_rec += [imp[i].impute(y_masked, draw=True)]
    if verbose:
        ich=2
        nt=len(y[ich])
        dt = 1/(fs[-1]*2)
        ts=np.arange(nt)*dt
        print('Time comparison chan '+str(ich)+':')
        tdiffs=np.abs(y[ich]-y_rec[ich])>1e-24
        count = np.count_nonzero(tdiffs)
        print('  ',count,'of',nt,'different.')
        #for i in range(len(y[ich])):
        #    if tdiffs[i]:
        #        print('  ',i,ts[i],y[ich][i],y_rec[ich][i])
    if verbose:view_gaps(ts,[y,y_rec],gapinfo,labels=['orig','new'])    
        
    result=construct_specialized_data(y_rec,nchan,df,fs[0])
    if verbose and psd is not None:
        control=construct_specialized_data(y,nchan,df,fs[0])
        plot_compare_spectra([resid_data,resid,control,result],PSDset=psd,labels=['orig','fuzzed','control','result'])
    return result
