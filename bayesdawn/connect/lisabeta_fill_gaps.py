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
    #assume data cols are freq, ch1.real, ch1.imag, ch2.real,... 
    #evenly spaced in freq, over some band
    fs=np.real(data[:,0])
    nd=len(data)
    df=(fs[-1]-fs[0])/(nd-1)
    nf=nd+int(fs[0]/df+.5)
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
    ioff=int(f0/df+.5)
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

def plot_compare_spectra(datasets,LISAnoise=None,nchan=3,labels=None,fs=None,PSDset=None,TDItype='TDIAET'):
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
            noises += [pyLISAnoise.LISANoisePSDFunction(LISAnoise,ich+1,TDIrescaled=resc,TDI=TDItype).apply(fs)]
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


def splinePSDs_from_time_data(tdataset,fsamp):
    n_chan=len(tdataset)
    n_data=len(tdataset[0])
    t_obs = n_data / fs
    # Lower frequency for the PSD estimation
    fmin = 1 / t_obs * 1.05
    # Upper frequency
    fmax=fs/2
    # Instantiate PSD estimator class
    psd_sets=[]
    for ichan in range(n_chan):
        psd_cls = psdmodel.PSDSpline(n_data, fsamp,
                                     n_knots=20,
                                     d=3,
                                     fmin=fmin,
                                     fmax=fmax,
                                     ext=0)
        psd_cls.estimate(tdataset[ichan])
        psd_sets.append(psd_cls)
    return psd_sets
        

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

class PSD_model:
    '''
    Construct a PSD model using the psdmodel.ModelFDDataPSD class
    The value added here is just that we make models for each of the data channels, not just one.
    Currently the model is constructed from data by ML methods, but we plan to generalize that to allow updating by spline param MCMC.
    '''
    def __init__(self, data, channels, **model_args):
        self.channels=channels
        if 'savefilebase' in model_args:
            savefilebase=model_args.pop('savefilebase')
        else: savefilebase=None
        if 'real_data' in model_args:
            self.have_real_data=model_args.pop('real_data')
        else: self.have_real_data=False
        self.args=model_args
        self.ML_update(data,savefilebase=savefilebase)

    def ML_update(self, FD_noise_data,savefilebase=None):
        PSDmodels=[]
        #Our interface for ModelFDDataPSD expects a recarray
        #or a dict so  we have to construct it
        datadict={'f':np.real(FD_noise_data[:,0])}            
        for ich in range(len(self.channels)):
            if self.have_real_data:
                datadict[self.channels[ich]]=FD_noise_data[:,2*ich+1]+1j*FD_noise_data[:,2*ich+2]
            else:
                datadict[self.channels[ich]]=FD_noise_data[:,ich+1]

        for chan in self.channels:
            chanmodel=psdmodel.ModelFDDataPSD(datadict, chan, **self.args)
            PSDmodels.append(chanmodel)
            if savefilebase is not None:
                chanmodel.plot()
                plt.savefig(savefilebase+"_"+chan+".png")
                plt.clf()
        self.PSDs=PSDmodels

    def get_params(self):
        #Get the underlying PSD model spline data in dict form
        pardict={}
        for ich in range(len(self.channels)):
            chan=self.channels[ich]
            pardict[chan]=self.PSDs[ich].get_spline_data()
        return pardict
    
    def update_params(self,pardict):
        #Reset the underlying PSD model spline data
        for ich in range(len(self.channels)):
            chan=self.channels[ich]
            self.PSDs[ich].set_spline_data(pardict[chan])
                                           
class FDimputation:
    
    def __init__(self,gap_intervals,PSDmodel,t0=0,method='nearest',nab=60,intergap_min=None,verbose=False):
        self.verbose=verbose
        self.channels=PSDmodel.channels
        self.PSDmodel=PSDmodel
        nchan=len(self.channels)
        self.t0=t0
        if self.verbose: print('constructing FD imputation with',len(gap_intervals),'gaps\n gap_intervals=',gap_intervals)
        self.gap_intervals=gap_intervals
        self.have_gap_info=False
        self.have_imps=False
        
        args={'method':method}
        if method=='woodbury':
            pass
        elif method=='nearest':
            args['na']=nab
            args['nb']=nab
        else: raise ValueError('Did not recognize method="'+method+'"')
        self.impargs=args
        if intergap_min is None: intergap_min=nab
        self.intergap_min=intergap_min

    def compute_imps(self):
        assert(self.have_gap_info)
        if self.have_imps: return self.imps
        gapinfo=self.gap_info
        mask=gapinfo['mask']
        s=np.zeros(len(mask))  #for residual 'signal' is zero
        imps = [ datamodel.GaussianStationaryProcess(s, mask, psd, **self.impargs) for psd in self.PSDmodel.PSDs]

        # perform offline computations
        for imp in imps: imp.compute_offline()
        self.imps=imps
        self.have_imps=True
        return imps

    def compute_gap_info(self,fs):
        '''
        Function to compute temporally indexed gap-info, including the gap mask, from time-valued gap info. If this has already been computed then it is reused.  We use fs (and stored t0) to define the temporal grid.
        '''
        # Saved args
        gap_intervals=self.gap_intervals
        t0=self.t0
        
        if self.have_gap_info: return self.gap_info
        if self.verbose: print('constructing FD imputation with',len(gap_intervals),'gaps\n gap_intervals=',gap_intervals)
        self.gap_intervals=gap_intervals
        self.have_gap_info=False
        self.have_imps=False
        
        args={'method':method}
        if method=='woodbury':
            pass
        elif method=='nearest':
            args['na']=nab
            args['nb']=nab
        else: raise ValueError('Did not recognize method="'+method+'"')
        self.impargs=args
        if intergap_min is None: intergap_min=nab
        self.intergap_min=intergap_min

    def compute_imps(self):
        assert(self.have_gap_info)
        if self.have_imps: return self.imps
        gapinfo=self.gap_info
        mask=gapinfo['mask']
        s=np.zeros(len(mask))  #for residual 'signal' is zero
        imps = [ datamodel.GaussianStationaryProcess(s, mask, psd, **self.impargs) for psd in self.PSDmodel.PSDs]

        # perform offline computations
        for imp in imps: imp.compute_offline()
        self.imps=imps
        self.have_imps=True
        return imps

    def compute_gap_info(self,fs):
        '''
        Function to compute temporally indexed gap-info, including the gap mask, from time-valued gap info. If this has already been computed then it is reused.  We use fs (and stored t0) to define the temporal grid.
        '''
        # Saved args
        gap_intervals=self.gap_intervals
        t0=self.t0
        
        if self.have_gap_info: return self.gap_info
        if self.verbose:
            print('computing gap info for',len(gap_intervals),'gaps\n initial gap_intervals =',gap_intervals)
        
        if not len(gap_intervals.shape)==2 or not gap_intervals.shape[1]==2:
            raise ValueError('obs_params.gap_intervals should provide a list of pairs [[tstart,tend],[...]] indicating the temporal location of data gaps.')
        
        # First sort the gaps by the start time
        gap_intervals = gap_intervals[gap_intervals[:,0].argsort()]
        #for i in range(len(gap_intervals)-1):
        #    if gap_intervals[i,1]>=gap_intervals[i+1,0]:
        #        raise ValueError('Not ready to handlle overlapping gaps')

        # Quantize the gaps on the temporal grid
        df=(fs[-1]-fs[0])/(len(fs)-1)
        nf=len(fs)+int(fs[0]/df+.5)
        nt=2*(nf-1)
        dt=0.5/fs[-1]
        #ts=t0+np.arange(nt)*dt
        igap_starts=((gap_intervals[:,0]-t0)/dt+0.5).astype(int)        
        igap_ends=((gap_intervals[:,1]-t0)/dt+0.5).astype(int)+1
        #print('gaps 1',list(zip(igap_starts,igap_ends)))
        # Ensure gaps are in domain
        igap_starts=igap_starts[igap_ends>0]
        igap_ends=igap_ends[igap_ends>0]
        #print('gaps 2',list(zip(igap_starts,igap_ends)))
        igap_ends=igap_ends[igap_starts<nt]
        igap_starts=igap_starts[igap_starts<nt]
        #print('gaps 3',list(zip(igap_starts,igap_ends)))
        igap_starts[igap_starts<0]=0
        igap_ends[igap_ends>nt]=nt
        #print('gaps 4',list(zip(igap_starts,igap_ends)))


        # Ensure that gaps are not too close, merging if needed
        i=0
        while i<len(igap_starts)-1:            
            if igap_ends[i]-igap_starts[i+1]>=-self.intergap_min:
                #print('was:',gap_starts[i:i+2],gap_ends[i:i+2])
                igap_starts=np.delete(igap_starts,i+1)
                igap_ends=np.delete(igap_ends,i)
                #print('now:',gap_starts[i:i+2],gap_ends[i:i+2])
            else: i+=1
        #print('gaps 5',list(zip(igap_starts,igap_ends)))
            
        # Define gap mask
        mask=np.ones(nt,dtype=int)
        for i in range(len(igap_starts)):
            mask[igap_starts[i]:igap_ends[i]]=0
            
        if self.verbose:
            print('Applying',len(igap_starts),'gaps. Gap fraction is', 1-sum(mask)/len(mask))

        # Store
        gapinfo={}
        gapinfo['gap_starts']=igap_starts
        gapinfo['gap_ends']=igap_ends
        gapinfo['mask']=mask
        self.gap_info=gapinfo
        self.have_gap_info=True
        
        return gapinfo
    
    def apply_imputation(self, resid_data,psd=None,complex_data=False,report_mean_squares=False):
        '''
        Function to apply a set of imputation models to  multi-channel Fourier-domain residual data.

        To understand what we do, note the big picture logic here:
           -There is some fixed original FD data (possibly in a transformed/restricted working format).
           -This is then interpreted as full domain data with junk for gap infill.
           -We subtract from that a signal model to get a residual which is passed in here
           -(That residual represents the FD version of some time domain model with the gap data corresponding to 
            the original junk minus the signal, making it new junk.  We don't care about the new junk though 
            because it will be masked.)
           -We convert the FD residual to something in the time domain in an invertible way.  This may not correspond
            identically to the actual initial time domain data because of band selection and differences between
            whatever initial sort of Fourier windowing or filtering may have been applied.  We don't try to match that
            exactly, but differences like that will impact the transform space adjusted by the gap filling.  In our case 
            it is most important that the transform is clearly invertible on the Fourier domain of interest.  
           -For this time-domain data, we apply imputation to reset the gap data.
           -We expect that this more-or-less physically corresponds to resetting the original gap data, but that is
            something we need to understand better.
           -We then apply the fourier transform to revert to FD and return the result.
        '''
        #Expecting residual data in the format (eg for nchan=3)
        #[[f0,Ar0,Ai0,Er0,Ei0,Tr0,Ti0],[f1,Ar1,Ai1,...],...]
        #print('applying imputation')
        resid=resid_data.copy()

        #This routine is expecting data in real,imag split form
        #so have to convert complex data
        if complex_data:
            nchan=len(resid.T)-1
            resid=np.zeros((len(resid),nchan*2+1))
            if self.verbose:
                print('Converting resid to real. New shape=',resid.shape)
            resid[:,0]=np.real(resid_data[:,0])
            for ich in range(nchan):
                resid[:,1+2*ich]=np.real(resid_data[:,1+ich])
                resid[:,2+2*ich]=np.real(resid_data[:,1+ich])

        fs=resid[:,0]

        gapinfo=self.compute_gap_info(fs)
        mask=gapinfo['mask']

        imps=self.compute_imps()

        nd=len(resid)
        df=(fs[-1]-fs[0])/(nd-1)
        nchan=len(self.channels)
        if self.verbose:
            print('constructing TD, nchan=',nchan)
            print('resid.shape',resid.shape)
        y=reconstruct_time_domain(resid,nchan=nchan)
        y_rec=[]
        
        for i in range(nchan):
            # Impute missing data
            #print('len check',len(mask),len(y[i]))
            #print('imputing channel',i)
            y_rec += [imps[i].impute(y[i], draw=True)]
        #y_rec=np.array(y_rec).T
        if report_mean_squares:
            #For diagnostic, compute the ratio of masked to unmasked data meansq
            maskedmeansq=np.mean(np.array([x[mask==1] for x in y_rec])**2)
            unmaskedmeansq=np.mean(np.array([x[mask==0] for x in y_rec])**2)
            print('masked,unmasked mean-sq and ratio:',maskedmeansq,unmaskedmeansq,maskedmeansq/unmaskedmeansq)            
            #print('y_rec.shape',y_rec.shape)
        if self.verbose:
            print('reconstructing FD')
        result=construct_specialized_data(y_rec,nchan,df,fs[0])

        #Put the data back how we got it if needed
        if complex_data:
            res=np.zeros((len(result),nchan+1),dtype=complex)
            if self.verbose:
                print('Converting result to complex. New shape=',res.shape)
            res[:,0]=result[:,0]
            for ich in range(nchan):
                res[:,1+ich]=result[:,1+2*ich]+1j*result[:,2+2*ich]
            result=res

        return result


