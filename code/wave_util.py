''' wave_util.py

    Use spatio-temporal spectral analysis to estimate Phase speed spectra ("Hayashi spectra") follwing Randel & Held (1991).
    
'''
import numpy as np
import xarray as xr
import numba

from scipy import signal



@numba.guvectorize(
    "(float64[:,:],float64[:],float64[:],complex128[:,:])",
    "(m,n), (k), (l) -> (k,l)",
    forceobj=True
)
def fft2d(a,freq0,freq1,out):
    '''
        vectorized two-dimensional fast fourier transfrom
        
        - take care of the normalization
    '''
    padded = np.pad(a,((0,len(freq0)-a.shape[0]),(0,len(freq1)-a.shape[1])))
    out[:,:] = np.fft.fft2(padded,norm=None)
    
    
    
@numba.guvectorize(   
    "(float64[:],int16,float64[:])",
    "(n), () -> (n)",
    forceobj=True
)
def filtering(a,n,out):
    tmp = np.append(a,n*[a[-1]])
    tmp = np.insert(tmp,0,n*[tmp[0]])
    
    for i in range(n):
        tmp = 0.25*tmp[:-2] + 0.5*tmp[1:-1] + 0.25*tmp[2:]
        
    out[:] = tmp
    
    
    
@numba.guvectorize(
    "(float64[:],float64[:],float64[:],float64[:])",
    "(n), (n), (m) -> (m)",
    nopython=True
)
def interp_freq(spect,freq,fc,out):
    '''
    '''
    out[:] = np.interp(fc,freq,spect)

    
    
####################################
    
    
    
def wavenum_freq_spect(da,timestep=86400,window=np.hanning,window_args=(),pad=0,nfilter=0,verbose=False):
    '''
        wavenumber-frequency power spectrum
        
        - takes xarray.DataArray input
        - sampling is sample spacing in seconds
        - tapering produces a smoother spectrum but increases leakage
        
        returns power spectral density in units of [variance] / [frequency] = m^2 / s (this is not latitude-dependant)
    '''
    if verbose:
        print('\n Time-mean, zonal-mean variance')
        print(da.var(('time','longitude')))

    # prepare arrays for frequency, wavenumber, tapering
    freq = xr.DataArray(np.fft.fftfreq(len(da.time)+pad,d=timestep),dims=('frequency'))
    wavenum = xr.DataArray(np.fft.fftfreq(len(da.longitude),d=1/len(da.longitude)),dims=('wavenumber'))
    taper = xr.DataArray(window(len(da.time),*window_args),coords=dict(time=da.time),dims=('time'))
    
    # apply tapering
    da = da * taper
    
    # fft2d does not allow chunking along core dimensions
    spect = xr.apply_ufunc(fft2d,
                           *(da,freq,wavenum),
                           input_core_dims=[['time','longitude'],['frequency'],['wavenumber']],
                           output_core_dims=[['frequency','wavenumber']],
                           dask='parallelized',
                           output_dtypes=[np.complex128])
    spect = spect.assign_coords(dict(frequency=freq,wavenumber=wavenum))
    
    # Compute power spectrum
    spect = spect * np.conjugate(spect)
    spect = np.real(spect)
    
    # 1-2-1 filtering
    if nfilter > 0:
        spect = xr.apply_ufunc(filtering,
                              *(spect,nfilter),
                              input_core_dims=[['frequency'],[]],
                              output_core_dims=[['frequency']],
                              dask='parallelized',
                              output_dtypes=[np.float64])
    
    # this is in units of power spectral denstity times frequency resolution times wavenumber resolution
    #spect = spect / len(da.time)**2 / len(da.longitude)**2
    # frequency resolution is one over timeseries length
    spect = spect / len(da.longitude)**2 / len(da.time) * timestep
    # use positive wavenumber only
    spect = spect.where(wavenum >= 0,drop=True) * 2
    
    # account for tapering
    spect = spect / (taper**2).mean()
    
    if verbose:
        print('\n Variance retainded by integration over frequency and wavenumber')
        print(spect.integrate(('frequency','wavenumber')))
    
    return spect



def freq2phase_speed_interp(spect,dc=1,verbose=False):
    '''
        Calculate wavenumber-phase speed spectra from wavenumber-frequency spectra
        following Randel & Held (1991)
        
        -positive phase speed is eastward
        
        Divides power spectral density by wavelength to conserve variance after integration and get units of [variance] / [phase speed] = m / s (which is latitude-dependant).
        This is because during interpolation, many frequency bins at high latitudes will fall in only a few phase speed bins (for 6-hourly data, high phase speeds are not resolved at high latitudes).
    '''
    # Define an array of phase speed
    N = len(spect.frequency)
    c = xr.DataArray(np.arange(-dc*int(N/2),dc*int(N/2),dc),dims=('phase_speed')) 
    c = c.assign_coords(phase_speed=c)
    
    # Define the array of frequencies that correspond that phase speeds
    a = 6371000
    factor = 1 / (2*np.pi*a)
    factor = factor / np.cos(spect.latitude/180*np.pi)
    fc = factor * c * spect.wavenumber # this has dimenstions latitude, phase speed, wavenumber
    
    # Interpolate linearly to these frequencies
    spect = spect.sortby('frequency')
    new_spect = xr.apply_ufunc(interp_freq,
                               *(spect,spect.frequency,fc),
                               input_core_dims=[['frequency'],['frequency'],['phase_speed']],
                               output_core_dims=[['phase_speed']],
                               dask='parallelized',
                               output_dtypes=[spect.dtype])
    
    # scale power spectral density into units of phase speed
    new_spect = new_spect * spect.wavenumber * factor
    # positive phase speed is eastward
    new_spect['phase_speed'] = -1* new_spect['phase_speed']
    
    if verbose:
        print('\n Variance retainded by integration over phase speed and wavenumber')
        print((-1)*new_spect.integrate(('phase_speed','wavenumber')))
    
    return new_spect


#########################################


def construct_rolling_dataset(files,
                              selection=None,
                              n_per_window = 30*4,
                              n_per_day = 4, # 6-hourly
                              season_days = 30+31+31, # June-July-Agust
                             ):
    '''
        Construct a time series of rolling windows of length N time steps with 50% overlap
        
        - use nested list of filenames with one list of files per seasson
    '''

    
    length = int(n_per_day*season_days / n_per_window*2) * int(n_per_window/2)
    start = range(0, length - int(n_per_window/2), int(n_per_window/2))
    end = range(n_per_window, length + int(n_per_window/2), int(n_per_window/2))

    slices = [slice(start,end) for start, end in zip(start,end)]
    
    outer_rolling = []

    for season in files:
        
        # open, select, rechunk xr.Dataset
        ds = xr.open_mfdataset(season,combine='nested',concat_dim='time')
        if not(selection is None):
            ds = ds.sel(**selection)
        ds = ds.chunk(time=-1)
        
        # select rolling slices
        inner_rolling = [ds.isel(time=s) for s in slices]
        inner_rolling = [da.drop('time').assign_coords(rolling=da.time.mean()) for da in inner_rolling]
        
        outer_rolling.append(xr.concat(inner_rolling,dim='rolling'))

    return xr.concat(outer_rolling,dim='rolling')


def remove_climatology(da):
    '''
        Subtract a zonally varying climatology
        
        - this is in contrast to Jiménez-Esteve et al. (2022) https://doi.org/10.1029/2021GL096337
    '''
    day = da['rolling.day'].values
    month = da['rolling.month'].values
    index = ['%02d-%02d'%(d,m) for d,m in zip(day,month)]
    
    indexed = da.assign_coords(my_index=xr.DataArray(index,dims='rolling'))
    
    clim = indexed.groupby('my_index').mean(('rolling','time')).compute()
    
    anomalies = indexed.groupby('my_index') - clim 
    
    return anomalies


def compute_spectra(array,
                    timestep=6*3600,
                    wavenumber=slice(1,10),
                    dc=1/3,
                    phase_speed=slice(30,-30)):
    '''
        Perform spectral analysis (no meridional average)
    '''
    # compute wavenumber-frequency spectrum
    spect = wavenum_freq_spect(array,timestep=timestep,window=signal.tukey,window_args=(0.5,),pad=2*len(array['time']))
    spect = spect.sel(wavenumber=wavenumber)

    # interpolate to frequency to phase speed spectrum
    spect = freq2phase_speed_interp(spect,dc=dc)

    spect = spect.sel(phase_speed=phase_speed)
    
    return spect


def centroid(da,m=1):
    
    return (da * da.phase_speed**m).sum('phase_speed') / (da * da.phase_speed**(m-1)).sum('phase_speed')


def integral(da,upper,lower):
    
    return (da.sel(phase_speed=slice(upper,lower)).sum('phase_speed') * np.abs(da['phase_speed'].diff('phase_speed').mean()))