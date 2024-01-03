import numpy as np
import xarray as xr

from numba import guvectorize, vectorize, float64


@guvectorize(   
    "(float64[:],int16,complex128[:])",
    "(n), () -> (n)",
    forceobj=True
)
def _hilbert(y,N,out):
    '''
        Analytic signal using the Hilbert transform technique (Marple, 1999)
    '''
    # Check whether the signal has even or odd length
    if N%2==0:
        a = int(N/2)
    else:
        if N>1:
            a = int((N-1)/2)
        else:
            a = 0
        
    # FFT of y
    z = np.fft.fft(y)
    
    # Zero-out the negative frequencies
    z[a+1:N] = 0
    # Double the positive frequencies except from the 0th and (N/2)th ones
    z = 2*z
    z[0] = z[0]/2
    if N%2==0: 
        # For the even-length case, we also have the Nyquist frequency in the spectrum. 
        # This is shared between the positive and negative frequencies so we need to keep it once (see Marple 1999). 
        # For odd lengths, there is no Nyquist frequency in the spectrum.
        z[a] = z[a]/2

    # Inverse FFT to get the analytic signal
    out[:] = np.fft.ifft(z)
    
    
@vectorize([float64(float64, float64)])
def _rad_diff(a,b):
    '''
        In cases where the upstream and downstream phase differ more than pi or -pi, add/subtract 2pi where needed.
    '''
    diff = a - b
    if diff > np.pi:
        diff -= 2*np.pi
    elif diff < -np.pi:
        diff += 2*np.pi
        
    return diff
    
    
@guvectorize(   
    "(float64[:],float64[:])",
    "(n) -> (n)",
    forceobj=True
)    
def _finite_difference(a,out):
    '''
        Use centered differences in the interior, one-sided differences at the boundaries
    '''
    out[1:-1] = _rad_diff(a[2:],a[:-2])/2.
    out[-1] = _rad_diff(a[-1],a[-2])
    out[0] = _rad_diff(a[1],a[0])
    
    
@guvectorize(
    "(complex128[:,:], complex128[:], complex128[:,:], complex128[:,:], float64[:])",
    "(m,n), (k) -> (m,k), (n,k), (k)",
    forceobj=True
)
def _vectorized_svd(X,dummy,U,VS,S2):
    '''
        Vectorized singular value decomposition  -> generalized NumPy universal function
        
        - X = U @ np.diag(S) @ VH
        - U is standardized
        - m is dimension of time, n is stacked dimension, k = min(m,n)
    '''
    u, s, vh = np.linalg.svd(X,full_matrices=False)
    u_std = np.std(u,axis=0)
    U[:,:] = u/u_std
    VS[:,:] = vh.transpose() * s * u_std
    S2[:] = s**2   
    
    
#############################    
    


def complex_pca(anomalies):
    '''
        Principal component analysis using a sigular value decomposition algorithm
        
        - stack spatial dimensions
        - all dimensions apart from index and allpoints are broadcasted over
        - no chunking along core-dimensions allowed
        - for now, no area weighting is applied
    '''
    # apply area weighting
    # exclude poles for data on regular grid to avoid zero-devision
    #weights = np.sqrt(np.cos(anomalies['lat'] * np.pi/180))
    #anomalies = anomalies * weights
    
    
    signal = xr.apply_ufunc(_hilbert,
                         anomalies,len(anomalies.lon),
                         input_core_dims=[['lon'],[]],
                         output_core_dims=[['lon']],
                         dask='parallelized',
                         output_dtypes=[np.dtype('complex128')]
                       )
    
    
    # stack spatial dimensions
    stacked = signal.stack(allpoints=('lat','lon'))

    # singular value decomposition
    dummy = min(len(stacked.allpoints),len(stacked.index))
    dummy = xr.DataArray(np.zeros(dummy),dims=('number'))
    pc, eof_stacked, expl = xr.apply_ufunc(_vectorized_svd,
                                           stacked,dummy,
                                           input_core_dims=[['index','allpoints'],['number']],
                                           output_core_dims=[['index','number'],
                                                             ['allpoints','number'],
                                                             ['number']],
                                           dask='parallelized',
                                           output_dtypes=[np.dtype('complex128'),np.dtype('complex128'),np.dtype('float64')])
    
    
    # ratio of explained variance
    expl = expl/expl.sum('number')
    
    eof = eof_stacked.unstack('allpoints')
    
    # invert area weighting
    #eof = eof / weights
    
    ds = xr.Dataset({'pc':pc,'eof':eof,'expl':expl})
    
    return ds


def wavenumber(eof):
    '''
        Compute local wavenumber for EOF pattern using finite differences
    '''
    phase = np.arctan2(np.imag(eof),np.real(eof))
    
    # radians per grid spacing
    wavenum = xr.apply_ufunc(_finite_difference,
                             phase,
                             input_core_dims=[['lon']],
                             output_core_dims=[['lon']],
                             dask='parallelized',
                             output_dtypes=[np.double]
                            )
    
    # cycles per circumference
    wavenum = wavenum / (2*np.pi) * (len(wavenum['lon'])-1)
    
    return wavenum
   
    
def phase_speed(eof,pc,timestep=6*3600):
    '''
        Compute local wavenumber for EOF pattern and local frequency of PC time series using finite differences
    '''
    wavenum = wavenumber(eof)
    
    phase = np.arctan2(np.imag(pc),np.real(pc))
    
    # radians per time step
    freq = xr.apply_ufunc(_finite_difference,
                          phase,
                          input_core_dims=[['time']],
                          output_core_dims=[['time']],
                          dask='parallelized',
                          output_dtypes=[np.double]
                         )
    
    # cycles per second
    freq = (-1) * freq / (2*np.pi) / timestep
    
    # meters per circumference
    circ = 2*np.pi * 6371000 * np.cos(np.radians(eof['lat']))
    
    
    speed = freq / wavenum * circ
    
    return speed