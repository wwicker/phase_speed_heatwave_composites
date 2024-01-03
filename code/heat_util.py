''' heat_util.py

    Routines to compute heatwave metrics which comprises
    
    - creating predictor time series for detrending
    - detrending temperature data
    - estimating percentile for a rolling X-day-of-year window
    - checking length of threshold exceedance to identify heatwave
    - binning time series from heatwave DataFrame into rolling windows
    
    Shell script to compute daily maxima and global monthly means already exist.
'''
import numpy as np
import xarray as xr
import pandas
import numba
import scipy.ndimage


##################################
## CREATING PREDICTOR TIME SERIES
## don't forget to remove time mean

@numba.guvectorize(
    "(float64[:], float64[:], float64[:])",
    "(m), (n) -> (m)",
    forceobj=True
)
def vectorized_convolution(x,kernel,out):
    '''
        Vectorized convolution -> generalized NumPy universal function
        
        - mode='wrap' means that input is assumed being periodic
        - mode='mirror' means that input is extended by reflectinf about the center of the last pixel
    '''
    out[:] = scipy.ndimage.convolve(x,kernel,mode='mirror')


def lowpass(da,dim,co,valid=False):
    '''
        convolution in time space is multiplication in frequency space
        
        - no chunking along core dimension dim
    '''
    # transform of Hanning window as better spectral properties than transfrom of box-car-window
    n = 2 * co  + 1
    hann = np.hanning(n)
    hann /= hann.sum()
    hann = xr.DataArray(hann,dims=('kernel'))
    
    filtered = xr.apply_ufunc(vectorized_convolution,
                              da,hann,                                           
                              input_core_dims=[[dim,],['kernel']],
                              output_core_dims=[[dim,],],
                              dask='parallelized',
                              output_dtypes=[da.dtype])
    
    # input is assumed to be periodic
    # remove beginning and end if unvalid
    if valid:
        valid_slice = slice((n - 1) // 2, -(n - 1) // 2)
        filtered = filtered.isel(time=valid_slice)
        
    return filtered


##############################
## DETRENDING TEMPERATURE DATA
## coeff = covariance / predictor.var()


def file_by_file_covariance(f,predictor):
    '''
        Compute contribution of covariance by data in one file
        - requires predictor time series with zero mean
        - covariance = sum(weight)**-1 * sum(weight*sumOfSquares)
    '''
    Y = xr.open_dataset(f)['var167']
    X = predictor.interp_like(Y,method='linear')
    
    weight = len(Y['time'])
    sumOfSquares = (Y * X).sum('time')
    
    return (weight,sumOfSquares)
    
    
def file_by_file_detrend(f,predictor,coeff,outdir='./'):
    '''
    '''
    Y = xr.open_dataset(f)['var167']
    X = predictor.interp_like(Y,method='linear')
    
    trend = coeff * X
    detrended = Y - trend
    
    xr.Dataset(dict(var167=detrended)).to_netcdf(outdir+f.split('/')[-1])
    
    
#########################
## ESTIMATING PERCENTILES


@numba.guvectorize(
    "(float64[:],float64[:],float64[:])",
    "(n), (m) -> (m)",
    forceobj=True
)    
def ecdf(a,p,out):
    '''
        Emperical cummulative distribution function of array
        at percentiles p
    '''
    sort = np.sort(a)
    out[:] = sort[np.int64(p*len(a))]

    
def estimate_percentiles(da):

    da = da.chunk(dict(time=-1))
    p = xr.DataArray([0.25,0.75,0.9],dims=('p'))
    
    dist = xr.apply_ufunc(ecdf,
                          *(da,p),
                          input_core_dims=[['time'],['p']],
                          output_core_dims=[['p']],
                          dask='parallelized',
                          output_dtypes=[da.dtype])
    dist['p'] = p
    
    return dist


################################
## CHECKING LENGTH OF EXCEEDANCE

@numba.jit()
def check_length(array):
    '''
        Check exceedance for length >= 3 days
    '''
    start = []
    length = []
    mean = []
    count = 0
    
    for i in range(len(array)):
        if np.isnan(array[i]):
            if count > 2:
                start.append(i-count)
                length.append(count)
                mean.append(sum([array[j] for j in range(i-count,i)]))
                
            count = 0  
            
        else:
            count +=1
                            
    return start, length, mean


@numba.jit(nopython=True,parallel=False)
def cell_loop(array):
    '''
        Loop check_length over all grid cells
    '''
    # First loop to infer number of elements
    count = 0
    for j in numba.prange(array.shape[1]):
        s,l,m = check_length(array[:,j])
        
        count += len(s)
        
    # Create empty arrays with correct length    
    ncells = np.zeros(count,np.int_)
    start = np.zeros(count,np.int_)
    length = np.zeros(count,np.int_)
    mean = np.zeros(count,np.float_)
    
    # Second loop to fill arrays
    count = 0
    for j in numba.prange(array.shape[1]):
        s,l,m = check_length(array[:,j])

        for i in range(len(s)):
            ncells[count+i] = j
            start[count+i] = s[i]
            length[count+i] = l[i]
            mean[count+i] = m[i]
        
        count += len(s)
        
    return ncells, start, length, mean


def one_year_heat(da,filename,dist):
    
    time = da['time']
    
    # create dayofyear index, mask, stack, and load into memory
    tmp = da.assign_coords(dayofyear=da['time.dayofyear']).set_index(time='dayofyear').rename(time='dayofyear')
    tmp = tmp.where(tmp >= dist.sel(p=0.9))
    tmp = tmp.stack(ncells=('lat','lon'))
    tmp = tmp.compute()
    
    # identify heatwaves
    ncells, start, length, mean = cell_loop(tmp.values)

    # create DataFrame
    start = time.isel(time=start).values
    start = np.datetime_as_string(start)
    lon = tmp['lon'].isel(ncells=ncells).values
    lat = tmp['lat'].isel(ncells=ncells).values

    heat = pandas.DataFrame(dict(lat=lat,lon=lon,start=start,length=length,mean=mean),
                            columns=['lat','lon','start','length','mean'])
    
    # I think, this line is redundant
    heat = heat[heat['length'] > 2] 
    # turn sum into mean temperature
    heat['mean'] = heat['mean'] / heat['length']
    
    if not(filename is None):
        heat.to_json(filename,orient='records')
        
    return heat


######################
## BINNING TIME SERIES

def bin_rolling(data,start_month=6,days_per_window=30,season_days=30+31+30):
    '''
        Define time windows similar to wave_util.construct_rolling_dataset but different
        - make main_bins and side_bins (centers are edges and edges are centers)
    '''
    # define window labels
    year = np.unique(np.array([start[:4] for start in data['start']]))[0]
    length = int(season_days / days_per_window*2) * int(days_per_window/2)
    labels = range(0, length + int(days_per_window/2), int(days_per_window/2))
    labels = [np.datetime64(year+'-%02d-01'%start_month)+np.timedelta64(l,'D') for l in labels]
    
    # bin pandas.DataFrame
    if len(labels)%2:
        data['main_bins'] = pandas.cut(pandas.to_datetime(data['start']),bins=labels[::2],labels=labels[1::2])
        data['side_bins'] = pandas.cut(pandas.to_datetime(data['start']),bins=labels[1::2],labels=labels[2:-1:2])
    else:
        data['main_bins'] = pandas.cut(pandas.to_datetime(data['start']),bins=labels[::2],labels=labels[1:-1:2])
        data['side_bins'] = pandas.cut(pandas.to_datetime(data['start']),bins=labels[1::2],labels=labels[2:-1:2])
        
    return data


def basic_metrics(files,start_month=6,days_per_window=30,season_days=30+31+30,verbose=False):
    
    frequency = []
    length = []
    mean = []

    for f in files:
        if verbose:
            print(f)
            
        data = pandas.read_json(f)
        data = bin_rolling(data,start_month,days_per_window,season_days)
        
        frequency.append(data.groupby(['lat','lon','main_bins'])['start'].count().to_xarray().rename(main_bins='rolling'))
        frequency.append(data.groupby(['lat','lon','side_bins'])['start'].count().to_xarray().rename(side_bins='rolling'))
        
        length.append(data.groupby(['lat','lon','main_bins'])['length'].mean().to_xarray().rename(main_bins='rolling'))
        length.append(data.groupby(['lat','lon','side_bins'])['length'].mean().to_xarray().rename(side_bins='rolling'))
    
        mean.append(data.groupby(['lat','lon','main_bins'])['mean'].mean().to_xarray().rename(main_bins='rolling'))
        mean.append(data.groupby(['lat','lon','side_bins'])['mean'].mean().to_xarray().rename(side_bins='rolling'))
    
    frequency = xr.concat(frequency,dim='rolling').sortby('rolling')
    length = xr.concat(length,dim='rolling').sortby('rolling')
    mean = xr.concat(mean,dim='rolling').sortby('rolling')
    
    return xr.Dataset(dict(frequency=frequency,length=length,mean=mean))
    