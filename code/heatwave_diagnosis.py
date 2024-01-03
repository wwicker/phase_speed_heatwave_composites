import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import numba
import scipy.ndimage
import pandas
import os

from dask.distributed import Client



def avg_time(da,axis=None,refdate='1979-01-01'):
    '''
        Average an array of np.datetime64 objects
    '''
    return ((da - np.datetime64(refdate))/10**9).mean(axis)*10**9+np.datetime64(refdate)



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
    
    
    
def file_by_file_detrend(f,predictor,coeff,outdir=work+'/wolfgang/detrended/'):
    '''
    '''
    Y = xr.open_dataset(f)['var167']
    X = predictor.interp_like(Y,method='linear')
    
    trend = coeff * X
    detrended = Y - trend
    
    xr.Dataset(dict(var167=detrended)).to_netcdf(outdir+f.split('/')[-1])
    

    
def estimate_percentiles(da,p):
    '''
    '''
    da = da.chunk(dict(time=-1))
    
    dist = xr.apply_ufunc(ecdf,
                          *(da,p),
                          input_core_dims=[['time'],['p']],
                          output_core_dims=[['p']],
                          dask='parallelized',
                          output_dtypes=[da.dtype])
    dist['p'] = p
    
    return dist



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



if __name__ == '__main__':
    
    work = os.environ['WORK']
    
    client = Client() # potentionally use a SLURMCluster with dask_jobqueue
    
    
    # create 9 year lowpass filtered global mean temperature as predictor for detrending
    directory = work+'/wolfgang/ERA5_surf_day_max/global_monthly_mean/'
    files = [directory+f for f in os.listdir(directory) if f.endswith('.nc')]
    files.sort()

    ds = xr.open_mfdataset(files,combine='nested',concat_dim='time')
    
    filtered = lowpass(ds['var167'].compute(),dim='time',co=9*12)
    filtered = filtered.squeeze().drop(('lon','lat'))
    predictor = filtered - filtered.mean('time')
    
    
    # list of files with daily maxima
    directory = work+'/wolfgang/ERA5_surf_day_max/'
    files = [directory+f for f in os.listdir(directory) if f.startswith('era5_an_t2m_reg05_1h_')]
    files.sort()
    
    
    # compute covariance and regression coefficient with predictor time series

    futures = client.map(file_by_file_covariance,files,predictor=predictor)

    results = client.gather(futures)

    covariance = [sum(a) for a in zip(*results)]
    covariance = covariance[1] / covariance[0]

    coeff = (covariance / predictor.var())
    
    
    # store predictor and filed of coefficients to disk
    xr.Dataset(dict(predictor=predictor,coeff=coeff)).to_netcdf(work+'/wolfgang/detrended/meta.nc')
    
    
    # store detrended data to disk
    client.map(file_by_file_detrend,files,predictor=predictor,coeff=coeff)
    
    
    # load detrended temperature data
    directory = work+'/wolfgang/detrended/'
    files = [directory+f for f in os.listdir(directory) if f.startswith('era5_an_t2m_reg05_1h_')]
    files.sort()

    data = xr.open_mfdataset(files,combine='nested',concat_dim='time')['var167']
    data = data.chunk(dict(time=365))
    
    
    # construct rolling dayofyear window (union of multiple years)
    rolling = []
    for day in range(1,367):
        days = [i%(367) for i in range(day-15,day+16)]
        rolling.append(data.where(data['time.dayofyear'].isin(days),drop=True).assign_coords(dayofyear=day))
        
        
    # distribute estimation of examples onto cluster
    p = xr.DataArray([0.25,0.75,0.9],dims=('p'))
    dist = client.map(estimate_percentiles,rolling,p=p)
    result = client.map(xr.DataArray.compute,dist)
    
    
    # gather estimates from cluster and store to disk
    percentiles = client.gather(result)
    percentiles = xr.concat(percentiles,dim='dayofyear')
    
    xr.Dataset(dict(var167=percentiles)).to_netcdf(work+'/wolfgang/detrended/percentiles_31days.nc')
    
    
    # load detrended temperature data
    directory = work+'/wolfgang/detrended/'
    files = [directory+f for f in os.listdir(directory) if (f.endswith('06.nc') or
                                                        f.endswith('07.nc') or
                                                        f.endswith('08.nc'))]
    files.sort()

    data = xr.open_mfdataset(files,combine='nested',concat_dim='time')['var167']
    
    dist = xr.open_dataarray(work+'/wolfgang/detrended/percentiles_31days.nc')
    
    # group for year-by-year processing
    groups = list(data.groupby('time.year'))
    filenames = [work+'/wolfgang/detrended/heat/heat_%d.json'%year[0] for year in groups]
    groups = [year[1] for year in groups]
    
    
    # serial execution is just fine
    for g, f in zip(groups,filenames):
        print(f)
        one_year_heat(g,f,dist=dist)