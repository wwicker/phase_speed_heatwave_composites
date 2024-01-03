''' composite_util.py


    wolfgang.wicker@unil.ch
'''
import numpy as np
import xarray as xr
import numba

from scipy import signal, stats


def tapered_mean(da):
    '''
        rolling window of meridional wind is tapered during phase spectra compution
        
        do the same with other variables
    '''
    
    taper = xr.DataArray(signal.tukey(len(da.time),0.5),coords=dict(time=da.time),dims=('time'))
    
    da = da * taper
    da = da.mean('time')
    da = da / taper.mean('time')
    
    return da


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
    
    
    
def composite_dates(da,percentile=0.9):
    '''
    '''
    p = xr.DataArray(np.linspace(0,0.99,100),dims=('percentile'))
    dist = xr.apply_ufunc(ecdf,
                      *(da,p),
                      input_core_dims=[['time'],['percentile']],
                      output_core_dims=[['percentile']],
                      output_dtypes=[[da.dtype]])
    dist['percentile'] = p

    threshold = dist.sel(percentile=percentile,method='nearest')

    dates = da['time'].where(da > threshold,drop=True)

    return dates


##########################
# Significance assesment


@numba.guvectorize(
    "(float64[:],float64[:],float64[:],float64[:,:])",
    "(k), (n), (m) -> (n,m)",
    forceobj=True
)
def random_sample(a,nt,nb,out):
    '''
        Draw len(nb) random samples from array a
        'ziehen mit zuruecklegen'
        
        - nb is a dummy array to get dimension size
    '''
    index = stats.uniform.rvs(0,len(a),len(nt)*len(nb))
    index = index.astype(int).reshape((len(nt),len(nb)))
    out[:,:] = a[index]
    
    
t_statistic = lambda x, ref: x.mean('time') / x.std('time') * len(x['time'])**(1/2)


def parametric_bootstrap(composite,population,nrandom=100,steps=1,pvalue=0.05,parameter=t_statistic,one_sided=False):
    '''
        Adapt this routine to allow for a different parameter
        - the f parameter depends not only on the composite but also on the population
    '''
    
    n = len(composite['time'])
    n = xr.DataArray(np.arange(n),dims=('time'))
    
    # Resample control
    bootstrap = xr.DataArray(np.arange(nrandom/steps),dims=('random'))
    
    dist = []
    
    for i in range(steps):

        sample = xr.apply_ufunc(random_sample,
                             *(population,n,bootstrap),
                             input_core_dims=[['rolling'],['time'],['random']],
                             output_core_dims=[['time','random']],
                             dask='parallelized',
                             output_dtypes=[[population.dtype]])
    
        # t statistic for the resampled data
        dist.append(parameter(sample,population))
    
    dist = xr.concat(dist,dim='random')
    
    # emperical cumulative distribution function
    p = xr.DataArray(np.linspace(0,0.999,1000),dims=('percentile'))
    dist = xr.apply_ufunc(ecdf,
                          *(dist,p),
                          input_core_dims=[['random'],['percentile']],
                          output_core_dims=[['percentile']],
                          dask='parallelized',
                          output_dtypes=[[dist.dtype]])
    dist['percentile'] = p
    
    # check whether Null hypothesis can be rejected
    param = parameter(composite,population)
    if one_sided:
        sig = param > dist.sel(percentile=1-pvalue,method='nearest')
    else:
        sig = np.add(param < dist.sel(percentile=pvalue/2,method='nearest'), 
                     param > dist.sel(percentile=1-pvalue/2,method='nearest'))
    
    return sig



def two_sample_bootstrap(sample1,sample2,dim='time',nrandom=100,confid=0.05):
    '''
        Test mean difference between two composites
    '''
    # Produce control samples that fulfill the Null hypothesis
    c1 = sample1 - sample1.mean(dim)
    c2 = sample2 - sample2.mean(dim)
    
    n1 = xr.DataArray(np.arange(len(sample1[dim])),dims=dim)
    n2 = xr.DataArray(np.arange(len(sample2[dim])),dims=dim)
    
    # Resample control
    bootstrap = xr.DataArray(np.arange(nrandom),dims=('random'))
    c1 = xr.apply_ufunc(random_sample,
                         *(c1,n1,bootstrap),
                         input_core_dims=[[dim],[dim],['random']],
                         output_core_dims=[[dim,'random']],
                         dask='parallelized',
                         output_dtypes=[[c1.dtype]])
    c2 = xr.apply_ufunc(random_sample,
                         *(c2,n2,bootstrap),
                         input_core_dims=[[dim],[dim],['random']],
                         output_core_dims=[[dim,'random']],
                         dask='parallized',
                         output_dtypes=[[c2.dtype]])
    
    # t statistic for the resampled data
    statistic = lambda x1, x2,dim: (x1.mean(dim)-x2.mean(dim)) / (x1.var(dim)/len(n1) + x2.var(dim)/len(n2))**(1/2)
    dist = statistic(c1,c2,dim)
    
    # emperical cumulative distribution function
    p = xr.DataArray(np.linspace(0,0.999,1000),dims=('percentile'))
    dist = xr.apply_ufunc(ecdf,
                          *(dist,p),
                          input_core_dims=[['random'],['percentile']],
                          output_core_dims=[['percentile']],
                          dask='parallelized',
                          output_dtypes=[[dist.dtype]])
    dist['percentile'] = p
    
    # check whether Null hypothesis can be rejected
    param = statistic(sample1,sample2,dim)
    sig = np.add(param < dist.sel(percentile=confid/2,method='nearest'), 
                 param > dist.sel(percentile=1-confid/2,method='nearest'))
    
    return sig


def bootstrap_correlation(predictor,series,nrandom=10000):
    
    N = len(series)
    index = stats.uniform.rvs(0,N,N*nrandom)
    index = index.astype(int).reshape(N,nrandom)
    index = xr.DataArray(index,dims=('rolling','random'))
    
    random_sample = series.drop('rolling').isel(rolling=index)
    random_corr = xr.corr(predictor,random_sample,dim='rolling')
    
    percentile = xr.DataArray(np.linspace(0,0.9999,10000),dims=('percentile'))
    
    dist = xr.apply_ufunc(ecdf,
                          random_corr,percentile,
                          input_core_dims=[['random'],['percentile']],
                          output_core_dims=[['percentile'],],
                          output_dtypes=[np.double])
    dist['percentile'] = percentile
    
    corr = xr.corr(predictor,series,dim='rolling')
    
    try:
        pvalue = dist.where(dist < corr,drop=True)

        pvalue = pvalue['percentile'].isel(percentile=-1).drop('percentile')
        
    except IndexError:
        
        pvalue = 0
    
    
    return xr.Dataset(dict(correlation=corr,p_value=pvalue))
