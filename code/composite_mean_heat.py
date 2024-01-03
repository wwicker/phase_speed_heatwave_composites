import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
import os

from heat_util import basic_metrics
from wave_util import centroid, integral
from composite_util import composite_dates, parametric_bootstrap





def estimate_composite_significance(series,field,percentile=0.9,nrandom=10000,steps=1):
    
    dates = composite_dates(series.rename(rolling='time'),percentile=percentile)
    composite = field.sel(rolling=dates)
    sig = parametric_bootstrap(composite,field,nrandom=nrandom,pvalue=0.1,steps=steps)
    
    return xr.Dataset(dict(mean=composite.mean('time'),sig=sig))


def plot(da_slow, da_fast, duration, filename):
    
    fig = plt.figure(figsize=(8,6))
    
    ax1 = fig.add_subplot(3,1,1,projection=ccrs.EqualEarth())
    da_slow['mean'].plot(ax=ax1,levels=np.linspace(-3,3,21),
                                              transform=ccrs.PlateCarree(),
                                              cmap=cmocean.cm.balance,
                                              extend='both',
                                              add_colorbar=False)
    ax1.contourf(da_slow['lon'],da_slow['lat'],da_slow['sig'].astype(np.double),
             transform=ccrs.PlateCarree(),
             levels=[0,0.5,1],hatches=['','..'],
             alpha=0)

    ax1.coastlines()
    ax1.set_title('')
    ax1.text(-0.05,0.5,'Amplfied Slow',weight='bold',
         va='bottom', ha='center',
         rotation='vertical', rotation_mode='anchor',
         transform=ax1.transAxes)
    ax1.set_extent([-180,180,0,90],ccrs.PlateCarree())


    ax2 = fig.add_subplot(3,1,2,projection=ccrs.EqualEarth())

    C = da_fast['mean'].plot(ax=ax2,levels=np.linspace(-3,3,21),
                                              transform=ccrs.PlateCarree(),
                                              cmap=cmocean.cm.balance,
                                              extend='both',
                                              add_colorbar=False)

    ax2.contourf(da_fast['lon'],da_fast['lat'],da_fast['sig'].astype(np.double),
             transform=ccrs.PlateCarree(),
             levels=[0,0.5,1],hatches=['','..'],
             alpha=0)

    ax2.coastlines()
    ax2.set_title('')
    ax2.text(-0.05,0.5,'Amplfied Fast',weight='bold',
         va='bottom', ha='center',
         rotation='vertical', rotation_mode='anchor',
         transform=ax2.transAxes)
    ax2.set_extent([-180,180,0,90],ccrs.PlateCarree())
    
    
    ax3 = fig.add_subplot(3,1,3,projection=ccrs.EqualEarth())
    
    C2 = duration.plot(ax=ax3,levels=np.arange(3,6,0.5),
                       transform=ccrs.PlateCarree(),
                                              cmap=cmocean.cm.rain,
                                              extend='max',
                                              add_colorbar=False)
    ax3.coastlines()
    ax3.set_title('')
    ax3.text(-0.05,0.5,'Amplfied Fast',weight='bold',
         va='bottom', ha='center',
         rotation='vertical', rotation_mode='anchor',
         transform=ax2.transAxes)
    ax3.set_extent([-180,180,0,90],ccrs.PlateCarree())

    

    fig.subplots_adjust(0,0,1,1,0.2,0.2)

    cbar = plt.colorbar(C,ax=[ax1,ax2],orientation='vertical',shrink=0.95,pad=0.05,aspect=30)
    cbar.set_label('Anomalous heatwave frequency [days / window]',size=10)
    
    cbar = plt.colorbar(C2,ax=ax3,orientation='vertical',shrink=0.95,pad=0.05,aspect=15)
    cbar.set_label('Mean heatwave length [days]',size=9)

    plt.savefig(filename,bbox_inches='tight',dpi=300)



def main():
    
    work = os.environ.get('WORK')+'/'
    plt.rcParams.update({'font.size': 14})
    
    
    spectra = xr.open_dataarray(work+'wolfgang/spectra_30days_65-35N_wave5-8.nc')
    
    ## aggregate slow and fast wave energy for multiple wavenumbers
    slow = []
    fast = []
    
    for w in spectra['wavenumber']:                       
        
        # create time series for composite construction
        bound = centroid(spectra.sel(wavenumber=w).mean('rolling'))
        
        slow.append(integral(spectra.sel(wavenumber=w),bound.values,-30))
        fast.append(integral(spectra.sel(wavenumber=w),30,bound.values))
    
    slow = xr.concat(slow,dim='wavenumber')
    fast = xr.concat(fast,dim='wavenumber')

    slow = slow.sum('wavenumber')
    fast = fast.sum('wavenumber')
    
    print(slow)
    
    
    ## HEATWAVE metric

    # list of json files
    directory = work+'wolfgang/detrended_9ylowpass/heat/'
    files = [directory+f for f in os.listdir(directory) if f.endswith('.json')]
    files.sort()

    # binned time series of heatwave metric
    metric = basic_metrics(files,start_month=6,days_per_window=30,season_days=30+31+30)
    
    duration = metric['length'].mean('rolling')
    
    print(duration)
    
    days = metric['frequency'] * metric['length']
    days = days.where(np.logical_not(np.isnan(days)),other=0)
    
    print(days)
    
    
    composite_heat_slow = estimate_composite_significance(slow,days-days.mean('rolling'),percentile=0.9,nrandom=10000,steps=10)
    composite_heat_fast = estimate_composite_significance(fast,days-days.mean('rolling'),percentile=0.9,nrandom=10000,steps=10)
    
    print(composite_heat_slow)
    
    
    ## Plotting
    
    plot(composite_heat_slow,composite_heat_fast,duration,
         '/users/wwicker/ERA5_heatwaves/plots/composite_mean_heat_duration.png')



if __name__ == '__main__':
    
    main()
    
    
    
