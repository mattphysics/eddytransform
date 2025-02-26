from pathlib import Path

import intake
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

import xarray as xr
xr.set_options(keep_attrs=True)

import eddytransform
import importlib
importlib.reload(eddytransform)

def lon_360_to_180(da,lon='lon',inplace=False):
    '''
    Convert longitude from (0-360 E) to (180W - 180E)
    Reference: https://gis.stackexchange.com/a/201793
    '''
    if inplace == True:
        raise ValueError('option inplace=True in not functional, does not sort longitude appropriately.')
    if not lon in da.coords:
        print('Try to infer: longitude is named "longitude?"')
        if not 'longitude' in da.coords:
            print('Cannot find coordinate named "lon" or "longitude"')
            raise ValueError 
        else:
            print("Found coordinate 'longitude'")
            lon = 'longitude'
    if inplace == True:
        da[lon] = np.mod(da[lon] + 180, 360) - 180
        da = da.sortby(lon)
    else:
        lon_attrs = da[lon].attrs
        da_out = da.assign_coords({lon:np.mod(da[lon] + 180, 360) - 180}).sortby(lon)
        da_out[lon].attrs = lon_attrs
        return da_out
#     return da.sortby('lon')

def lon_180_to_360(da,lon='lon',inplace=False):
    '''
    Convert longitude from (180W - 180E) to (0-360 E)
    Reference: https://gis.stackexchange.com/a/201793
    '''
    if inplace == True:
        raise ValueError('option inplace=True in not functional, does not sort longitude appropriately.')
    if not lon in da.coords:
        print('Try to infer: longitude is named "longitude?"')
        if not 'longitude' in da.coords:
            print('Cannot find coordinate named "lon" or "longitude"')
            raise ValueError 
        else:
            lon = 'longitude'
    if inplace == True:
        da[lon] = np.mod(da[lon], 360)
        da = da.sortby(lon)
    else:
        lon_attrs = da[lon].attrs
        da_out = da.assign_coords({lon:np.mod(da[lon], 360)}).sortby(lon)
        da_out[lon].attrs = lon_attrs
        return da_out
#     return da.sortby('lon')

# Specify EERIE data
CATALOG = "/home/necr/EERIE/intake_atos/eerie.yaml"
MODEL = "ifs-amip"
DATASET = "2D_24h_0.1deg"
EXP = 'tco1279-eerie_production_202407'
ROOTDIR = "/hpcperm/necr/EERIE/"
DOMAIN_HALF_WIDTH_IN_DEGREES = 10 # Used to select of composite data before resampling to avoid loading global data.
TIME_IDX = 10 # User-specified time index to select data from catalogue (e.g. from tracking algorithm)
EDDY_LON = -71 # User-specified eddy location (e.g. from tracking algorithm)
EDDY_LAT = 39.5


# Read data and select domain centr
catalog = intake.open_catalog(CATALOG)
ds = eddytransform.reshape_latlon_1d_to_latlon_2d(catalog[MODEL][EXP][DATASET].to_dask())

# ds = gs.lon_180_to_360(ds)
ds = lon_180_to_360(ds)

# Select a region surrounding eddy with sufficient buffer for resampling in rotated/scaled coordinates.
ds_region = ds.isel(time=TIME_IDX).sel(
    lon=slice(EDDY_LON-DOMAIN_HALF_WIDTH_IN_DEGREES, EDDY_LON+DOMAIN_HALF_WIDTH_IN_DEGREES),
    lat=slice(EDDY_LAT-DOMAIN_HALF_WIDTH_IN_DEGREES, EDDY_LAT+DOMAIN_HALF_WIDTH_IN_DEGREES),
)

tracks0 = eddytransform.load_eddy_tracks('anticyclonic')
tracks2021 = tracks0.sel(obs=tracks0['year'] == 2021)
tracks2021_AG = tracks2021.sel(obs = 
      (tracks2021['latitude'] > -50)
    & (tracks2021['latitude'] < -37)
    & (tracks2021['longitude'] > 10)
    & (tracks2021['longitude'] < 90)
)

dates = pd.date_range('2021-01-01 12:00','2021-12-31',freq='10D')
tracks2021_AG_10days = tracks2021_AG.sel(obs=[t in dates for t in tracks2021_AG['time'].values])
tracks2021_AG_10days_10points = tracks2021_AG_10days.isel(obs=slice(0,None,10))


transform_settings = dict(
    DOMAIN_HALF_WIDTH_IN_DEGREES = 10, # domain half width
    EDDY_RADIUS = 100, # User-specified eddy radius in km.
    AVG_WIND_EDDY_RADIUSES = 2, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u", 
    VPARAM = "avg_10v"
)

# varnames = ['avg_sst','avg_ci','mean2t','tprate','mslhf','msshf','mean10ws','avg_2sh','avg_10u','avg_10v']
varname = 'avg_2sh' #'mean10ws' #'msshf' # 'mslhf' # 'tprate' # 'mean2t' # 'avg_ci' # 'avg_sst' #'mean2t' #'avg_2sh' 'avg_10v'


eddies_AG = eddytransform.loop_over_eddies(
    ds,
    #tracks2021_SO,
    #tracks2021_AG_10days_10points,
    tracks2021_AG_10days, #_10points,
    # ['avg_sst','mean2t','tprate','mslhf','msshf','mean10ws','meantcc','avg_10u','avg_10v'],
    # varnames,
    [varname],
    rotate_winds = False,
    **transform_settings
)

# eddies_AG.to_netcdf('/ec/fws5/lb/project/eerie/testing/eddies_SO.nc')
eddies_AG.to_netcdf('/ec/fws5/lb/project/eerie/testing/eddies_AG_%s.nc' % varname)
#eddies_AG.to_netcdf('/ec/fws5/lb/project/eerie/testing/eddies_AG_%s_short.nc' % varname)

print('Done')
