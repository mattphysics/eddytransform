from pathlib import Path

import intake
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

import xarray as xr
xr.set_options(keep_attrs=True)

# import eddytransform
import eddytransform3 as eddytransform
import os
import importlib
importlib.reload(eddytransform)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year')
parser.add_argument('kind')
args = parser.parse_args()

year = int(args.year)
kind = args.kind

print('Process year %i, kind %s' % (year,kind))

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
#DATASET = "2D_24h_0.1deg"!
DATASET = "2D_24h_0.25deg"
# EXP = 'tco1279-eerie_production_202407'
# EXP = 'tco1279-eerie_production_202408-c_0_a_LR20'
# EXP = 'tco399-eerie_production_202407'
EXP = 'tco399-eerie_production_202408-c_0_a_LR20'
realization = 1
# kind = 'anticyclonic'
# kind = 'cyclonic'
assert kind in ['cyclonic','anticyclonic']
#ROOTDIR = "/hpcperm/necr/EERIE/"
# DOMAIN_HALF_WIDTH_IN_DEGREES = 10 # Used to select of composite data before resampling to avoid loading global data.
#DOMAIN_HALF_WIDTH_IN_DEGREES = 20 # Used to select of composite data before resampling to avoid loading global data.
#TIME_IDX = 10 # User-specified time index to select data from catalogue (e.g. from tracking algorithm)
#EDDY_LON = -71 # User-specified eddy location (e.g. from tracking algorithm)
#EDDY_LAT = 39.5

if kind == 'anticyclonic':
    fname_root_dir = '/ec/fws5/lb/project/eerie/output/%s/processed/composites_rot/' % (EXP)
    fname_root = fname_root_dir + 'eddy_r_%i_' % (realization)
elif kind == 'cyclonic':
    fname_root_dir = '/ec/fws5/lb/project/eerie/output/%s/processed/composites_rot_cyc/' % (EXP)
    fname_root = fname_root_dir + 'eddy_r_%i_' % (realization)
else:
    raise ValueError('kind %s is not defined' % kind)

if not os.path.exists(fname_root_dir):
    print('Directory does not exists yet, create: %s' % fname_root_dir)
    os.system('mkdir -p %s' % fname_root_dir)

# Read data and select domain centr
print('Open datasets')
catalog = intake.open_catalog(CATALOG)
ds = eddytransform.reshape_latlon_1d_to_latlon_2d(catalog[MODEL][EXP][DATASET].to_dask())

if 'realization' in ds.dims:
    ds = ds.sel(realization=realization)

# ds = gs.lon_180_to_360(ds)
ds = lon_180_to_360(ds)

# Select a region surrounding eddy with sufficient buffer for resampling in rotated/scaled coordinates.
#ds_region = ds.isel(time=TIME_IDX).sel(
#    lon=slice(EDDY_LON-DOMAIN_HALF_WIDTH_IN_DEGREES, EDDY_LON+DOMAIN_HALF_WIDTH_IN_DEGREES),
#    lat=slice(EDDY_LAT-DOMAIN_HALF_WIDTH_IN_DEGREES, EDDY_LAT+DOMAIN_HALF_WIDTH_IN_DEGREES),
#)

tracks0 = eddytransform.load_eddy_tracks(kind)
#tracks2021 = tracks0.sel(obs=tracks0['year'] == 2021)
#tracks2021 = tracks0.sel(obs=tracks0['year'] == 2010)
tracks2021 = tracks0.sel(obs=tracks0['year'] == year)
tracks2021_SO = tracks2021.sel(obs = 
      (tracks2021['latitude'] > -60)
    & (tracks2021['latitude'] < -25)
    & (tracks2021['longitude'] > 0)
)

#dates = pd.date_range('2021-01-01 12:00','2021-12-31',freq='10D')
#dates = pd.date_range('2010-01-01 12:00','2021-12-31',freq='10D')
dates = pd.date_range('%i-01-01 12:00' % year,'%i-12-31' % year,freq='10D')
tracks2021_SO_10days = tracks2021_SO.sel(obs=[t in dates for t in tracks2021_SO['time'].values])
# tracks2021_SO_10days_10points = tracks2021_SO_10days.isel(obs=slice(0,None,10))
# tracks2021_SO_10days_10points = tracks2021_SO_10days.isel(obs=slice(1,None,10))
# tracks2021_SO_10days_10points = tracks2021_SO_10days.isel(obs=slice(2,None,10))
tracks2021_SO_10days_10points = tracks2021_SO_10days#.isel(obs=slice(1,None,10))


transform_settings = dict(
    #DOMAIN_HALF_WIDTH_IN_DEGREES = 10, # domain half width
    DOMAIN_HALF_WIDTH_IN_DEGREES = 20, # domain half width
    #EDDY_RADIUS = 100, # User-specified eddy radius in km.
    # AVG_WIND_EDDY_RADIUSES = 2, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    AVG_WIND_EDDY_RADIUSES = 3, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    # AVG_WIND_EDDY_RADIUSES = 5, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    # AVG_WIND_EDDY_RADIUSES = 10, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    # AVG_WIND_EDDY_RADIUSES = 14, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u", 
    VPARAM = "avg_10v",
    #fname_root='/ec/fws5/lb/project/eerie/output/%s/processed/composites_rot/eddy_' % EXP
    fname_root=fname_root
)

varnames = ['avg_sst','avg_ci','mean2t','tprate','mslhf','msshf','mean10ws','avg_2sh']#,'avg_10u','avg_10v']
#varname = 'avg_2sh' # 'mean10ws' #'msshf' # 'mslhf' # 'tprate' #'mean2t' # 'avg_ci'#'avg_sst' #'mean2t' #'avg_2sh' 'avg_10v'

print('Start looping')
eddies_AG = eddytransform.loop_over_eddies(
    ds,
    # tracks2021_SO,
    tracks2021_SO_10days_10points,
    # tracks2021_SO_10days, #_10points,
    # ['avg_sst','mean2t','tprate','mslhf','msshf','mean10ws','meantcc','avg_10u','avg_10v'],
    varnames,
    #[varname],
    #rotate_winds = False,
    rotate_winds = True,
    **transform_settings
)

print(eddies_AG)

# eddies_AG.to_netcdf('/ec/fws5/lb/project/eerie/testing/eddies_SO_1279a_10d_2020.nc')
# eddies_AG.to_netcdf('/ec/fws5/lb/project/eerie/testing/eddies_SO_a007.nc')
#eddies_AG.to_netcdf('/ec/fws5/lb/project/eerie/testing/eddies_SO_a008.nc')
#eddies_AG.to_netcdf('/ec/fws5/lb/project/eerie/testing/eddies_SO_%s.nc' % varname)
#eddies_AG.to_netcdf('/ec/fws5/lb/project/eerie/testing/eddies_SO_%s_short.nc' % varname)

print('Done')
