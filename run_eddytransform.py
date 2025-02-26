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
import os
import importlib
importlib.reload(eddytransform)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year')
args = parser.parse_args()

year = int(args.year)

print('Process year %i' % year)

# Specify eddy track data
PLATFORM = 'ATOS'
SOURCE = 'AVISO'
# kind = 'anticyclonic'
kind = 'cyclonic'
EDDY_PATH = None # file path to eddy source if not through platform,source,kind

# bounding box to select eddies
eddy_lon_min = 0
eddy_lon_max = 360
eddy_lat_min = -60 #-90
eddy_lat_max = -25 # 90

# select every nth day
# eddy_freq = '1D'
eddy_freq = '10D'

# select every 10th eddy, starting with <eddy_thin> (if not None)
eddy_thin = None
# eddy_thin = 1

# Specify data to be composited
CATALOG = "/home/necr/EERIE/intake_atos/eerie.yaml"
MODEL = "ifs-amip"
#DATASET = "2D_24h_0.1deg"!
DATASET = "2D_24h_0.25deg"
# EXP = 'tco1279-eerie_production_202407'
# EXP = 'tco1279-eerie_production_202408-c_0_a_LR20'
EXP = 'tco399-eerie_production_202407'
realization = 1

varnames = ['avg_sst','avg_ci','mean2t','tprate','mslhf','msshf','mean10ws','avg_2sh']#,'avg_10u','avg_10v']

transform_settings = dict(
    DOMAIN_HALF_WIDTH_IN_DEGREES = 20, # domain half width
    AVG_WIND_EDDY_RADIUSES = 3, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u", 
    VPARAM = "avg_10v"
)

ROOTDIR_OUTPUT = '/ec/fws5/lb/project/eerie/output'

filename_out = None # if not None, save all composited eddies into this file, not each eddy separately following the <fname_root> logic below.

# ============================================================================

if filename_out == None:
    if kind == 'anticyclonic':
        fname_root_dir = '%s/%s/processed/composites_rot/' % (ROOTDIR_OUTPUT,EXP)
        fname_root = fname_root_dir + 'eddy_r_%i_' % (realization)
    elif kind == 'cyclonic':
        fname_root_dir = '%s/%s/processed/composites_rot_cyc/' % (ROOTDIR_OUTPUT,EXP)
        fname_root = fname_root_dir + 'eddy_r_%i_' % (realization)
    else:
        raise ValueError('kind %s is not defined' % kind)

    if not os.path.exists(fname_root_dir):
        print('Directory does not exists yet, create: %s' % fname_root_dir)
        os.system('mkdir -p %s' % fname_root_dir)

    transform_settings['fname_root']=fname_root
else:
    fname_root = None
    transform_settings['fname_root']=fname_root

# Read data
print('Open datasets')
catalog = intake.open_catalog(CATALOG)
ds = eddytransform.reshape_latlon_1d_to_latlon_2d(catalog[MODEL][EXP][DATASET].to_dask())

if 'realization' in ds.dims:
    ds = ds.sel(realization=realization)

ds = eddytransform.lon_180_to_360(ds)

tracks0 = eddytransform.load_eddy_tracks(kind=kind,source=SOURCE,platform=PLATFORM,path=EDDY_PATH)

tracks_year = tracks0.sel(obs=tracks0['year'] == year)
tracks_year_region = tracks_year.sel(obs = 
      (tracks_year['latitude'] > eddy_lat_min)
    & (tracks_year['latitude'] < eddy_lat_max)
    & (tracks_year['longitude'] > eddy_lon_min)
    & (tracks_year['longitude'] < eddy_lon_max)
)

dates = pd.date_range('%i-01-01 12:00' % year,'%i-12-31 12:00' % year,freq=eddy_freq)
tracks_year_region_days = tracks_year_region.sel(obs=[t in dates for t in tracks_year_region['time'].values])
if eddy_thin is not None:
    tracks_select = tracks_year_region_days.isel(obs=slice(eddy_thin,None,10))
else:
    tracks_select = tracks_year_region_days

print('Start looping')
eddies_composited = eddytransform.loop_over_eddies(
    ds,
    tracks_select,
    varnames,
    rotate_winds = True,
    **transform_settings
)

print(eddies_composited)

if eddies_composited != 0:
    print('Save composited dataset to %s' % filename_out)
    eddies_composited.to_netcdf(filename_out)

print('Done')
