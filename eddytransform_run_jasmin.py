from pathlib import Path

import intake
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

import xarray as xr
xr.set_options(keep_attrs=True)

import eddytransform as et
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year')
parser.add_argument('kind')
args = parser.parse_args()

year = int(args.year)
kind = args.kind

print('Process year %i, kind %s' % (year,kind))

# Specify EERIE data
MODEL='HadGEM3-GC5-EERIE-N640'
EXP='highresSST-present'
DATASET='Aday'
gridtype = 'regular' # irregular # is the data on a 'regular' lat on grid or other?
realization = 1
extent = [0,360,-60,-25] # None # geographical region to only match eddies in
freq = '10D' # '1D' # frequency of sampling - default is '10D': pick every 10th day
Nmax = None #10 # None # maximum number of eddies. if not None, pick the first <Nmax> after the other selections have been done. Meant for testing. 

varnames = ['ts','pr','hfls']#,'hfss'] # ,'uas','vas'

# varnames = ['avg_sst','avg_ci','mean2t','tprate','mslhf','msshf','mean10ws','avg_2sh']#,'avg_10u','avg_10v']
#varname = 'avg_2sh' # 'mean10ws' #'msshf' # 'mslhf' # 'tprate' #'mean2t' # 'avg_ci'#'avg_sst' #'mean2t' #'avg_2sh' 'avg_10v'

# OUTPUT PROCESSING OPTIONS
# 1. output = 'single'
#          outname is a pattern, individual eddies get written to <outname>_%i.nc . 
# 2. output = 'all'
#          outname is a filename, all eddies are concatenated and writen to <outname>.nc

output  = 'single_var' # 'single' # 'all' # write output to one file per eddy, or 'all' at once
outname = 'eddy_r_%i_' % (realization)  # 'eddies_test.nc'

# output  = 'all' # write output to one file per eddy, or 'all' at once
# outname = 'eddies_test.nc'

OUTPUTROOTDIR = '/gws/nopw/j04/eerie/aengenh/output/'

# ===============

transform_settings = dict(
    DOMAIN_HALF_WIDTH_IN_DEGREES = 20, # domain half width
    AVG_WIND_EDDY_RADIUSES = 3, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "uas", # zonal surface wind velocity, for eddy rotation
    VPARAM = "vas",  # meridional surface wind velocity, for eddy rotation
    output = output  # output mode ('single','single_var','all')
)

# ===============

assert output in ['single','single_var','all']

assert kind in ['cyclonic','anticyclonic']

if output in ['single','single_var']:
    print('Write each eddy into a file, individually: output = %s' % output)
    if kind == 'anticyclonic':
        fname_root_dir = OUTPUTROOTDIR + '%s/%s/processed/composites_rot_acyc/' % (MODEL,EXP)
    elif kind == 'cyclonic':
        fname_root_dir = OUTPUTROOTDIR + '%s/%s/processed/composites_rot_cyc/' % (MODEL,EXP)
    else:
        raise ValueError('kind %s is not defined' % kind)
    fname_root = fname_root_dir + outname
elif output == 'all':
    print('Write all eddies into one file')
    fname_root = None
    fname_root_dir = OUTPUTROOTDIR

transform_settings['fname_root'] = fname_root

if not os.path.exists(fname_root_dir):
    print('Directory does not exists yet, create: %s' % fname_root_dir)
    os.system('mkdir -p %s' % fname_root_dir)

# Read data
print('Open datasets')
ds = et.open_mohc_jasmin(
    MODEL,
    EXP,
    DATASET,
    varnames + [transform_settings['UPARAM'],transform_settings['VPARAM']],
    sel=dict(time=slice('2000','2009'))
)

# if gridtype == 'regular':
#     print('Reshape regular catalog to 2D coordinates')
#     ds = et.reshape_latlon_1d_to_latlon_2d(ds)
#     sort = True
# else:
#     sort = False

if 'realization' in ds.dims:
    ds = ds.sel(realization=realization)

lontype = et.identify_lontype(ds=ds)
if lontype == '180':
    print('Change longitude to 0-360 to align with eddy tracks')
    ds = et.lon_180_to_360(ds,sort=sort)

print('Load eddy tracks')
tracks0 = et.load_eddy_tracks(kind,platform='Jasmin')
tracks_year = tracks0.sel(obs=tracks0['year'] == year)
if extent is not None:
    x0, x1, y0, y1 = extent
    tracks_year_extent = tracks_year.sel(obs = 
        (tracks_year['latitude'] > y0)
        & (tracks_year['latitude'] < y1)
        & (tracks_year['longitude'] > x0)
        & (tracks_year['longitude'] < x1)
    )
else:
    tracks_year_extent = tracks_year

dates = pd.date_range('%i-01-01 12:00' % year,'%i-12-31' % year,freq=freq)
tracks_year_extent_freq = tracks_year_extent.sel(obs=[t in dates for t in tracks_year_extent['time'].values])

if Nmax:
    print('Picking only the first Nmax = %i eddies' % Nmax)
    tracks_year_extent_freq = tracks_year_extent_freq.isel(obs=slice(Nmax))

print('Start looping')
eddies = et.loop_over_eddies(
    ds,
    tracks_year_extent_freq,
    varnames,
    rotate_winds = True,
    **transform_settings
)

if output in ['single','single_var']:
    assert eddies == 0
    print('Saving eddies individually has been successful: fname_root = %s' % fname_root)
elif output == 'all':
    print('Result of compositing: ')
    print(eddies)
    print('Saving eddies to: %s%s' % (OUTPUTROOTDIR,outname))
    eddies.to_netcdf('%s%s' % (OUTPUTROOTDIR,outname))

print('Done')
