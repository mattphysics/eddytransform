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


def query_mapping_varnames(from_col, to_col, value):
    '''
    Translate variable names from AMIP to FESOM, and GRIB2 paramIds
    # Example usage:
    query_mapping_varnames('amip', 'fesom', 'avg_sst')  # 'msst'
    query_mapping_varnames('fesom', 'amip', 'msst')     # 'avg_sst'
    '''
    data = {
        'param': [235089, 235083, 235084, 228005, 228004, 235034, 235033, 172228, 
                  228006, 235159, 235137, 235136, 235038, 235037, 235040, 235039,
                  235165, 235166],
        'amip': ['avg_2sh', 'avg_ci', 'avg_sst', 'mean10ws', 'mean2t', 'mslhf','msshf', 'tprate', 
                 'meantcc', 'avg_blh', 'avg_tcwv', 'avg_tcw','msnlwrf', 'msnswrf', 'mtnlwrf', 
                 'mtnswrf', 'avg_10u', 'avg_10v'],
        'fesom': ['m2sh', 'mci', 'msst', 'mean10ws', 'mean2t', 'mslhf', 'msshf', 'tprate', 
                  'meantcc', 'mblh', 'mtcwv', 'mtcw', 'msnlwrf', 'msnswrf', 'mtnlwrf', 
                  'mtnswrf', 'm10u', 'm10v'],
        'cmor': [None] * 18
    }
    df = pd.DataFrame(data)
    return df.loc[df[from_col] == value, to_col].squeeze()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year')
parser.add_argument('kind')
parser.add_argument('model')
parser.add_argument('exp')
parser.add_argument('dataset')
parser.add_argument('realm')
parser.add_argument('grid')
parser.add_argument('--vn',default=None)
parser.add_argument('--rw',default=None)
args = parser.parse_args()

year = int(args.year)
kind = args.kind
MODEL = args.model
EXP = args.exp
DATASET = args.dataset
REALM = args.realm
GRID = args.grid
vn = args.vn
rw = args.rw

print(args)

# sys.exit()

if rw == '0':
    print('rotate_wind = False')
    rotate_winds = False
else:
    print('rotate_wind = True')
    rotate_winds = True

print('Process year %i, kind %s, vn %s, rotate_winds= %s' % (year,kind,vn,rotate_winds))

# Specify EERIE data
CATALOG = "https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml"
# MODEL = 'ifs-fesom2-sr'
# DATASET = '2D_daily_avg'
# EXP = 'hist-1950'
# EXP = 'highres-future-ssp245'
if MODEL == 'ifs-fesom2-sr':
    VERSION = 'v20240304'
# REALM = 'atmos'
# GRID = 'gr025'


gridtype = 'regular' # irregular # is the data on a 'regular' lat on grid or other?
realization = 1
# extent = [0,360,-60,-25] # None # geographical region to only match eddies in
# extent = [0,360,-60,-25] # None # geographical region to only match eddies in
extent = [360-82,360-20,28,60] # None # geographical region to only match eddies in
print('Only look for eddies in region: %s' % extent)
freq = '10D' # '1D' # frequency of sampling - default is '10D': pick every 10th day
Nmax = None #10 # None # maximum number of eddies. if not None, pick the first <Nmax> after the other selections have been done. Meant for testing. 

if vn != None:
    varnames = [vn]
else:
    varnames = ['avg_2sh','avg_ci','avg_sst','mean10ws','mean2t','mslhf','msshf','tprate','meantcc','avg_blh','avg_tcwv','avg_tcw','msnlwrf','msnswrf','mtnlwrf','mtnswrf']

if vn == 'sfcWind':
    print("vn == 'sfcWind' : construct from uas and vas")
    varnames = []

# varnames = ['avg_sst','avg_ci','mean2t','tprate','mslhf','msshf','mean10ws','avg_2sh']#,'avg_10u','avg_10v']

# OUTPUT PROCESSING OPTIONS
# 1. output = 'single'
#          outname is a pattern, individual eddies get written to <outname>_%i.nc . 
# 2. output = 'all'
#          outname is a filename, all eddies are concatenated and writen to <outname>.nc

# output  = 'single_var' # 'single' # 'all' # write output to one file per eddy, or 'all' at once
output  = 'single' # 'single' # 'all' # write output to one file per eddy, or 'all' at once
outname = 'eddy_r_%i_' % (realization)  # 'eddies_test.nc'

# output  = 'all' # write output to one file per eddy, or 'all' at once
# outname = 'eddies_test.nc'

# OUTPUTROOTDIR = '/ec/fws5/lb/project/eerie/output/'
OUTPUTROOTDIR = '/scratch/b/b382473/composites/'

# ===============

transform_settings = dict(
    DOMAIN_HALF_WIDTH_IN_DEGREES = 20, # domain half width
    AVG_WIND_EDDY_RADIUSES = 3, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u", # zonal surface wind velocity, for eddy rotation
    VPARAM = "avg_10v",  # meridional surface wind velocity, for eddy rotation
    output = output,  # output mode ('single','single_var','all')
    rotate_winds = rotate_winds # also computed rotated wind field and save as well
)

# ===============

assert output in ['single','single_var','all']

assert kind in ['cyclonic','anticyclonic']

if output in ['single','single_var']:
    print('Write each eddy into a file, individually: output = %s' % output)
    if kind == 'anticyclonic':
        # fname_root_dir = OUTPUTROOTDIR + '%s/processed/composites_rot_acyc/' % (EXP)
        # fname_root_dir = OUTPUTROOTDIR + '%s/processed/composites_GS_rot_acyc/' % (EXP)
        fname_root_dir = OUTPUTROOTDIR + '%s/%s/%s/%s/%s/%s/composites_GS_rot_acyc/' % (MODEL,EXP,VERSION,REALM,GRID,DATASET)
    elif kind == 'cyclonic':
        # fname_root_dir = OUTPUTROOTDIR + '%s/processed/composites_rot_cyc/' % (EXP)
        # fname_root_dir = OUTPUTROOTDIR + '%s/processed/composites_GS_rot_cyc/' % (EXP)
        fname_root_dir = OUTPUTROOTDIR + '%s/%s/%s/%s/%s/%s/composites_GS_rot_cyc/' % (MODEL,EXP,VERSION,REALM,GRID,DATASET)
    else:
        raise ValueError('kind %s is not defined' % kind)
    fname_root = fname_root_dir + outname
elif output == 'all':
    print('Write all eddies into one file')
    fname_root = None
    fname_root_dir = OUTPUTROOTDIR

transform_settings['fname_root'] = fname_root
print(transform_settings)
# sys.exit()

if not os.path.exists(fname_root_dir):
    print('Directory does not exists yet, create: %s' % fname_root_dir)
    os.system('mkdir -p %s' % fname_root_dir)

# Read data
print('Open datasets')
catalog = intake.open_catalog(CATALOG)
ds = catalog['dkrz']['disk']['model-output'][MODEL][EXP][VERSION][REALM][GRID][DATASET](chunks={}).to_dask()

if REALM == 'atmos':
    print('Rename variables to fit IFS-AMIP conventions')
    if 'fesom' in MODEL:
        varnames_rename = varnames + [transform_settings['UPARAM'],transform_settings['VPARAM']]
        varnames_model = [query_mapping_varnames('amip', 'fesom', v) for v in varnames_rename]
    elif 'ifs-amip' in MODEL:
        print('Already IFS-AMIP, assume correct')
    else:
        raise ValueError('renaming not defined for model %s' % MODEL)

    ds = ds.rename(dict(zip(varnames_model, varnames_rename)))
else:
    print('Do not rename')

# ds = catalog['dkrz']['disk']['model-output']['ifs-fesom2-sr']['highres-future-ssp245']['v20240304']['atmos']['gr025']['2D_daily_avg'](chunks={}).to_dask()
if gridtype == 'regular':
    print('Reshape regular catalog to 2D coordinates')
    ds = et.reshape_latlon_1d_to_latlon_2d(ds)
    sort = True
else:
    sort = False

if 'realization' in ds.dims:
    ds = ds.sel(realization=realization)

lontype = et.identify_lontype(ds=ds)
if lontype == '180':
    print('Change longitude to 0-360 to align with eddy tracks')
    ds = et.lon_180_to_360(ds,sort=sort)

print('Load eddy tracks')
tracks0 = et.load_eddy_tracks(kind,platform='Levante')
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

print('Start looping over %i eddies' % tracks_year_extent_freq['obs'].size)
eddies = et.loop_over_eddies(
    ds,
    tracks_year_extent_freq,
    varnames,
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
