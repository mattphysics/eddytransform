import intake

import xarray as xr
xr.set_options(keep_attrs=True)

import eddytransform as et
import os

# Specify EERIE data
CATALOG = "/home/neam/code/intake_atos/eerie.yaml"
MODEL = "ifs-amip"
DATASET = "2D_24h_0.25deg"
gridtype = 'regular' # irregular # is the data on a 'regular' lat on grid or other?
EXP = 'tco1279-eerie_production_202407'
realization = 1

# time/lon/lat coordinates of imagined eddy
TIME = '2021-01-01 12:00:00'
EDDY_LON = -71
EDDY_LAT = 39.5

varname = 'avg_sst'

outfilename = 'composite_test.nc'
OUTPUTROOTDIR = '/ec/fws5/lb/project/eerie/output/'

# ===============

# Settings for eddy transformation - decent default values
DOMAIN_HALF_WIDTH_IN_DEGREES = 20 # domain half width
AVG_WIND_EDDY_RADIUSES = 3 # Number of eddy radiuses used for calculating direction of large-scale winds. 
RESAMPLE_EDDY_RADIUSES = 3 # Number of eddy radiuses to sample in transformed composite coordinates.
RESAMPLE_DENSITY = 30 # Number of data points per eddy radius in transformed composite coordinates.
UPARAM = "avg_10u" # zonal surface wind velocity, for eddy rotation
VPARAM = "avg_10v" # meridional surface wind velocity, for eddy rotation

# ===============

fname_out = OUTPUTROOTDIR + outfilename

# Read & process data 
print('Open datasets')
catalog = intake.open_catalog(CATALOG)
ds = catalog[MODEL][EXP][DATASET].to_dask()

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


eddy_centered = et.transform_eddy(
    ds,
    COMPOSITE_PARAM = varname,
    TIME_DX = TIME,
    EDDY_LON = EDDY_LON,
    EDDY_LAT = EDDY_LAT,
    DOMAIN_HALF_WIDTH_IN_DEGREES = DOMAIN_HALF_WIDTH_IN_DEGREES,
    EDDY_RADIUS = 100,
    AVG_WIND_EDDY_RADIUSES = AVG_WIND_EDDY_RADIUSES,
    RESAMPLE_EDDY_RADIUSES = RESAMPLE_EDDY_RADIUSES,
    RESAMPLE_DENSITY = RESAMPLE_DENSITY,
    UPARAM = UPARAM,
    VPARAM = VPARAM,
    # ds_wind = ds_wind
)
eddy_centered_winds = et.transform_winds(
    ds[[UPARAM,VPARAM]],
    TIME_DX = TIME,
    EDDY_LON = EDDY_LON,
    EDDY_LAT = EDDY_LAT,
    DOMAIN_HALF_WIDTH_IN_DEGREES = DOMAIN_HALF_WIDTH_IN_DEGREES,
    EDDY_RADIUS = 100,
    AVG_WIND_EDDY_RADIUSES = AVG_WIND_EDDY_RADIUSES,
    RESAMPLE_EDDY_RADIUSES = RESAMPLE_EDDY_RADIUSES,
    RESAMPLE_DENSITY = RESAMPLE_DENSITY,
    UPARAM = UPARAM,
    VPARAM = VPARAM
)

eddy_centered = xr.merge(
    [
        eddy_centered,
        eddy_centered_winds
    ]
)

print('Saving to netcdf')
eddy_centered.to_netcdf(fname_out)

print('Done')
