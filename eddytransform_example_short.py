import intake

import xarray as xr
xr.set_options(keep_attrs=True)

import eddytransform as et
import os

import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import cartopy.crs as ccrs

# Specify EERIE data
gridtype = 'regular' # irregular # is the data on a 'regular' lat on grid or other?
res = '0.25deg' # 0.1deg
infilename = 'data/data_0.25deg.nc'

# gridtype = 'regular' # irregular # is the data on a 'regular' lat on grid or other?
# res = '0.1deg'
# infilename = 'data/data_0.1deg.nc'

# gridtype = 'irregular' # is the data on a 'regular' lat on grid or other?
# res = 'native'
# infilename = 'data/data_native.nc'

# time/lon/lat coordinates of imagined eddy
TIME = '2021-01-11 12:00:00'
EDDY_LON = 289
EDDY_LAT = 39.5

varname = 'avg_sst'

outfilename = 'composite_test_%s.nc' % res
OUTPUTROOTDIR = 'output/'
PLOTDIR = 'plots/'

fname_out = OUTPUTROOTDIR + outfilename

# ===============

# Settings for eddy transformation - decent default values
DOMAIN_HALF_WIDTH_IN_DEGREES = 20 # domain half width
AVG_WIND_EDDY_RADIUSES = 3 # Number of eddy radiuses used for calculating direction of large-scale winds. 
RESAMPLE_EDDY_RADIUSES = 3 # Number of eddy radiuses to sample in transformed composite coordinates.
RESAMPLE_DENSITY = 30 # Number of data points per eddy radius in transformed composite coordinates.
UPARAM = "avg_10u" # zonal surface wind velocity, for eddy rotation
VPARAM = "avg_10v" # meridional surface wind velocity, for eddy rotation

# ===============

# Read & process data 
print('Open datasets')
ds = xr.open_dataset(
    infilename
)

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


# PLOTTING
print('Plotting')

# Plot original data on lat-lon grid
print('Plot original data on lat-lon grid')
if gridtype == 'irregular':
    fig, ax = plt.subplots(constrained_layout=True,subplot_kw=dict(projection=ccrs.PlateCarree()))
    ds_region = et.sel_region(
        ds.sel(
            time=TIME
        ),
        EDDY_LON,EDDY_LAT,5
    )
    p = ax.scatter(ds_region['lon'],ds_region['lat'],c=ds_region[varname],s=5)
    cbar = fig.colorbar(p,ax=ax,label='avg_sst [K]')

    ax.set_facecolor('grey')
    ax.gridlines(draw_labels=True)
    fig.savefig('%s/eddy_%s_original_grid.png' % (PLOTDIR,res),dpi=200)
else:
    fig, ax = plt.subplots(constrained_layout=True,subplot_kw=dict(projection=ccrs.PlateCarree()))
    cf = et.sel_region(
        ds[varname].sel(
            time=TIME
        ),
        EDDY_LON,EDDY_LAT,5
    ).plot(transform=ccrs.PlateCarree())

    quiver = et.sel_region(
        ds[[UPARAM,VPARAM]].sel(
            time=TIME
        ),
        EDDY_LON,EDDY_LAT,5
    ).isel(lon=slice(None,None, 2),lat=slice(None,None, 2)).plot.quiver(
        x='lon', y='lat', u='avg_10u', v='avg_10v', scale=100
    ) 

    ax.set_facecolor('grey')
    ax.gridlines(draw_labels=True)
    fig.savefig('%s/eddy_%s_original_grid.png' % (PLOTDIR,res),dpi=200)

print('Plot rotated & scaled eddy on eddy-centric grid')
fig, ax = plt.subplots(constrained_layout=True)
eddy_centered['avg_sst'].plot(ax=ax)

theta = np.linspace(0,2*np.pi,100)

plt.plot(
    np.cos(theta),
    np.sin(theta),
    'w'
)

eddy_centered.isel(
    x=slice(None,None, 5),y=slice(None,None, 5)
).plot.quiver(
    x='x',y='y',u='avg_10u',v='avg_10v',scale=100,
    ax=ax
)

ax.set_facecolor('grey')
ax.grid(True)
fig.savefig('%s/eddy_%s_rotated_scaled.png' % (PLOTDIR,res),dpi=200)


print('Done')
