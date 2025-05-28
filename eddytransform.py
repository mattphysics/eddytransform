""" 
Functions to resample ocean eddy composites in an eddy-centric 
rotated/rescaled transformed coordinate system.

"""

import xarray as xr
import numpy as np
import scipy
import pandas as pd
import os
from glob import glob

def open_mohc_jasmin(
    model,
    exp,
    stream,
    varnames=None,
    sel={}
):
    '''
    Load Met Office output from netcdf files
    '''
    path_to_data = '/gws/nopw/j04/eerie/public/data/EERIE/EERIE/MOHC/{model}/{exp}/r1i1p1f1/{stream}'.format(model=model,exp=exp,stream=stream)
    if varnames is None:
        varnames = [d.split('/')[-1] for d in sorted(glob('{path}/*'.format(path=path_to_data)))]
        print('Get all variables')
    if not isinstance(varnames,list):
        varnames = [varnames]
    if sel != {}:
        print('Sub-select by: %s' % sel)
        
    if len(varnames) == 1:
        varname = varnames[0]
        print('Get one variable: %s' % varname)
        pattern = '{path}/{varname}/*/*/*.nc'.format(path=path_to_data,varname=varname)
        fnames = sorted(glob(pattern))
        print('Get %i files' % len(fnames))
        ds = xr.open_mfdataset(fnames,decode_timedelta=True)
        if 'height' in ds.coords:
            ds[varname].attrs['height'] = ds['height'].values
            ds = ds.drop_vars(['height'])
            ds = ds.sel(sel)
    else:
        print('Get variables: %s' % varnames)
        ds = []
        for varname in varnames:
            print(varname)
            pattern = '{path}/{varname}/*/*/*.nc'.format(path=path_to_data,varname=varname)
            fnames = sorted(glob(pattern))
            print('Get %i files' % len(fnames))
            dsi = xr.open_mfdataset(fnames,decode_timedelta=True)
            if 'height' in dsi.coords:
                dsi[varname].attrs['height'] = dsi['height'].values
                dsi = dsi.drop_vars(['height'])
            dsi = dsi.sel(sel)
            ds.append(dsi)
        print('Merging variables')
        ds = xr.merge(ds)
        if 'latitude_longitude' in ds:
            ds = ds.drop_vars('latitude_longitude')
            
    return ds

def lon_360_to_180(da,lon='lon',inplace=False,sort=True):
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
        if sort:
            print('Re-sorting by new longitudes (-180 - 180 E)')
            da = da.sortby(lon)
    else:
        lon_attrs = da[lon].attrs
        da_out = da.assign_coords({lon:np.mod(da[lon] + 180, 360) - 180})
        if sort:
            print('Re-sorting by new longitudes (-180 - 180 E)')
            da_out = da_out.sortby(lon)
        da_out[lon].attrs = lon_attrs
        return da_out

def lon_180_to_360(da,lon='lon',inplace=False,sort=True):
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
        if sort:
            print('Re-sorting by new longitudes (0 - 360 E)')
            da = da.sortby(lon)
    else:
        lon_attrs = da[lon].attrs
        da_out = da.assign_coords({lon:np.mod(da[lon], 360)})
        if sort:
            print('Re-sorting by new longitudes (0 - 360 E)')
            da_out = da_out.sortby(lon)
        da_out[lon].attrs = lon_attrs
        return da_out


def identify_lontype(ds,lon='lon',lat='lat'):
    '''
    Identify if a Dataset is of type [0 - 360] degree longitude ('360'), or of type [-180 - 180] degree longitude ('180')
    '''
    if (ds[lon].max() > 180 and ds[lon].min() >= 0):
        lontype = '360'
    elif (ds[lon].max() <= 180 and ds[lon].min() <  0):
        lontype = '180'
    else:
        raise ValueError('cannot reliably identify lontype')
    if (ds[lon].max() > 180 and ds[lon].min() < 0):
        raise ValueError('inconsistent')
    elif (ds[lon].max() > 180 and ds[lon].min() < 0):
        raise ValueError('inconsistent')
    print('lontype identified as %s' % lontype)
    return lontype

def sel_region(ds,LON_CENTER,LAT_CENTER,DOMAIN_HALF_WIDTH_IN_DEGREES,lontype=None,lon='lon',lat='lat'):
    '''
    Select a latitude-longitude box from a regular grid Dataset
    '''
    # identify type
    lontype0 = lontype
    if lontype is None:
        lontype = identify_lontype(ds=ds,lon=lon,lat=lat)
    else:
        print('lontype = %s is enforced' % lontype)
        
    if lontype == '180':
        # ds_region = sel_region_180(ds,LON_CENTER,LAT_CENTER,DOMAIN_HALF_WIDTH_IN_DEGREES)#! # FIX THIS!!!
        ds_region = sel_region_180(ds,np.mod(LON_CENTER + 180, 360) - 180,LAT_CENTER,DOMAIN_HALF_WIDTH_IN_DEGREES)
    elif lontype == '360':
        # ds_region = sel_region_360(ds,LON_CENTER,LAT_CENTER,DOMAIN_HALF_WIDTH_IN_DEGREES)#! # FIX THIS!!!
        ds_region = sel_region_360(ds,np.mod(LON_CENTER,360),LAT_CENTER,DOMAIN_HALF_WIDTH_IN_DEGREES)
    else:
        raise ValueError('lontype = %s is not defined' % lontype)

    if not (lontype0 is None):
        assert identify_lontype(ds_region) == lontype
    return ds_region

def identify_gridtype(ds,lon='lon',lat='lat'):
    '''
    decide on whether regular or non-regular grid
    '''
    if lon in ds.dims and lat in ds.dims: # regular
        print('assume regular grid')
        gridtype = 'regular'
    else: # irregular
        print('assume non-regular grid')
        gridtype = 'irregular'
    return gridtype

def sel_region_180(ds,LON_CENTER,LAT_CENTER,DOMAIN_HALF_WIDTH_IN_DEGREES):
    '''
    select region centered on a coordinate if longitude is [-180, 180]
    '''
    x0,x1 = LON_CENTER-DOMAIN_HALF_WIDTH_IN_DEGREES, LON_CENTER+DOMAIN_HALF_WIDTH_IN_DEGREES
    y0,y1 = LAT_CENTER-DOMAIN_HALF_WIDTH_IN_DEGREES, LAT_CENTER+DOMAIN_HALF_WIDTH_IN_DEGREES

    gridtype = identify_gridtype(ds,lon='lon',lat='lat')
    if gridtype == 'regular':
        append_dim = 'lon'
    else:
        append_dim = 'value'

    ds_region = sel_box(ds,x0,x1,y0,y1,lon='lon',lat='lat',gridtype=gridtype)

    if x0 < -180:
        print('concat left')
        ds_left = sel_box(ds,360+x0,180,y0,y1,lon='lon',lat='lat',gridtype=gridtype)
        ds_left = ds_left.assign_coords(lon=ds_left['lon']-360)
    
        ds_region = xr.concat(
            [
                ds_left,
                ds_region
            ],
            dim=append_dim
        )
    elif x1 > 180:
        print('concat right')
        ds_right = sel_box(ds,-180,x1-360,y0,y1,lon='lon',lat='lat',gridtype=gridtype)
        ds_right = ds_right.assign_coords(lon=ds_right['lon']+360)    
        
        ds_region = xr.concat(
            [
                ds_region,
                ds_right
            ],
            dim=append_dim
        )
    return ds_region


def sel_region_360(ds,LON_CENTER,LAT_CENTER,DOMAIN_HALF_WIDTH_IN_DEGREES):
    '''
    select region centered on a coordinate if longitude is [0, 360]
    '''

    x0,x1 = LON_CENTER-DOMAIN_HALF_WIDTH_IN_DEGREES, LON_CENTER+DOMAIN_HALF_WIDTH_IN_DEGREES
    y0,y1 = LAT_CENTER-DOMAIN_HALF_WIDTH_IN_DEGREES, LAT_CENTER+DOMAIN_HALF_WIDTH_IN_DEGREES

    gridtype = identify_gridtype(ds,lon='lon',lat='lat')
    if gridtype == 'regular':
        append_dim = 'lon'
    else:
        append_dim = 'value'

    ds_region = sel_box(ds,x0,x1,y0,y1,lon='lon',lat='lat',gridtype=gridtype)
    
    if x0 < 0:
        print('concat left')
        # print(gridtype)
        ds_left = sel_box(ds,360+x0,360,y0,y1,lon='lon',lat='lat',gridtype=gridtype)
        ds_left = ds_left.assign_coords(lon=ds_left['lon']-360)
        ds_region = xr.concat(
            [
                ds_left,
                ds_region
            ],
            dim=append_dim
        )
    elif x1 > 360:
        print('concat right')
        ds_right = sel_box(ds,0,x1-360,y0,y1,lon='lon',lat='lat',gridtype=gridtype)
        ds_right = ds_right.assign_coords(lon=ds_right['lon']+360)        
        ds_region = xr.concat(
            [
                ds_region,
                ds_right
            ],
            dim=append_dim
        )
    return ds_region

def sel_box(ds,x0,x1,y0,y1,lon='lon',lat='lat',gridtype=None):
    '''
    select a lat-lon box from a dataset, decide on whether regular or non-regular grid
    '''
    if gridtype == None:
        gridtype = identify_gridtype(ds,lon=lon,lat=lat)
        print('assume %s grid' % gridtype)

    if gridtype == 'regular':
        ds_out = ds.sel({lon:slice(x0,x1),lat:slice(y0,y1)})
    elif gridtype == 'irregular':
        ds_out = get_box_native(ds,x0,x1,y0,y1,lon=lon,lat=lat)
    else: 
        raise ValueError('gridtype %s is not valid' % gridtype)

    return ds_out

def get_box_native(ds,x0,x1,y0,y1,lon='lon',lat='lat',dim='value',to180=False):
    '''
    Extract a box on a native grid
    '''
    print('get box: [x0,x1,y0,y1]: [%s,%s,%s,%s]' % (x0,x1,y0,y1))
    coords = {
        lon:ds[lon].load(),
        lat:ds[lat].load()
    }
    if to180:
        coords[lon] = np.mod(coords[lon] + 180, 360) - 180
    return ds.sel(
        {
            dim : ( 
                (coords[lat] >= y0) 
                & (coords[lat] <= y1) 
                & (coords[lon] >= x0) 
                & (coords[lon] <= x1) 
            ) 
        }
    )


def fix_longitude(ds,EDDY_LON):
    '''Ensure that EDDY_LON is in identified format (180 or 360)'''
    lontype = identify_lontype(ds=ds)
    if lontype == '180':
        if not ( (EDDY_LON >= -180) & (EDDY_LON <= 180) ):
            print('adjust EDDY_LON to fit lontype == \'180\'')
            EDDY_LON = np.mod(EDDY_LON + 180, 360) - 180
            print('EDDY_LON = %.1f' % EDDY_LON)
            assert (EDDY_LON >= -180) & (EDDY_LON <= 180)
    elif lontype == '360':
        if not ( (EDDY_LON >= 0) & (EDDY_LON <= 360) ):
            print('adjust EDDY_LON to fit lontype == \'360\'')
            EDDY_LON = np.mod(EDDY_LON, 360)
            print('EDDY_LON = %.1f' % EDDY_LON)
            assert (EDDY_LON >= 0) & (EDDY_LON <= 360)
    return EDDY_LON


def all_equal(arr, tolerance=1e-6):
    """ Return True if all values are equal within the specified tolerance.

    :param np.array arr: aData rray containing values to be compared.
    """
    if len(arr) >0:
        return np.all(np.abs(np.array(arr) - np.array(arr)[0]) < tolerance)
    else:
        raise ValueError("Cannot test equality on empty array/list.")


def check_lat_lon_coords(lats, lons, tolerance=1e-4):
    """ Raises error if latitude/longitude coordinates do not have the assumed properties.
    
    :param np.array lats: One-dimensional array containing latitude values.
    :param np.array lons: One-dimensional array containing longitude values.
    """
    # Checks that lat/lon coordinates are regularly spaced
    if not all_equal(np.diff(lats),tolerance=tolerance):
        raise ValueError(f"Latitude values are not regularly spaced.")
        
    if not all_equal(np.diff(lons),tolerance=tolerance):
        raise ValueError(f"Longitude values are not regularly spaced.")


def reshape_latlon_1d_to_latlon_2d(ds, dim_name="value", lat_name="lat", lon_name="lon"):
    """ Reshapes data from 1D lat-lon to 2D lat-lon representation. 
    
    :param xarray.Dataset ds: An xarray dataset containing data with shape=(...,nvalues) \
        and corresponding one-dimensional latitude/longitude coordinates with shape=(nvalues).
    :param str dim_name: Name of original 1D index dimension.
    :param str lat_name: Name for latitude coordinate.
    :param str lon_name: Name for longitude coordinate.    
    :return xarray.Dataset ds: An xarray dataset containing two-dimensional data arrays with shape=(...,nlats, nlons) and 
        one-dimensional latitude/longitude coordinates with shape=(nlats) and shape=(nlons), respectively.
    """
    # Check shape of dims/coords
    for coord in [dim_name, lat_name, lon_name]:
        if ds[coord].ndim != 1:
            raise ShapeError(f"'{coord}' must be one-dimensional.")
        if coord != dim_name:
            if ds[coord].size != ds[dim_name].size:
                raise ShapeError(f"'{coord}' must be same length as '{dim_name}'.")
    ds_reshaped = ds.rename({dim_name:'latlon'}).set_index(latlon=(lat_name, lon_name)).unstack("latlon")
    check_lat_lon_coords(ds_reshaped[lat_name].values, ds_reshaped[lon_name].values)
    return ds_reshaped


def get_local_cartesian_coords(lats, lons, central_lat=None, central_lon=None, earth_radius=6.378e6):
    """ Return distances (km) along constant latitude/longitude from the centre of domain for
    a local cartesian coordinate system. 
    
    :param np.array lats: One-dimensional array containing latitude values. 
    :param np.array lons: One-dimensional array containing longitude values.
    :param float central_lat: Optional. Number giving central latitude. If not provided, compute as mean of lats. 
    :param float central_lon: Optional. Number giving central longitude. If not provided, compute as mean of lons. 
    :return tuple xy: A tuple of (x,y) distances (km), where x and y correspond to distances along lines \
        of constant latitude and longitude, respectively.
    """
    # check_lat_lon_coords(lats, lons)    
    if central_lat == None:
        central_lat = np.mean(lats)
        print('Derive central_lat as mean of lats = %.1f' % central_lat)
    if central_lon == None:
        central_lon = np.mean(lons)
        print('Derive central_lon as mean of lons = %.1f' % central_lon)
    dlon = lons - central_lon
    dlat = lats - central_lat

    # Return distance in km
    y = earth_radius * np.deg2rad(dlat) / 1000.
    x = earth_radius * np.cos(np.deg2rad(central_lat)) * np.deg2rad(dlon) / 1000.
    
    return x, y 

def calc_direction_of_average_winds(x, y, u, v, distance_from_eddy_threshold, lats=None):
    """ Returns the direction of the average winds as the anti-clockwise angle (in radians) 
    from x-axis. Wind component values are averaged over region specified. N.B. The direction
    of averaged wind components is not the same as the average of all wind directions, which is not
    well defined! There may be a better way to do this... 

    :param np.array x: One-dimensional array containing distances (km) along x-coord relative to eddy centre. 
    :param np.array y: One-dimensional array containing distances (km) along y-coord relative to eddy centre.
    :param np.array u: Two-dimensional array containing zonal wind data. 
    :param np.array v: Two-dimensional array containing meridional wind data. 
    :param np.array lats: One-dimensional array of latitudes for calculating cos(lat) grid weights (optional).
    :param float distance_from_eddy_threhsold: Threshold distance (km) for calculation of average wind components.
    :return float wind_direction_in_radians: Direction of the averaged winds as the anti-clockwise \
        angle (in radians) from x-axis.
    """
    if (u.ndim != 2) or (v.ndim != 2):
        print('Assume irregular grid')
        gridtype = 'irregular'
    else:
        print('Assume regular grid')
        gridtype = 'regular'
    
    # Values that exceed threshold distance are excluded.
    if gridtype == 'regular':
        print('wind direction, assuming regular grid')
        x_2d, y_2d = np.meshgrid(x, y) 
        distance_from_eddy = np.sqrt(x_2d**2 + y_2d**2)
        weights = np.array(distance_from_eddy <= distance_from_eddy_threshold, dtype=float)
        
        if lats is not None:
            _, lats_2d = np.meshgrid(x, lats)
            coslat = np.cos(np.deg2rad(lats_2d))
            weights = weights * coslat
        
        u_avg = (u * weights).sum() / weights.sum()
        v_avg = (v * weights).sum() / weights.sum()
    elif gridtype == 'irregular':
        print('wind direction, assuming irregular grid')
        # x_2d, y_2d = np.meshgrid(x, y) 
        distance_from_eddy = np.sqrt(x**2 + y**2)
        weights = np.array(distance_from_eddy <= distance_from_eddy_threshold, dtype=float)
        
        if lats is not None:
            # _, lats_2d = np.meshgrid(x, lats)
            coslat = np.cos(np.deg2rad(lats))
            weights = weights * coslat
        
        u_avg = (u * weights).sum() / weights.sum()
        v_avg = (v * weights).sum() / weights.sum()
    return np.arctan2(v_avg, u_avg)


def calc_direction_of_average_winds_backup(x, y, u, v, distance_from_eddy_threshold, lats=None):
    """ Returns the direction of the average winds as the anti-clockwise angle (in radians) 
    from x-axis. Wind component values are averaged over region specified. N.B. The direction
    of averaged wind components is not the same as the average of all wind directions, which is not
    well defined! There may be a better way to do this... 

    :param np.array x: One-dimensional array containing distances (km) along x-coord relative to eddy centre. 
    :param np.array y: One-dimensional array containing distances (km) along y-coord relative to eddy centre.
    :param np.array u: Two-dimensional array containing zonal wind data. 
    :param np.array v: Two-dimensional array containing meridional wind data. 
    :param np.array lats: One-dimensional array of latitudes for calculating cos(lat) grid weights (optional).
    :param float distance_from_eddy_threhsold: Threshold distance (km) for calculation of average wind components.
    :return float wind_direction_in_radians: Direction of the averaged winds as the anti-clockwise \
        angle (in radians) from x-axis.
    """
    if (u.ndim != 2) or (v.ndim != 2):
        raise ValueError("U and V data must have two dimensions.")
    
    # Values that exceed threshold distance are excluded.
    x_2d, y_2d = np.meshgrid(x, y) 
    distance_from_eddy = np.sqrt(x_2d**2 + y_2d**2)
    weights = np.array(distance_from_eddy <= distance_from_eddy_threshold, dtype=float)
    
    if lats is not None:
        _, lats_2d = np.meshgrid(x, lats)
        coslat = np.cos(np.deg2rad(lats_2d))
        weights = weights * coslat
    
    u_avg = (u * weights).sum() / weights.sum()
    v_avg = (v * weights).sum() / weights.sum()
    return np.arctan2(v_avg, u_avg)


def create_transformed_coordinates(
    resample_eddy_radiuses,
    resample_density,
):
    """ Returns locations to be sampled in transformed composite coordinates.

    :param float resample_eddy_radiuses: The number of eddy radiuses to be sampled when creating composites. \
        For example, resample_eddy_radiuses=3 will produce composites that sample data within +/- 3 eddy \
        radiuses.
    :param float resample_density: The number of sample locations per eddy radius in transformed \
        composite coordinates.     
    :return tuple xy: A tuple of (x', y') distances (with units of 'eddy radiuses'), where x' and y' correspond \
        to distances along transformed composite coordinates such that x' is in the direction of large-scale \
        winds.
    """
    x = np.linspace(-resample_eddy_radiuses, resample_eddy_radiuses, resample_density * resample_eddy_radiuses+1)
    y = np.linspace(-resample_eddy_radiuses, resample_eddy_radiuses, resample_density * resample_eddy_radiuses+1)
    return x, y


def create_position_vectors_in_original_coords(
    transformed_x,
    transformed_y,
    wind_direction_in_radians_from_x,
    eddy_radius,
):
    """ Returns matrix of position vectors to be sampled in local cartesian coordinates. 

    :param np.array transformed_x: One-dimensional array of sample locations in the x-direction \
        of the transformed composite coordinate.
    :param np.array transformed_y: One-dimensional array of sample locations in the y-direction \
        of the transformed composite coordinate.
    :param float wind_direction_in_radians_from_x: Direction of the wind in the local cartesian coordinate \
        anti-clockwise in radians from x-axis.
    :param float eddy_radius: Radius of eddy (km).
    :return np.matrix position_vectors: A matrix of position vectors for sample locations in the original \
        local cartesian coordinates. 
    """
    transformed_x_2d, transformed_y_2d = np.meshgrid(transformed_x, transformed_y) 
    position_vectors_in_transformed_coords = np.matrix([
        transformed_x_2d.reshape(transformed_x_2d.size),
        transformed_y_2d.reshape(transformed_y_2d.size),
    ])
    # Rotate and scale back to original coords
    position_vectors_in_original_coords = eddy_radius * rotate_vectors(
        position_vectors_in_transformed_coords,
        wind_direction_in_radians_from_x,
        transpose=False,
    )
    return position_vectors_in_original_coords


def rotate_vectors(position_vectors_in_transformed_coords, angle_in_rad, transpose=False):
    """ Returns position vectors with components in original cartesian coordinates when supplied
    with position vectors with components in basis that has been rotated by angle_in_rad. This is
    equivalent to the transform that converts the original basis vectors to the rotated basis vectors.

    :param np.matrix position_vectors: A matrix of position vectors for sample locations in the transformed \
        composite coordinates.
    :param float angle_in_rad: Angle in radians, which corresponds to the direction of the wind in the
    local cartesian coordinate as measured anti-clockwise in radians from original x-axis.
    :param bool transpose: If True, apply the inverse rotation transformation.
    :return np.matrix position_vectors: A matrix of position vectors for sample locations in the unrotated \
        coordinates.
    """
    
    # Rotation matrix that converts original basis vectors to rotated basis vectors
    rotation_matrix = np.matrix([
        [np.cos(angle_in_rad), -np.sin(angle_in_rad)],
        [np.sin(angle_in_rad), np.cos(angle_in_rad)],
    ])

    if transpose:
        return np.matmul(np.transpose(rotation_matrix), position_vectors_in_transformed_coords)
    else:
        return np.matmul(rotation_matrix, position_vectors_in_transformed_coords)

def interpolate_data_to_sample_locations(
    data,
    x,
    y,
    transformed_x,
    transformed_y,
    sample_position_vectors_in_original_coords,
    interpolation_method="linear",
):
    """ Returns composite data in the transformed composite coordinates interpolated from the original \
    local cartesian coordinates.

    :param np.array data: Two-dimensional array containing data to be resampled. 
    :param np.array x: One-dimensional array containing distances (km) along x-coord relative to eddy centre. 
    :param np.array y: One-dimensional array containing distances (km) along y-coord relative to eddy centre.
    :param np.array transformed_x: One-dimensional array of sample locations in the x-direction \
        of the transformed composite coordinate.
    :param np.array transformed_y: One-dimensional array of sample locations in the y-direction \
        of the transformed composite coordinate.
    :param np.matrix position_vectors: A matrix of position vectors to be sampled with components in the original \
        local cartesian coordinates. 
    :param str interpolation_method: Interpolation method supported by ``scipy.interpolate.griddata`` (default="linear").
    :return np.array composite_data: A two-dimensional array of data in the transformed composite coordinates.
    """
    if (x.shape == data.shape) and (y.shape == data.shape) and (x.size == data.size): # coords and data already one-dimensional
        x_1d = x
        y_1d = y
        data_1d = data
    else: # coords and data not yet one-dimensional
        # Create x,y,z vectors of data 
        x_2d, y_2d = np.meshgrid(x, y) 
        x_1d = x_2d.reshape(x_2d.size)
        y_1d = y_2d.reshape(y_2d.size)
        data_1d = data.reshape(data.size)

    xsample_1d = np.array(sample_position_vectors_in_original_coords[0,:]).squeeze()
    ysample_1d = np.array(sample_position_vectors_in_original_coords[1,:]).squeeze()
    #print('cubic')
    resampled_data_in_transformed_coords_1d = scipy.interpolate.griddata(
        (x_1d, y_1d),
        data_1d,
        (xsample_1d, ysample_1d),
        #method="cubic",
        method="linear",
    )
    # Return reshaped transformed data to 2D 
    transformed_x_2d, _ = np.meshgrid(transformed_x, transformed_y) 
    return resampled_data_in_transformed_coords_1d.reshape(transformed_x_2d.shape)

def transform_sample_point(
    ds,
    COMPOSITE_PARAM, # = "avg_sst"
    TIME_DX, # time-index
    EDDY_LON, # eddy center longitude
    EDDY_LAT, # eddy center latitude
    DOMAIN_HALF_WIDTH_IN_DEGREES = 10, # domain half width
    EDDY_RADIUS = 100, # User-specified eddy radius in km.
    AVG_WIND_EDDY_RADIUSES = 2, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u", 
    VPARAM = "avg_10v",
    ds_wind = None
):
    ds_region = sel_region(
        ds.sel(time=TIME_DX),
        LON_CENTER=EDDY_LON,
        LAT_CENTER=EDDY_LAT,
        DOMAIN_HALF_WIDTH_IN_DEGREES=DOMAIN_HALF_WIDTH_IN_DEGREES,
        lontype=None,lon='lon',lat='lat'
    )

    # Create a local cartesian coordinate system with (x,y) specifing distance from eddy in km.
    x, y = get_local_cartesian_coords(
        lats=ds_region["lat"].data,
        lons=ds_region["lon"].data,
        central_lat=EDDY_LAT,
        central_lon=EDDY_LON
    )

    # Create the grid for transformed coordinate system
    transformed_x, transformed_y = create_transformed_coordinates(
        RESAMPLE_EDDY_RADIUSES,
        RESAMPLE_DENSITY,
    )

    if ds_wind is None:
        ds_region_wind = ds_region
        x_wind, y_wind = x, y
    else:
        ds_region_wind = sel_region(
            ds_wind.sel(time=TIME_DX),
            LON_CENTER=EDDY_LON,
            LAT_CENTER=EDDY_LAT,
            DOMAIN_HALF_WIDTH_IN_DEGREES=DOMAIN_HALF_WIDTH_IN_DEGREES,
            lontype=None,lon='lon',lat='lat'
        )
        # Create a local cartesian coordinate system with (x,y) specifing distance from eddy in km.
        x_wind, y_wind = get_local_cartesian_coords(
            lats=ds_region_wind["lat"].data,
            lons=ds_region_wind["lon"].data,
            central_lat=EDDY_LAT,
            central_lon=EDDY_LON
        )

    # Find direction of large-scale wind (averaged over within specified number of km from eddy centre)
    if UPARAM is None or VPARAM is None:
        print('u and/or v are not provided. DO NOT rotate eddies')
        wind_direction_in_radians_from_x = 0.
    else:
        wind_direction_in_radians_from_x = calc_direction_of_average_winds(
            # x,
            # y,
            x_wind,
            y_wind,
            # ds_region[UPARAM].values,
            # ds_region[VPARAM].values,
            ds_region_wind[UPARAM].values,
            ds_region_wind[VPARAM].values,
            distance_from_eddy_threshold=AVG_WIND_EDDY_RADIUSES*EDDY_RADIUS,
            # lats = ds_region.lat.values,
            lats = ds_region_wind.lat.values,
        )

    # Find coordinates of requested locations in local cartesian coordinate system
    sample_position_vectors_in_original_coords = create_position_vectors_in_original_coords(
        transformed_x,
        transformed_y,
        wind_direction_in_radians_from_x,
        EDDY_RADIUS,
    )

    # Resample data into the transformed coordinate system
    resampled_data_in_transformed_coords = interpolate_data_to_sample_locations(
        ds_region[COMPOSITE_PARAM].values,
        x,
        y,
        transformed_x,
        transformed_y,
        sample_position_vectors_in_original_coords,
    )

    return transformed_x, transformed_y, resampled_data_in_transformed_coords

def eddy_calc_wind_direction(
    ds,
    TIME_DX, # time-index
    EDDY_LON, # eddy center longitude
    EDDY_LAT, # eddy center latitude
    DOMAIN_HALF_WIDTH_IN_DEGREES = 10, # domain half width
    EDDY_RADIUS = 100, # User-specified eddy radius in km.
    AVG_WIND_EDDY_RADIUSES = 2, # Number of eddy radiuses used for calculating direction of large-scale winds.
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u",
    VPARAM = "avg_10v"
):

    ds_region = sel_region(
        ds.sel(time=TIME_DX),
        LON_CENTER=EDDY_LON,
        LAT_CENTER=EDDY_LAT,
        DOMAIN_HALF_WIDTH_IN_DEGREES=DOMAIN_HALF_WIDTH_IN_DEGREES,
        lontype=None,lon='lon',lat='lat'
    )

    x, y = get_local_cartesian_coords(
        lats=ds_region["lat"].data,
        lons=ds_region["lon"].data,
        central_lat=EDDY_LAT,
        central_lon=EDDY_LON
    )

    # Create the grid for transformed coordinate system
    transformed_x, transformed_y = create_transformed_coordinates(
        RESAMPLE_EDDY_RADIUSES,
        RESAMPLE_DENSITY,
    )

    wind_direction_in_radians_from_x = calc_direction_of_average_winds(
        x,
        y,
        ds_region[UPARAM].values,
        ds_region[VPARAM].values,
        distance_from_eddy_threshold=AVG_WIND_EDDY_RADIUSES*EDDY_RADIUS,
        lats = ds_region.lat.values,
    )
    print('Wind angle: %.2f rad, %.2f deg' % (wind_direction_in_radians_from_x,np.rad2deg(wind_direction_in_radians_from_x)))
    return wind_direction_in_radians_from_x

def rotate_winds_point(
    ds,
    TIME_DX, # time-index
    EDDY_LON, # eddy center longitude
    EDDY_LAT, # eddy center latitude
    DOMAIN_HALF_WIDTH_IN_DEGREES = 10, # domain half width
    EDDY_RADIUS = 100, # User-specified eddy radius in km.
    AVG_WIND_EDDY_RADIUSES = 2, # Number of eddy radiuses used for calculating direction of large-scale winds.
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u",
    VPARAM = "avg_10v"
):

    ds_region = sel_region(
        ds.sel(time=TIME_DX),
        LON_CENTER=EDDY_LON,
        LAT_CENTER=EDDY_LAT,
        DOMAIN_HALF_WIDTH_IN_DEGREES=DOMAIN_HALF_WIDTH_IN_DEGREES,
        lontype=None,lon='lon',lat='lat'
    )

    x, y = get_local_cartesian_coords(
        lats=ds_region["lat"].data,
        lons=ds_region["lon"].data,
        central_lat=EDDY_LAT,
        central_lon=EDDY_LON
    )

    # Create the grid for transformed coordinate system
    transformed_x, transformed_y = create_transformed_coordinates(
        RESAMPLE_EDDY_RADIUSES,
        RESAMPLE_DENSITY,
    )

    wind_direction_in_radians_from_x = calc_direction_of_average_winds(
        x,
        y,
        ds_region[UPARAM].values,
        ds_region[VPARAM].values,
        distance_from_eddy_threshold=AVG_WIND_EDDY_RADIUSES*EDDY_RADIUS,
        lats = ds_region.lat.values,
    )
    print('Wind angle: %.2f rad, %.2f deg' % (wind_direction_in_radians_from_x,np.rad2deg(wind_direction_in_radians_from_x)))

    sample_position_vectors_in_original_coords = create_position_vectors_in_original_coords(
        transformed_x,
        transformed_y,
        wind_direction_in_radians_from_x,
        EDDY_RADIUS,
    )

    # rotate wind vectors
    wind_vector_components_transformed_coords = rotate_vectors(
            np.matrix([
                ds_region[UPARAM].values.reshape(ds_region[UPARAM].values.size),
                ds_region[VPARAM].values.reshape(ds_region[VPARAM].values.size)
            ]),
            wind_direction_in_radians_from_x,
            transpose=True,
        )

    # Resample data into the transformed coordinate system - u-wind
    resampled_data_in_transformed_coords_u = interpolate_data_to_sample_locations(
        np.squeeze(np.array(wind_vector_components_transformed_coords[0])),#.reshape(80,80),
        x,
        y,
        transformed_x,
        transformed_y,
        sample_position_vectors_in_original_coords,
    )
    # Resample data into the transformed coordinate system - v-wind
    resampled_data_in_transformed_coords_v = interpolate_data_to_sample_locations(
        np.squeeze(np.array(wind_vector_components_transformed_coords[1])),#.reshape(80,80),
        x,
        y,
        transformed_x,
        transformed_y,
        sample_position_vectors_in_original_coords,
    )

    return transformed_x, transformed_y, resampled_data_in_transformed_coords_u, resampled_data_in_transformed_coords_v


def transform_winds(
    ds,
    TIME_DX, # time-index
    EDDY_LON, # eddy center longitude
    EDDY_LAT, # eddy center latitude
    DOMAIN_HALF_WIDTH_IN_DEGREES = 10, # domain half width
    EDDY_RADIUS = 100, # User-specified eddy radius in km.
    AVG_WIND_EDDY_RADIUSES = 2, # Number of eddy radiuses used for calculating direction of large-scale winds.
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u",
    VPARAM = "avg_10v"
):
    print([
        'rotate u/v',
        TIME_DX,
        EDDY_LON,
        EDDY_LAT,
        DOMAIN_HALF_WIDTH_IN_DEGREES,
        EDDY_RADIUS,
        AVG_WIND_EDDY_RADIUSES,
        RESAMPLE_EDDY_RADIUSES,
        RESAMPLE_DENSITY,
        UPARAM,
        VPARAM
    ])

    # Ensure consistent longitude formatting
    EDDY_LON = fix_longitude(ds,EDDY_LON)

    transformed_x, transformed_y, resampled_data_in_transformed_coords_u, resampled_data_in_transformed_coords_v = rotate_winds_point(
        ds = ds,
        TIME_DX = TIME_DX,
        EDDY_LON = EDDY_LON,
        EDDY_LAT = EDDY_LAT,
        DOMAIN_HALF_WIDTH_IN_DEGREES = DOMAIN_HALF_WIDTH_IN_DEGREES,
        EDDY_RADIUS = EDDY_RADIUS,
        AVG_WIND_EDDY_RADIUSES = AVG_WIND_EDDY_RADIUSES,
        RESAMPLE_EDDY_RADIUSES = RESAMPLE_EDDY_RADIUSES,
        RESAMPLE_DENSITY = RESAMPLE_DENSITY,
        UPARAM = UPARAM,
        VPARAM = VPARAM
    )

    out = xr.Dataset(coords={'x': transformed_x, 'y': transformed_y})
    out[UPARAM] = xr.DataArray(dims=['x','y'],coords={'x':transformed_x,'y':transformed_y},data=resampled_data_in_transformed_coords_u)
    out[VPARAM] = xr.DataArray(dims=['x','y'],coords={'x':transformed_x,'y':transformed_y},data=resampled_data_in_transformed_coords_v)

    out[UPARAM].attrs = ds[UPARAM].attrs
    out[VPARAM].attrs = ds[VPARAM].attrs

    out['x'].attrs['long_name'] = 'x_coordinate'
    out['x'].attrs['units'] = 'Eddy radii'
    out['y'].attrs['long_name'] = 'y_coordinate'
    out['y'].attrs['units'] = 'Eddy radii'

    out = out.transpose()

    out['time'] = TIME_DX
    out['lon'] = EDDY_LON
    out['lat'] = EDDY_LAT

    out.attrs['DOMAIN_HALF_WIDTH_IN_DEGREES'] = DOMAIN_HALF_WIDTH_IN_DEGREES
    out.attrs['EDDY_RADIUS'] = EDDY_RADIUS
    out.attrs['AVG_WIND_EDDY_RADIUSES'] = AVG_WIND_EDDY_RADIUSES
    out.attrs['RESAMPLE_EDDY_RADIUSES'] = RESAMPLE_EDDY_RADIUSES
    out.attrs['RESAMPLE_DENSITY'] = RESAMPLE_DENSITY
    out.attrs['UPARAM'] = UPARAM
    out.attrs['VPARAM'] = VPARAM

    return out

def transform_eddy(
    ds,
    COMPOSITE_PARAM, # = "avg_sst"
    TIME_DX, # time-index
    EDDY_LON, # eddy center longitude
    EDDY_LAT, # eddy center latitude
    DOMAIN_HALF_WIDTH_IN_DEGREES = 10, # domain half width
    EDDY_RADIUS = 100, # User-specified eddy radius in km.
    AVG_WIND_EDDY_RADIUSES = 2, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u", 
    VPARAM = "avg_10v",
    ds_wind = None
):
    print([
        COMPOSITE_PARAM,
        TIME_DX,
        EDDY_LON,
        EDDY_LAT,
        DOMAIN_HALF_WIDTH_IN_DEGREES,
        EDDY_RADIUS,
        AVG_WIND_EDDY_RADIUSES,
        RESAMPLE_EDDY_RADIUSES,
        RESAMPLE_DENSITY,
        UPARAM,
        VPARAM
    ])

    # Ensure consistent longitude formatting
    EDDY_LON = fix_longitude(ds,EDDY_LON)

    transformed_x, transformed_y, resampled_data_in_transformed_coords = transform_sample_point(
        ds = ds,
        COMPOSITE_PARAM = COMPOSITE_PARAM,
        TIME_DX = TIME_DX,
        EDDY_LON = EDDY_LON,
        EDDY_LAT = EDDY_LAT,
        DOMAIN_HALF_WIDTH_IN_DEGREES = DOMAIN_HALF_WIDTH_IN_DEGREES,
        EDDY_RADIUS = EDDY_RADIUS,
        AVG_WIND_EDDY_RADIUSES = AVG_WIND_EDDY_RADIUSES,
        RESAMPLE_EDDY_RADIUSES = RESAMPLE_EDDY_RADIUSES,
        RESAMPLE_DENSITY = RESAMPLE_DENSITY,
        UPARAM = UPARAM,
        VPARAM = VPARAM,
        ds_wind = ds_wind
    )

    out = xr.Dataset()
    out[COMPOSITE_PARAM] = xr.DataArray(dims=['x','y'],coords={'x':transformed_x,'y':transformed_y},data=resampled_data_in_transformed_coords.T)

    out[COMPOSITE_PARAM].attrs = ds[COMPOSITE_PARAM].attrs

    out['x'].attrs['long_name'] = 'x_coordinate'
    out['x'].attrs['units'] = 'Eddy radii'
    out['y'].attrs['long_name'] = 'y_coordinate'
    out['y'].attrs['units'] = 'Eddy radii'

    out = out.transpose()

    out['time'] = TIME_DX
    out['lon'] = EDDY_LON
    out['lat'] = EDDY_LAT

    out.attrs['DOMAIN_HALF_WIDTH_IN_DEGREES'] = DOMAIN_HALF_WIDTH_IN_DEGREES
    out.attrs['EDDY_RADIUS'] = EDDY_RADIUS
    out.attrs['AVG_WIND_EDDY_RADIUSES'] = AVG_WIND_EDDY_RADIUSES
    out.attrs['RESAMPLE_EDDY_RADIUSES'] = RESAMPLE_EDDY_RADIUSES
    out.attrs['RESAMPLE_DENSITY'] = RESAMPLE_DENSITY
    out.attrs['UPARAM'] = UPARAM
    out.attrs['VPARAM'] = VPARAM

    return out



def get_tracks(kind,source='AVISO',platform='ATOS',path=None):
    '''
    Load eddy tracks 
    Option 1: specify 
        - kind (cyclonic, anticyclonic)
        - source (AVISO)
        - platform (ATOS, Levante, Jasmin)
    Option 2: specify
        - path: filepath on the current system
    '''

    if path is not None:
        tracks = xr.open_dataset(path)

    if not kind in ['cyclonic','anticyclonic']:
        raise ValueError('kind %s is not defined' % kind)
    if not source in ['AVISO']:
        raise ValueError('source %s is not defined' % source)
    if not platform in ['ATOS','Levante','Jasmin']:
        raise ValueError('platform %s is not defined' % platform)

    if source == 'AVISO':
        if platform == 'ATOS':
            if kind == 'anticyclonic':
                tracks = xr.open_dataset('/ec/fws5/lb/project/eerie/data/AVISO/META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc')
            elif kind == 'cyclonic':
                tracks = xr.open_dataset('/ec/fws5/lb/project/eerie/data/AVISO/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc')
        elif platform == 'Jasmin':
            if kind == 'anticyclonic':
                tracks = xr.open_dataset('/gws/nopw/j04/eerie/observations/AVISO/META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc')
            elif kind == 'cyclonic':
                tracks = xr.open_dataset('/gws/nopw/j04/eerie/observations/AVISO/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc')

        elif platform == 'Levante':
            import intake
            catobs = intake.open_catalog('https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml')\
                                    ['dkrz']['disk']['observations']
            tracks = catobs['AVISO']['eddy-tracks'][kind].to_dask()

    tracks['time'].values = pd.to_datetime(tracks.time)
    tracks['obs'] = np.arange(tracks['obs'].size)
    tracks['year'] = tracks['time.year']
    return tracks

def load_eddy_tracks(
    kind,
    source='AVISO',
    platform='ATOS',
    path=None,
    thin_by_obs=1,  # keep every nth observation
    thin_by_date=1, # keep every nth date (in the tracks) to better sample variability
    thin_by_eddy=1  # for each eddy, keep only every nth date
):
    tracks0 = get_tracks(kind=kind,source=source,platform=platform,path=path)
    tracks0['time'] = tracks0['time'] + pd.Timedelta('12h')

    if thin_by_obs > 1:
        tracks0 = tracks0.sel(obs=slice(0,None,thin_by_obs))

    if thin_by_date > 1:
        raise ValueError('not yet defined')

    if thin_by_eddy > 1:
        raise ValueError('not yet defined')

    return tracks0



def compute_eddy_tracks():
    print('TBD')


def hpfilter(
    ds,
    filterparams
):
    # TODO: question: filtering 
    #   - globally before selecting?
    #   - on the regional selection before rotation / scaling?
    #   - on the resampled data in eddy-space?
    print('TBD')


def loop_over_eddies(
    ds,
    tracks,
    params,
    # COMPOSITE_PARAM,
    DOMAIN_HALF_WIDTH_IN_DEGREES = 10, # domain half width
    AVG_WIND_EDDY_RADIUSES = 2, # Number of eddy radiuses used for calculating direction of large-scale winds. 
    RESAMPLE_EDDY_RADIUSES = 3, # Number of eddy radiuses to sample in transformed composite coordinates.
    RESAMPLE_DENSITY = 30, # Number of data points per eddy radius in transformed composite coordinates.
    UPARAM = "avg_10u", 
    VPARAM = "avg_10v",
    rotate_winds = True,
    ds_wind = None,
    fname_root = None,
    output = 'all', # 'single', 'single_var'
    **kwargs
):

    if not isinstance(params,list):
        params = [params]
    if UPARAM is None or VPARAM is None:
        rotate_winds = False
    if rotate_winds is True and ds_wind is None:
        ds_wind = ds
    print('Composite parameters: ', params)
    N = tracks['obs'].size
    N_saved = 0
    print('Loop over %i eddies' % N)

    if fname_root == None:
        output = 'all'

    eddies = []
    print(output)
    for i in range(N):
        tracki = tracks.isel(obs=i)
        print('\n%i , obs = %i' % (i, tracki['obs']))
        eddy = []
        if fname_root is not None:
            assert isinstance(fname_root,str)
            fname = fname_root + '%i.nc' % tracki['obs']

            # Check if already moved into 'saved' subdirectory
            fname_saved_split = fname.split('/')
            fname_saved_split.insert(-1,'saved')
            fname_saved = '/'.join(fname_saved_split)

            if output != 'single_var':
                if os.path.exists(fname) or os.path.exists(fname_saved):
                    print('File exists for obs=%i, continue...' % tracki['obs'])
                    continue

        for param in params:
            if output == 'single_var':
                fnamei = fname.replace('.nc','_%s.nc' % param)
                if os.path.exists(fnamei):
                    print('File exists for obs=%i, param=%s, continue...' % (tracki['obs'],param))
                    continue
            eddyi = transform_eddy(
                ds,
                COMPOSITE_PARAM = param,
                TIME_DX = tracki['time'].values,
                EDDY_LON = tracki['longitude'].values,
                EDDY_LAT = tracki['latitude'].values,
                DOMAIN_HALF_WIDTH_IN_DEGREES = DOMAIN_HALF_WIDTH_IN_DEGREES,
                EDDY_RADIUS = np.round(tracki['effective_radius'].values / 1000),
                AVG_WIND_EDDY_RADIUSES = AVG_WIND_EDDY_RADIUSES,
                RESAMPLE_EDDY_RADIUSES = RESAMPLE_EDDY_RADIUSES,
                RESAMPLE_DENSITY = RESAMPLE_DENSITY,
                UPARAM = UPARAM,
                VPARAM = VPARAM,
                ds_wind = ds_wind
            )
            if output == 'single_var':
                print(fname, param)
                # fnamei = fname.replace('.nc','_%s.nc' % param)
                eddyi.coords['obs'] = tracki['obs']
                print('Saving eddy obs %i, variable %s to fname %s' % (eddyi['obs'], param, fnamei))
                eddyi.to_netcdf(fnamei)
            else:
                eddy.append(eddyi)
        if rotate_winds:
            print('Rotate winds')
            if output == 'single_var':
                fnamei = fname.replace('.nc','_%s.nc' % 'uv')
                if os.path.exists(fnamei):
                    print('File exists for obs=%i, param=%s, continue...' % (tracki['obs'],'uv'))
                    continue
            eddyi = transform_winds(
                # ds,
                ds_wind,
                TIME_DX = tracki['time'].values,
                EDDY_LON = tracki['longitude'].values,
                EDDY_LAT = tracki['latitude'].values,
                DOMAIN_HALF_WIDTH_IN_DEGREES = DOMAIN_HALF_WIDTH_IN_DEGREES,
                EDDY_RADIUS = np.round(tracki['effective_radius'].values / 1000),
                AVG_WIND_EDDY_RADIUSES = AVG_WIND_EDDY_RADIUSES,
                RESAMPLE_EDDY_RADIUSES = RESAMPLE_EDDY_RADIUSES,
                RESAMPLE_DENSITY = RESAMPLE_DENSITY,
                UPARAM = UPARAM,
                VPARAM = VPARAM
            )
            if output == 'single_var':
                fnamei = fname.replace('.nc','_%s.nc' % 'uv')
                eddyi.coords['obs'] = tracki['obs']
                print('Saving eddy obs %i, variable %s to fname %s' % (eddyi['obs'], 'uv', fnamei))
                eddyi.to_netcdf(fnamei)
            else:
                eddy.append(eddyi)

        if output != 'single_var':
            eddy = xr.merge(eddy)
            eddy.coords['obs'] = tracki['obs']
            if fname_root is not None:
                print('Saving eddy obs %i to fname %s' % (eddy['obs'], fname))
                eddy.to_netcdf(fname)
            else:
                eddies.append(eddy)
        N_saved += 1
    if fname_root is not None:
        print('Saved %i eddies to disk using pattern %s' % (N_saved,fname_root))
        return 0
    else:
        eddies = xr.concat(eddies,dim='obs')
        return eddies

