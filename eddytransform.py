""" 
Functions to resample ocean eddy composites in an eddy-centric 
rotated/rescaled transformed coordinate system.

"""


import numpy as np
import scipy


def all_equal(arr, tolerance=1e-6):
    """ Return True if all values are equal within the specified tolerance.

    :param np.array arr: aData rray containing values to be compared.
    """
    if len(arr) >0:
        return np.all(np.abs(np.array(arr) - np.array(arr)[0]) < tolerance)
    else:
        raise ValueError("Cannot test equality on empty array/list.")


def check_lat_lon_coords(lats, lons, tolerance=1e-6):
    """ Raises error if latitude/longitude coordinates do not have the assumed properties.
    
    :param np.array lats: One-dimensional array containing latitude values.
    :param np.array lons: One-dimensional array containing longitude values.
    """
    # Checks that lat/lon coordinates are regularly spaced
    if not all_equal(np.diff(lats)):
        raise ValueError(f"Latitude values are not regularly spaced.")
        
    if not all_equal(np.diff(lons)):
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


def get_local_cartesian_coords(lats, lons, earth_radius=6.378e6):
    """ Return distances (km) along constant latitude/longitude from the centre of domain for
    a local cartesian coordinate system. 
    
    :param np.array lats: One-dimensional array containing latitude values. 
    :param np.array lons: One-dimensional array containing longitude values.
    :return tuple xy: A tuple of (x,y) distances (km), where x and y correspond to distances along lines \
        of constant latitude and longitude, respectively.
    """
    check_lat_lon_coords(lats, lons)    
    central_lat = np.mean(lats)
    central_lon = np.mean(lons)
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
    # Create x,y,z vectors of data 
    x_2d, y_2d = np.meshgrid(x, y) 
    x_1d = x_2d.reshape(x_2d.size)
    y_1d = y_2d.reshape(y_2d.size)
    data_1d = data.reshape(data.size)

    xsample_1d = np.array(sample_position_vectors_in_original_coords[0,:]).squeeze()
    ysample_1d = np.array(sample_position_vectors_in_original_coords[1,:]).squeeze()
    resampled_data_in_transformed_coords_1d = scipy.interpolate.griddata(
        (x_1d, y_1d),
        data_1d,
        (xsample_1d, ysample_1d),
        method="linear",
    )
    # Return reshaped transformed data to 2D 
    transformed_x_2d, _ = np.meshgrid(transformed_x, transformed_y) 
    return resampled_data_in_transformed_coords_1d.reshape(transformed_x_2d.shape)

