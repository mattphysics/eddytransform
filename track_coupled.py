"""py-eddy-tracker on the coupled IFS-FESOM runs (daily avg_zos, HEALPix zoom 9 -> 0.25 deg).

Stages (each resumable, skips existing outputs):
  regrid    H512 avg_zos -> regular 0.25 deg grid, one NetCDF per day (area-weighted bin mean)
  identify  eddy identification per day (Bessel high-pass, closed SSH contours), cyc + anticyc
  track     stitch the daily detections into tracks (AreaTracker), write *_tracks.nc

Parameters follow the EERIE model-eddy recipe (eddy_analysis/eddy_identification_paralelised_share.ipynb):
  wavelength 700 km, step 2 mm, shape error 70 %, no pixel limit; tracking cmin 0.05, 4 virtual
  steps, long tracks >= 10 obs.

Time: the ocean daily means are stamped at 00:00 starting 2020-01-02 and ending 2021-03-01, i.e. the
stamp is the END of the averaging day. Fields are labelled with the day they represent (stamp - 1 day),
so the tracks' dates match the atmospheric daily means (stamped D 12:00) used for the composites.
"""
import argparse
import glob
import os
import sys
import time as T
from datetime import datetime, timedelta

import numpy as np

K = '/work/bm1344/DKRZ/kerchunks_batched'
RUNS = {
    'ng5_rcbmf': f'{K}/NGC-IFS-tco3999-ng5-production/2D_daily_healpix512_ocean.parq',
    'ng5_deepoff': f'{K}/NGC-IFS-tco3999-ng5-production-deep-off/2D_daily_healpix512_ocean.parq',
}
ROOT = '/work/bk1377/b382473/eddytracks'
STAMP_SHIFT = timedelta(days=-1)   # ocean stamp is end of day -> represented day = stamp - 1 day
VAR = 'avg_zos'
FILL = -9999.0

WAVELENGTH = 700      # km, high-pass cutoff
STEP_HT = 0.002       # m
PIXEL_LIMIT = None
SHAPE_ERROR = 70      # %
NB_OBS_MIN = 10
CMIN = 0.05
VIRTUAL = 4

NLON, NLAT, RES = 1440, 720, 0.25


def paths(run):
    base = os.path.join(ROOT, run, '0.25deg')
    d = dict(base=base, zos=base + '/zos', idd=base + '/eddyID', trk=base + '/eddyTracks')
    return d


def day_file(run, day):
    return '%s/zos_%s.nc' % (paths(run)['zos'], day.strftime('%Y%m%d'))


def open_ocean(run):
    import xarray as xr
    return xr.open_dataset('reference::' + RUNS[run], engine='zarr', consolidated=False, chunks={},
                           backend_kwargs=dict(storage_options=dict(remote_protocol='file', lazy=True)))


def days_of(run, ds=None):
    ds = ds or open_ocean(run)
    stamps = ds['time'].values.astype('datetime64[s]').astype(object)
    return [s + STAMP_SHIFT for s in stamps]


# ---------------------------------------------------------------------------------------------
def stage_regrid(run, i0, i1):
    from netCDF4 import Dataset
    p = paths(run)
    os.makedirs(p['zos'], exist_ok=True)
    ds = open_ocean(run)
    days = days_of(run, ds)
    cache = p['base'] + '/binindex_h512_to_0.25deg.npy'
    if os.path.exists(cache):
        binidx = np.load(cache)
    else:
        lat = np.asarray(ds['lat'].values, 'float64')
        lon = np.mod(np.asarray(ds['lon'].values, 'float64'), 360.0)
        ilat = np.clip(np.floor((lat + 90.0) / RES).astype(np.int64), 0, NLAT - 1)
        ilon = np.floor(lon / RES).astype(np.int64) % NLON
        binidx = ilat * NLON + ilon
        np.save(cache, binidx)
    nbin = NLAT * NLON
    tot = np.bincount(binidx, minlength=nbin).astype('float64')
    lons = RES * (np.arange(NLON) + 0.5)
    lats = -90.0 + RES * (np.arange(NLAT) + 0.5)
    i1 = min(i1, len(days))
    print('regrid %s: steps %d..%d of %d' % (run, i0, i1 - 1, len(days)), flush=True)
    for i in range(i0, i1):
        out = day_file(run, days[i])
        if os.path.exists(out):
            continue
        z = np.asarray(ds[VAR].isel(time=i).values, 'float64')
        ok = np.isfinite(z)
        s = np.bincount(binidx, weights=np.where(ok, z, 0.0), minlength=nbin)
        n = np.bincount(binidx, weights=ok.astype('float64'), minlength=nbin)
        with np.errstate(invalid='ignore', divide='ignore'):
            g = np.where((tot > 0) & (n >= 0.5 * tot) & (n > 0), s / n, np.nan).reshape(NLAT, NLON)
        tmp = out + '.tmp'
        with Dataset(tmp, 'w') as h:
            h.createDimension('lat', NLAT); h.createDimension('lon', NLON)
            vlat = h.createVariable('lat', 'f8', ('lat',)); vlat[:] = lats
            vlat.units = 'degrees_north'; vlat.standard_name = 'latitude'
            vlon = h.createVariable('lon', 'f8', ('lon',)); vlon[:] = lons
            vlon.units = 'degrees_east'; vlon.standard_name = 'longitude'
            v = h.createVariable(VAR, 'f4', ('lat', 'lon'), fill_value=FILL, zlib=True, complevel=1)
            v[:] = np.ma.masked_invalid(g).astype('f4')
            v.units = 'm'
            v.comment = 'bin mean of HEALPix H512 cells onto 0.25 deg; >=50%% valid cells required; represents day %s' % days[i].date()
        os.replace(tmp, out)
        print('  wrote %s (nan frac %.3f)' % (os.path.basename(out), np.isnan(g).mean()), flush=True)


# ---------------------------------------------------------------------------------------------
def detect(run, day):
    from netCDF4 import Dataset
    from py_eddy_tracker.dataset.grid import RegularGridDataset
    p = paths(run)
    os.makedirs(p['idd'], exist_ok=True)
    tag = day.strftime('%Y%m%d')
    outs = {t: '%s/eddyID_%s_%s.nc' % (p['idd'], t, tag) for t in ('anticyclonic', 'cyclonic')}
    if all(os.path.exists(o) for o in outs.values()):
        return False
    g = RegularGridDataset(day_file(run, day), 'lon', 'lat', centered=True)
    g.add_uv(VAR)
    g.bessel_high_filter(VAR, WAVELENGTH, order=1)
    a, c = g.eddy_identification(VAR, 'u', 'v', datetime(day.year, day.month, day.day), STEP_HT,
                                 pixel_limit=PIXEL_LIMIT, shape_error=SHAPE_ERROR)
    for t, obs in (('anticyclonic', a), ('cyclonic', c)):
        tmp = outs[t] + '.tmp'
        with Dataset(tmp, 'w') as h:
            obs.to_netcdf(h)
        os.replace(tmp, outs[t])
    print('  %s: %d anticyclonic, %d cyclonic' % (tag, len(a), len(c)), flush=True)
    return True


def stage_identify(run, stripe, nstripes, limit=None):
    # days come from the regridded files, so this stage needs no zarr/kerchunk stack (pyeddytracker env)
    fs = sorted(glob.glob('%s/zos_????????.nc' % paths(run)['zos']))
    days = [datetime.strptime(os.path.basename(f)[4:12], '%Y%m%d') for f in fs]
    mine = days[stripe::nstripes][:limit]
    print('identify %s: stripe %d/%d, %d days' % (run, stripe, nstripes, len(mine)), flush=True)
    for day in mine:
        if not os.path.exists(day_file(run, day)):
            print('  missing regridded field for %s, skip' % day.date()); continue
        t0 = T.time()
        done = detect(run, day)
        if done:
            print('    %.1f s' % (T.time() - t0), flush=True)


# ---------------------------------------------------------------------------------------------
def stage_track(run, eddy_type):
    from netCDF4 import Dataset
    from py_eddy_tracker.featured_tracking.area_tracker import AreaTracker
    from py_eddy_tracker.tracking import Correspondances
    p = paths(run)
    os.makedirs(p['trk'], exist_ok=True)
    files = sorted(glob.glob('%s/eddyID_%s_????????.nc' % (p['idd'], eddy_type)))
    print('track %s %s: %d daily files (%s .. %s)' % (run, eddy_type, len(files), os.path.basename(files[0]), os.path.basename(files[-1])), flush=True)
    c = Correspondances(datasets=files, class_method=AreaTracker, class_kw=dict(cmin=CMIN), virtual=VIRTUAL)
    c.track()
    c.prepare_merging()
    with Dataset('%s/%s_correspondance.nc' % (p['trk'], eddy_type), 'w') as h:
        c.to_netcdf(h)
    raw = False
    c.get_unused_data(raw_data=raw).write_file(filename='%s/%s_untracked.nc' % (p['trk'], eddy_type))
    short_c = c._copy()
    short_c.shorter_than(size_max=NB_OBS_MIN)
    short_track = short_c.merge(raw_data=raw)
    if c.longer_than(size_min=NB_OBS_MIN) is False:
        long_track = short_track.empty_dataset()
    else:
        long_track = c.merge(raw_data=raw)
    if c.virtual:
        long_track['virtual'][:] = long_track['time'] == 0
        long_track.normalize_longitude()
        long_track.filled_by_interpolation(long_track['virtual'] == 1)
        short_track['virtual'][:] = short_track['time'] == 0
        short_track.normalize_longitude()
        short_track.filled_by_interpolation(short_track['virtual'] == 1)
    print('longest track %d obs, mean length %.1f' % (c.nb_obs_by_tracks.max(), c.nb_obs_by_tracks.mean()))
    long_track.write_file(filename='%s/%s_tracks.nc' % (p['trk'], eddy_type))
    short_track.write_file(filename='%s/%s_short.nc' % (p['trk'], eddy_type))
    print('wrote %s/%s_tracks.nc (%d obs long, %d obs short)' % (p['trk'], eddy_type, len(long_track), len(short_track)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['regrid', 'identify', 'track', 'days'])
    ap.add_argument('run', choices=list(RUNS))
    ap.add_argument('--i0', type=int, default=0)
    ap.add_argument('--i1', type=int, default=10 ** 6)
    ap.add_argument('--stripe', type=int, default=0)
    ap.add_argument('--nstripes', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--type', default='cyclonic', choices=['cyclonic', 'anticyclonic'])
    a = ap.parse_args()
    if a.stage == 'regrid':
        stage_regrid(a.run, a.i0, a.i1)
    elif a.stage == 'identify':
        stage_identify(a.run, a.stripe, a.nstripes, a.limit)
    elif a.stage == 'track':
        stage_track(a.run, a.type)
    else:
        d = days_of(a.run); print(len(d), d[0], d[-1])
