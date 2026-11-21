"""Eddy composites for the 2020-2021 IFS simulations on Levante (daily means, 0.25 deg).

One task = one source, one date (daily mean stamped 12:00, like the AVISO track times),
one eddy kind. Writes one NetCDF per eddy, same layout/variables as the older
eddytransform_run_levante_date_*.py drivers (variables keep the AMIP daily names).

Sources
  tco2559_deepon   v20240901  native daily means (2D_24h)              AVISO tracks
  tco2559_rcbmf    v20250101  native daily means (2D_24h)              AVISO tracks
  tco3999_amip                daily means built from hourly 0.25 deg   AVISO tracks
  ng5_rcbmf        coupled    daily means built from hourly 0.25 deg   own tracks (track_coupled.py)
  ng5_deepoff      coupled    daily means built from hourly 0.25 deg   own tracks (track_coupled.py)

Daily means from hourly data, for the day D (stamp D 12:00):
  instantaneous fields : mean of the 24 hourly values D 00:00 .. D 23:00 (the model's own
                         daily-mean convention)
  accumulated fields   : sum of the 24 hourly accumulations stamped D 01:00 .. D+1 00:00
                         (accumulation covers the hour *preceding* the stamp) / 86400 s
  tprate is built from tp (hourly tprate is not an hour mean) in m s-1, like the native daily files.
  Fluxes (J m-2 per hour) -> W m-2, same sign convention as the native mslhf, msshf, ...
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import eddytransform as et

xr.set_options(keep_attrs=True)

P = '/work/bm1235/b382473/IFS_AMIP/production'
K = '/work/bm1344/DKRZ/kerchunks_batched'
AVISO = '/work/bk1377/b382473/obs/AVISO/META3.2_DT_allsat_%s_long_19930101_20220209.nc'

# first / last daily-mean date (stamp 12:00) that is fully covered, from the real time axes
SOURCES = {
    'tco2559_deepon': dict(kind='daily', start='2020-01-01', end='2020-06-30', tracks='AVISO',
                           path=f'{P}/parquets/ifs-amip-tco2559/hist/v20240901/atmos/gr025/2D_24h.parq'),
    'tco2559_rcbmf': dict(kind='daily', start='2020-01-01', end='2021-02-28', tracks='AVISO',
                          path=f'{P}/parquets/ifs-amip-tco2559/hist/v20250101/atmos/gr025/2D_24h.parq'),
    'tco3999_amip': dict(kind='hourly', start='2020-01-01', end='2021-02-28', tracks='AVISO',
                         path=f'{P}/parquets_2d3d/tco3999-ng-rcbmf_end/hourly_0.25deg.parq'),
    'ng5_rcbmf': dict(kind='hourly', start='2020-01-01', end='2021-02-28',
                      tracks='/work/bk1377/b382473/eddytracks/ng5_rcbmf/0.25deg/eddyTracks/{kind}_tracks.nc',
                      path=f'{K}/NGC-IFS-tco3999-ng5-production/2D_hourly_0.25deg.parq'),
    'ng5_deepoff': dict(kind='hourly', start='2020-01-01', end='2021-02-28',
                        tracks='/work/bk1377/b382473/eddytracks/ng5_deepoff/0.25deg/eddyTracks/{kind}_tracks.nc',
                        path=f'{K}/NGC-IFS-tco3999-ng5-production-deep-off/2D_hourly_0.25deg.parq'),
}

DEFAULT_VARNAMES = ['avg_2sh', 'avg_ci', 'avg_sst', 'mean10ws', 'mean2t', 'mslhf', 'msshf', 'tprate',
                    'meantcc', 'avg_blh', 'avg_tcwv', 'avg_tcw', 'msnlwrf', 'msnswrf', 'mtnlwrf', 'mtnswrf']
UV = ['avg_10u', 'avg_10v']

# ---- hourly -> daily recipes -------------------------------------------------------------
# instantaneous fields: output name -> hourly name
INST = {'avg_10u': '10u', 'avg_10v': '10v', 'mean2t': '2t', 'avg_sst': 'sst', 'avg_ci': 'ci',
        'meantcc': 'tcc', 'avg_blh': 'blh', 'avg_tcwv': 'tcwv', 'mean10ws': '10si'}
# accumulated J m-2 per hour -> W m-2 (output name -> hourly name)
FLUX = {'mslhf': 'slhf', 'msshf': 'sshf', 'msnlwrf': 'str', 'msnswrf': 'ssr', 'mtnlwrf': 'ttr', 'mtnswrf': 'tsr'}
UNITS = {'avg_10u': 'm s**-1', 'avg_10v': 'm s**-1', 'mean2t': 'K', 'avg_sst': 'K', 'avg_ci': '(0 - 1)',
         'meantcc': '(0 - 1)', 'avg_blh': 'm', 'avg_tcwv': 'kg m**-2', 'avg_tcw': 'kg m**-2',
         'mean10ws': 'm s**-1', 'avg_2sh': 'kg kg**-1', 'tprate': 'm s**-1',
         **{k: 'W m**-2' for k in FLUX}}


def open_src(path):
    return xr.open_dataset(
        'reference::' + path, engine='zarr', consolidated=False, chunks={},
        backend_kwargs=dict(storage_options=dict(remote_protocol='file', lazy=True)),
    )


def date_list(source, freq):
    s = SOURCES[source]
    all_days = pd.date_range(s['start'], s['end'], freq='1D')
    step = int(pd.Timedelta(freq) / pd.Timedelta('1D'))
    return all_days[::step]


def q_from_dewpoint(td, sp):
    """IFS-style (Tetens over water) specific humidity from dew point [K] and pressure [Pa]."""
    e = 611.21 * np.exp(17.502 * (td - 273.16) / (td - 32.19))
    return 0.622 * e / (sp - (1 - 0.622) * e)


def daily_from_hourly(ds, date, varnames):
    """Build the daily-mean dataset (one time step, stamp date 12:00) for `varnames`."""
    t0 = np.datetime64(pd.Timestamp(date).to_datetime64(), 'ns')
    hrs = np.arange(25) * np.timedelta64(1, 'h')
    t_inst = t0 + hrs[:24]                  # 00:00 .. 23:00 (24 instantaneous values)
    t_acc = t0 + hrs[1:]                    # 01:00 .. next 00:00 (24 accumulations, end-stamped)
    w_inst = np.full(24, 1 / 24.0)

    def load(v, times):
        da = ds[v].sel(time=times)
        assert da.sizes['time'] == len(times), '%s: missing times for %s' % (v, date)
        return np.asarray(da.values, dtype='float64')

    def inst_mean(v):
        return np.tensordot(w_inst, load(v, t_inst), axes=(0, 0))

    out, skipped = {}, []
    for name in varnames:
        try:
            if name in INST and INST[name] in ds:
                out[name] = inst_mean(INST[name])
            elif name == 'mean10ws':  # no 10si: from components
                u, v = load('10u', t_inst), load('10v', t_inst)
                out[name] = np.tensordot(w_inst, np.hypot(u, v), axes=(0, 0))
            elif name in FLUX and FLUX[name] in ds:
                out[name] = load(FLUX[name], t_acc).sum(axis=0) / 86400.0
            elif name == 'tprate':
                out[name] = load('tp', t_acc).sum(axis=0) / 86400.0   # m s-1 of water
            elif name == 'avg_2sh':
                out[name] = np.tensordot(w_inst, q_from_dewpoint(load('2d', t_inst), load('sp', t_inst)), axes=(0, 0))
            elif name == 'avg_tcw':
                total = inst_mean('tcwv')
                for part in ('tclw', 'tciw', 'tcrw', 'tcsw'):
                    total = total + inst_mean(part)
                out[name] = total
            else:
                skipped.append(name)
        except KeyError:
            skipped.append(name)
    if skipped:
        print('WARNING: not available/derivable from this hourly product, skipped: %s' % skipped)
    t12 = (t0 + np.timedelta64(12, 'h'))
    dsd = xr.Dataset(
        {k: (('time', 'value'), v[None].astype('float32'), {'units': UNITS.get(k, '')}) for k, v in out.items()},
        coords={'time': [t12], 'lat': ('value', np.asarray(ds['lat'].values)), 'lon': ('value', np.asarray(ds['lon'].values))},
    )
    return dsd, [k for k in varnames if k in out]


def load_tracks(source, kind, tracks_path, cache_dir):
    """Return a Dataset of tracks and the per-obs day number used for fast date selection."""
    path = tracks_path or (AVISO % kind.capitalize())
    tr = xr.open_dataset(path)
    if tracks_path:   # own tracks: key by source (file names like cyclonic_tracks.nc are not unique)
        key = os.path.join(cache_dir, 'trackdays_%s_%s.npy' % (source, kind))
    else:             # AVISO: unchanged name, shared with the running AMIP jobs
        key = os.path.join(cache_dir, 'trackdays_%s_%s.npy' % (os.path.basename(path).replace('.nc', ''), kind))
    ref = np.datetime64('1900-01-01', 'D')
    if os.path.exists(key):
        days = np.load(key)
    else:
        print('Build track-day cache %s' % key)
        days = (tr['time'].values.astype('datetime64[D]') - ref).astype('int32')
        os.makedirs(cache_dir, exist_ok=True)
        tmp = key + '.%d.tmp.npy' % os.getpid()
        np.save(tmp, days)
        os.replace(tmp, key)
    return tr, days, ref


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=list(SOURCES))
    parser.add_argument('idate', type=int, help='index into the date list (every --freq days from the start of the run)')
    parser.add_argument('kind', choices=['cyclonic', 'anticyclonic'])
    parser.add_argument('--region', default='SO', choices=['SO', 'GS', 'global'])
    parser.add_argument('--freq', default='2D')
    parser.add_argument('--tracks', default=None, help='py-eddy-tracker netcdf (needed for the coupled runs)')
    parser.add_argument('--vn', default=None, help='single variable (default: all)')
    parser.add_argument('--rw', default='1')
    parser.add_argument('--nmax', type=int, default=None, help='only the first N eddies (testing)')
    parser.add_argument('--outroot', default='/work/bk1377/b382473/composites/')
    parser.add_argument('--check-only', action='store_true', help='build the daily dataset, print stats, do not composite')
    parser.add_argument('--ndates', action='store_true', help='print the number of dates and exit')
    args = parser.parse_args()
    print(args, flush=True)

    src = SOURCES[args.source]
    dates = date_list(args.source, args.freq)
    if args.ndates:
        print(len(dates))
        sys.exit(0)
    date = dates[args.idate]
    date12 = date + pd.Timedelta('12h')
    print('source %s: date index %d -> %s (daily mean stamped %s)' % (args.source, args.idate, date.date(), date12), flush=True)

    rotate_winds = args.rw != '0'
    extent = {'SO': [0, 360, -60, -25], 'GS': [360 - 82, 360 - 20, 28, 60], 'global': None}[args.region]
    varnames = [args.vn] if args.vn else list(DEFAULT_VARNAMES)

    # ---- data ---------------------------------------------------------------------------
    ds_raw = open_src(src['path'])
    need = varnames + (UV if rotate_winds else [])
    if src['kind'] == 'daily':
        missing = [v for v in need if v not in ds_raw]
        if missing:
            print('WARNING: not in native daily file, skipped: %s' % missing)
        keep = [v for v in need if v in ds_raw]
        ds = ds_raw[keep].sel(time=[np.datetime64(date12, 'ns')]).load()
        varnames = [v for v in varnames if v in keep]
    else:
        ds, built = daily_from_hourly(ds_raw, date, need)
        varnames = [v for v in varnames if v in built]
    print('Daily dataset:', {k: tuple(ds[k].dims) for k in ds.data_vars}, flush=True)
    for v in ds.data_vars:
        x = ds[v].values
        print('  %-10s [%s] min %.4g mean %.4g max %.4g  nan %d' % (
            v, ds[v].attrs.get('units'), np.nanmin(x), np.nanmean(x), np.nanmax(x), np.isnan(x).sum()))
    if args.check_only:
        sys.exit(0)

    ds = et.reshape_latlon_1d_to_latlon_2d(ds)
    if et.identify_lontype(ds=ds) == '180':
        ds = et.lon_180_to_360(ds, sort=True)

    # ---- tracks -------------------------------------------------------------------------
    if args.tracks:
        tracks_path = args.tracks
    elif src['tracks'] == 'AVISO':
        tracks_path = None
    else:
        tracks_path = src['tracks'].format(kind=args.kind)   # own py-eddy-tracker tracks (track_coupled.py)
    print('tracks: %s' % (tracks_path or 'AVISO'), flush=True)
    cache_dir = os.path.join(args.outroot, '_cache')
    tr, days, ref = load_tracks(args.source, args.kind, tracks_path, cache_dir)
    want = int((np.datetime64(date, 'D') - ref).astype(int))
    idx = np.where(days == want)[0]
    sub = tr[['time', 'longitude', 'latitude', 'effective_radius']].isel(obs=idx).load()
    sub['obs'] = idx  # same obs numbering as et.get_tracks (np.arange over the file)
    sub['longitude'] = sub['longitude'] % 360  # own tracks run continuously across 0 (e.g. -7.8 .. 363.9)
    sub['time'] = sub['time'] + pd.Timedelta('12h')
    if extent is not None:
        x0, x1, y0, y1 = extent
        m = ((sub['latitude'] > y0) & (sub['latitude'] < y1)
             & (sub['longitude'] > x0) & (sub['longitude'] < x1)).values
        sub = sub.isel(obs=np.where(m)[0])
    if args.nmax:
        sub = sub.isel(obs=slice(args.nmax))
    print('%d %s eddies on %s in region %s' % (sub['obs'].size, args.kind, date.date(), args.region), flush=True)

    # ---- output -------------------------------------------------------------------------
    tag = {'cyclonic': 'cyc', 'anticyclonic': 'acyc'}[args.kind]
    reg = '' if args.region == 'SO' else '%s_' % args.region   # SO keeps the old (unprefixed) name
    outdir = os.path.join(args.outroot, args.source, 'atmos', 'gr025', '2D_daily', 'composites_%srot_%s' % (reg, tag)) + '/'
    os.makedirs(outdir, exist_ok=True)
    print('Output: %s' % outdir, flush=True)

    transform_settings = dict(
        DOMAIN_HALF_WIDTH_IN_DEGREES=20, AVG_WIND_EDDY_RADIUSES=3, RESAMPLE_EDDY_RADIUSES=3, RESAMPLE_DENSITY=30,
        UPARAM='avg_10u', VPARAM='avg_10v', output='single', rotate_winds=rotate_winds,
        fname_root=outdir + 'eddy_r_1_',
    )
    et.loop_over_eddies(ds, sub, varnames, **transform_settings)
    print('Done')
