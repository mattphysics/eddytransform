"""Read-only discovery of the five IFS simulations (time axes, variables, units, grids).

Run on a compute node (sbatch discover_sims.slurm). Reads metadata and one small
slice per source; writes nothing except stdout.
"""
import sys
import traceback

import numpy as np
import xarray as xr

P = '/work/bm1235/b382473/IFS_AMIP/production/parquets'
K = '/work/bm1344/DKRZ/kerchunks_batched'
AMIP3999 = '/work/bm1235/b382473/IFS_AMIP/production/parquets_2d3d/tco3999-ng-rcbmf_end'

SOURCES = {
    'tco2559_v20240901_gr025_2D_24h': f'{P}/ifs-amip-tco2559/hist/v20240901/atmos/gr025/2D_24h.parq',
    'tco2559_v20250101_gr025_2D_24h': f'{P}/ifs-amip-tco2559/hist/v20250101/atmos/gr025/2D_24h.parq',
    'tco2559_v20250101_gr025_2D_1h': f'{P}/ifs-amip-tco2559/hist/v20250101/atmos/gr025/2D_1h.parq',
    'tco3999_amip_gr025_2D_1h': f'{P}/ifs-amip-tco3999/hist/v20250101/atmos/gr025/2D_1h.parq',
    'tco3999_amip_gr025_2D_1h_acc': f'{P}/ifs-amip-tco3999/hist/v20250101/atmos/gr025/2D_1h_acc.parq',
    'tco3999_amip_hp2048_hourly': f'{AMIP3999}/hourly_healpix2048.parq',
    'ng5_rcbmf_0.25deg_hourly': f'{K}/NGC-IFS-tco3999-ng5-production/2D_hourly_0.25deg.parq',
    'ng5_rcbmf_hp2048_hourly': f'{K}/NGC-IFS-tco3999-ng5-production/2D_hourly_healpix2048.parq',
    'ng5_deepoff_0.25deg_hourly': f'{K}/NGC-IFS-tco3999-ng5-production-deep-off/2D_hourly_0.25deg.parq',
    'ng5_deepoff_hp2048_hourly': f'{K}/NGC-IFS-tco3999-ng5-production-deep-off/2D_hourly_healpix2048.parq',
}


def open_src(path):
    return xr.open_dataset(
        'reference::' + path, engine='zarr', consolidated=False, chunks={},
        backend_kwargs=dict(storage_options=dict(remote_protocol='file', lazy=True)),
    )


def describe(name, path):
    print('\n' + '=' * 100)
    print(name, path, flush=True)
    ds = open_src(path)
    print('dims:', dict(ds.sizes))
    print('coords:', list(ds.coords))
    if 'time' in ds.coords:
        t = ds['time'].values
        print('time: n=%d  %s .. %s' % (t.size, t[0], t[-1]))
        if t.size > 2:
            dt = np.unique(np.diff(t))
            print('time steps (unique):', dt[:5], 'first stamps:', t[:3])
            print('calendar/encoding:', ds['time'].encoding.get('calendar'), ds['time'].attrs)
    for v in ds.data_vars:
        a = ds[v].attrs
        print('  %-14s %-28s units=%-12s long=%s chunks=%s' % (
            v, ds[v].dims, a.get('units'), (a.get('long_name') or a.get('GRIB_name') or '')[:40],
            ds[v].chunks if ds[v].chunks is None else tuple(c[0] for c in ds[v].chunks)))
    for c in ('lat', 'lon', 'crs'):
        if c in ds.variables:
            x = ds[c]
            print('  %s: dims=%s attrs=%s' % (c, x.dims, dict(x.attrs)))
    # one small slice of the first 2D variable that looks like 2 m temperature
    for v in ('2t', 'mean2t', 'avg_2t', 'tas'):
        if v in ds:
            sl = ds[v].isel(time=ds.sizes['time'] // 2)
            if 'value' in sl.dims and sl.sizes['value'] > 10_000_000:
                sl = sl.isel(value=slice(0, 1_000_000))
            x = sl.load()
            print('  slice %s @ %s: min %.2f mean %.2f max %.2f' % (
                v, str(sl['time'].values)[:19], float(x.min()), float(x.mean()), float(x.max())))
            break
    return ds


if __name__ == '__main__':
    only = sys.argv[1:]
    for name, path in SOURCES.items():
        if only and not any(o in name for o in only):
            continue
        try:
            describe(name, path)
        except Exception:
            print('FAILED', name)
            traceback.print_exc()
