"""Bundle per-eddy composite NetCDFs into one zarr store per (source, kind).

Same layout as save_eddies_rot_to_zarr_exp.py (obs-chunked, zstd, store next to the directory:
.../composites_rot_cyc/ -> .../composites_rot_cyc.zarr), with differences:
  * the last partial block is written too,
  * source files are left in place (nothing is moved or deleted),
  * resumable: eddies already in the store are skipped, and the final check compares
    #files against #obs in the store.
Usage: bundle_composites_zarr.py <source> <cyc|acyc> [--root ...] [--block 1000]
"""
import argparse
import glob
import os
import re
import sys
import time as T

import numpy as np
import xarray as xr
import zarr

ap = argparse.ArgumentParser()
ap.add_argument('source')
ap.add_argument('tag', choices=['cyc', 'acyc'])
ap.add_argument('--root', default='/work/bk1377/b382473/composites/')
ap.add_argument('--region', default='')          # '' for SO (old naming), 'GS_' for Gulf Stream
ap.add_argument('--block', type=int, default=1000)
a = ap.parse_args()

d = os.path.join(a.root, a.source, 'atmos', 'gr025', '2D_daily', 'composites_%srot_%s' % (a.region, a.tag))
zpath = d + '.zarr'
files = glob.glob(os.path.join(d, 'eddy_r_1_*.nc'))
obs_of = lambda f: int(re.search(r'eddy_r_1_(\d+)\.nc$', f).group(1))
files = sorted(files, key=obs_of)
print('%s: %d eddy files -> %s' % (d, len(files), zpath), flush=True)
if not files:
    sys.exit('no files')

done = set()
if os.path.exists(zpath):
    done = set(xr.open_zarr(zpath)['obs'].values.tolist())
    print('%d obs already in store' % len(done))
todo = [f for f in files if obs_of(f) not in done]
print('%d to add' % len(todo), flush=True)

compressor = zarr.Blosc(cname='zstd', clevel=7, shuffle=2)
bad = []
for b0 in range(0, len(todo), a.block):
    t0 = T.time()
    block = todo[b0:b0 + a.block]
    parts = []
    for f in block:
        try:
            with xr.open_dataset(f) as dsi:
                dsi = dsi.load()
            if dsi.sizes.get('x') != 91 or dsi.sizes.get('y') != 91:
                raise ValueError('unexpected shape %s' % dict(dsi.sizes))
            dsi = dsi.assign(radius_km=float(dsi.attrs['EDDY_RADIUS']))   # per-eddy radius, else lost in the concat
            dsi.attrs = {k: v for k, v in dsi.attrs.items() if k != 'EDDY_RADIUS'}
            parts.append(dsi.expand_dims(obs=[obs_of(f)]) if 'obs' not in dsi.dims else dsi)
        except Exception as e:
            print('  BAD %s: %s' % (f, e), flush=True)
            bad.append(f)
    if not parts:
        continue
    ds = xr.concat(parts, dim='obs', coords='minimal', compat='override')
    # scalar per-eddy variables (time, lon, lat) must be indexed by obs only
    for v in ('time', 'lon', 'lat', 'radius_km'):
        if v in ds and 'obs' not in ds[v].dims:
            ds[v] = ds[v].expand_dims(obs=ds['obs'])
    ds = ds.chunk({'obs': ds.sizes['obs'], 'x': ds.sizes['x'], 'y': ds.sizes['y']})
    for v in ds.data_vars:
        if ds[v].ndim >= 3:
            ds[v].encoding.update({'compressor': compressor})
    for v in ('x', 'y'):
        ds[v].encoding.update({'chunks': (-1,), 'compressor': compressor})
    if not os.path.exists(zpath):
        ds.to_zarr(zpath)
    else:
        ds.to_zarr(zpath, append_dim='obs')
    print('[%s] block %d: obs %d..%d (%d eddies) %.1fs' % (
        T.strftime('%H:%M:%S'), b0 // a.block + 1, ds['obs'].values[0], ds['obs'].values[-1], ds.sizes['obs'], T.time() - t0), flush=True)

z = xr.open_zarr(zpath)
n_store = z.sizes['obs']
print('FINAL: %d files, %d obs in store, %d bad files; obs unique: %s' % (
    len(files), n_store, len(bad), len(np.unique(z['obs'].values)) == n_store))
sys.exit(0 if (n_store == len(files) - len(bad) and not bad) else 3)
