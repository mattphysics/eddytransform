"""Read-only look at the coupled-run daily ocean product (avg_zos, HEALPix zoom 9 / 7)."""
import numpy as np, xarray as xr, traceback
K = '/work/bm1344/DKRZ/kerchunks_batched'
SRC = {
 'ng5_rcbmf_h512': f'{K}/NGC-IFS-tco3999-ng5-production/2D_daily_healpix512_ocean.parq',
 'ng5_rcbmf_h128': f'{K}/NGC-IFS-tco3999-ng5-production/2D_daily_healpix128_ocean.parq',
 'ng5_deepoff_h512': f'{K}/NGC-IFS-tco3999-ng5-production-deep-off/2D_daily_healpix512_ocean.parq',
}
def op(p):
    return xr.open_dataset('reference::'+p, engine='zarr', consolidated=False, chunks={},
        backend_kwargs=dict(storage_options=dict(remote_protocol='file', lazy=True)))
for k,p in SRC.items():
    print('='*90+'\n'+k, flush=True)
    try:
        ds = op(p)
        print('dims', dict(ds.sizes)); t = ds.time.values
        print('time n=%d %s .. %s  steps %s' % (t.size, t[0], t[-1], np.unique(np.diff(t))[:4]))
        for v in ds.data_vars: print('  %-14s %s units=%s chunks=%s' % (v, ds[v].dims, ds[v].attrs.get('units'), ds[v].chunks and tuple(c[0] for c in ds[v].chunks)))
        for c in ('lat','lon','crs'):
            if c in ds.variables: print(' ', c, ds[c].dims, dict(ds[c].attrs), ds[c].shape)
        z = ds['avg_zos'].isel(time=t.size//2).load()
        x = z.values; print('zos mid-time %s: nan frac %.3f  min %.3f mean %.3f max %.3f  std %.3f' % (str(z.time.values)[:16], np.isnan(x).mean(), np.nanmin(x), np.nanmean(x), np.nanmax(x), np.nanstd(x)))
        lat = ds['lat'].values; lon = ds['lon'].values
        print('lon range %.2f..%.2f  lat range %.2f..%.2f' % (lon.min(), lon.max(), lat.min(), lat.max()))
        for i in (0, t.size-1):
            xx = ds['avg_zos'].isel(time=i).values
            print('  step %d %s nan frac %.3f global-mean(zos) %.4f' % (i, str(t[i])[:16], np.isnan(xx).mean(), np.nanmean(xx)))
    except Exception: traceback.print_exc()
