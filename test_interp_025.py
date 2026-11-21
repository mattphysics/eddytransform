"""Was a 0.25 deg product remapped conservatively or point-sampled (bilinear-like)?

For each (0.25deg product, HEALPix product) pair of the same run, over one SO box:
  A = 0.25deg field
  C = conservative coarsening of the fine HEALPix field (mean of all fine cells whose centre
      falls in the 0.25deg cell; HEALPix cells are equal-area, so this is the area mean)
  N = nearest fine cell to each 0.25deg centre (what point sampling / bilinear from a much
      finer grid returns)
Conservative remap => A matches C better than N. Point sampling => A matches N better than C.
Smooth fields (2t) cannot discriminate; noisy fields (tprate, 10u) can.

Read-only. Run on a compute node.
"""
import sys
import traceback

import numpy as np
import xarray as xr

P = '/work/bm1235/b382473/IFS_AMIP/production'
K = '/work/bm1344/DKRZ/kerchunks_batched'
CMB = f'{P}/parquets/tco2559-eerie_production_202407-rcbmf_combination'
A3999 = f'{P}/parquets_2d3d/tco3999-ng-rcbmf_end'
NG = f'{K}/NGC-IFS-tco3999-ng5-production'
NGD = f'{K}/NGC-IFS-tco3999-ng5-production-deep-off'

BOX = dict(lon0=0.0, lon1=20.0, lat0=-60.0, lat1=-40.0)
MARGIN = 1.0  # deg, drop box edge where coarse cells are incompletely covered
T = np.datetime64('2020-07-31T12:00')

# name: (coarse path, fine path, kind, variables)
HOURLY_VARS = ['tprate', '2t', '10u', 'tcwv']
DAILY_VARS = ['tprate', 'mean2t', 'avg_10u', 'avg_tcwv']
PAIRS = {
    'tco2559_hourly_zoom10': (f'{CMB}/2D_hourly_0.25deg.parq', f'{CMB}/2D_hourly_healpix1024.parq', 'hourly', HOURLY_VARS),
    'tco2559_daily_gr025_vs_zoom10': (f'{P}/parquets/ifs-amip-tco2559/hist/v20250101/atmos/gr025/2D_24h.parq',
                                      f'{CMB}/2D_daily_healpix1024.parq', 'daily', DAILY_VARS),
    'tco3999_amip_hourly_zoom11': (f'{A3999}/hourly_0.25deg.parq', f'{A3999}/hourly_healpix2048.parq', 'hourly', HOURLY_VARS),
    'ng5_rcbmf_hourly_zoom11': (f'{NG}/2D_hourly_0.25deg.parq', f'{NG}/2D_hourly_healpix2048.parq', 'hourly', HOURLY_VARS),
    'ng5_deepoff_hourly_zoom11': (f'{NGD}/2D_hourly_0.25deg.parq', f'{NGD}/2D_hourly_healpix2048.parq', 'hourly', HOURLY_VARS),
}


def open_src(path):
    return xr.open_dataset(
        'reference::' + path, engine='zarr', consolidated=False, chunks={},
        backend_kwargs=dict(storage_options=dict(remote_protocol='file', lazy=True)),
    )


def lon360(x):
    return np.mod(np.asarray(x, dtype='float64'), 360.0)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def stats(label, a, ref):
    m = np.isfinite(a) & np.isfinite(ref)
    a, ref = a[m], ref[m]
    return ('%-4s corr=%.5f  rmse/std(ref)=%.4f  std ratio=%.4f  mean ratio=%.4f  max ratio=%.4f  p99.9 ratio=%.4f' % (
        label, np.corrcoef(a, ref)[0, 1], np.sqrt(np.mean((a - ref) ** 2)) / ref.std(),
        a.std() / ref.std(), a.mean() / ref.mean() if ref.mean() != 0 else np.nan,
        a.max() / ref.max() if ref.max() != 0 else np.nan,
        np.percentile(a, 99.9) / np.percentile(ref, 99.9) if np.percentile(ref, 99.9) != 0 else np.nan))


def prepare(ds_c, ds_f):
    """index sets and the C/N mapping, computed once per pair"""
    # coarse (regular, flat 'value')
    clat = np.asarray(ds_c['lat'].values, 'float64')
    clon = lon360(ds_c['lon'].values)
    inbox_c = ((clat >= BOX['lat0'] + MARGIN) & (clat <= BOX['lat1'] - MARGIN)
               & (clon >= BOX['lon0'] + MARGIN) & (clon <= BOX['lon1'] - MARGIN))
    ci = np.where(inbox_c)[0]
    # fine (HEALPix nested)
    flat = np.asarray(ds_f['lat'].values, 'float64')
    flon = lon360(ds_f['lon'].values)
    inbox_f = ((flat >= BOX['lat0']) & (flat <= BOX['lat1'])
               & (flon >= BOX['lon0']) & (flon <= BOX['lon1']))
    fi = np.where(inbox_f)[0]
    flat, flon = flat[fi], flon[fi]
    # bin fine cells to the nearest 0.25 deg centre (cells are centred on multiples of 0.25)
    blat = np.round(flat / 0.25) * 0.25
    blon = np.round(flon / 0.25) * 0.25
    key_f = np.round((blat + 90) / 0.25).astype('int64') * 100000 + np.round(blon / 0.25).astype('int64')
    key_c = np.round((clat[ci] + 90) / 0.25).astype('int64') * 100000 + np.round(clon[ci] / 0.25).astype('int64')
    uniq, inv = np.unique(key_f, return_inverse=True)
    pos = np.searchsorted(uniq, key_c)
    ok = (pos < uniq.size) & (uniq[np.minimum(pos, uniq.size - 1)] == key_c)
    ci, key_c, pos = ci[ok], key_c[ok], pos[ok]
    # nearest fine cell per bin
    d = (flat - blat) ** 2 + ((flon - blon) * np.cos(np.deg2rad(flat))) ** 2
    order = np.lexsort((d, inv))
    first = np.r_[True, inv[order][1:] != inv[order][:-1]]
    nearest = fi[order[first]]  # one per uniq bin, same ordering as uniq
    counts = np.bincount(inv, minlength=uniq.size)
    print('  box cells: coarse used=%d, fine in box=%d, median fine cells per 0.25deg cell=%d' % (
        ci.size, fi.size, np.median(counts[pos])), flush=True)
    return dict(ci=ci, fi=fi, inv=inv, pos=pos, nearest=nearest, counts=counts, nbin=uniq.size)


def get_CN(vals_f_box, m):
    """conservative (bin mean) and nearest-cell samples at the coarse cells"""
    sums = np.bincount(m['inv'], weights=vals_f_box.astype('float64'), minlength=m['nbin'])
    C = (sums / m['counts'])[m['pos']]
    return C


def read(ds, v, t, idx):
    da = ds[v].sel(time=t)
    return np.asarray(da.isel(value=idx).values, 'float64')


def run_pair(name, cp, fp, kind, variables):
    print('\n' + '=' * 100 + '\n' + name + '\n  coarse: %s\n  fine:   %s' % (cp, fp), flush=True)
    ds_c, ds_f = open_src(cp), open_src(fp)
    print('  coarse time %s .. %s ; fine time %s .. %s' % (
        str(ds_c.time.values[0])[:16], str(ds_c.time.values[-1])[:16],
        str(ds_f.time.values[0])[:16], str(ds_f.time.values[-1])[:16]))
    m = prepare(ds_c, ds_f)
    # position of 'nearest' index within the fine box array
    pos_in_box = np.searchsorted(m['fi'], m['nearest'])
    for v in variables:
        if v not in ds_c or v not in ds_f:
            print('  %-8s missing (coarse %s, fine %s)' % (v, v in ds_c, v in ds_f))
            continue
        try:
            print('  --- %s  (coarse units=%s, fine units=%s)' % (v, ds_c[v].attrs.get('units'), ds_f[v].attrs.get('units')))
            A = read(ds_c, v, T, m['ci'])
            F = read(ds_f, v, T, m['fi'])
            C = get_CN(F, m)
            N = F[pos_in_box][m['pos']]
            print('    ' + stats('A~C', A, C))
            print('    ' + stats('A~N', A, N))
            print('    -> A closer to %s (rmse C=%.4g N=%.4g, in units of std(C))' % (
                'CONSERVATIVE (C)' if np.sqrt(np.nanmean((A - C) ** 2)) < np.sqrt(np.nanmean((A - N) ** 2)) else 'NEAREST/POINT (N)',
                np.sqrt(np.nanmean((A - C) ** 2)) / np.nanstd(C), np.sqrt(np.nanmean((A - N) ** 2)) / np.nanstd(C)))
        except Exception:
            print('  FAILED', v)
            traceback.print_exc()
    # tp: also resolve the time stamp by comparing hourly accumulations
    if kind == 'hourly' and 'tp' in ds_c and 'tp' in ds_f:
        print('  --- tp (accumulated): which fine stamp matches the coarse stamp %s ?' % str(T)[:16])
        A = read(ds_c, 'tp', T, m['ci'])
        for dh in (-1, 0, 1):
            tf = T + np.timedelta64(dh, 'h')
            F = read(ds_f, 'tp', tf, m['fi'])
            C = get_CN(F, m)
            print('    fine stamp %+dh: corr(A,C)=%.5f  mean ratio A/C=%.4f' % (dh, corr(A, C), A.mean() / C.mean()))
        # is tprate an instantaneous rate or hourly mean? compare to tp/3600 at the end-stamp
        if 'tprate' in ds_f:
            r = read(ds_f, 'tprate', T, m['fi'])
            for dh in (0, 1):
                tp_ = read(ds_f, 'tp', T + np.timedelta64(dh, 'h'), m['fi']) * 1000 / 3600.0
                print('    fine: corr(tprate@T, tp@T%+dh /3600 [kg m-2 s-1])=%.5f  ratio of means=%.4f  ratio of stds=%.4f' % (
                    dh, corr(r, tp_), r.mean() / tp_.mean(), r.std() / tp_.std()))


if __name__ == '__main__':
    only = sys.argv[1:]
    for name, (cp, fp, kind, variables) in PAIRS.items():
        if only and not any(o in name for o in only):
            continue
        try:
            run_pair(name, cp, fp, kind, variables)
        except Exception:
            print('FAILED PAIR', name)
            traceback.print_exc()
