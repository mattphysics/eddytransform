"""Evaluate the eddy composites of the five 2020-21 simulations.

  compute : read the zarr stores, write eval/results.pkl (anomaly maps, centre values per eddy)
  report  : print tables, write figures to eval/

Anomaly = field minus that eddy's mean over the outer ring (r > 2.5 R). Centre = mean over r < 0.5 R.
Paired comparisons use identical eddies (same AVISO observation index). Standard errors are
clustered by track (repeated observations of one eddy are not independent).
Usage: evaluate_composites.py compute|report [SO|GS]   (GS writes to eval/GS/)
"""
import os
import pickle
import sys

import numpy as np
import xarray as xr

ROOT = '/work/bk1377/b382473/composites/'
REGION = sys.argv[2] if len(sys.argv) > 2 else 'SO'      # SO (default, old names) or GS
assert REGION in ('SO', 'GS')
OUT = '/home/b/b382473/code/eddytransform/eval/' + ('' if REGION == 'SO' else 'GS/')
AVISO = '/work/bk1377/b382473/obs/AVISO/META3.2_DT_allsat_%s_long_19930101_20220209.nc'
TRK = '/work/bk1377/b382473/eddytracks/%s/0.25deg/eddyTracks/%s_tracks.nc'
SOURCES = ['tco2559_deepon', 'tco2559_rcbmf', 'tco3999_amip', 'ng5_rcbmf', 'ng5_deepoff']
KINDS = {'cyc': 'cyclonic', 'acyc': 'anticyclonic'}
# variable: (label, unit, factor)
VARS = {
    'avg_sst': ('SST', 'K', 1.0),
    'mean2t': ('2 m temperature', 'K', 1.0),
    'mean10ws': ('10 m wind speed', 'm/s', 1.0),
    'avg_10u': ('wind along mean-wind dir.', 'm/s', 1.0),
    'tprate': ('precipitation', 'mm/day', 86400e3),
    'mslhf': ('latent heat flux (down +)', 'W/m2', 1.0),
    'msshf': ('sensible heat flux (down +)', 'W/m2', 1.0),
    'avg_blh': ('boundary layer height', 'm', 1.0),
    'avg_tcwv': ('column water vapour', 'kg/m2', 1.0),
    'meantcc': ('total cloud cover', '1', 1.0),
    'avg_2sh': ('2 m specific humidity', 'g/kg', 1e3),
}
PAIRS = [('tco3999_amip', 'tco2559_rcbmf'), ('tco2559_deepon', 'tco2559_rcbmf'), ('ng5_rcbmf', 'ng5_deepoff')]


def zpath(src, tag):
    return os.path.join(ROOT, src, 'atmos', 'gr025', '2D_daily', 'composites_%srot_%s.zarr' % ('' if REGION == 'SO' else 'GS_', tag))


def track_ids(src, kind, obs):
    if src.startswith('ng5'):
        path = TRK % (src, kind)
    else:
        path = AVISO % kind.capitalize()
    tr = xr.open_dataset(path)['track'].values
    return np.asarray(tr[obs])


def cluster_se(x, g):
    """clustered SE of the mean of x with cluster labels g (NaNs dropped)"""
    m = np.isfinite(x)
    x, g = x[m], g[m]
    n = x.size
    if n < 3:
        return np.nan, n, 0
    xc = x - x.mean()
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=xc)
    ncl = s.size
    se = np.sqrt((s ** 2).sum() * ncl / max(ncl - 1, 1)) / n
    return se, n, ncl


def compute():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for tag, kind in KINDS.items():
        stores = {s: xr.open_zarr(zpath(s, tag)) for s in SOURCES if os.path.exists(zpath(s, tag))}
        obs = {s: z['obs'].values for s, z in stores.items()}
        subsets = {}
        for a, b in PAIRS:
            if a in obs and b in obs:
                subsets[(a, b)] = np.intersect1d(obs[a], obs[b]) if not a.startswith('ng5') else None
        for s, z in stores.items():
            x, y = z['x'], z['y']
            r = np.sqrt(x ** 2 + y ** 2)
            ring, core = r > 2.5, r < 0.5
            print('%s %s: %d eddies' % (s, tag, z.sizes['obs']), flush=True)
            # subsets of this store: all eddies, and those paired with another store
            sel = {'all': obs[s]}
            for (a, b), common in subsets.items():
                if common is not None and s in (a, b):
                    sel['paired_%s_%s' % (a, b)] = common
            meta = dict(obs=obs[s], lon=z['lon'].values, lat=z['lat'].values, radius_km=z['radius_km'].values,
                        time=z['time'].values.astype('datetime64[D]').astype(str), track=track_ids(s, kind, obs[s]))
            res[(s, tag, 'meta')] = meta
            for v, (_, _, fac) in VARS.items():
                if v not in z:
                    continue
                f = z[v] * fac
                anom = f - f.where(ring).mean(('x', 'y'))
                centre = anom.where(core).mean(('x', 'y')).compute().values       # per eddy
                res[(s, tag, v, 'centre')] = centre
                for name, ids in sel.items():
                    idx = np.searchsorted(obs[s], ids)
                    res[(s, tag, v, 'amap', name)] = anom.isel(obs=idx).mean('obs').compute().values
                    res[(s, tag, v, 'rawmap', name)] = f.isel(obs=idx).mean('obs').compute().values
                print('  %-9s done' % v, flush=True)
    res['x'] = xr.open_zarr(zpath('tco2559_rcbmf', 'cyc'))['x'].values
    with open(OUT + 'results.pkl', 'wb') as h:
        pickle.dump(res, h)
    print('wrote', OUT + 'results.pkl')


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(OUT + 'results.pkl', 'rb') as h:
        res = pickle.load(h)
    x = res['x']
    ext = [x[0], x[-1], x[0], x[-1]]
    lines = []

    def P(s=''):
        print(s); lines.append(s)

    # ---- tables of centre anomalies --------------------------------------------------------
    for tag in KINDS:
        P('\n=== centre anomaly (r < 0.5 R, minus outer ring), %s eddies;  mean +- track-clustered SE  [N eddies / tracks] ===' % KINDS[tag])
        P('%-26s' % 'variable' + ''.join('%-30s' % s for s in SOURCES))
        for v, (lab, unit, _) in VARS.items():
            row = '%-26s' % ('%s [%s]' % (lab, unit))
            for s in SOURCES:
                if (s, tag, v, 'centre') not in res:
                    row += '%-30s' % '-'; continue
                c = res[(s, tag, v, 'centre')]; g = res[(s, tag, 'meta')]['track']
                se, n, ncl = cluster_se(c, g)
                row += '%-30s' % ('%+.3f +- %.3f [%d/%d]' % (np.nanmean(c), se, n, ncl))
            P(row)
    # ---- paired differences on identical eddies --------------------------------------------
    for a, b in PAIRS[:2]:
        for tag in KINDS:
            m_a, m_b = res[(a, tag, 'meta')], res[(b, tag, 'meta')]
            common = np.intersect1d(m_a['obs'], m_b['obs'])
            ia, ib = np.searchsorted(m_a['obs'], common), np.searchsorted(m_b['obs'], common)
            P('\n=== PAIRED %s minus %s, %s eddies, %d identical eddies: centre anomaly difference ===' % (a, b, KINDS[tag], common.size))
            for v, (lab, unit, _) in VARS.items():
                if (a, tag, v, 'centre') not in res or (b, tag, v, 'centre') not in res:
                    continue
                d = res[(a, tag, v, 'centre')][ia] - res[(b, tag, v, 'centre')][ib]
                se, n, ncl = cluster_se(d, m_a['track'][ia])
                ca, cb = np.nanmean(res[(a, tag, v, 'centre')][ia]), np.nanmean(res[(b, tag, v, 'centre')][ib])
                sig = abs(np.nanmean(d)) > 2 * se
                P('  %-28s %s=%+.3f  %s=%+.3f  diff %+.3f +- %.3f %s' % ('%s [%s]' % (lab, unit), a[:9], ca, b[:9], cb, np.nanmean(d), se, '*' if sig else ''))
    # ---- unpaired coupled comparison ---------------------------------------------------------
    for tag in KINDS:
        P('\n=== UNPAIRED ng5_rcbmf minus ng5_deepoff (different eddies), %s: centre anomaly ===' % KINDS[tag])
        for v, (lab, unit, _) in VARS.items():
            if (('ng5_rcbmf', tag, v, 'centre') not in res) or (('ng5_deepoff', tag, v, 'centre') not in res):
                continue
            c1, c2 = res[('ng5_rcbmf', tag, v, 'centre')], res[('ng5_deepoff', tag, v, 'centre')]
            s1, _, _ = cluster_se(c1, res[('ng5_rcbmf', tag, 'meta')]['track'])
            s2, _, _ = cluster_se(c2, res[('ng5_deepoff', tag, 'meta')]['track'])
            d = np.nanmean(c1) - np.nanmean(c2); se = np.hypot(s1, s2)
            P('  %-28s rcbmf=%+.3f deepoff=%+.3f diff %+.3f +- %.3f %s' % ('%s [%s]' % (lab, unit), np.nanmean(c1), np.nanmean(c2), d, se, '*' if abs(d) > 2 * se else ''))
    # ---- eddy populations ----------------------------------------------------------------------
    P('\n=== eddy populations (%s box, every 10 days)' % REGION + ' ===')
    for tag in KINDS:
        for s in SOURCES:
            m = res.get((s, tag, 'meta'))
            if m is None: continue
            P('  %-15s %-5s N=%6d  tracks=%6d  radius km median %.0f (p10-p90 %.0f-%.0f)  mean |lat| %.1f' % (
                s, tag, m['obs'].size, len(np.unique(m['track'])), np.median(m['radius_km']), *np.percentile(m['radius_km'], [10, 90]), np.abs(m['lat']).mean()))
    open(OUT + 'report.txt', 'w').write('\n'.join(lines))

    # ---- figures -------------------------------------------------------------------------------
    def fig_maps(varlist, cols, fname, title, diff_pairs=()):
        nrow, ncol = len(varlist) * 2, len(cols)
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow), squeeze=False)
        for iv, v in enumerate(varlist):
            for it, tag in enumerate(KINDS):
                row = 2 * iv + it
                maps = []
                for col in cols:
                    if col[0] == 'src':
                        m = res.get((col[1], tag, v, 'amap', 'all'))
                    else:
                        _, a, b = col
                        ma = res.get((a, tag, v, 'amap', 'paired_%s_%s' % (a, b))) if (a, tag, v, 'amap', 'paired_%s_%s' % (a, b)) in res else res.get((a, tag, v, 'amap', 'all'))
                        mb = res.get((b, tag, v, 'amap', 'paired_%s_%s' % (a, b))) if (b, tag, v, 'amap', 'paired_%s_%s' % (a, b)) in res else res.get((b, tag, v, 'amap', 'all'))
                        m = None if ma is None or mb is None else ma - mb
                    maps.append(m)
                vmax = max([np.nanpercentile(np.abs(m), 99.5) for c, m in zip(cols, maps) if m is not None and c[0] == 'src'] or [1])
                dmax = max([np.nanpercentile(np.abs(m), 99.5) for c, m in zip(cols, maps) if m is not None and c[0] == 'diff'] or [1])
                for ic, (col, m) in enumerate(zip(cols, maps)):
                    ax = axs[row, ic]
                    if m is None:
                        ax.axis('off'); continue
                    lim = vmax if col[0] == 'src' else dmax
                    im = ax.imshow(m, origin='lower', extent=ext, cmap='RdBu_r', vmin=-lim, vmax=lim)
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_title('%s\n%s, %s [%s]' % (col[1] if col[0] == 'src' else '%s - %s' % (col[1][:11], col[2][:11]), VARS[v][0][:22], tag, VARS[v][1]), fontsize=7.5)
                    ax.set_aspect('equal'); ax.tick_params(labelsize=6)
        fig.suptitle(title, fontsize=10); fig.tight_layout()
        fig.savefig(OUT + fname, dpi=110); plt.close(fig)
    cols_amip = [('src', 'tco2559_rcbmf'), ('src', 'tco3999_amip'), ('diff', 'tco3999_amip', 'tco2559_rcbmf')]
    fig_maps(['avg_sst', 'mean10ws', 'tprate'], cols_amip, 'maps_amip_a.png', 'AMIP: eddy composites (anomaly vs outer ring), same AVISO eddies; right = tco3999 - tco2559')
    fig_maps(['mslhf', 'avg_blh', 'avg_tcwv'], cols_amip, 'maps_amip_b.png', 'AMIP (continued)')
    cols_cpl = [('src', 'ng5_rcbmf'), ('src', 'ng5_deepoff'), ('diff', 'ng5_rcbmf', 'ng5_deepoff')]
    fig_maps(['avg_sst', 'mean10ws', 'tprate'], cols_cpl, 'maps_coupled_a.png', 'Coupled: composites on own tracked eddies; right = rcbmf - deepoff (different eddies)')
    fig_maps(['mslhf', 'avg_blh', 'avg_tcwv'], cols_cpl, 'maps_coupled_b.png', 'Coupled (continued)')
    cols_x = [('src', 'tco3999_amip'), ('src', 'ng5_rcbmf'), ('src', 'ng5_deepoff')]
    fig_maps(['avg_sst', 'mean10ws', 'tprate'], cols_x, 'maps_amip_vs_coupled.png', 'tco3999 AMIP (AVISO eddies) vs coupled (own eddies)')
    print('figures in', OUT)


if __name__ == '__main__':
    {'compute': compute, 'report': report}[sys.argv[1]]()
