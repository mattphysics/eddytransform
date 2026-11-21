import sys, numpy as np, xarray as xr
for run in ['ng5_rcbmf','ng5_deepoff']:
    for k in ['cyclonic','anticyclonic']:
        ds = xr.open_dataset(f'/work/bk1377/b382473/eddytracks/{run}/0.25deg/eddyTracks/{k}_tracks.nc')
        lon, lat, t, r = ds.longitude.values, ds.latitude.values, ds.time.values, ds.effective_radius.values
        so = (lat>-60)&(lat<-25)
        d = t.astype('datetime64[D]'); nd = len(np.unique(d))
        print(f'{run:12s} {k:13s} obs {lon.size}  lon {lon.min():.2f}..{lon.max():.2f}  lat {lat.min():.1f}..{lat.max():.1f}  time {str(t.min())[:16]}..{str(t.max())[:16]} ({nd} days)  SO obs/day {so.sum()/nd:.0f}  radius km median {np.median(r)/1e3:.0f} p5-p95 {np.percentile(r,5)/1e3:.0f}-{np.percentile(r,95)/1e3:.0f}  all-day: {nd}', flush=True)
