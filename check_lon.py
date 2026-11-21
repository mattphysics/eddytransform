import numpy as np, xarray as xr, sys
sys.path.insert(0,'.')
import eddytransform_run_levante_sim as d
for kind in ['Cyclonic','Anticyclonic']:
    tr = xr.open_dataset(d.AVISO % kind)
    t = tr['time'].values.astype('datetime64[D]'); lat = tr['latitude'].values; lon = tr['longitude'].values
    dates = d.date_list('tco2559_rcbmf','10D').values.astype('datetime64[D]')
    m = np.isin(t, dates) & (lat > -60) & (lat < -25)
    out = m & ((lon <= 0) | (lon >= 360))
    print(kind, 'SO obs on the 43 dates:', int(m.sum()), ' lon outside (0,360):', int(out.sum()), ' lon range overall %.2f..%.2f' % (lon.min(), lon.max()), ' lon<0: %d lon>360: %d' % ((m & (lon<0)).sum(), (m & (lon>360)).sum()))
