import sys, numpy as np, pandas as pd, xarray as xr
sys.path.insert(0,'/home/b/b382473/code/eddytransform')
import eddytransform_run_levante_sim as d
ref=np.datetime64('1900-01-01','D')
for kind in ['Cyclonic','Anticyclonic']:
    tr=xr.open_dataset(d.AVISO % kind)
    t=tr['time'].values.astype('datetime64[D]'); lat=tr['latitude'].values; lon=tr['longitude'].values
    so=(lat>-60)&(lat<-25)
    for src in ['tco2559_deepon','tco2559_rcbmf']:
        for fr in ['10D','5D','2D']:
            dates=d.date_list(src,fr).values.astype('datetime64[D]')
            m=np.isin(t,dates)
            print(kind,src,fr,'dates',len(dates),'SO eddies',int((m&so).sum()),'global',int(m.sum()),flush=True)
