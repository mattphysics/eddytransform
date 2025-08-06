# Fix zarr stores where saving failed

import xarray as xr
import numpy as np
xr.set_options(keep_attrs=True)
import pandas as pd
import zarr
import os, sys
from glob import glob
from datetime import datetime
import time as T
from dask.diagnostics import ProgressBar

def find_failed_obs(ds):
    obs_failed = []
    for vname in list(ds0.data_vars):
        # if vname in ['avg_sst','avg_ci']:
        #     print('not using sst or sea ice')
        #     continue
        if 'x' in ds0[vname].dims:
            dai = ds0[vname].max(['x','y'])
        else:
            dai = ds0[vname]
        with ProgressBar():
            dai.load()
        print(vname,np.isnan(dai).sum().values)
        obs_failed_i = dai.sel(obs=np.isnan(dai)).obs.values
        for obsi in obs_failed_i:
            if not obsi in obs_failed:
                obs_failed.append(obsi)
    return obs_failed

#EXP = 'GPM-IMERG'
#EXP = 'surface_wind'
#EXP = 'ESA-CCI-v3'
# EXP = 'tco1279-eerie_production_202407'
# EXP = 'tco1279-eerie_production_202408-c_0_a_LR20'
EXP = 'tco399-eerie_production_202407'
# MODEL = 'HadGEM3-GC5-EERIE-N640'
# EXP   = 'highresSST-SmoothAnom'
# EXP   = 'highresSST-present'
# kind  = 'anticyclonic'
kind = 'cyclonic'
realization = 1

# Already moved failed obs to saved? 
saved = False
# saved = True

print(EXP,kind,realization,saved)


block_size = 1000 # chunk length in obs

print('Fix saved composites for (EXP, kind, realization) = (%s, %s, %s)' % (EXP, kind, realization))
print('Saved: %s, block_size: %i' % (saved, block_size))

if kind == 'anticyclonic':
    # path = '/ec/fws5/lb/project/eerie/output/{EXP}/processed/composites_rot_acyc/'.format(EXP=EXP)
    path = '/ec/fws5/lb/project/eerie/output/{EXP}/processed/composites_GS_rot_acyc/'.format(EXP=EXP)
elif kind == 'cyclonic':
    # path = '/ec/fws5/lb/project/eerie/output/{EXP}/processed/composites_rot_cyc/'.format(EXP=EXP)
    path = '/ec/fws5/lb/project/eerie/output/{EXP}/processed/composites_GS_rot_cyc/'.format(EXP=EXP)
else:
    raise ValueError('kind %s is not defined' % kind)

if realization == 1:
    zarr_path = path[:-1] + '.zarr'
    print('Using zarr: %s' % (zarr_path))
else:
    zarr_path = path[:-1] + '_r_%i' % realization + '.zarr'
    print('Using zarr: %s' % (zarr_path))

assert os.path.exists(zarr_path)

ds0 = xr.open_zarr(zarr_path)
obs_failed = find_failed_obs(ds0)
# obs_failed = []
# for vname in list(ds0.data_vars):
#     # if vname in ['avg_sst','avg_ci']:
#     #     print('not using sst or sea ice')
#     #     continue
#     if 'x' in ds0[vname].dims:
#         dai = ds0[vname].max(['x','y'])
#     else:
#         dai = ds0[vname]
#     with ProgressBar():
#         dai.load()
#     print(vname,np.isnan(dai).sum().values)
#     obs_failed_i = dai.sel(obs=np.isnan(dai)).obs.values
#     for obsi in obs_failed_i:
#         if not obsi in obs_failed:
#             obs_failed.append(obsi)

if len(obs_failed) == 0:
    print('Found no broken obs! Do not execute further')
    sys.exit()

obs_block = [int(i) for i in obs_failed]

assert len(obs_block) == block_size
assert np.all(ds0['obs'].isel(obs=slice(-block_size,None)).values == obs_failed)
isel = dict(
    obs = slice(len(ds0['obs']) - block_size, len(ds0['obs']))
)
assert np.all(ds0['obs'].isel(isel).values == obs_failed)


if saved:
    fnames = glob('%ssaved/*.nc' % path)
else:
    fnames = glob('%s*.nc' % path)

df = pd.DataFrame(fnames,columns=['fname'])
# df['obs'] = df['fname'].str.split('_').str[-2]
df['obs'] = df['fname'].str.split('/').str[-1].str.split('_').str[3]
df['varname'] = df['fname'].str.split('_').str[-1].str.split('.').str[0]
df['realization'] = df['fname'].str.split('/').str[-1].str.split('_').str[2]
varnames = df['varname'].unique().tolist()
Nvar = len(varnames)
df_filtered = df.groupby('obs').filter(lambda x: len(x) == Nvar)#.sortby('obs')
df_filtered['obs'] = df_filtered['obs'].astype('int')

# check that data is complete
df_no = df.groupby('obs').filter(lambda x: len(x) != Nvar)
assert len(df_no)==0

# fnames = df_filtered['fname'].values.tolist()
# obs = sorted(df_filtered['obs'].unique().tolist())

# N = len(df_filtered)
# print('Looking over %i eddy files over %i variables' % (N,Nvar))

# T.sleep(3)


# ==========================================
# Merge .nc to .zarr

t0 = T.time()
df_block = df_filtered[df_filtered['obs'].isin(obs_block)]
fnames_block = df_block['fname'].values.tolist()
assert len(fnames_block) == (Nvar * len(obs_block))

try:
    ds = []
    for obs_id, group in df_block.groupby('obs'):
        files = group['fname'].tolist()
        dsi = xr.open_mfdataset(files, combine='by_coords')
        dsi = dsi.expand_dims(obs=[obs_id])  # Add obs dimension
        ds.append(dsi)
    
    # Concatenate all datasets along 'obs'
    ds = xr.concat(ds, dim='obs')
except Exception as e:
    print('Opening failed. Diagnosing...')
    for f in fnames_block:
        try:
            ds = xr.open_dataset(f)
            if not 'x' in ds:
                print('not x in: %s' % f)
        except:
            print('cannot open: %s' % f)
    raise type(e)(f"Error opening dataset: {e}")


print('Obs range: ' + str(ds.obs.values[[0,-1]]))

coordnames = [d for d in ds.coords]
varnames = [d for d in ds if not d in coordnames]

compressor = zarr.Blosc(cname="zstd", clevel=7, shuffle=2)

ds = ds.chunk({'obs':ds.obs.size,'x':ds.x.size,'y':ds.y.size})
for varname in varnames:
    ds[varname].encoding.update({"compressor":compressor})
for varname in ['x','y']:
    ds[varname].encoding.update({"chunks": (-1), "compressor":compressor})

print('append to store')
ds.drop_vars(['x','y']).to_zarr(
    zarr_path ,
    region=isel
)

# Move saved files to other directory
if not saved:
    for f in fnames_block:
        fparts = f.split('/')
        # saved_dir = '/'.join(fparts[:-1]) + '/saved_2000-2009/'
        saved_dir = '/'.join(fparts[:-1]) + '/saved/'
        if not os.path.exists(saved_dir):
            os.system('mkdir -p %s' % saved_dir)
        command = 'mv %s %s' % (f,saved_dir)
        s = os.system(command)
        assert s == 0
tf = T.time()
print('Processing block took %.2f seconds' % (tf - t0))

print('Done with EXP = %s, kind = %s' % (EXP,kind))
print('DONE')
