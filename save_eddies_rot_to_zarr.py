# Save eddies to zarr

import xarray as xr
import numpy as np
xr.set_options(keep_attrs=True)
import pandas as pd
import zarr
import os
from glob import glob
from datetime import datetime
import time as T


#EXP = 'GPM-IMERG'
#EXP = 'surface_wind'
#EXP = 'ESA-CCI-v3'
#EXP = 'tco1279-eerie_production_202407'
#EXP = 'tco1279-eerie_production_202408-c_0_a_LR20'
# EXP = 'tco399-eerie_production_202407'
MODEL = 'HadGEM3-GC5-EERIE-N640'
EXP   = 'highresSST-SmoothAnom'
# EXP   = 'highresSST-present'
# kind  = 'anticyclonic'
kind = 'cyclonic'
realization = 1
print(EXP,kind)

T.sleep(3)
# print('USING HACKY VERSION: DO NOT MOVE ALREADY EXISTING EDDIES, CODE IS COMMENTED OUT!!!') 
# print('THIS IS INTENDED ONLY WHILE COMPOSITING IS STILL RUNNING! IS THIS WHAT YOU WANT?')
# T.sleep(8)


if kind == 'anticyclonic':
    path = '/gws/nopw/j04/eerie/aengenh/output/{MODEL}/{EXP}/processed/composites_rot_acyc/'.format(MODEL=MODEL,EXP=EXP)
elif kind == 'cyclonic':
    path = '/gws/nopw/j04/eerie/aengenh/output/{MODEL}/{EXP}/processed/composites_rot_cyc/'.format(MODEL=MODEL,EXP=EXP)
else:
    raise ValueError('kind %s is not defined' % kind)

fnames = glob('%s*.nc' % path)

df = pd.DataFrame(fnames,columns=['fname'])
df['obs'] = df['fname'].str.split('_').str[-2]
df['varname'] = df['fname'].str.split('_').str[-1].str.split('.').str[0]
df['realization'] = df['fname'].str.split('/').str[-1].str.split('_').str[2]
varnames = df['varname'].unique().tolist()
Nvar = len(varnames)
df_filtered = df.groupby('obs').filter(lambda x: len(x) == Nvar)#.sortby('obs')
df_filtered['obs'] = df_filtered['obs'].astype('int')

# check that data is complete
df_no = df.groupby('obs').filter(lambda x: len(x) != Nvar)
assert len(df_no)==0

fnames = df_filtered['fname'].values.tolist()
obs = sorted(df_filtered['obs'].unique().tolist())

N = len(obs)
print('Looking over %i eddy files' % N)

if realization == 1:
    zarr_path = path[:-1] + '.zarr'
    print('Save to %s' % (zarr_path))
else:
    zarr_path = path[:-1] + '_r_%i' % realization + '.zarr'
    print('Save realization %i to %s' % (realization,zarr_path))

# TODO: 2025-05-28: GOOD UNTIL HERE, CONTINUE below!

if os.path.exists(zarr_path):
    print('Store already exists. Scan for existing observations within, and skip files accordingly')
    # obs_fnames = [int(f.split('/')[-1].split('.')[0].split('_')[-1]) for f in fnames]
    # obs_fnames = obs
    ds0 = xr.open_zarr(zarr_path)
    obs_exist = ds0['obs'].load().values.tolist()
    print('%i obs already exist in store' % len(obs_exist))

    obs_new   = [o for o in obs if not o in obs_exist]
    obs_saved = [o for o in obs if     o in obs_exist]

    # df_fnames = pd.Series(index=obs_fnames,data=fnames)
    # df_fnames = df_filtered[df_filtered['obs'].isin(obs)]

    # fnames_new = df_fnames[obs_new].values
    fnames_saved = df_filtered[df_filtered['obs'].isin(obs_saved)]['fname'].values
    # N_new = len(fnames_new)
    N_new = len(obs_new)

    if len(fnames_saved) > 0:
        print('Move %i already saved files' % len(fnames_saved))
        for f in fnames_saved:
            fparts = f.split('/')
            saved_dir = '/'.join(fparts[:-1]) + '/saved/'
            if not os.path.exists(saved_dir):
                os.system('mkdir -p %s' % saved_dir)
            command = 'mv %s %s' % (f,saved_dir)
            s = os.system(command)
            assert s == 0

    print('Process %i new eddies' % len(obs_new))

    N = N_new
    # fnames = fnames_new
    obs = obs_new
    df_filtered = df_filtered[df_filtered['obs'].isin(obs_new)]

    ds0.close()
else:
    print('Store does not exist, process all files')


# ==========================================
# Merge .nc to .zarr

block_size = 1000 # chunk length in obs

if len(obs) < block_size:
    raise ValueError('Need at least %i files to process, have only %i' % (block_size, len(fnames)))

n_blocks = len(obs) // block_size

# n_blocks = 5

for ni in range(n_blocks):
    t0 = T.time()
    print(datetime.today().strftime('%Y-%m-%d %H:%M:%S') + ' : ' + 'Processing block %i / %i' % (ni+1, n_blocks))
    # fnames_block = fnames[ni * block_size : (ni + 1) * block_size]
    obs_block = obs[ni * block_size : (ni + 1) * block_size]
    df_block = df_filtered[df_filtered['obs'].isin(obs_block)]
    fnames_block = df_block['fname'].values.tolist()
    try:
        # ds = xr.open_mfdataset(fnames_block,combine='nested',concat_dim='obs')
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


    # Make sure obs don't already exist - check for naming issues
    if os.path.exists(zarr_path):
        ds0 = xr.open_zarr(zarr_path)
        obs_exist = ds0['obs'].load().values.tolist()
        obs_new = ds['obs'].load().values.tolist()

        obs_add    = [d for d in obs_new if not d in obs_exist]
        obs_double = [d for d in obs_new if d in obs_exist]
        assert len(obs_add) == block_size, 'Some to-be-added obs already exist in zarr. Something went wrong, this should have been caught! Check manually. Duplicates: %s' % obs_double
        ds0.close()

    print('Obs range: ' + str(ds.obs.values[[0,-1]]))

    coordnames = [d for d in ds.coords]
    varnames = [d for d in ds if not d in coordnames]

    compressor = zarr.Blosc(cname="zstd", clevel=7, shuffle=2)

    ds = ds.chunk({'obs':ds.obs.size,'x':ds.x.size,'y':ds.y.size})
    for varname in varnames:
        ds[varname].encoding.update({"compressor":compressor})
    for varname in ['x','y']:
        ds[varname].encoding.update({"chunks": (-1), "compressor":compressor})

    if not os.path.exists(zarr_path):
        print('create store')
        ds.to_zarr(
            zarr_path
        )
    else:
        print('append to store')
        ds.to_zarr(
            zarr_path ,
            append_dim='obs',
        )

    # Move saved files to other directory
    for f in fnames_block:
        fparts = f.split('/')
        saved_dir = '/'.join(fparts[:-1]) + '/saved/'
        if not os.path.exists(saved_dir):
            os.system('mkdir -p %s' % saved_dir)
        command = 'mv %s %s' % (f,saved_dir)
        s = os.system(command)
        assert s == 0
    tf = T.time()
    print('Processing block took %.2f seconds' % (tf - t0))

print('Done with EXP = %s' % EXP)
print('DONE')
