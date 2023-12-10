import argparse
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import os
from glob import glob
import pvlib

from utils.data import read_rds
from utils.logging import get_logger

logger = get_logger(name='data')


def pd2xr(x: pd.DataFrame, mach_id: str) -> xr.Dataset:
    x['date'] = pd.to_datetime(x.date)
    x = x.rename(columns={'date': 'time'})
    x = x.set_index('time')
    ds: xr.Dataset = x.to_xarray()

    for var in ds.data_vars:
       ds[var] = ds[var].astype('float32')

    return ds.expand_dims(station=[mach_id])


def add_static(ds: xr.Dataset) -> xr.Dataset:
    stat = read_rds(path='/data/william/data/RawFromMichael/obs/static/static_fields.rds')

    # Fix data issue:
    stat = read_rds(path='/data/william/data/RawFromMichael/obs/static/static_fields.rds')
    stat = stat[stat['field.y'] == 'abb'].rename(columns={'field.x': 'field'})

    # abb = soil topographic index
    # atb = hydraulic topographic index
    # btk = soil depth
    # dhm = digital heigt model
    # glm = glacier morphology
    # kwt = hydraulic conductivity
    # pfc = sotrage capacity
    # pus = landuse classified
    # slp = slope

    stat = stat.pivot(index='mach_ID', columns='field', values='mean').drop(columns=[
        'exp',
    ]).reset_index('mach_ID').rename(
        columns=dict(
            mach_ID='station'
        )
    ).set_index('station')

    stat = stat.to_xarray()
    stat['station'] = stat['station'].astype('str')

    for var in stat.data_vars:
        ds[var] = stat[var].astype('float32')

    # Add basin area.
    sf = gpd.read_file('/data/william/data/RawFromMichael/obs/shapefile/shape_tabsd.shp').set_index('mach_ID').to_xarray()
    ds['area'] = sf.Shape_Area.sel(mach_ID=ds.station)

    ds = ds.drop('mach_ID')

    return ds


def add_folds(ds: xr.Dataset) -> xr.Dataset:
    folds = pd.read_csv('/data/william/data/Toy3/Toy3_8foldCV.csv')
    ds['folds'] = xr.full_like(ds.abb, -1, dtype=int)

    for mach_id, fold in zip(folds.mach_id, folds.fold):
        ds['folds'].loc[{'station': mach_id}] = fold

    return ds


def add_clearsky_rad(ds : xr.Dataset) -> xr.Dataset:
    ds['Prad'] = xr.full_like(ds['P'], np.nan).compute()
    for station in tqdm(ds.station, ncols=100, desc='>>> Add Clearsky radiation'):
        height = ds.sel(station=station).dhm.item()
        loc = pvlib.location.Location(46.8182, 8.2275, altitude=height, tz='Etc/GMT+1')

        rad = loc.get_clearsky(
            times=pd.date_range(
                start=ds.indexes['time'][0],
                end=ds.indexes['time'][-1] + pd.Timedelta('1D'),
                freq='H',
                inclusive='left'
            )
        )
        rad = rad['ghi'].to_xarray()

        ds['Prad'].loc[{'station': station}] = rad.resample(index='D').mean().values

    return ds


def main(zarr_path: str, include_clearsky: bool):
    logger.info('Harmonizing PREVAH data')
    logger.info('-' * 79)

    # CHANGE PATH BACK!!
    # obs_basins = [
    #     os.path.basename(basin).split('.')[0] for basin in glob('/data/william/data/RawFromMichael/obs/prevah/*.rds')
    # ]
    obs_basins = [
        os.path.basename(basin).split('.')[0] for basin in glob('/data/william/data/RawFromMichael/obs/with_q/*.rds')
    ]
    do_basins = list(read_rds(
        path='/data/william/data/RawFromMichael/droughtch_operational_catchments/drought_IDs.rds'
    ).mach_ID.values)
    obs_basins_is_do = [basin in do_basins for basin in obs_basins]

    logger.info(f'PREVAH ({len(obs_basins)}) and drought operational ({len(do_basins)}) basin lists loaded')

    station_list = []

    for mach_id in tqdm(obs_basins, ncols=100, desc='>>> Load stations'):

        prevah = pd2xr(
            x=read_rds(
                path=f'/data/william/data/RawFromMichael/obs/with_q/{mach_id}.rds'
            ).drop(columns=['Qm3s', 'Qls', 'Qmm', 'source']),
            mach_id=mach_id
        )

        additional_vars = []
        for var in ['Tmin', 'Tmax', 'Srel', 'direct']:
            path = f'/data/william/data/RawFromMichael/obs/additional_meteoswiss/{var}/binary/{mach_id}.rds'
            additional_vars.append(pd2xr(x=read_rds(path=path), mach_id=mach_id))
        additional_vars = xr.merge(additional_vars)
        additional_vars = additional_vars.rename({'direct': 'Drad'})

        prevah = xr.merge((prevah, additional_vars))

        prevah_sim = pd2xr(
            x=read_rds(
                path=f'/data/william/data/RawFromMichael/obs/prevah/{mach_id}.rds'
            ),
            mach_id=mach_id
        )

        prevah['Qmm_prevah'] = prevah_sim.Q

        if mach_id in do_basins:
            obs = pd2xr(
                x=read_rds(
                    path=f'/data/william/data/RawFromMichael/obs/with_q/{mach_id}.rds'
                ).drop(columns=['source', 'Qm3s']),
                mach_id=mach_id
            )
            obs.station.attrs = {'note': 'station ID corresponds to mach_ID'}

            prevah = xr.merge((prevah, obs))

        station_list.append(prevah)

    logger.info(f'PREVAH featurtes, target, and simulations loaded.')

    ds = xr.concat(station_list, dim='station')
    ds = ds.assign_coords(do_subset=('station', obs_basins_is_do))

    ds['station'] = ds.station.astype(str)

    encoding = {}
    for var in ds.data_vars:
        encoding.update(
            {
                var: {
                    'chunks': (1, -1)
                }
            }
        )

    ds = add_static(ds)

    ds = add_folds(ds)

    if include_clearsky:
        ds = add_clearsky_rad(ds)

    ds.to_zarr(zarr_path, mode='w', encoding=encoding)

    logger.info(f'PREVAH data harmonized and saved to \'{zarr_path}\'.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', type=str, default='/data/basil/harmonized_basins.zarr')
    parser.add_argument('--include_clearsky', action='store_true')
    args = parser.parse_args()

    main(zarr_path=args.out_path, include_clearsky=args.include_clearsky)
