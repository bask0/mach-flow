"""Data structure

base_dir = '/data/william/data/RawFromMichael/'
|
+-- obs: observational catchments
    |
    +-- with_q: dynamic features and observed Q
        |
        +-- <mach_ID>.rds: date, T, P, E, Qm3s, Qls, Qmm
    |
    +-- prevah: prevah simulated Q (mm)
        |
        +-- <mach_ID>.rds: date, Q
    |
    +-- static: various
        |
        +-- static_fields.rds: mach_ID, field, mean
        |
        +-- pus_field.rds: ?

+-- prevah_307: observational catchments
    |
    +-- binary: prevah catchments
        |
        +-- <HSU_ID>.rds: date, T, P, E
    |
    +-- static: various
        |
        +-- static_fields.rds: mach_ID, field, mean
        |
        +-- pus_field.rds: ?

"""

import argparse
import xarray as xr
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np
from sklearn.manifold import Isomap
import warnings

from utils.data import read_rds
from utils.logging import get_logger
from utils.data import convert_q

logger = get_logger(name='data')


def pd2xr(x: pd.DataFrame, basin_id: str) -> xr.Dataset:
   x['date'] = pd.to_datetime(x.date)
   x = x.rename(columns={'date': 'time'})
   x = x.set_index('time')
   ds: xr.Dataset = x.to_xarray()

   for var in ds.data_vars:
      with warnings.catch_warnings():
         warnings.filterwarnings('ignore', 'invalid value encountered in cast')
         ds[var] = ds[var].astype('float32')

   return ds.expand_dims(station=[basin_id])


def add_folds(ds: xr.Dataset, fold_file: str | os.PathLike) -> xr.Dataset:
    folds = pd.read_csv(fold_file)

    for mach_id, fold in zip(folds.mach_id, folds.fold):
        ds['folds'].loc[{'station': mach_id}] = fold

    return ds


def merge_timeseries(a: pd.DataFrame, b: pd.DataFrame, merge_columns: list[str]) -> pd.DataFrame:
    a = pd.merge_ordered(a, b, on='date')

    for column in merge_columns:
        a[column] = np.mean(a[[f'{column}_x', f'{column}_y']], axis=1)
        a = a.drop(columns=[f'{column}_x', f'{column}_y'])

    return a


def main(zarr_path: str):

    logger.info('Harmonizing PREVAH data.')
    logger.info('-' * 79)

    base_dir = Path('/data/william/data/RawFromMichael')

    obs_dir = base_dir / 'obs'
    ch_dir = base_dir / 'prevah_307'

    obs_data_dir = obs_dir / 'with_q'
    obs_data_add_dir = obs_dir / 'recent_P_T_E'
    obs_prevah_data_dir = obs_dir / 'prevah'
    obs_prevah_data_80s_dir = obs_dir / 'prevah_80ies'
    ch_data_dir = ch_dir / 'binary'
    ch_data_add_dir = ch_dir / 'recent_P_T_E'
    ch_prevah_data_dir = ch_dir / 'prevah'
    ch_prevah_data_80s_dir = ch_dir / 'prevah_80ies'

    obs_stat_dir = obs_dir / 'static'
    ch_stat_dir = ch_dir / 'static'

    for path in [
            obs_data_dir,
            obs_data_add_dir,
            obs_prevah_data_dir,
            obs_prevah_data_80s_dir,
            obs_stat_dir,
            ch_data_dir,
            ch_data_add_dir,
            ch_prevah_data_dir,
            ch_prevah_data_80s_dir,
            ch_stat_dir]:
        if not path.is_dir():
            raise FileNotFoundError(f'dir does not exist: {path}')

    obs_data_files = list(obs_data_dir.glob('*.rds'))
    obs_prevah_data_files = list(obs_prevah_data_dir.glob('*.rds'))
    obs_prevah_data_80s_files = list(obs_prevah_data_80s_dir.glob('*.rds'))
    ch_data_files = list(ch_data_dir.glob('*.rds'))
    ch_prevah_data_files = list(ch_prevah_data_dir.glob('*.rds'))
    ch_prevah_data_80s_files = list(ch_prevah_data_80s_dir.glob('*.rds'))

    for path_list, search_dir in zip(
        [obs_data_files, obs_prevah_data_files, obs_prevah_data_80s_files,
         ch_data_files, ch_prevah_data_files, ch_prevah_data_80s_files],
        [obs_data_dir, obs_prevah_data_dir, obs_prevah_data_80s_dir,
         ch_data_dir, ch_prevah_data_dir, ch_prevah_data_80s_files]):
        if len(path_list) < 1:
            raise FileNotFoundError(f'no files found with pattern \'*.rds\' in {search_dir}')

    obs_stat_file = Path('/data/basil/static_harmonized/obs/harmonized.rds')
    obs_stat_shapefile = obs_dir / 'shapefile/shape_tabsd.shp'
    ch_stat_file = Path('/data/basil/static_harmonized/prevah_307/harmonized.rds')
    ch_stat_shapefile = ch_dir / 'shapefile/Regions_307.shp'
    fold_cv_file = Path('/data/william/data/Toy3/Toy3_8foldCV_with_glacier.csv')

    for path in [obs_stat_file, obs_stat_shapefile, ch_stat_file, ch_stat_shapefile, fold_cv_file]:
        if not path.is_file():
            raise FileNotFoundError(f'file does not exist: {path}')

    logger.info('Loading observational basin data.')

    obs_data_list = []

    for obs_basin_file in tqdm(obs_data_files, ncols=100, desc=f'{">>> Load obs data":<21}'):
        basin = obs_basin_file.stem
        prevah_basin = basin
        prevah_basin_file = [
            prevah_data_file for prevah_data_file in obs_prevah_data_files if prevah_basin in prevah_data_file.stem
        ]
        if (n := len(prevah_basin_file)) != 1:
            raise RuntimeError(
                f'found {n} prevah files for basin {basin}, expected 1.'
            )
        prevah_basin_file = prevah_basin_file[0]

        prevah_basin_80s_file = [
            prevah_data_file for prevah_data_file in obs_prevah_data_80s_files if prevah_basin in prevah_data_file.stem
        ]
        if (n := len(prevah_basin_80s_file)) != 1:
            raise RuntimeError(
                f'found {n} prevah 80s files for basin {basin}, expected 1.'
            )
        prevah_basin_80s_file = prevah_basin_80s_file[0]

        additional_obs_file = obs_data_add_dir / f'{basin}.rds'
        if not additional_obs_file.exists():
            raise FileNotFoundError(
                f'additional meteo file not found: {additional_obs_file}.'
            )

        obs_data = read_rds(obs_basin_file)
        obs_add_data = read_rds(additional_obs_file)
        obs_data = merge_timeseries(obs_data, obs_add_data, merge_columns=['P', 'T', 'E'])

        prevah_data = read_rds(prevah_basin_file)
        prevah_80s_data = read_rds(prevah_basin_80s_file)

        prevah_data = merge_timeseries(prevah_data, prevah_80s_data, merge_columns=['Q'])

        # Join observations and prevah simulations.
        merged_data = pd.merge(obs_data, prevah_data, how='left', on='date')
        # Rename prevah simulations from Q to Qmm_prevah.
        merged_data = merged_data.rename(columns={'Q': 'Qmm_prevah'})
        # Drop source field.
        merged_data = merged_data.drop(columns=['source'])
        # Convert pandas Dataframe to xarray Dataset.
        merged_data = pd2xr(x=merged_data, basin_id=basin)

        obs_data_list.append(merged_data)

    obs_data = xr.concat(obs_data_list, dim='station')
    obs_data['station'] = obs_data.station.astype(str)
    obs_data['folds'] = xr.full_like(obs_data['T'].isel(time=0).drop('time'), -1, dtype=int)

    logger.info('Loading ch307 basin data.')

    ch_data_list = []

    for ch_basin_file in tqdm(ch_data_files, ncols=100, desc=f'{">>> Load ch307 data":<21}'):
        basin = ch_basin_file.stem
        basin_nr = int(basin.split('_')[1])
        prevah_basin_file = [
            prevah_data_file for prevah_data_file in ch_prevah_data_files if f'{basin_nr}.rds' == prevah_data_file.name
        ]
        if (n := len(prevah_basin_file)) != 1:
            raise RuntimeError(
                f'found {n} prevah files for basin {basin}, expected 1.'
            )
        prevah_basin_file = prevah_basin_file[0]

        prevah_basin_80s_file = [
            prevah_data_file for prevah_data_file in ch_prevah_data_80s_files if  f'{basin_nr}.rds' == prevah_data_file.name
        ]
        if (n := len(prevah_basin_80s_file)) != 1:
            raise RuntimeError(
                f'found {n} prevah 80s files for basin {basin}, expected 1.'
            )
        prevah_basin_80s_file = prevah_basin_80s_file[0]

        additional_obs_file = ch_data_add_dir / f'{basin_nr}.rds'
        if not additional_obs_file.exists():
            raise FileNotFoundError(
                f'additional meteo file not found: {additional_obs_file}.'
            )

        ch_data = read_rds(ch_basin_file)
        ch_add_data = read_rds(additional_obs_file)
        ch_data = merge_timeseries(ch_data, ch_add_data, merge_columns=['P', 'T', 'E'])

        prevah_data = read_rds(prevah_basin_file)
        prevah_80s_data = read_rds(prevah_basin_80s_file)

        prevah_data = merge_timeseries(prevah_data, prevah_80s_data, merge_columns=['Q'])

        # Join observations and prevah simulations.
        merged_data = pd.merge(ch_data, prevah_data, how='left', on='date')
        # Rename prevah simulations from Q to Qmm_prevah.
        merged_data = merged_data.rename(columns={'Q': 'Qmm_prevah'})
        # Convert pandas Dataframe to xarray Dataset.
        merged_data = pd2xr(x=merged_data, basin_id=f'HSU_{basin_nr:03d}')

        ch_data_list.append(merged_data)

    ch_data = xr.concat(ch_data_list, dim='station')
    ch_data['station'] = ch_data.station.astype(str)
    ch_data['folds'] = xr.full_like(ch_data['T'].isel(time=0).drop('time'), -2, dtype=int)

    logger.info('Combining observational and PREVAH basin data.')

    combined = xr.merge((obs_data, ch_data)).drop('Qls')

    combined = add_folds(ds=combined, fold_file=fold_cv_file)

    logger.info('Add static variables.')

    combined = add_static(
        ds=combined,
        obs_static_file=obs_stat_file,
        ch_static_file=ch_stat_file,
        obs_shapefile=obs_stat_shapefile,
        ch_shapefile=ch_stat_shapefile)

    combined['Qm3s_prevah'] = convert_q(combined['Qmm_prevah'], combined['area'], from_unit='mm', to_unit='m3s')

    encoding = {}
    for var in combined.data_vars:
        encoding.update(
            {
                var: {
                    'chunks': (1, -1)
                }
            }
        )

    combined['folds'].attrs= {
        'legend': 'fold=-2: PREVAH basins, no observations | fold>-2: Observational basins; ' + \
            '0 to n are cross validation folds for drought operational catchments, -1 is others.'
    }

    logger.info('Saving data.')

    combined.to_zarr(zarr_path, mode='w', encoding=encoding)

    logger.info(f'Data harmonized and saved to \'{zarr_path}\'.')


def add_shape_area(ds: xr.Dataset, shapefile_path: str | os.PathLike, object_id: str) -> xr.Dataset:
    # Add basin area.
    sf = gpd.read_file(shapefile_path).set_index(object_id).to_xarray()
    sf = sf.rename({object_id: 'station'})
    if object_id == 'OBJECTID':
        sf['station'] = [f'HSU_{oid:03d}' for oid in sf['station']]

    if 'area' not in ds.data_vars:
        ds['area'] = ds['folds'].copy().compute()

    common_stations = np.intersect1d(sf.station, ds.station)

    for station in common_stations:
        ds['area'].loc[{'station': station}] = sf.sel(station=station).Shape_Area

    return ds


def add_static(
        ds: xr.Dataset,
        obs_static_file: str | os.PathLike,
        ch_static_file: str | os.PathLike,
        obs_shapefile: str | os.PathLike,
        ch_shapefile: str | os.PathLike) -> xr.Dataset:
    ds0 = read_rds(obs_static_file)
    ds1 = read_rds(ch_static_file)
    stat = pd.concat((ds0, ds1)).drop(columns=['exp'])

    stat_vars = [stat_var for stat_var in stat.columns if stat_var not in ['OBJECTID', 'mach_ID']]

    for var in stat_vars:
        ds[var] = xr.full_like(ds.folds, np.nan, dtype='float32')

    for station in tqdm(ds.station.values, total=len(ds.station), desc=f'{">>> Adding stat vars":<21}', ncols=100):
        if station.startswith('HSU'):
            stat_values = stat.loc[stat.OBJECTID==station.split('_')[1].lstrip('0')]
        else:
            stat_values = stat.loc[stat.mach_ID==station]

        if (n := len(stat_values)) != 1:
            raise ValueError(
                f'expected one row to match station \'{station}\', got {n}'
            )

        for var in stat_vars:
            ds[var].loc[{'station': station}] = stat_values[var].item()

    isomap = Isomap(n_neighbors=12, n_components=5)
    stat_arr = ds[stat_vars].to_array().values.T
    stat_arr_norm = (stat_arr - stat_arr.mean(0, keepdims=True)) / stat_arr.std(0, keepdims=True)
    stat_red = isomap.fit_transform(stat_arr_norm)

    for component in range(stat_red.shape[1]):
        var = f'c{component}'
        da = ds.folds.copy().compute()
        da.values[:] = stat_red[:, component]
        ds[var] = da

    ds = add_shape_area(ds=ds, shapefile_path=obs_shapefile, object_id='mach_ID')
    ds = add_shape_area(ds=ds, shapefile_path=ch_shapefile, object_id='OBJECTID')

    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', type=str, default='/data/basil/harmonized_basins.zarr')
    args = parser.parse_args()

    main(zarr_path=args.out_path)
