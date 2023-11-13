import xarray as xr
import pandas as pd
from tqdm import tqdm
import os
from glob import glob

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


def main(zarr_path: str = '/data/basil/harmonized_basins.zarr'):
    logger.info('Harmonizing PREVAH data')
    logger.info('-' * 79)

    prevah_basins = [
        os.path.basename(basin).split('.')[0] for basin in glob('/data/william/data/RawFromMichael/obs/prevah/*.rds')
    ]
    do_basins = list(read_rds(
        path='/data/william/data/RawFromMichael/droughtch_operational_catchments/drought_IDs.rds'
    ).mach_ID.values)
    prevah_basins_is_do = [basin in do_basins for basin in prevah_basins]

    logger.info(f'PREVAH ({len(prevah_basins)}) and drought operational ({len(do_basins)}) basin lists loaded')

    station_list = []

    for mach_id in tqdm(prevah_basins, ncols=100, desc='>>> Load stations'):

        prevah = pd2xr(
            x=read_rds(
                path=f'/data/william/data/RawFromMichael/obs/with_q/{mach_id}.rds'
            ).drop(columns=['Qm3s', 'Qls', 'Qmm', 'source']),
            mach_id=mach_id
        )

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
    ds = ds.assign_coords(do_subset=('station', prevah_basins_is_do))

    encoding = {}
    for var in ds.data_vars:
        encoding.update(
            {
                var: {
                    'chunks': (1, -1)
                }
            }
        )

    ds.to_zarr(zarr_path, mode='w', encoding=encoding)

    logger.info(f'PREVAH data harmonized and saved to \'{zarr_path}\'.')

if __name__ == '__main__':
    main()
