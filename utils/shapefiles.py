
import xarray as xr
import geopandas as gpd
from pathlib import Path
import os


def get_shapefile(
        source: str,
        datapath: str | os.PathLike = '/mydata/machflow/basil/data/'
        ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    datapath = Path(datapath)

    if source == 'obs':
        obs = gpd.read_file(datapath / 'obs/shapefile/shape_tabsd.shp')
        obs_point = gpd.read_file(datapath / 'obs/shapefile/stations_tabsd.shp')
        id_field = 'mach_ID'
    elif source == 'prevah':
        obs = gpd.read_file(datapath / 'prevah/shapefile/Regions_307.shp')
        obs_point = obs
        id_field = 'OBJECTID'
        obs[id_field] = [f'HSU_{int(id):03d}' for id in obs[id_field].astype(str)]
    else:
        raise ValueError('`source` must be on fo \'obs\', \'prevah\', is \'{source}\'.')

    obs = obs.rename(columns={id_field: 'OID'}).set_index('OID')
    obs_point = obs_point.rename(columns={id_field: 'OID'}).set_index('OID')
    obs_point = obs_point.reindex_like(obs)
    obs = obs.reset_index()
    obs_point = obs_point.reset_index()

    ds = xr.open_zarr(datapath / 'harmonized_basins.zarr/')
    common_ids = [oid for oid in obs['OID'].values if oid in ds.station.values]

    obs = obs[obs['OID'].isin(common_ids)]
    obs_point = obs_point[obs_point['OID'].isin(common_ids)]
    ds = ds.sel(station=common_ids)

    obs['is_train'] = (ds.folds >= 0).sel(station=common_ids)
    obs_point['is_train'] = (ds.folds >= 0).sel(station=common_ids)

    obs['Shape_Area'] = obs['Shape_Area'] / 1000000

    return obs, obs_point
