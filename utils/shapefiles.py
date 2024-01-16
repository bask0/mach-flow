
import xarray as xr
import geopandas as gpd
from pathlib import Path
import os


def get_shapefile(source: str, datapath: str | os.PathLike = '/mydata/machflow/basil/data/') -> gpd.GeoDataFrame:

    datapath = Path(datapath)

    if source == 'obs':
        obs = gpd.read_file('/mydata/machflow/basil/data/obs/shapefile/shape_tabsd.shp')
        id_field = 'mach_ID'
    elif source == 'prevah':
        obs = gpd.read_file('/mydata/machflow/basil/data/prevah/shapefile/Regions_307.shp')
        id_field = 'OBJECTID'
        obs[id_field] = [f'HSU_{int(id):03d}' for id in obs[id_field].astype(str)]
    else:
        raise ValueError('`source` must be on fo \'obs\', \'prevah\', is \'{source}\'.')

    ds = xr.open_zarr('/mydata/machflow/basil/data/harmonized_basins.zarr/')

    common_ids = [oid for oid in obs[id_field].values if oid in ds.station.values]
    obs = obs[obs[id_field].isin(common_ids)]
    ds = ds.sel(station=common_ids)
    obs['is_train'] = (ds.folds >= 0).sel(station=common_ids)

    obs = obs.rename(columns={id_field: 'OID'})

    obs['Shape_Area'] = obs['Shape_Area'] / 1000000

    return obs
