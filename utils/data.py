import pyreadr
import pandas as pd
import xarray as xr
from glob import glob
import os


def read_rds(path: str) -> pd.DataFrame:
    return pyreadr.read_r(path)[None]


def sel_cv_set(ds: xr.Dataset, cv_sets: int | list[int]) -> xr.Dataset:
    cv_sets = [cv_sets] if isinstance(cv_sets, int) else cv_sets

    for cv_set in cv_sets:
        if cv_set not in range(-1, 3):
            raise ValueError(
                f'\'cv_sets\' must contain integers in range [-1, 2], but is `{cv_set}`, where '
                '-1=not used, 0=training, 1=validation, 2=test.'
            )

    return ds.sel(station=ds.cv_set.isin(cv_sets))


def load_xval_test_set(xval_dir: str) -> xr.Dataset:
    ds = xr.open_mfdataset(
        paths=glob(os.path.join(xval_dir, 'fold_*/preds.zarr')),
        engine='zarr',
        concat_dim='station',
        combine='nested', 
        preprocess=lambda x: sel_cv_set(ds=x, cv_sets=2))
    return ds
