import pyreadr
import pandas as pd
import xarray as xr
from glob import glob
import os


def read_rds(path: str | os.PathLike) -> pd.DataFrame:
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


def load_xval_test_set(xval_dir: str | os.PathLike) -> xr.Dataset:
    ds = xr.open_mfdataset(
        paths=glob(os.path.join(xval_dir, 'fold_*/preds.zarr')),
        engine='zarr',
        concat_dim='station',
        combine='nested', 
        preprocess=lambda x: sel_cv_set(ds=x, cv_sets=2))
    return ds


def convert_q(
        x: float | xr.DataArray,
        area: float | xr.DataArray,
        from_unit: str, 
        to_unit: str) -> float | xr.DataArray:
    """Convert Q. Area is in m2."""

    valid_units = ['mm', 'mmd', 'm3s', 'ls']

    for unit_arg_name, unit_arg in zip(['from_unit', 'to_unit'], [from_unit, to_unit]):
        if unit_arg not in valid_units:
            raise ValueError(
                f'`{unit_arg_name}` must be one of `{"` | `".join(valid_units)}``, is `{unit_arg}`.'
            )

    from_unit = 'mmd' if from_unit == 'mm' else from_unit
    to_unit = 'mmd' if to_unit == 'mm' else to_unit

    if from_unit == to_unit:
        return x

    mm_to_m3s_factor = 60 * 60 * 24 * 1000 / area

    cm = dict(
        mmd=dict(
            m3s=1 / mm_to_m3s_factor,
            ls=1 / mm_to_m3s_factor / 1000
        ),
        m3s=dict(
            mmd=mm_to_m3s_factor,
            ls=1 / 1000
        ),
        ls=dict(
            mmd=mm_to_m3s_factor * 1000,
            m3s=1000
        ),
    )

    return x * cm[from_unit][to_unit]