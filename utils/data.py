import pyreadr
import pandas as pd
import xarray as xr
from glob import glob
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from dataset.machflowdata import MachFlowDataModule


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


def load_xval_test_set(
        xval_dir: str | os.PathLike,
        num_expected_folds : int = 8,
        time_slices: list[str] | None = None,
        **subset) -> xr.Dataset:

    paths = glob(os.path.join(xval_dir, 'fold_*/preds.zarr'))

    if (n := len(paths))!= num_expected_folds:
        raise ValueError(
            f'number of files found ({n}) not equal \'num_expected_folds\' ({num_expected_folds}) in \'{xval_dir}\'.'
        )

    ds = xr.open_mfdataset(
        paths=paths,
        engine='zarr',
        concat_dim='station',
        combine='nested', 
        preprocess=lambda x: sel_cv_set(ds=x, cv_sets=2))

    if time_slices is not None:
        ds = MachFlowDataModule.mask_time_slices(
                        mask=ds,
                        tranges=time_slices,
                        mask_is_ds=True)

    return ds.sel(**subset).compute()


@dataclass
class XvalResult():
    model: str
    config: str
    xrdata: xr.Dataset
    path: os.PathLike 

    @property
    def ds(self) -> xr.Dataset:
        return self.xrdata.sel(tau=0.5)

    @property
    def config_parts(self) -> list[str]:
        return self.config.split('_')

    @property
    def data_vars(self) -> list[str]:
        return list(self.ds.data_vars)

    @property
    def mod_vars(self) -> list[str]:
        return [var for var in list(self.ds.data_vars) if var.endswith('_mod')]

    def expand_to_config(self, new_dims: dict[str, Any], vars: list[str] | None = None) -> xr.Dataset:
        ds = self.ds

        if vars is not None:
            ds = self.ds[vars]

        return ds.expand_dims(dim=new_dims)

    def __contains__(self, item: str) -> bool:
        return item in self.config_parts

    def __eq__(self, other: "XvalResult") -> bool:
        a = self.ds
        b = other.ds

        # Check structure.
        if (not a.coords.equals(b.coords)) or (list(a.data_vars) != list(b.data_vars)):
            return False

        # check all data variables.
        for var in a.data_vars:
            if a[var].dims != b[var].dims:
                return False

        return True

class XvalResultCollection():
    def __init__(
            self,
            path: os.PathLike | str,
            nonbool_kwords: list[str] | None = None,
            exclude_kword: list[str] | None = None,
            time_slices: list[str] | None = None,
            **subset) -> None:
        self.nonbool_kwords = nonbool_kwords
        self.exclude_kword = exclude_kword
        self.time_slices = time_slices
        self.subset = subset
        self.xval_results = self.load_all_xval_test_sets(path)
        self.unique_configs = self.get_unique_configs()
        self.mod_vars = self.infer_mod_vars(xval_results=self.xval_results)

    def get_unique_configs(self) -> list[str]:
        unique_configs = []

        for xval_res in self.xval_results:
            new_parts = [
                part for part in xval_res.config_parts if (part not in unique_configs) and (part != 'default')
            ]

            unique_configs.extend(new_parts)

        return unique_configs

    def handle_nonbool_config_mapping(self, expand_mapping: dict[str, Any]) -> dict[str, Any]:
        if self.nonbool_kwords is None:
            return expand_mapping

        for nonbool_item in self.nonbool_kwords:
            affected_keys = {key: value for key, value in expand_mapping.items() if key.startswith(nonbool_item)}

            matching_key = [key for key, val in affected_keys.items() if val == [True]]

            if len(matching_key) == 0:
                expand_mapping[nonbool_item] = ['none']
            elif len(matching_key) == 1:
                expand_mapping[nonbool_item] = [matching_key[0].split(nonbool_item)[1]]
            else:
                raise RuntimeError(
                    f'for the non-bool item \'{nonbool_item}\', more than one value was [True], indicating that the '
                    'non-bool item is not mutually exlusive.'
                )

            for key in affected_keys.keys():
                del expand_mapping[key]

        return expand_mapping

    def get_expanded_xval(self) -> xr.Dataset:
        unified_ds = self.xval_results[0].ds.copy()

        mod_vars_list = []

        for el in self.xval_results:
            # Get mapping config_item: present.
            expand_mapping: dict[str, Any] = {var: [var in el.config_parts] for var in self.unique_configs}
            expand_mapping = self.handle_nonbool_config_mapping(expand_mapping=expand_mapping)

            # Add model.
            expand_mapping['model'] = [el.model]

            mod_vars = el.expand_to_config(new_dims=expand_mapping, vars=self.mod_vars)
            mod_vars_list.append(mod_vars)

        mod_vars_cobined = xr.combine_by_coords(mod_vars_list)

        for var in self.mod_vars:
            unified_ds[var] = mod_vars_cobined[var]

        return unified_ds

    def infer_mod_vars(self, xval_results: list[XvalResult]) -> list[str]:
        self.assert_all_equal(xval_results=xval_results)
        return self[0].mod_vars

    def assert_all_equal(self, xval_results: list[XvalResult]) -> None:
        for el in xval_results:
            if xval_results[0] != el:
                raise RuntimeError(
                    'not all XvalResults are equal. Hint: compare these xval results: '
                    f'\'{xval_results[0].path}\' and \'{el.path}\''
                )

    def load_all_xval_test_sets(self, path: os.PathLike | str) -> list[XvalResult]:

        path = Path(path)

        xval_res_list = []

        for xval_path in path.glob('**/xval/'):

            rel_path = xval_path.relative_to(path)

            conf, model, _ = rel_path.parts

            if self.exclude_kword is not None:
                skip = False
                if self.exclude_kword == model:
                    skip = True
                for kword in self.exclude_kword:
                    if kword in conf:
                        skip = True

                if skip:
                    continue

            xval_test_ds = load_xval_test_set(xval_path, time_slices=self.time_slices, **self.subset)

            xval_res = XvalResult(model=model, config=conf, xrdata=xval_test_ds, path=xval_path)

            xval_res_list.append(xval_res)

        return xval_res_list

    def __getitem__(self, idx: int) -> XvalResult:
        return self.xval_results[idx]

    def __iter__(self):
        for xval_res in self.xval_results:
            yield xval_res


def load_config_xval_test_set(
        path: os.PathLike | str,
        nonbool_kwords: list[str] | None = None,
        exclude_kword: list[str] | None = None,
        time_slices: list[str] | None = None,
        **subset) -> xr.Dataset:
    """
    
    Args:
        path (str): path to an experiment (top level).
        nonbool_kwords (str | None): Optional configuration keywords for non-bool configurations. For example, if we
            have a config 'transsqrt' and 'translog', which can be both False of one them True, we can use
            `nonbool_kwords=['trans']` and the resulting dataset has a dimension 'trans' with either 'none' if both
            configs are not used, and 'sqrt' or 'log' otherwise.
        exclude_kword (str | None): Optional keywords to exclude, referring to configurations. E.g., 'translog' to
            exclude all models with 'translog' configuration.
        time_slices (list[str]): List of comma-separated slices to subset the respective set in time, e.g.,
            ['2001,2003', '2005-01-01,'2010-06'].
        **subset: xr.Dataset-style 'sel' arguments to select subset of predictions. For example `time='2009'`.

    Returns:
        A xarray.Dataset with all configurations merged into dimensions.
    """
    return XvalResultCollection(
        path=path,
        nonbool_kwords=nonbool_kwords,
        exclude_kword=exclude_kword,
        time_slices=time_slices,
        **subset).get_expanded_xval().compute()


def sources_to_folds(sources: int | list[int]) -> list[int] | None:
    """Select basin by source:

    sources:
        0: PREVAH
        1: Observational but not used for training
        2: Observational used for training

    """
    sources = [sources] if isinstance(sources, int) else sources

    if sources is not None:
        actual_sources = []

        for source in sources:
            if source not in [0, 1, 2]:
                raise ValueError(
                    f'\'sources\' must contain integers in range [0, 2], but is `{source}`, where '
                    '0=PREVAH, 1=Observational but not used for training, 2=Observational used for training.'
                )
            if source == 2:
                actual_sources.extend(list(range(8)))
            else:
                actual_sources.append(source - 2)

    else:
        actual_sources = None

    return actual_sources


def load_xval(xval_dir: str | os.PathLike, sources: int | list[int] | None = None):
    """Select basin by source:

    sources:
        0: PREVAH
        1: Observational but not used for training
        2: Observational used for training

    """

    pattern = os.path.join(xval_dir, 'fold_*/preds.zarr')
    paths = sorted(glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(
            f'no files found with pattern: `{pattern}`'
        )
    ds = xr.open_zarr(paths[0])
    mod_vars = [var for var in list(ds.data_vars) if var.endswith('_mod')]
    folds = sources_to_folds(sources)

    ds_mod = xr.open_mfdataset(
        paths=paths,
        engine='zarr',
        concat_dim='cv',
        combine='nested',
        preprocess=lambda x: x[mod_vars].sel(tau=0.5))

    for var in mod_vars:
        ds[var] = ds_mod[var]

    if folds is not None:
        ds = ds.sel(station=ds.folds.isin(folds))

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