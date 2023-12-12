import numpy as np
import xarray as xr
from typing import Iterable


def common_mask(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None,
        drop_empty: bool = True) -> tuple[xr.DataArray, xr.DataArray]:
    mask = mod.notnull() & obs.notnull()

    if drop_empty:
        all_miss = mask.any(dim).compute()
        mask = mask.where(all_miss, drop=True)

    obs = obs.where(mask)
    mod = mod.where(mask)

    return obs, mod

def _bias(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None) -> xr.DataArray:

    bias = mod.mean(dim=dim, skipna=True) - obs.mean(dim=dim, skipna=True)

    return bias.compute()

def _absbias(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None) -> xr.DataArray:

    absbias: xr.DataArray = np.abs(_bias(obs=obs, mod=mod, dim=dim))

    return absbias.compute()

def _mse(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None) -> xr.DataArray:

    return ((mod - obs)**2).mean(dim=dim).compute()

def _rmse(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None) -> xr.DataArray:

    return (_mse(obs=obs, mod=mod, dim=dim) ** 0.5).compute()

def _r(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None) -> xr.DataArray:

    return xr.corr(obs, mod, dim=dim).compute()

def _nse(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None) -> xr.DataArray:

    top = ((mod - obs)**2).sum(dim=dim, skipna=True)
    bottom = ((obs - obs.mean(dim=dim, skipna=True))**2).sum(dim=dim, skipna=True)

    nse = 1 - top / bottom

    return nse.compute()

def _nnse(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None) -> xr.DataArray:

    nse = _nse(obs=obs, mod=mod, dim=dim)

    return (1 / (2 - nse)).compute()

def _kge(
        obs: xr.DataArray,
        mod: xr.DataArray,
        dim: str | Iterable[str] | None) -> xr.DataArray:

    r = _r(obs, mod, dim=dim)
    alpha = mod.std(dim=dim) / obs.std(dim=dim)
    beta = mod.mean(dim=dim) / obs.mean(dim=dim)

    value = ((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return (1 - value ** 0.5).compute()

METRIC_MAPPING = dict(
    bias={'func': _bias, 'name': 'Bias', 'direction': 'min'},
    absbias={'func': _absbias, 'name': 'Absolute bias', 'direction': 'min'},
    mse={'func': _mse, 'name': 'Mean squared error', 'direction': 'min'},
    rmse={'func': _rmse, 'name': 'Root mean squared error', 'direction': 'min'},
    nse={'func': _nse, 'name': 'Modeling efficiency', 'direction': 'max'},
    nnse={'func': _nnse, 'name': 'Normalized modeling efficiency', 'direction': 'max'},
    kge={'func': _kge, 'name': 'Kling–Gupta efficiency', 'direction': 'max'},
    r={'func': _r, 'name': 'Pearson\'s correlation', 'direction': 'max'},
)


def compute_metrics(
    obs: xr.DataArray,
    mod: xr.DataArray,
    metrics: str | list[str] = 'all',
    dim: str | Iterable[str] | None = None,
    drop_empty: bool = True) -> xr.Dataset:

    if metrics == 'all':
        metrics = list(METRIC_MAPPING.keys())

    metrics = [metrics] if isinstance(metrics, str) else metrics

    missing_metrics = []
    for metric in metrics:
        if metric not in METRIC_MAPPING:
            missing_metrics.append(metric)

    if len(missing_metrics) > 0:
        raise ValueError(
            f'the following metrics are not implemented: `{"`, `".join(missing_metrics)}`'
        )

    obs_da, mod_da = common_mask(obs=obs, mod=mod, dim=dim, drop_empty=drop_empty)

    ds = xr.Dataset()

    for metric in metrics:
        func = METRIC_MAPPING[metric]['func']
        name = METRIC_MAPPING[metric]['name']
        direction = METRIC_MAPPING[metric]['direction']
        da = func(obs=obs_da, mod=mod_da, dim=dim)
        da.attrs['long_name'] = name
        da.attrs['direction'] = direction
        ds[metric] = da

    return ds
