
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import xarray as xr
import numpy as np
from pathlib import Path

from utils.data import load_xval_test_set
from utils.metrics import compute_metrics
from utils.plotting import load_default_mpl_config, savefig

load_default_mpl_config()

PLOT_PATH = Path('/mydata/machflow/basil/mach-flow/analysis/figures/')
TEST_SLICES = [slice('1995', '1999'), slice('2016', '2020')]
runoff_vars = ['Qmm', 'Qmm_mod', 'Qmm_prevah']

def common_mask_da(*da):
    mask = True
    for el in da:
        mask *= el.notnull()

    return mask


def get_masked_runoff(ds: xr.Dataset, kw: str = 'Qmm') -> xr.Dataset:
    ds_subset = ds[[var for var in list(ds.data_vars) if var.startswith(kw)]]
    
    mask = ds_subset.to_array().notnull().all('variable').compute()

    ds_subset = ds_subset.where(mask)

    return ds_subset.compute()


def resample_yearly(ds: xr.Dataset, min_frac_present: float = 0.01) -> xr.Dataset:
    ds_yearly = ds.resample(time='1Y').mean()
    ds_yearly_valid = ds.notnull().resample(time='1Y').sum()
    ds_yearly_count = ds.resample(time='1Y').count()

    ds_yearly = ds_yearly.where(ds_yearly_valid / ds_yearly_count > min_frac_present)

    return ds_yearly.compute()

def merged_df(mod, prevah):
    x_mod = mod.to_dataframe().reset_index()
    x_mod['source'] = 'mod'
    x_pre = prevah.to_dataframe().reset_index()
    x_pre['source'] = 'prevah'
    x_both = pd.concat((x_mod, x_pre)).reset_index()

    return x_both

ds = load_xval_test_set(
    xval_dir='/mydata/machflow/basil/runs/basin_level/staticall_allbasins_sqrttrans/LSTM/xval/',
    time_slices=[
        f'{TEST_SLICES[0].start},{TEST_SLICES[0].stop}',
        f'{TEST_SLICES[1].start},{TEST_SLICES[1].stop}'])[runoff_vars].sel(tau=0.5).drop_vars('tau')
ds = get_masked_runoff(ds)
ds_yearly = resample_yearly(ds)

x_mod = compute_metrics(obs=ds_yearly.Qmm, mod=ds_yearly.Qmm_mod, dim='time')
x_prevah = compute_metrics(obs=ds_yearly.Qmm, mod=ds_yearly.Qmm_prevah, dim='time')

x_both = merged_df(mod=x_mod, prevah=x_prevah)

BOX_PROPS = {
    'boxprops':{'edgecolor': 'k'},
    'medianprops':{'color': 'k'},
    'whiskerprops':{'color': 'k'},
    'capprops':{'color': 'k'}
}

ds_yearly_diff = (ds_yearly.sel(time=TEST_SLICES[1]).mean('time') - ds_yearly.sel(time=TEST_SLICES[0]).mean('time')).compute()
df_yearly_diff = ds_yearly_diff.to_dataframe().melt(ignore_index=False, var_name='source', value_name='delta').reset_index()
df_yearly_diff_wide = ds_yearly_diff.to_dataframe().reset_index()

yearly_corr_mod = df_yearly_diff_wide.corr(numeric_only=True).values[0, 1]
yearly_corr_prevah = df_yearly_diff_wide.corr(numeric_only=True).values[0, 2]

print(f'LSTM trend correlation: {yearly_corr_mod:0.2f}')
print(f'PREVAH trend correlation: {yearly_corr_prevah:0.2f}')

q0 = f'{TEST_SLICES[0].start}-{TEST_SLICES[0].stop}'
q1 = f'{TEST_SLICES[1].start}-{TEST_SLICES[1].stop}'

ds = load_xval_test_set(
    xval_dir='/mydata/machflow/basil/runs/basin_level/staticall_allbasins_sqrttrans/LSTM/xval/',
    time=slice('1995', None))[runoff_vars].sel(tau=0.5).drop_vars('tau')
ds = get_masked_runoff(ds)
ds_yearly = resample_yearly(ds)

ds_yearly_tid = ds_yearly.copy()
ds_yearly_tid['time'] = np.arange(len(ds_yearly_tid['time']))

metric_labels = ['r (-)', 'bias (mm d$^{-1}$)']
metrics = ['r', 'bias']

fig, axes = plt.subplots(1, 3, figsize=(8, 2), gridspec_kw={'wspace': 0.6, 'width_ratios': [1, 1, 1.7]})

for ax in axes:
    ax.spines[['right', 'top']].set_visible(False)

# METRICS

for i, (metric_label, metric, ax) in enumerate(zip(metric_labels, metrics, axes[:2])):
    bplot = sns.boxplot(
        data=x_both,
        x='source',
        y=metric,
        showfliers=False,
        hue='source',
        ax=ax,
        palette={'mod': 'tab:red', 'prevah': '0.5'},
        width=0.5,
        **BOX_PROPS)

    ax.set_ylabel(metric_label)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r'LSTM$_\mathrm{best}$', 'PREVAH'])
    ax.set_xlabel('')

    if i == 1:
        ax.axhline(color='k', lw=1.0, ls=':', zorder=-1)

# Q diff

ax = axes[2]

sns.boxplot(
    data=df_yearly_diff,
    x='source',
    y='delta',
    hue='source',
    showfliers=False,
    ax=ax,
    width=0.6,
    color='w',
    palette={'Qmm': 'w', 'Qmm_mod': 'tab:red', 'Qmm_prevah': '0.5'},
    **BOX_PROPS)

ax.axhline(color='k', lw=1.0, ls=':', zorder=-1)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Observed', r'LSTM$_\mathrm{best}$', 'PREVAH'])
ax.set_ylabel(f'$\Delta Q = \\bar{{Q}}_2 - \\bar{{Q}}_1$ (mm d$^-1$)')
ax.set_xlabel('')
ax.spines[['right', 'top']].set_visible(False)



for axes_ in axes.T:
    fig.align_ylabels(axes_)

for i, ax in enumerate(axes.flat):
    ax.text(0.01, 0.99, ['a)', 'b)', 'c)'][i], ha='left', va='top', transform=ax.transAxes)


savefig(fig, path=PLOT_PATH / 'fig07.png')
