
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import xarray as xr
import numpy as np
from pathlib import Path
from scipy.stats import theilslopes
from scipy.stats.mstats import winsorize

from utils.data import load_xval_test_set
from utils.metrics import compute_metrics
from utils.plotting import load_default_mpl_config, savefig

load_default_mpl_config()

PLOT_PATH = Path('/net/argon/landclim/kraftb/machflow/mach-flow/analysis/figures/')
RUNS_PATH = Path('/net/argon/landclim/kraftb/machflow/runs/')
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


def dfunc(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    return (theilslopes(x, y).slope, theilslopes(x, y).intercept)


def robust_regression(x):
    trend = xr.apply_ufunc(dfunc, x, x.time,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[], []],
        output_dtypes=[float, float],
        vectorize=True)
    return trend


def plot_linreg(x, y, color, ax, **kwargs):
    sort_indices = np.argsort(x)
    x_o = x[sort_indices]
    y_o = y[sort_indices]


    x = winsorize(x_o, limits=(0.05, 0.05))
    y = winsorize(y_o, limits=(0.05, 0.05))

    X_o = sm.add_constant(x_o)
    X = sm.add_constant(x)
    ols_model = sm.OLS(y, X)
    est = ols_model.fit()
    out = est.conf_int(alpha=0.05, cols=None)

    ax.scatter(x_o, y_o, color=color, zorder=101, **kwargs)
    y_pred = est.predict(X_o)
    x_pred = x_o
    ax.plot(x_pred, y_pred, color=color, lw=1.7, zorder=100, ls='-')

    pred = est.get_prediction(X_o).summary_frame()
    ax.fill_between(x_pred, pred['mean_ci_lower'], pred['mean_ci_upper'],
                    facecolor=color, edgecolor='none', alpha=0.4)

    return est


ds = load_xval_test_set(
    xval_dir=RUNS_PATH / 'staticall_allbasins_sqrttrans/LSTM/xval/',
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

ds = load_xval_test_set(
    xval_dir=RUNS_PATH / 'staticall_allbasins_sqrttrans/LSTM/xval/',
    time=slice('1995', None))[runoff_vars].sel(tau=0.5).drop_vars('tau')
ds = get_masked_runoff(ds)
ds_yearly = resample_yearly(ds)

ds_yearly_tid = ds_yearly.copy()
ds_yearly_tid['time'] = np.arange(len(ds_yearly_tid['time']))

metric_labels = ['r (-)', 'bias (mm d$^{-1}$)']
metrics = ['r', 'bias']

ds_trends = load_xval_test_set(
    xval_dir=RUNS_PATH / 'staticall_allbasins_sqrttrans/LSTM/xval/',
    time=slice('1995', '2020'))[runoff_vars].sel(tau=0.5).drop_vars('tau')

ds_trends = get_masked_runoff(ds_trends)
ds_trends = resample_yearly(ds_trends)

ds_trends['time'] = np.arange(len(ds_trends['time']))

fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), gridspec_kw={'wspace': 0.3, 'width_ratios': [1, 1, 1.5]})

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
    ax.set_xlabel('Model')

    if i == 1:
        ax.axhline(color='k', lw=1.0, ls=':', zorder=-1)

ax = axes[-1]

pf = ds_trends.polyfit(dim='time', deg=1).sel(degree=1)
pf_obs = pf.Qmm_polyfit_coefficients
pf_mod = pf.Qmm_mod_polyfit_coefficients
pf_prevah = pf.Qmm_prevah_polyfit_coefficients

obs_slope, obs_intercept = robust_regression(ds_trends.Qmm)
mod_slope, mod_intercept = robust_regression(ds_trends.Qmm_mod)
prevah_slope, prevah_intercept = robust_regression(ds_trends.Qmm_prevah)

est_mod = plot_linreg(
    x=obs_slope.values, y=mod_slope.values,
    color='tab:red', ax=ax, s=5, label='LSTM', facecolor='tab:red', edgecolor='k', lw=0.3)
est_prevah = plot_linreg(
    x=obs_slope.values, y=prevah_slope.values,
    color='0.4', ax=ax, s=5, label='PREVAH', facecolor='0.4', edgecolor='k', lw=0.3)

ax.text(0.01, 0.91,
    f'LSTM (r={np.sqrt(est_mod.rsquared):0.2f}): y={est_mod.params[0]:0.2f} + {est_mod.params[1]:0.2f}x',
        ha='left', va='top', transform=ax.transAxes, color='tab:red', size=7)
ax.text(0.01, 0.83,
    f'PREVAH (r={np.sqrt(est_prevah.rsquared):0.2f}): y={est_prevah.params[0]:0.2f} + {est_prevah.params[1]:0.2f}x',
        ha='left', va='top', transform=ax.transAxes, color='k', size=7)

xmin, xmax = pf_obs.quantile([0, 1])
ymin0, ymax0 = pf_mod.quantile([0, 1])
ymin1, ymax1 = pf_mod.quantile([0, 1])
ymin = min(ymin0, ymin1)
ymax = max(ymax0, ymax1)

ax.plot([xmin, xmax], [xmin, xmax], color='k', lw=0.7, ls='--', zorder=-1)

extra_f = 0.1
xrng = (xmax - xmin) * extra_f
xmin -= xrng
xmax += xrng
yrng = (ymax - ymin) * extra_f
ymin -= yrng
ymax += yrng

xmax = ymax = 0.14

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.axvline(0, color='k', lw=1.0, ls=':', zorder=-1)
ax.axhline(0, color='k', lw=1.0, ls=':', zorder=-1)
ax.set_xlabel('Observed trends (mm y$^{-2}$)')
ax.set_ylabel('Simulated trends (mm y$^{-2}$)')

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

for ax in axes[:-1]:
    ax.spines[['right', 'top']].set_visible(False)
axes[-1].spines[['top', 'left']].set_visible(False)

for i, ax in enumerate(axes.flat):
    ax.text(0.01, 0.99, ['a)', 'b)', 'c)'][i], ha='left', va='top', transform=ax.transAxes)

# Print trend comparison:

x = pf_obs.values
y = pf_mod.values - pf_prevah.values

sort_indices = np.argsort(x)
x = x[sort_indices]
y = y[sort_indices]

X = sm.add_constant(x)
ols_model = sm.OLS(y, X)
est = ols_model.fit()
out = est.conf_int(alpha=0.05, cols=None)

print(est.summary())

savefig(fig, path=PLOT_PATH / 'fig06.png')
