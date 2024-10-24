
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import xarray as xr
import numpy as np
from scipy.stats import theilslopes, spearmanr
from scipy.stats.mstats import winsorize

from utils.data import load_xval_test_set
from utils.metrics import compute_metrics
from utils.plotting import load_default_mpl_config, savefig
from config import get_path_dict

load_default_mpl_config()

paths = get_path_dict()

# TEST_SLICES = [slice('1995', '1999'), slice('2016', '2020')]
TEST_SLICES = ['1995,2020']
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
    ds_yearly = ds.resample(time='1YE').sum()
    ds_yearly_valid = ds.notnull().resample(time='1YE').sum()
    ds_yearly_count = ds.resample(time='1YE').count()

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
    res = theilslopes(x, y)
    return (res.slope, res.intercept)


def robust_regression(x):
    trend = xr.apply_ufunc(dfunc, x, x.time,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[], []],
        output_dtypes=[float, float],
        vectorize=True)
    return trend


def plot_linreg(x, y, text_y_pos, mod_name, color, ax, **kwargs):

    rlm = sm.RLM(y, sm.add_constant(x.reshape(-1, 1)), M=sm.robust.norms.HuberT())
    rlm_results = rlm.fit()

    x_new = np.linspace(x.min(), x.max(), 10)
    y_new = rlm_results.predict(sm.add_constant(x_new))

    ax.scatter(x, y, color=color, zorder=101, **kwargs)
    ax.plot(x_new, y_new, color=color, lw=1.7, zorder=100, ls='-')

    r = spearmanr(x, y).statistic

    s = f'{mod_name} ($\\rho$={r:0.2f}): y={rlm_results.params[0]:0.2f} + {rlm_results.params[1]:0.2f}x'
    # f'{mod_name} (r={r:0.2f}): y={rlm_results.params[0]:0.2f} + {rlm_results.params[1]:0.2f}x'
    ax.text(0.01, text_y_pos, s, ha='left', va='top', transform=ax.transAxes, color=color, size=7,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1))


ds = load_xval_test_set(
    xval_dir=paths['runs'] / 'staticall_allbasins_sqrttrans/LSTM/xval/',
    time_slices=TEST_SLICES)[runoff_vars].sel(tau=0.5).drop_vars('tau')
ds = get_masked_runoff(ds)
ds_yearly = resample_yearly(ds)

x_mod = compute_metrics(obs=ds_yearly.Qmm, mod=ds_yearly.Qmm_mod, dim='time')
x_prevah = compute_metrics(obs=ds_yearly.Qmm, mod=ds_yearly.Qmm_prevah, dim='time')

x_both = merged_df(mod=x_mod, prevah=x_prevah)

BOX_PROPS = {
    'boxprops':{'edgecolor': 'k', 'alpha': 0.7},
    'medianprops':{'color': 'k'},
    'whiskerprops':{'color': 'k'},
    'capprops':{'color': 'k'}
}

ds = load_xval_test_set(
    xval_dir=paths['runs'] / 'staticall_allbasins_sqrttrans/LSTM/xval/',
    time=slice('1995', None))[runoff_vars].sel(tau=0.5).drop_vars('tau')
ds = get_masked_runoff(ds)
ds_yearly = resample_yearly(ds)

ds_yearly_tid = ds_yearly.copy()
ds_yearly_tid['time'] = np.arange(len(ds_yearly_tid['time']))

metric_labels = ['r (-)', 'bias (mm y$^{-1}$)']
metrics = ['r', 'bias']

ds_trends = load_xval_test_set(
    xval_dir=paths['runs'] / 'staticall_allbasins_sqrttrans/LSTM/xval/',
    time=slice('1995', '2020'))[runoff_vars].sel(tau=0.5).drop_vars('tau')

ds_trends = get_masked_runoff(ds_trends)
ds_trends = resample_yearly(ds_trends)

ds_trends['time'] = np.arange(len(ds_trends['time']))

fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), gridspec_kw={'wspace': 0.3, 'width_ratios': [1, 1, 1.3]})

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
        ax.axhline(color='k', lw=1.0, ls=':', zorder=99)

ax = axes[-1]

obs_slope, obs_intercept = robust_regression(ds_trends.Qmm)
mod_slope, mod_intercept = robust_regression(ds_trends.Qmm_mod)
prevah_slope, prevah_intercept = robust_regression(ds_trends.Qmm_prevah)

r_mod = plot_linreg(
    x=obs_slope.values, y=mod_slope.values, mod_name='LSTM',
    text_y_pos=0.91, color='tab:red', ax=ax, s=8, label='LSTM', facecolor='tab:red', edgecolor='k', lw=0.3, alpha=0.7)
r_prevah = plot_linreg(
    x=obs_slope.values, y=prevah_slope.values, mod_name='PREVAH',
    text_y_pos=0.85, color='0.4', ax=ax, s=8, label='PREVAH', facecolor='0.4', edgecolor='k', lw=0.3, alpha=0.7)

xmin, xmax = obs_slope.quantile([0, 1])
ymin0, ymax0 = mod_slope.quantile([0, 1])
ymin1, ymax1 = prevah_slope.quantile([0, 1])
ymin = min(ymin0, ymin1)
ymax = max(ymax0, ymax1)

plt_min = max(xmin, ymin)
plt_max = min(xmax, ymax)

ax.plot([plt_min, plt_max], [plt_min, plt_max], ls='--', color='k')

ax.text(plt_max - 0.03 * (xmax - xmin), plt_max, '1:1', va='center_baseline', ha='right', fontsize=7)

# ax.plot([-0.06, 0.06], [-0.06, 0.06], color='k', lw=0.7, ls='--', zorder=-1)

extra_f = 0.1
xrng = (xmax - xmin) * extra_f
xmin -= xrng
xmax += xrng
yrng = (ymax - ymin) * extra_f
ymin -= yrng
ymax += yrng

xmin = -0.07
xmax = -xmin
ymin = -0.07
ymax = 0.15

# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)

ax.axvline(0, color='k', lw=1.0, ls=':', zorder=-1)
ax.axhline(0, color='k', lw=1.0, ls=':', zorder=-1)
ax.set_xlabel('Observed trends (mm y$^{-1}$)')
ax.set_ylabel('Simulated trends (mm y$^{-1}$)')

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

for ax in axes[:-1]:
    ax.spines[['right', 'top']].set_visible(False)
axes[-1].spines[['top', 'left']].set_visible(False)

for i, ax in enumerate(axes.flat):
    ax.text(0.01, 0.99, ['a)', 'b)', 'c)'][i], ha='left', va='top', transform=ax.transAxes)

# Print trend comparison:

x = obs_slope.values
y = mod_slope.values - prevah_slope.values

X = sm.add_constant(x)
ols_model = sm.OLS(y, X)
est = ols_model.fit()

print(est.summary())

savefig(fig, path=paths['figures'] / 'fig05.png')
