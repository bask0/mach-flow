
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d

from utils.plotting import savefig, load_default_mpl_config
from utils.data import load_xval
from utils.shapefiles import get_shapefile

load_default_mpl_config()

PLOT_PATH = Path('/mydata/machflow/basil/mach-flow/analysis/figures/')

ds = load_xval(
    xval_dir='/mydata/machflow/basil/runs/basin_level/staticall_allbasins_sqrttrans/LSTM/xval/',
    sources=0).sel(time=slice('1962', '2023'))
prevah, _ = get_shapefile(source='prevah')

def add_data(
        da: xr.DataArray,
        gd: gpd.GeoDataFrame, 
        new_col='da_var',
        allow_missing_id: bool = False) -> gpd.GeoDataFrame:

    da = da.load()

    gd[new_col] = da.sel(station=gd.OID.values).values

    return gd


def digitize(x: xr.DataArray) -> xr.DataArray:
    num_bins = 13
    q = x.sel(time=slice('1971', '2000')).quantile(np.linspace(0.0, 1.0, num_bins), dim='time')
    # q = xr.apply_ufunc(np.digitize, x, q.values, input_core_dims=[('time')], vectorize = True)
    r = xr.apply_ufunc(
        np.digitize, x, q,
        input_core_dims=[['time'], ['quantile']],
        output_core_dims=[['time']], vectorize=True) / num_bins

    return r


def plot_linreg(x, y, color, ax, **kwargs):
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    y = y[sort_indices]

    X = sm.add_constant(x)
    ols_model = sm.OLS(y, X)
    est = ols_model.fit()
    out = est.conf_int(alpha=0.05, cols=None)

    y_pred = est.predict(X)
    x_pred = x
    ax.plot(x_pred, y_pred, color=color, zorder=100, **kwargs)


    return est

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


qmm_year = ds.Qmm_mod.resample(time='1Y').sum('time').median('cv').compute()
qmm_year_ref = qmm_year.sel(time=slice('1971', '2000')).mean('time').compute()
qmm_year_ano = (qmm_year - qmm_year_ref).compute()

qmm_year_ano_dig = digitize(qmm_year_ano)

qmm_dec_mean = qmm_year_ano.copy()
year = qmm_dec_mean['time'].dt.year
decade = ((year - 1960) // 10).astype(int)
qmm_dec_mean['decade'] = decade
qmm_dec_mean = qmm_dec_mean.groupby('decade').mean('time')

qmm_dec_ano = (qmm_year_ano * ds.area).sum('station') / ds.area.sum()
qmm_dec_ano = qmm_dec_ano - qmm_dec_ano.sel(time=slice('1971', '2000')).mean('time')
year = qmm_dec_ano['time'].dt.year
decade = ((year - 1960) // 10).astype(int)
qmm_dec_ano['decade'] = decade
qmm_dec_ano = qmm_dec_ano.groupby('decade').mean('time')


fig = plt.figure(tight_layout=True, figsize=(10, 10))
gs = gridspec.GridSpec(11, 7, wspace=0, hspace=0)

min_year = 1962
max_year = 2023
# max_year = 1972
all_years = range(1960, 2024)

cmap = 'BrBG'

for i, year in tqdm(enumerate(all_years), total=len(all_years), ncols=120, desc='plotting year means'):
    row = year % 10
    col = (year - 1960) // 10

    ax = fig.add_subplot(gs[row, col])

    if (year < min_year) or (year > max_year):
        ax.remove()
        continue

    prevah_mean_qmm = add_data(da=qmm_year_ano_dig.sel(time=str(year)), gd=prevah, new_col='Qmm_sim')

    im = prevah_mean_qmm.plot(
        ax=ax,
        column='Qmm_sim',
        legend=False,
        facecolor='tab:blue',
        edgecolor='0.4',
        cmap=cmap,
        vmin=0,
        vmax=1,
        lw=0.2,
        alpha=1.0,
    )

    ax.axis('off')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.margins(0)
    ax.set_title(str(year), x=0, y=0.7, ha='left', size=7)

def gradientbars(bars):

    ax = bars[0].axes

    ymin, ymax = ax.get_ylim()
    yabs = max(np.abs(ymin), ymax)

    cm = plt.get_cmap(cmap)

    for i, bar in enumerate(bars):
        bar.set_zorder(1)
        y = bar.get_height()
        y_norm = (y + yabs) / 2 / yabs
        
        bar.set_facecolor(cm(y_norm))
        bar.set_edgecolor('k')
        ax.text(
            i, 30 if y < 0 else -30, f'{y:0.1f}',
            va='bottom' if y < 0 else 'top', ha='center', color='k',
            )

    return -yabs, yabs

cax = fig.add_subplot(gs[0, 0])
patch_col = im.collections[0]
cb = fig.colorbar(patch_col, ax=cax, orientation='horizontal', shrink=0.8)
cax.axis('off')
cb.ax.set_title('P(Q | Q$_{1971-2000}$)', size=9)
cb.ax.set_xticks([0, 0.5, 1])
cb.ax.set_xticklabels(['0', '0.5', '1'])
cb.ax.tick_params(labelsize=9)

ax_dec = fig.add_subplot(gs[-1, :])
bar = ax_dec.bar([f'{d}s' for d in range(1960, 2021, 10)], qmm_dec_ano, width=0.5)
ymin, ymax = gradientbars(bar)
ax_dec.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
ax_dec.set_ylim(ymin, ymax)
ax_dec.axhline(0, color='k', ls='--', lw=1)
ax_dec.text(0.01, 0.1, 'Decadal Q dev. 1971-2000 (mm y$^{-1}$)', va='bottom', ha='left', transform=ax_dec.transAxes)
ax_dec.margins(x=0.04)
ax_dec.set_yticks([])

lax = fig.add_subplot(gs[-2, -1])
lax.axis('off')
lax.text(0.1, 0.0, 'Kraft et al. (2024), HESS', va='bottom', ha='left', transform=lax.transAxes, size=8, rotation=90)

savefig(fig, PLOT_PATH / 'fig07.png', dpi=450)

qmm_year_mod = ds.Qmm_mod.resample(time='1Y').sum('time', skipna=False).median('cv').compute()
qmm_year_prevah = ds.Qmm_prevah.resample(time='1Y').sum('time', skipna=False).compute()

qmm_year_abs_mod = (qmm_year_mod * ds.area).sum('station', skipna=False) / ds.area.sum()
qmm_year_abs_prevah = (qmm_year_prevah * ds.area).sum('station', skipna=False) / ds.area.sum()

qmm_year_ref = qmm_year_abs_mod.sel(time=slice('1971', '2000')).mean('time').compute()

qmm_year_mod = qmm_year_abs_mod - qmm_year_ref
qmm_year_prevah = qmm_year_abs_prevah - qmm_year_ref

fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.25})

for ax in axes:
    ax.spines[['right', 'top']].set_visible(False)

ax = axes[0]

cmap = plt.get_cmap('BrBG', 256)


color_dry = cmap(0.2)
color_wet = cmap(0.8)

time = qmm_year_mod.time.dt.strftime('%Y').values.astype('int')

ax.bar(time, qmm_year_mod.where(qmm_year_mod >= 0).values, color=color_wet, label='Year above average')
ax.bar(time, qmm_year_mod.where(qmm_year_mod < 0).values, color=color_dry, label='Year below average')

ax.scatter(
    time, qmm_year_prevah.where(qmm_year_prevah >= 0).values, facecolor=color_wet, edgecolor='k', s=25,
    label='Year above average ')
ax.scatter(
    time, qmm_year_prevah.where(qmm_year_prevah < 0).values, facecolor=color_dry, edgecolor='k', s=25,
    label='Year below average ')

ax.set_xticks(time[::5])

n = 5
n_cut = int((n - 1) / 2)

ma_mod = gaussian_filter1d(qmm_year_mod.values, 2)

qmm_year_prevah_subset = qmm_year_prevah.sel(time=slice('1981', '2022'))
ma_prevah = gaussian_filter1d(qmm_year_prevah_subset.values, 2)
time_prevah = qmm_year_prevah_subset.time.dt.strftime('%Y').values.astype('int')

# plot_linreg(x=time.astype('float'), y=qmm_year_mod.values, ax=ax, color='tab:red', lw=1.3, ls='-', label='Linear trend ')
# plot_linreg(x=time_prevah.astype('float'), y=qmm_year_prevah_subset.values, ax=ax, color='0.4', lw=1.3, ls='-', label='Linear trend')


ax.plot(time, ma_mod, lw=2.0, color='tab:red', ls='-', label='Trend (low-pass filter) ')
ax.plot(time_prevah, ma_prevah, lw=2.0, color='0.4', ls='-', label='Trend (low-pass filter)')

ax.set_ylabel('Annual Q dev. 1971-2000 (mm y$^{-1}$)')

handles, labels = ax.get_legend_handles_labels()
handle_list, label_list = [], []

order = [4, 0,  5, 1, 2, 3]

for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)

handles = [handle_list[idx] for idx in order]
labels = [label_list[idx] for idx in order]

ph = [plt.plot([],marker="", ls="")[0]] * 2
handles = ph + handles
labels = ['CH-RUN:', 'PREVAH:'] + labels

ax.legend(
    handles,
    labels,
    ncols=4,
    frameon=False,
    fontsize=9,
    handlelength=2,
    columnspacing=0.7,
    loc='center',
    bbox_to_anchor=(0.45, -0.2))

ax.text(0.01, 0.99, 'a)', va='top', ha='left', transform=ax.transAxes)

ax = axes[1]

ax.scatter(qmm_year_abs_prevah.values, qmm_year_abs_mod.values, edgecolor='k', facecolor='none', s=15)

qmean = qmm_year_ref.item()

ax.axhline(qmean, color='k', ls=':', lw=0.8)
ax.axvline(qmean, color='k', ls=':', lw=0.8)

ax.text(qmean, 620, r'$\leftarrow$ dry  ', ha='right', va='center', color=color_dry, weight='bold')
ax.text(qmean, 620, r' wet $\rightarrow$', ha='left', va='center', color=color_wet, weight='bold')

ax.text(620, qmean, r'$\leftarrow$ dry  ', va='top', ha='center', color=color_dry, rotation=90, weight='bold')
ax.text(620, qmean, r' wet $\rightarrow$', va='bottom', ha='center', color=color_wet, rotation=90, weight='bold')

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

ax.plot(lims, lims, color='k', ls='--')
# ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_ylabel('CH-RUN: Annual Q (mm y$^{-1}$)')
ax.set_xlabel('PREVAH: Annual Q (mm y$^{-1}$)')

e =plot_linreg(
    x=qmm_year_abs_prevah.sel(time=slice('1981', '2022')).values,
    y=qmm_year_abs_mod.sel(time=slice('1981', '2022')).values,
    ax=ax, color='k', lw=1., ls='-', label='Linear fit')

ax.text(0.01, 0.99, f'b)\n$\mathrm{{Q_{{CH–RUN}}}} = {e.params[0]:0.2f} + {e.params[1]:0.2f}\,\mathrm{{Q_{{PREVAH}}}}$', va='top', ha='left', transform=ax.transAxes)

corr = np.corrcoef(qmm_year_abs_prevah.sel(time=slice('1981', '2022')).values, qmm_year_abs_mod.sel(time=slice('1981', '2022')).values)[0, 1]

ax.text(0.01, 0.99, f'b)\n$\mathrm{{Q_{{CH–RUN}}}} = {e.params[0]:0.2f} + {e.params[1]:0.2f}\,\mathrm{{Q_{{PREVAH}}}}$\n$r={corr:0.2f}$', va='top', ha='left', transform=ax.transAxes)

ax.text(1.0, -0.25, 'Kraft et al. (2024), HESS', va='bottom', ha='right', transform=ax.transAxes, size=8)

savefig(fig, PLOT_PATH / 'fig08.png', dpi=450)


fig = plt.figure(tight_layout=True, figsize=(10, 5))
gs = gridspec.GridSpec(5, 8, wspace=0, hspace=0)

decades = [1960, 1970, 1980, 1990, 2000, 2010, 2020]
cmap = 'BrBG'

for s, season in enumerate([None, 'DJF', 'MAM', 'JJA', 'SON']):
# for s, season in enumerate([['DJF', 'JJA', 'MAM', 'SON']]):
    if season is not None:
        ds_s = ds.sel(time=ds.time.dt.season.isin(season))
    else:
        ds_s = ds

    qmm_year = ds_s.Qmm_mod.resample(time='1Y').sum('time', skipna=False).compute()

    if 'cv' in qmm_year.dims:
        qmm_year = qmm_year.median('cv')

    qmm_year = qmm_year.compute()

    qmm_year_ref = qmm_year.sel(time=slice('1971', '2000')).mean('time').compute()
    qmm_year_ano = (qmm_year - qmm_year_ref).compute()

    qmm_dec_mean = qmm_year_ano.copy()
    year = qmm_dec_mean['time'].dt.year
    decade = ((year - 1960) // 10).astype(int)
    qmm_dec_mean['decade'] = decade
    qmm_dec_mean = qmm_dec_mean.groupby('decade').mean('time')

    for i, decade in tqdm(enumerate(decades), total=7, ncols=120, desc='plotting year means'):

        ax = fig.add_subplot(gs[s, i + 1])

        data = add_data(da=qmm_dec_mean.isel(decade=i), gd=prevah, new_col='Qmm_sim')

        im = data.plot(
            ax=ax,
            column='Qmm_sim',
            legend=False,
            facecolor='tab:blue',
            edgecolor='0.4',
            cmap=cmap,
            vmin=-100,
            vmax=100,
            lw=0.2,
            alpha=1.0,
        )

        ax.axis('off')
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.margins(0)
        ax.set_title(f'{decade}s', x=0, y=0.7, ha='left', size=7)

        if i == 0:
            if s == 0:
                season_label = 'Year'
            else:
                season_label = season

            ax.text(0, 0.5, season_label, transform=ax.transAxes, ha='right', va='center', rotation=90)

cax = fig.add_subplot(gs[0, 0])
patch_col = im.collections[0]
cb = fig.colorbar(patch_col, ax=cax, orientation='horizontal', shrink=0.8)
cax.axis('off')
cb.ax.set_title('Q dev.\n1971-2000\n(mm y$^{-1}$)', size=9)
cb.ax.set_xticks([-100, 0, 100])
cb.ax.set_xticklabels(['-100', '0', '100'])
cb.ax.tick_params(labelsize=9)

lax = fig.add_subplot(gs[-1, 0])
lax.axis('off')
lax.text(0.8, 0.2, 'Kraft et al. (2024), HESS', va='bottom', ha='right', transform=lax.transAxes, size=8, rotation=90)

savefig(fig, PLOT_PATH / 'fig09.png', dpi=450)
