import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path

from utils.data import load_xval_test_set
from utils.plotting import load_default_mpl_config, savefig
from utils.metrics import compute_metrics

load_default_mpl_config()

PLOT_PATH = Path('./analysis/figures')
PLOT_PATH_ALT = Path('./analysis/helper_figures/')
runoff_vars = ['Qmm', 'Qmm_mod', 'Qmm_prevah']

def merged_df(mod, prevah):
    x_mod = mod.to_dataframe().reset_index()
    x_mod['source'] = 'mod'
    x_pre = prevah.to_dataframe().reset_index()
    x_pre['source'] = 'prevah'
    x_both = pd.concat((x_mod, x_pre)).reset_index()

    return x_both


xval_ds = load_xval_test_set(
        xval_dir='/Users/kraftb/Downloads/basin_level/staticall_allbasins_sqrttrans/LSTM/xval/'
    ).isel(tau=0).drop_vars('tau')

xval_ds = xval_ds.sortby(xval_ds.Qmm.notnull().sum('time'))

import pandas as pd

fig = plt.figure(frameon=False)
fig.set_size_inches(8, 3)
ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
fig.add_axes(ax)

cmap = mcolors.ListedColormap(['1.0', '0.8'])

xval_ds.Qmm.notnull().plot(ax=ax, cmap=cmap, add_colorbar=False)


eval_times = [
    ['1993-01-01', '1994-12-31'],
    ['2000-01-01', '2001-12-31'],
    ['2014-01-01', '2015-12-31'],
    ['2021-01-01', '2023-12-31'],
]

test_times = [
    ['1995-01-01', '1999-12-31'],
    ['2016-01-01', '2020-12-31'],
]

for s, e in eval_times:
    ax.axvspan(s, e, alpha=0.5, facecolor='0.4', zorder=1)
    ax.axvspan(s, e, alpha=0.5, facecolor='none', edgecolor='0.4', hatch='//', zorder=2)

for s, e in test_times:
    ax.axvspan(s, e, alpha=0.5, facecolor='0.2', zorder=1)
    ax.axvspan(s, e, alpha=0.5, facecolor='none', edgecolor='0.2', hatch='\\\\', zorder=2)

# ax.spines[['left', 'top', 'right', 'bottom']].set_visible(False)

ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xticklabels([])


#ax.axis('off')

fig.savefig(PLOT_PATH_ALT / 'catchment_timesplit.png', dpi=600, transparent=False, bbox_inches='tight', pad_inches=0)

catchments = {
    'CHFO-0181': 'Murg-Frauenfeld (ID: 2386), Rain dominated',
    'CHFO-0190': 'Aabach-Hitzkirch, Richensee (ID: 2416), Lake dominated',
    'CHFO-0112': 'Plessur-Chur (ID: 2185), Snow dominated',
    'CHFO-0126': 'Simme-Oberried/Lenk (ID: 2219), Glacier dominated',
}

fig, axes = plt.subplots(
    nrows=len(catchments),
    ncols=2,
    figsize=(10, 2 * len(catchments)),
    sharey='row',
    sharex='col',
    gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

time_subsets = [
    '1998',
    '2018'
]

for s, (station_id, station_desc) in enumerate(catchments.items()):
    for i, time_subset in enumerate(time_subsets):

        ax = axes[s, i]

        ds_sel = xval_ds.sel(time=time_subset, station=station_id).compute()

        nse_mod = compute_metrics(obs=ds_sel.Qmm, mod=ds_sel.Qmm_mod, metrics=['nse']).nse.item()
        nse_prevah = compute_metrics(obs=ds_sel.Qmm, mod=ds_sel.Qmm_prevah, metrics=['nse']).nse.item()

        ax.plot(
            ds_sel.time,
            ds_sel.Qmm,
            color='k',
            lw=1.2,
            ls='--',
            zorder=10,
            label='observations'
        )
        ax.plot(
            ds_sel.time,
            ds_sel.Qmm_prevah,
            color='0.4',
            lw=1.2,
            # ls='--',
            label='PREVAH'
        )
        ax.plot(
            ds_sel.time,
            ds_sel.Qmm_mod,
            color='tab:red',
            lw=1.2,
            label=r'LSTM$_\mathrm{best}$'
        )

        if i == 0:
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylabel('Q (mm d$^{-1}$)')
            ax.text(0.01, 0.99, f'{["a)", "b)", "c)", "d)"][s]} {station_desc}',
                    ha='left', va='top', transform=ax.transAxes, size=9)

        else:
            ax.spines[['right', 'top', 'left']].set_visible(False)
            ax.tick_params('y', length=0, width=0, which='major')


        ax.text(
            0.01 if i == 0 else 0.99,
            0.8,
            f'LSTM$_\mathrm{{best}}$ NSE={nse_mod:0.2f}\nPREVAH NSE={nse_prevah:0.2f}',
            ha='left' if i == 0 else 'right',
            va='top',
            size=9,
            transform=ax.transAxes, zorder=9999)

        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

        fmt_month = mdates.MonthLocator()
        ax.xaxis.set_major_locator(fmt_month)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        if s == len(catchments) - 1:
            ax.set_xlabel(time_subset)
        else:
            ax.set_xlabel('')

axes[0, 1].legend(frameon=False, loc=2, bbox_to_anchor=(0.15, 1.0))

savefig(fig, path=PLOT_PATH / 'fig05.png')
