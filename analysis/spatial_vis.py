
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.ticker as plticker
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
import contextily as cx

from utils.plotting import load_default_mpl_config, savefig
from utils.analysis import xval_station_metrics
from utils.shapefiles import get_shapefile
from config import get_path_dict

load_default_mpl_config()

paths = get_path_dict()

xval_ds = xval_station_metrics(
    paths['runs'],
    metrics=['nse', 'bias2', 'varerr', 'phaseerr'],
    time_slices=['1995,1999', '2016,2020'])


obs, obs_point = get_shapefile(source='obs', datapath=paths['data'])
obs = obs.set_index('OID')
obs_point = obs_point.set_index('OID')

obs = obs.loc[xval_ds.station]
obs_point = obs_point.loc[xval_ds.station]
prevah, _ = get_shapefile(source='prevah', datapath=paths['data'])

for var in xval_ds.data_vars:
    obs_point[var] = xval_ds[var].sel(run='LSTM-staticdred_allbasins_sqrttrans')
    obs_point[var + '_prevah'] = xval_ds[var].sel(run='prevah')
    obs_point[var + '_d'] = obs_point[var] - obs_point[var + '_prevah']


fig, axes = plt.subplots(2, 4, figsize=(12, 4.5), gridspec_kw={'wspace': 0})
nbins = 21

for i in range(2):
    for j, (ax, var, label) in enumerate(zip(
            axes[i, :],
            ['nse', 'bias2', 'varerr', 'phaseerr'],
            ['NSE', 'e$_\mathrm{bias}$', 'e$_\mathrm{variance}$', 'e$_\mathrm{phase}$'])):
        if (i > 0) or (j > 0):
            pass
        ax.plot(*prevah.unary_union.buffer(100).geoms[0].exterior.xy, color='k', zorder=0, lw=0.8)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.margins(0.04)

        base  = 770000
        total_len = 50000
        mid = base + 0.5 * total_len
        ax.plot([base, base + total_len], [100000, 100000], color='k', solid_capstyle='butt', lw=3)
        ax.plot([mid, base + total_len - 500], [100000, 100000], color='w', solid_capstyle='butt', lw=2)
        ax.text(mid, 94000, '50 km', va='top', ha='center', size=7)

        ax = obs.plot(
            ax=ax,
            legend=False,
            facecolor='k',
            edgecolor='none',
            alpha=0.2,
        )

        ax = obs.plot(
            ax=ax,
            legend=False,
            facecolor='none',
            edgecolor='k',
            lw=0.5,
        )

        cx.add_basemap(ax, source='Esri.WorldTerrain', crs=prevah.crs, zoom=9, zorder=-10, attribution_size=4)
        txt = ax.texts[-1]
        txt.set_position([0.01,0.98])
        txt.set_ha('left')
        txt.set_va('top')

        if i == 0:
            varname = var
            data = obs_point[var]
            vmin = 0
            vmax = data.quantile(0.95)
            if j == 0:
                cmap = 'viridis'
                loc = plticker.MultipleLocator(base=0.4)
            else:
                cmap = 'plasma_r'
                vmin = 0
                vmax = 6
                loc = plticker.MultipleLocator(base=2.5)

            label_ = label

        else:
            varname = var + '_d'
            data = obs_point[varname]
            vmin, vmax = data.quantile([0.05, 0.95])
            vabs = max(-vmin, vmax)
            vmin = -vabs
            vmax = vabs

            if j == 0:
                cmap = 'PiYG'
            else:
                cmap = 'RdBu_r'

            label_ = '$\Delta$' + label

        obs_point.plot(
            column=varname,
            legend=False,
            ax=ax,
            zorder=1,
            vmin=vmin,
            vmax=vmax,
            s=20,
            edgecolor='k',
            cmap=cmap,
            lw=0.8)

        # ax_inset = ax.inset_axes([0.05, 1, 0.4, 0.15])
        ax_inset = ax.inset_axes([0.05, 0, 0.4, 0.22])
        ax_inset.set_facecolor('none')

        # x = np.random.uniform(vmin, vmax, 1000)
        x = obs_point[varname]
        extra = (vmax - vmin) / (nbins - 1) / 2
        bins_bounds = np.linspace(vmin - extra, vmax + extra, nbins + 1)
        bin_values, bins = np.histogram(x, bins=bins_bounds)
        bin_centers = (bins_bounds[:-1] + bins_bounds[1:]) / 2
        bin_width = bin_centers[1] - bin_centers[0]

        cmap = mpl.colormaps[cmap]
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(bin_centers))
        ax_inset.hist(x, bins_bounds, histtype='step', color='w', lw=4, zorder=-5)
        ax_inset.hist(x, bins_bounds, histtype='step', color='k', lw=1)
        ax_inset.bar(bin_centers, bin_values, bin_width, color=colors)
        ax_inset.set_yticks([])
        ax_inset.set_yticklabels([])
        ax_inset.spines[['left', 'top', 'right']].set_visible(False)
        ax_inset.spines['bottom'].set_linewidth(1.5)
        ax_inset.tick_params(axis='x', which='major', pad=0., width=1.5)

        mx = 0
        mn = -bin_values.max() / 4
        bs = bins_bounds[-1] - bins_bounds[0]
        ax_inset.bar(bin_centers, mn, bin_width, color=colors)
        ax_inset.add_patch(
            Rectangle((bins_bounds[0], 0), bs, mn, ec='w', fc='none', lw=4, zorder=-5))
        ax_inset.add_patch(
            Rectangle((bins_bounds[0], 0), bs, mn, ec='k', fc='none', lw=1))
        ax_inset.set_xlabel(label_, labelpad=0.)

        median = x.median()
        # ms = (mx - mn) * 0.1
        # ax_inset.plot([median, median], [mn + ms, mx - ms], color='k', lw=3)
        # ax_inset.plot([median, median], [mn + ms * 2, mx - ms * 2], color='w', lw=1.3)
        ms = (mx - mn) * 0.15
        ax_inset.plot([median, median], [mn + ms, mx - ms], color='k', lw=4)
        ax_inset.plot([median, median], [mn + ms * 2, mx - ms * 2], color='w', lw=2.3)

        if i == 0:
            ax_inset.xaxis.set_major_locator(loc)

            if j == 0:
                ax_inset.annotate('median', 
                    xy=(median + bs * 0.015, (mx + mn) / 2), xycoords='data', color='black',
                    xytext=(bs * 1., mn * 6.5), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>',
                                    connectionstyle='arc3,rad=0.3',
                                    path_effects=[pe.withStroke(linewidth=3, foreground='w')],
                                    fc='w'))

                ax_inset.annotate('distribution', 
                    xy=(median + bs * 0.05, mx + 9 * ms), xycoords='data', color='black',
                    xytext=(bs * 1.3, mn * 4.2), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>',
                                    connectionstyle='arc3,rad=0.2',
                                    path_effects=[pe.withStroke(linewidth=3, foreground='w')],
                                    fc='w'))

savefig(fig, paths['figures'] / 'fig04.png', tight=True)
