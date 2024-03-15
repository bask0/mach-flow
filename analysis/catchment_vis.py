import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import contextily as cx

from utils.plotting import savefig, load_default_mpl_config
from utils.shapefiles import get_shapefile

load_default_mpl_config()

PLOT_PATH = Path('/mydata/machflow/basil/mach-flow/analysis/figures/')


obs, obs_point = get_shapefile(source='obs')
prevah, _ = get_shapefile(source='prevah')


fig, ax = plt.subplots(figsize=(6, 4))

prevah_props = dict(
    facecolor='none',
    edgecolor='k',
    alpha=1.0,
    lw=0.4,   
)

ax = prevah.plot(
    ax=ax,
    legend=False,
    zorder=-2,
    facecolor='k',
    edgecolor='none',
    alpha=0.1,
    lw=0.0
)

ax = prevah.plot(
    ax=ax,
    legend=False,
    zorder=-1,
    facecolor='none',
    edgecolor='k',
    alpha=1.0,
    lw=0.5
)

obs_props = dict(
    facecolor='tab:pink',
    edgecolor='k',
    lw=0.3,
    alpha=0.4,
)

ax = obs[obs.is_train].plot(
    ax=ax,
    legend=False,
    **obs_props
)

obs_point_props = dict(
    color='tab:pink',
    markersize=5,
    edgecolor='k',
    lw=0.7,
)

ax = obs_point[obs_point.is_train].plot(
    ax=ax,
    legend=False,
    **obs_point_props
)

obs_point_props['markeredgecolor'] = obs_point_props.pop('edgecolor')

# ax.axis('off')
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
ax.margins(0.04)

legend_elements = [
    Patch(**prevah_props, label='Reconstruction catchments'),
    Patch(**obs_props, label='Observational catchments'),
    Line2D([], [], marker='.', linestyle='None', **obs_point_props, label='Gauging stations')
]

leg = ax.legend(handles=legend_elements, loc='upper left',  bbox_to_anchor=(0, 1),
          frameon=True, fancybox=False, fontsize=7)

leg.get_frame().set_edgecolor('b')
leg.get_frame().set_linewidth(0.0)

cx.add_basemap(ax, source='Esri.WorldTerrain', crs=prevah.crs, zoom=9, zorder=-10, attribution_size=5)

savefig(fig, PLOT_PATH / 'fig01.png', tight=True)
