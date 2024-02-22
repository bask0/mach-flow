import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from utils.plotting import savefig, load_default_mpl_config
from utils.shapefiles import get_shapefile

load_default_mpl_config()

PLOT_PATH = Path('/mydata/machflow/basil/mach-flow/analysis/catchment_vis/')


obs, obs_point = get_shapefile(source='obs')
prevah, _ = get_shapefile(source='prevah')


fig, ax = plt.subplots(figsize=(6, 4))

prevah_props = dict(
    facecolor='0.7',
    edgecolor='w',
    lw=0.2,   
)

ax = prevah.plot(
    ax=ax,
    legend=False,
    zorder=-1,
    **prevah_props
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

ax.axis('off')
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
ax.margins(0)

legend_elements = [
    Patch(**prevah_props, label='Reconstruction catchments'),
    Patch(**obs_props, label='Observational catchments'),
    Line2D([], [], marker='.', linestyle='None', **obs_point_props, label='Gauging stations')
]

ax.legend(handles=legend_elements, loc='upper left',  bbox_to_anchor=(0, 1.02),
          frameon=False, fontsize=8)


savefig(fig, PLOT_PATH / 'catchments.png', tight=True)
