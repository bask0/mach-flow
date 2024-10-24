import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import contextily as cx

from utils.plotting import savefig, load_default_mpl_config
from utils.shapefiles import get_shapefile
from config import get_path_dict

load_default_mpl_config()

paths = get_path_dict()

obs, obs_point = get_shapefile(source='obs', datapath=paths['data'])
prevah, _ = get_shapefile(source='prevah', datapath=paths['data'])

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

leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(1)

# ax.plot([780000, 800000], [100000, 100000], color='k', solid_capstyle='butt', lw=3)
# ax.plot([790000, 799500], [100000, 100000], color='w', solid_capstyle='butt', lw=2)
# ax.text(790000, 96000, '20 km', va='top', ha='center', size=7)

base  = 770000
total_len = 50000
mid = base + 0.5 * total_len
ax.plot([base, base + total_len], [100000, 100000], color='k', solid_capstyle='butt', lw=3)
ax.plot([mid, base + total_len - 500], [100000, 100000], color='w', solid_capstyle='butt', lw=2)
ax.text(mid, 94000, '50 km', va='top', ha='center', size=7)

cx.add_basemap(ax, source='Esri.WorldTerrain', crs=prevah.crs, zoom=9, zorder=-10, attribution_size=5)

savefig(fig, paths['figures'] / 'fig01.png', tight=True)
