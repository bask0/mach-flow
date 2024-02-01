import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from utils.plotting import savefig, load_default_mpl_config
from utils.shapefiles import get_shapefile

load_default_mpl_config()

PLOT_PATH = Path('/mydata/machflow/basil/mach-flow/analysis/catchment_vis/')


obs = get_shapefile(source='obs')
prevah = get_shapefile(source='prevah')


fig, axes = plt.subplots(1, 2, figsize=(7, 3))

ax = obs[obs.is_train].plot(
    ax=axes[0],
    column='Shape_Area',
    legend=False,
    edgecolor='0.9',
    cmap='RdPu_r',
    lw=0.3,
    alpha=1.0,
    vmin=0,
    vmax=500,
    # legend_kwds={
    #     'label': 'Catchment area (km$^2$)',
    #     'orientation': 'horizontal',
    #     'aspect': 30,
    #     'shrink': 0.7,
    #     'pad': 0.01
    # }
)

ax = prevah.plot(
    ax=ax,
    legend=True,
    facecolor='0.7',
    edgecolor='none',
    lw=0.0,
    zorder=-1
)


ax = prevah.plot(
    ax=axes[1],
    column='Shape_Area',
    legend=False,
    facecolor='tab:blue',
    edgecolor='0.9',
    cmap='RdPu_r',
    lw=0.3,
    alpha=1.0,
    vmin=0,
    vmax=500,
    # legend_kwds={
    #     'label': 'Catchment area (km$^2$)',
    #     'orientation': 'horizontal',
    #     'aspect': 30,
    #     'shrink': 0.7,
    #     'pad': 0.01
    # }
)

for label, ax in zip(['a)', 'b)'], axes):
    ax.axis('off')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.margins(0)
    ax.text(0.12, 0.72, label, ha='left', va='top', transform=ax.transAxes)

cax = fig.add_axes((0.42, 0.73, 0.12, 0.02))
sm = plt.cm.ScalarMappable(cmap='RdPu_r', norm=plt.Normalize(vmin=0, vmax=500))
sm._A = []
cbar = fig.colorbar(
    sm,
    cax=cax,
    label='',
    orientation='horizontal',
    aspect=30,
    shrink=0.7,
    pad=0.01
)
cbar.ax.set_title('Catchment area (km$^2$)', size=9)

savefig(fig, PLOT_PATH / 'catchments.png', tight=True)
