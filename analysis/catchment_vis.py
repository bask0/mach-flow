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


fig, ax = plt.subplots(figsize=(8, 5))

ax = obs[obs.is_train].plot(
    ax=ax,
    column='Shape_Area',
    legend=True,
    edgecolor='0.9',
    lw=0.5,
    alpha=1.0,
    vmax=500,
    legend_kwds={
        'label': 'Catchment area (km$^2$)',
        'orientation': 'horizontal',
        'aspect': 30,
        'shrink': 0.7,
        'pad': 0.01
    }
)

obs.plot(
    ax=ax,
    facecolor='0.7',
    edgecolor='0.9',
    lw=0.5,
    zorder=-1
)

ax.axis('off')
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
ax.margins(0)
ax.set_title('Observational catchments with low human impact')

savefig(fig, PLOT_PATH / 'training_catchments.png')


fig, ax = plt.subplots(figsize=(8, 5))

ax = prevah.plot(
    ax=ax,
    column='Shape_Area',
    legend=True,
    facecolor='tab:blue',
    edgecolor='0.9',
    lw=0.5,
    alpha=1.0,
    vmax=500,
    legend_kwds={
        'label': 'Catchment area (km$^2$)',
        'orientation': 'horizontal',
        'aspect': 30,
        'shrink': 0.7,
        'pad': 0.01
    }
)

ax.axis('off')
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
ax.margins(0)
ax.set_title('PREVAH catchments')

savefig(fig, PLOT_PATH / 'prevah_catchments.png')
