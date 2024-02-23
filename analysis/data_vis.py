

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from utils.data import read_rds
from utils.plotting import load_default_mpl_config, savefig

PLOT_PATH = Path('/home/basil/mach-flow/analysis/helper_figures/')

load_default_mpl_config()

ds0 = read_rds('/data/basil/static_harmonized/prevah_307/harmonized.rds')
ds1 = read_rds('/data/basil/static_harmonized/obs/harmonized.rds')
ds = pd.concat((ds0, ds1)).drop(columns=['exp', 'mach_ID', 'OBJECTID'])


# Static inputs PCA
# ------------------------------------------------------------

stat_arr = ds.values
stat_arr_norm = (stat_arr - stat_arr.mean(0, keepdims=True)) / stat_arr.std(0, keepdims=True)

pca = PCA(
    n_components=None,
)
pca.fit(stat_arr_norm)
X_test_pca = pca.transform(stat_arr)
n_components = X_test_pca.shape[-1]

exp_var = pca.explained_variance_ratio_.cumsum() * 100
min_at = np.argwhere(exp_var > 95)[0][0]

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(exp_var)
ax.axvline(min_at, color='k', ls=':')
ax.set_title(f'PCA explained variance. Cut at PC{min_at}>95 ({exp_var[min_at]:0.1f})')
ax.set_xlabel('Principal component #')
ax.set_ylabel('Cummulative explained variance')

savefig(fig, PLOT_PATH / 'stat_pca.png')

# Static inputs Isomap
# ------------------------------------------------------------

n_neigh = range(8, 13)
n_comp = range(2, 20)

errors = np.full((len(n_neigh), len(n_comp)), np.nan)

for i, n in tqdm(enumerate(n_neigh), total=len(n_neigh), ncols=120, desc='Computing Isomaps'):
    for j, c in enumerate(n_comp):
        isomap = Isomap(n_neighbors=n, n_components=c)
        X_reduced = isomap.fit_transform(stat_arr_norm)
        errors[i, j] = isomap.reconstruction_error()

fig, ax = plt.subplots(figsize=(8, 8))

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

fig, ax = plt.subplots(figsize=(8, 6))

errors_t = ((errors - errors.min()) / (errors.max() - errors.min())) * 100
levels = np.linspace(0, 100, 21)

im = ax.imshow(errors_t, aspect='auto', interpolation='none', cmap='PiYG_r')
cs = ax.contour(errors_t, color='k', levels=levels)

ax.clabel(cs, cs.levels, fmt=fmt, inline=True, fontsize=10)

cbar = fig.colorbar(im)
cbar.set_label('Reconstruction error (min-max scaled)')
ax.set_xlabel('n_comp')
ax.set_ylabel('n_neigh')
ax.set_xticks(np.arange(len(n_comp)), labels=n_comp)
ax.set_yticks(np.arange(len(n_neigh)), labels=n_neigh);
savefig(fig, PLOT_PATH / 'stat_isomap.png')
