import optuna
import os
import numpy as np 
import matplotlib.pyplot as plt
import xarray as xr

from utils.metrics import compute_metrics
from utils.data import load_xval_test_set


OPTUNA_PLOTS = [
    'plot_slice',
    'plot_contour',
    'plot_rank',
    'plot_intermediate_values',
    'plot_optimization_history',
    'plot_parallel_coordinate',
    'plot_param_importances',
    'plot_timeline',
]


def study_plots(study: optuna.Study, out_dir: str):
    for optuna_plot in OPTUNA_PLOTS:
        try: 
            fig = getattr(optuna.visualization, optuna_plot)(study)
            fig.write_image(os.path.join(out_dir, optuna_plot + '.png'), scale=5)
        except RuntimeError as e:
            print(
                'The following RuntimeError was raised (and ignored) for plot \'{optuna_plot}\':\n', e
            )
        except ValueError as e:
            print(
                'The following ValueError: was raised (and ignored) for plot \'{optuna_plot}\':\n', e
            )

def study_summary(study_path: str, study_name: str):

    base_dir = os.path.dirname(study_path)
    plot_dir = os.path.join(base_dir, 'optuna_plots')
    os.makedirs(plot_dir, exist_ok=True)

    study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{study_path}')

    study_plots(study=study, out_dir=plot_dir)


def get_cdf(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray, float]:
    count, bins = np.histogram(da, bins=len(da))
    bins = (bins[:-1] + bins[1:]) / 2
    pdf = count / sum(count) 
    cdf = np.cumsum(pdf)
    xloc = np.interp([0.5], cdf, bins)

    return bins, cdf, xloc.item()


def plot_cdf(
        ds: xr.Dataset,
        ds_ref: xr.Dataset | None = None,
        ours_name: str = 'ML (ours)',
        ref_name: str = 'PREVAH',
        save_path: str | None = None,
        col: str = '#1E88E5',
        ref_col: str = '#D81B60') -> None:
    metrics = list(ds.data_vars)

    num_metrics = len(metrics)

    has_ref = ds_ref is not None

    fig, axes = plt.subplots(
        2 if has_ref else 1, num_metrics, figsize=(3 * num_metrics, 4 + (has_ref * 2)), sharey='row', squeeze=False,
        gridspec_kw={'height_ratios': [10, 5]} if has_ref else {})

    annot_kwargs = dict(
        xytext=(-20,+20), 
        textcoords='offset points', ha='center', va='bottom', color='0.2',
        fontsize=9
    )

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        da = ds[metric]

        name = da.attrs.get('long_name', metric)

        bins, cdf, xloc = get_cdf(da)
        ax.plot(bins, cdf, label=ours_name, color=col, alpha=0.7)
        ax.axvline(xloc, ymin=0, ymax=0.5, color=col, ls=':', alpha=0.7)
        ax.annotate(
            f'{xloc:0.2f}', xy=(xloc, 0.5),
            bbox=dict(boxstyle='round,pad=0.2', fc=col, ec='none', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color=col, alpha=0.7),
            **annot_kwargs)

        if ds_ref is not None:
            da_ref = ds_ref[metric]
            bins_ref, cdf_ref, xloc_ref = get_cdf(da_ref)
            ax.plot(bins_ref, cdf_ref, label=ref_name, color=ref_col, alpha=0.7)
            ax.axvline(xloc_ref, ymin=0, ymax=0.5, color=ref_col, ls=':', alpha=0.7)
            ax.annotate(
                f'{xloc_ref:0.2f}', xy=(xloc_ref, 0.5),
                bbox=dict(boxstyle='round,pad=0.2', fc=ref_col, ec='none', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color=ref_col, alpha=0.7),
                **annot_kwargs)

        ax.axhline(0.5, ls=':', color='k', alpha=0.7)

        ax.set_xlabel(name)
        ax.spines[['right', 'top']].set_visible(False)

        if ds_ref is not None:
            ax = axes[1, i]
            ax.boxplot(da - da_ref,
                    vert=False,
                    medianprops=dict(linestyle=':', linewidth=1.2, color='k'),
                    flierprops=dict(marker='.'),
                    notch=True,
                    widths=0.4
            )
            # ax.axvline(0, color='k', ls=':', alpha=0.7)
            ax.set_xlabel(f'{name} difference')
            ax.spines[['right', 'top']].set_visible(False)

            xmin, xmax = ax.get_xlim()
            xmax_sym = max(np.abs(xmin), np.abs(xmax))
            n_span = 200
            xspan = np.linspace(0, xmax_sym, n_span)
            n0 = 0.7
            l = 0.12
            for i, (a, b) in enumerate(zip(xspan[:-1], xspan[1:])):
                if -a > xmin:
                    alp = n0 * np.exp(-l * i)
                    ax.axvspan(-b, -a, facecolor=ref_col, alpha=alp)

                if b < xmax:
                    alp = n0 * np.exp(-l * i)
                    ax.axvspan(a, b, facecolor=col, alpha=alp)

            if np.abs(xmin) > xmax:
                ax.text(
                    -(xmax- xmin) * 0.05, 1.4, f'{ref_name} better',
                    bbox=dict(boxstyle='larrow,pad=0.2',
                            fc=ref_col, ec='none', alpha=0.4),
                    color='0.2',
                    va='center', ha='right', fontsize=9)
            else:
                ax.text(
                    (xmax- xmin) * 0.05, 1.4, f'{ours_name} better',
                    bbox=dict(boxstyle='rarrow,pad=0.2',
                            fc=col, ec='none', alpha=0.4),
                    color='0.2',
                    va='center', ha='left', fontsize=9)

    axes[0, 0].set_ylabel('Cummulative probability')
    axes[0, 1].legend(frameon=False, fontsize=9)

    if has_ref:
        axes[1, 0].set_ylabel('')
        axes[1, 0].set_yticks([])

    fig.suptitle('Station-level model comparison')

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_xval_cdf(
        xval_dir: str,
        obs_name: str = 'Qmm',
        mod_name: str = 'Qmm_mod',
        ref_name: str | None = 'Qmm_prevah',
        save_path: str | None = None,
        subset: dict = {},
        **kwargs) -> None:

    if save_path is None:
        raise ValueError(
            'argument `save_path` cannot be None.'
        )

    ds = load_xval_test_set(xval_dir=xval_dir)
    ds = ds.sel(**subset)

    met = compute_metrics(metrics=['r', 'nse'], obs=ds[obs_name], mod=ds[mod_name], dim='time')

    if ref_name is None:
        met_ref = None
    else:
        met_ref = compute_metrics(metrics=['r', 'nse'], obs=ds[obs_name], mod=ds[ref_name], dim='time')

    plot_cdf(
        ds=met,
        ds_ref=met_ref,
        save_path=save_path,
        **kwargs
    )
