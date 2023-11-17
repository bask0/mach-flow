import optuna
import os

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
        fig = getattr(optuna.visualization, optuna_plot)(study)
        fig.write_image(os.path.join(out_dir, optuna_plot + '.png'))

def study_summary(study_path: str, study_name: str):

    base_dir = os.path.dirname(study_path)
    plot_dir = os.path.join(base_dir, 'optuna_plots')
    os.makedirs(plot_dir, exist_ok=True)

    study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{study_path}')

    study_plots(study=study, out_dir=plot_dir)
