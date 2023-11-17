import optuna

from utils.tuning import SearchSpace, TuneConfig


class LSTMSearchSpace(SearchSpace):
    """Defines the search space for the LSTM model."""
    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'model': {
                'class_path': 'model_comp.lstm.LSTM',
                'init_args': {
                    'num_enc': trial.suggest_int('num_enc', 2, 10, step=2)
                }
            },
            'optimizer': {
                'class_path': 'torch.optim.AdamW',
                'init_args': {
                    'lr': trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2]),
                    'weight_decay': trial.suggest_categorical('weight_decay', [0, 1e-4, 1e-2]),
                }
            }
        }
        super().__init__(trial=trial, config=config)


if __name__ == '__main__':

    tuner = TuneConfig(log_dir='runs', exp_name='model_comp')
    tuner.register_search_spaces(
        lstm=LSTMSearchSpace
    )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=4) if True else optuna.pruners.No()
    study = tuner.get_study(pruner=pruner)

    study.optimize(
        tuner.get_objective(),
        n_trials=2,
        timeout=600)

    tuner.predict_with_best_model(study)

    tuner.summarize_tuning()
