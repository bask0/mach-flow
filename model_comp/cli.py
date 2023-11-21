import optuna

from utils.tuning import SearchSpace, RunConfig


class CVSearchSpace(SearchSpace):
    """Defines the cross validation search space for iteration across folds."""
    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'data': {
                'class_path': 'model_comp.machflowdata.MachFlowDataModule',
                'init_args': {
                    'fold_nr': trial.suggest_categorical('fold_nr', [0, 1, 2, 3, 4, 5])
                }
            },
            # 'fold': trial.suggest_int('fold', low=0, high=5, step=1)
        }
        super().__init__(trial=trial, config=config)


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

    # HP tuning
    # ----------------------
    tuner = RunConfig(log_dir='runs', exp_name='model_comp', is_tune=True)
    tuner.register_search_spaces(
        lstm=LSTMSearchSpace
    )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=4)
    sampler = optuna.samplers.RandomSampler()
    study = tuner.get_study(pruner=pruner)

    study.optimize(
        tuner.get_objective(),
        n_trials=1,
        timeout=600)

    tuner.summarize_tuning()

    best_config_path, _ = tuner.get_best_config_and_ckpt(study)

    # Cross-validation
    # ----------------------
    xval = RunConfig(log_dir='runs', exp_name='model_comp', is_tune=False)
    xval.register_search_spaces(
        xval=CVSearchSpace
    )

    sampler = optuna.samplers.BruteForceSampler()
    study = xval.get_study(sampler=sampler)

    study.optimize(
        xval.get_objective(config_path=best_config_path),
        timeout=600)

    xval.summarize_tuning()

    xval.predict_trials(study)

